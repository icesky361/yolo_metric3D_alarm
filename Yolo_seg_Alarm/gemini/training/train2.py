import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import os
import sys
import argparse
import json
from datetime import datetime
import time
import threading
# 强制添加命令行参数确保任务类型为检测
if '--task' not in sys.argv:
    sys.argv.extend(['--task', 'detect'])
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.environment import get_device_and_adjust_config

# 创建一个线程锁，用于安全地写入进度文件
progress_lock = threading.Lock()

# 获取进度文件路径
def get_progress_file_path():
    current_dir = Path(__file__).parent
    progress_dir = current_dir / 'progress'
    progress_dir.mkdir(exist_ok=True)
    return progress_dir / 'train_progress.json'

# 加载训练进度
def load_progress(progress_file):
    default_progress = {
        'last_epoch': 0,
        'model_path': None,
        'stats': {'epochs_completed': 0, 'time': 0}
    }
    if progress_file.exists() and progress_file.stat().st_size > 0:
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"进度文件读取失败，使用默认值: {e}")
    return default_progress

# 保存训练进度
def save_progress(progress_file, epoch, model_path, stats):
    progress_data = {
        'last_epoch': epoch,
        'model_path': str(model_path),
        'stats': stats
    }
    try:
        with progress_lock:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        print(f"进度已保存: 第 {epoch} 个 epoch 完成")
    except IOError as e:
        print(f"进度文件写入失败: {e}")

# 1. 加载YAML配置文件
def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 2. 准备数据集路径
def prepare_data_paths(config):
    base_path = Path(config['path'])
    
    # 创建标签目录并复制标签文件
    for split in ['train', 'val']:
        # 原始图片目录
        img_dir = base_path / split / 'images'
        # 创建YOLO期望的标签目录
        label_dir = base_path / split / 'labels'
        label_dir.mkdir(exist_ok=True)
        
        # 复制标签文件
        for screen_folder in img_dir.iterdir():
            if screen_folder.is_dir():
                # 目标标签文件夹
                target_label_folder = label_dir / screen_folder.name
                target_label_folder.mkdir(exist_ok=True)
                
                # 复制标签文件
                for img_file in screen_folder.glob('*.*'):
                    if img_file.suffix in ['.jpg', '.png']:
                        label_file = img_file.with_suffix('.txt')
                        if label_file.exists():
                            import shutil
                            shutil.copy2(label_file, target_label_folder / label_file.name)
    
    # 设置训练集和验证集路径
    train_path = base_path / 'train' / 'images'
    val_path = base_path / 'val' / 'images'
    
    print(f"最终训练集路径: {train_path}")
    print(f"最终验证集路径: {val_path}")
    
    # 验证路径是否存在
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {train_path} 或 {val_path}")
    
    return train_path, val_path

# 主训练函数
def main():
    # 定义一个在退出前强制保存进度的函数
    def final_save(epoch, model_path, stats):
        progress_file = get_progress_file_path()
        save_progress(progress_file, epoch, model_path, stats)
        print(f"在退出前，强制保存了第 {epoch} 个 epoch 的进度")

    # 加载配置
    config_path = Path(__file__).parent.parent / 'configs' / 'yolov11_seg.yaml'
    config = load_config(config_path)
    # 动态计算数据集路径
    # 从train2.py位置计算项目根目录
    project_root = Path(__file__).resolve().parents[3]  # 向上三级到Yolo_metric_alarm
    data_path = project_root / 'Data' / 'raw'
    config['path'] = str(data_path)
    # 计算base_path并保存为变量（关键修改）
    base_path = Path(config['path'])

    # 断点续训相关
    progress_file = get_progress_file_path()
    progress = load_progress(progress_file)
    last_epoch = progress['last_epoch']
    resume_model_path = progress['model_path']

    # 询问用户是否继续上次训练
    if last_epoch > 0 and resume_model_path and Path(resume_model_path).exists():
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        user_choice = messagebox.askyesno(
            title="发现未完成的训练",
            message=f"检测到存在未完成的训练进度（已完成 {last_epoch} 个 epoch）。\n\n是否要继续上次的训练？\n\n- 选择【是】将从第 {last_epoch + 1} 个 epoch 继续。\n- 选择【否】将开始一个全新的训练，并清空旧进度。"
        )
        if user_choice:
            print(f"继续上次训练，从第 {last_epoch + 1} 个 epoch 开始")
            config['resume'] = True
            config['model_path'] = resume_model_path
            config['start_epoch'] = last_epoch + 1
        else:
            print("用户选择开始新训练，清空旧进度...")
            save_progress(progress_file, 0, None, {'epochs_completed': 0, 'time': 0})
            last_epoch = 0
            resume_model_path = None
    else:
        print("未检测到可继续的训练进度，开始新训练")
    
    # 根据硬件调整配置
    device, config = get_device_and_adjust_config(config)
    
    # 准备数据集路径
    train_path, val_path = prepare_data_paths(config)
    config['train'] = train_path
    config['val'] = val_path
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='detect', help='任务类型')
    args = parser.parse_args()
    
    # 在模型加载代码前强制设置任务类型
    args.task = 'detect'
    
    # 加载模型
    model_path = Path(config['model_path']) / config['weights']
    # 显式指定检测任务类型加载模型
    model = YOLO(str(model_path), task='detect')
    # 调整训练参数以支持断点续训
    train_args = {
        'task': args.task,
        'data': config_path,
        'epochs': config['epochs'],
        'batch': config['batch'],
        'imgsz': config['imgsz'],
        'device': device.type,
        'project': 'yolo_seg_alarm',
        'name': 'train2_results',
        'exist_ok': True
    }

    # 如果是续训，设置相关参数
    if config.get('resume', False) and resume_model_path:
        train_args['resume'] = True
        train_args['epochs'] = config['epochs']  # 总epochs保持不变
        # 加载最近的模型
        model = YOLO(resume_model_path, task='detect')
        print(f"已加载模型: {resume_model_path}")

    # 记录训练开始时间
    train_start_time = time.time()
    total_epochs = config['epochs']
    stats = {'epochs_completed': last_epoch, 'time': 0}

    try:
        # 训练模型
        results = model.train(**train_args)

        # 训练完成后更新统计
        stats['epochs_completed'] = total_epochs
        stats['time'] = time.time() - train_start_time
        final_save(total_epochs, 'yolo_seg_alarm_train2.pt', stats)

    except KeyboardInterrupt:
        print("\n检测到用户中断 (Ctrl+C)，正在保存当前进度...")
        # 尝试获取当前训练的epoch数
        current_epoch = last_epoch + 1  # 假设至少完成了一个新epoch
        # 保存中断时的模型
        temp_model_path = 'interrupted_model.pt'
        model.save(temp_model_path)
        stats['epochs_completed'] = current_epoch
        stats['time'] = time.time() - train_start_time
        final_save(current_epoch, temp_model_path, stats)
        print("进度已保存，程序即将退出。")
    except Exception as e:
        error_msg = str(e)
        print(f"发生意外错误: {error_msg}")
        
        # 特别处理数据解析错误
        if 'tokenizing data' in error_msg.lower() or 'expected' in error_msg.lower() and 'fields' in error_msg.lower():
            print("\n错误分析: 这看起来是一个数据解析错误，可能是标签文件格式不正确或路径问题。")
            print("建议检查数据集目录下的标签文件格式和路径是否正确。")
            print(f"错误详情: {error_msg}")
        
        # 尝试保存进度
        current_epoch = last_epoch
        if 'model' in locals():
            temp_model_path = 'error_model.pt'
            model.save(temp_model_path)
            final_save(current_epoch, temp_model_path, stats)
        else:
            final_save(current_epoch, None, stats)
    
    # 保存训练后的模型
    model.save('yolo_seg_alarm_train2.pt')
    print("训练完成，模型已保存")

    # 训练完成后删除进度文件或重置进度
    progress_file = get_progress_file_path()
    if progress_file.exists():
        save_progress(progress_file, 0, None, {'epochs_completed': 0, 'time': 0})
        print("训练完全完成，已重置进度文件")

if __name__ == '__main__':
    main()
