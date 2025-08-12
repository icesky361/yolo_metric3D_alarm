import yaml
import torch
import json
import time
import shutil
from pathlib import Path
from ultralytics import YOLO
import os
import sys
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.environment import get_device_and_adjust_config

# 进度保存目录
PROGRESS_DIR = Path(__file__).parent / 'progress'
PROGRESS_FILE = PROGRESS_DIR / 'train_progress.json'

# 创建进度目录
PROGRESS_DIR.mkdir(exist_ok=True)

# 1. 加载YAML配置文件
def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 2. 准备数据集路径
def prepare_data_paths(config):
    base_path = Path(config['path'])

    # 设置训练集和验证集路径（标签文件与图片同目录）
    train_path = base_path / 'train' / 'images'
    val_path = base_path / 'val' / 'images'

    print(f"最终训练集路径: {train_path}")
    print(f"最终验证集路径: {val_path}")

    # 验证路径是否存在
    if not train_path.exists() or not val_path.exists():
         raise FileNotFoundError(f"数据集路径不存在: {train_path} 或 {val_path}")

    return train_path, val_path

# 保存训练进度
# 修改save_progress调用条件
def save_progress(epoch, model, results, config):
    # 每轮都保存进度信息
    progress = {
        'epoch': epoch,
        'epochs_completed': epoch,
        'time': time.time(),
        'good_epoch': results.good_epoch,
        'good_map': float(results.good_map),
        'last_epoch': results.epoch,
        'timestamp': datetime.now().isoformat(),
        'last_weights': str(PROGRESS_DIR / f'checkpoint_epoch{epoch}.pt'),
        'good_weights': str(PROGRESS_DIR / f'good_epoch{epoch}.pt')
    }

    # 保存进度文件
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    # 保存中间模型
    checkpoint_path = PROGRESS_DIR / f'checkpoint_epoch_{epoch}.pt'
    # 保存当前权重
    model.save(PROGRESS_DIR / f'checkpoint_epoch{epoch}.pt')
    # 保存最佳权重
    shutil.copyfile(results.good, str(PROGRESS_DIR / f'good_epoch{epoch}.pt'))
    # 同时更新根目录下的权重文件
    final_model_path = Path(config['model_path']) / 'weights' / 'yolo_seg_alarm_train3_final.pt'
    good_model_path = Path(config['model_path']) / 'weights' / 'yolo_seg_alarm_train3_good.pt'
    model.save(str(final_model_path))
    shutil.copyfile(results.good, str(good_model_path))
    print(f"已保存第 {epoch} 轮进度和模型到 {PROGRESS_DIR} 和根目录")

# 加载训练进度
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# 主训练函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='detect', help='任务类型')
    parser.add_argument('--resume', action='store_true', help='是否从上次进度继续训练')
    args = parser.parse_args()
    args.task = 'detect'  # 强制设置为检测任务

    # 加载配置
    config_path = Path(__file__).parent.parent / 'configs' / 'yolov11_seg.yaml'
    config = load_config(config_path)

    # 动态计算数据集路径
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / 'Data' / 'raw'
    config['path'] = str(data_path)
    base_path = Path(config['path'])

    # 根据硬件调整配置
    device, config = get_device_and_adjust_config(config)

    # 准备数据集路径
    train_path, val_path = prepare_data_paths(config)
    config['train'] = train_path
    config['val'] = val_path

    # 创建图表保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    charts_dir = Path(config['model_path']) / 'weights' / f'train3_{timestamp}'
    charts_dir.mkdir(exist_ok=True, parents=True)

    # 加载模型
    model_path = Path(config['model_path']) / config['weights']
    model = YOLO(str(model_path), task='detect')

    # 检查是否需要从进度继续
    start_epoch = 0
    if args.resume:
        progress = load_progress()
        if progress:
            start_epoch = progress['epoch'] + 1
            checkpoint_path = PROGRESS_DIR / f'checkpoint_epoch_{progress["epoch"]}.pt'
            if checkpoint_path.exists():
                model = YOLO(str(checkpoint_path), task='detect')
                print(f"已从第 {progress['epoch']} 轮继续训练")
            else:
                print(f"未找到 checkpoint 文件，将从头开始训练")
        else:
            print("未找到进度文件，将从头开始训练")

    # 训练模型
    results = model.train(
        task=args.task,
        data=config_path,
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        device=device.type,
        project=str(charts_dir.parent),
        name=charts_dir.name,
        exist_ok=True,
        resume=args.resume
    )

    # 保存训练后的模型
    final_model_path = Path(config['model_path']) / 'weights' / 'yolo_seg_alarm_train3_final.pt'
    model.save(str(final_model_path))
    print(f"最终模型已保存到 {final_model_path}")

    # 保存最佳模型
    good_model_path = Path(config['model_path']) / 'weights' / 'yolo_seg_alarm_train3_good.pt'
    shutil.copyfile(results.good, str(good_model_path))
    print(f"最佳模型已保存到 {good_model_path}")



    print("训练完成，已清理临时文件")

if __name__ == '__main__':
    main()