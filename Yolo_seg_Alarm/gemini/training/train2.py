import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import os
import sys
import argparse
import sys
import json
import logging
import time
import threading
from datetime import datetime
from tqdm import tqdm
# 强制添加命令行参数确保任务类型为检测
if '--task' not in sys.argv:
    sys.argv.extend(['--task', 'detect'])
# 配置日志
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# 创建线程锁
progress_lock = threading.Lock()
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.environment import get_device_and_adjust_config

# 获取进度文件路径
def get_progress_file_path():
    current_dir = Path(__file__).parent
    progress_dir = current_dir / 'progress'
    progress_dir.mkdir(exist_ok=True)
    return progress_dir / 'train_progress.json'

# 加载进度
def load_progress():
    progress_file = get_progress_file_path()
    default_progress = {
        'last_epoch': 0,
        'model_path': None,
        'stats': {'epochs_completed': 0, 'time': 0, 'best_metrics': None}
    }
    if progress_file.exists() and progress_file.stat().st_size > 0:
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'last_epoch': data.get('last_epoch', 0),
                    'model_path': data.get('model_path'),
                    'stats': data.get('stats', default_progress['stats'])
                }
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"进度文件读取失败: {e}")
    return default_progress

# 保存进度
def save_progress(last_epoch, model_path, stats):
    progress_file = get_progress_file_path()
    progress_data = {
        'last_epoch': last_epoch,
        'model_path': model_path,
        'stats': stats
    }
    try:
        with progress_lock:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        logging.info(f"已保存进度至: {progress_file}")
    except IOError as e:
        logging.error(f"进度文件写入失败: {e}")

# 1. 加载YAML配置文件
def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 2. 准备数据集路径
def prepare_data_paths(config):
    base_path = Path(config['path'])
    
    # 创建标签目录的符号链接
    for split in ['train', 'val']:
        # 原始图片目录
        img_dir = base_path / split / 'images'
        # 创建YOLO期望的标签目录
        label_dir = base_path / split / 'labels'
        label_dir.mkdir(exist_ok=True)
        
        # 为每个大屏编号子文件夹创建符号链接
        for screen_folder in img_dir.iterdir():
            if screen_folder.is_dir():
                # 标签目录中的链接指向图片目录中的对应子文件夹
                link_path = label_dir / screen_folder.name
                if not link_path.exists():
                    # 创建目录符号链接
                    os.symlink(screen_folder, link_path, target_is_directory=True)
    
    # 设置训练集和验证集路径
    train_path = base_path / 'train' / 'images'
    val_path = base_path / 'val' / 'images'
    
    print(f"最终训练集路径: {train_path}")
    print(f"最终验证集路径: {val_path}")
    
    # 验证路径是否存在
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {train_path} 或 {val_path}")
    
    return train_path, val_path

# 保存模型并更新进度
def save_model_and_progress(model, epoch, total_epochs, start_time, stats):
    # 保存当前模型
    model_path = f"intermediate_model_epoch_{epoch}.pt"
    model.save(model_path)
    logging.info(f"已保存中间模型: {model_path}")
    
    # 更新统计信息
    stats['epochs_completed'] = epoch
    stats['time'] = time.time() - start_time
    
    # 尝试获取最佳指标
    try:
        if hasattr(model, 'best') and model.best:
            stats['best_metrics'] = model.best.tojson()
    except Exception as e:
        logging.warning(f"无法获取最佳指标: {e}")
    
    # 保存进度
    progress_file = get_progress_file_path()
    save_progress(progress_file, epoch, model_path, stats)
    
    return model_path

# 保存模型并更新进度
def save_model_and_progress(model, epoch, total_epochs, start_time, stats, is_final=False):
    # 保存当前模型
    if is_final:
        model_path = 'yolo_seg_alarm_train2.pt'
    else:
        model_path = f"intermediate_model_epoch_{epoch}.pt"
    model.save(model_path)
    logging.info(f"已保存{'最终' if is_final else '中间'}模型: {model_path}")
    
    # 更新统计信息
    stats['epochs_completed'] = epoch
    stats['time'] = time.time() - start_time
    
    # 尝试获取最佳指标
    try:
        if hasattr(model, 'best') and model.best:
            stats['best_metrics'] = model.best.tojson()
            logging.info(f"当前最佳指标: {model.best}")
    except Exception as e:
        logging.warning(f"无法获取最佳指标: {e}")
    
    # 保存进度
    save_progress(epoch, model_path, stats)
    
    return model_path

# 主训练函数
def main():
    try:
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
        
        # 根据硬件调整配置
        device, config = get_device_and_adjust_config(config)
        
        # 准备数据集路径
        train_path, val_path = prepare_data_paths(config)
        config['train'] = train_path
        config['val'] = val_path
        
        # 解析命令行参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='detect', help='任务类型')
        parser.add_argument('--save_interval', type=int, default=5, help='每隔多少个epoch保存一次进度')
        parser.add_argument('--resume', action='store_true', help='是否从上次进度继续训练')
        args = parser.parse_args()
        
        # 在模型加载代码前强制设置任务类型
        args.task = 'detect'
        
        # 加载进度
        progress = load_progress()
        last_epoch = progress['last_epoch']
        saved_model_path = progress['model_path']
        stats = progress['stats']
        
        # 询问用户是否继续上次训练
        if saved_model_path and Path(saved_model_path).exists() and last_epoch > 0 and args.resume:
            logging.info(f"发现上次训练进度，从epoch {last_epoch+1} 开始")
            model = YOLO(saved_model_path, task='detect')
            # 设置从上次中断的epoch继续训练
            config['epochs'] = config['epochs'] - last_epoch
            logging.info(f"剩余训练epoch数: {config['epochs']}")
        else:
            model_path = Path(config['model_path']) / config['weights']
            # 显式指定检测任务类型加载模型
            model = YOLO(str(model_path), task='detect')
            last_epoch = 0
            stats = {'epochs_completed': 0, 'time': 0, 'best_metrics': None}
            logging.info("开始新的训练")
        
        total_epochs = config['epochs']
        start_time = time.time() - stats.get('time', 0)
        
        # 训练模型
        logging.info(f"开始训练，共 {total_epochs} 个epoch")
        
        # 使用tqdm创建进度条
        with tqdm(total=total_epochs, initial=0, desc="训练进度") as pbar:
            # 使用Ultralytics的训练函数
            results = model.train(
                task=args.task,
                data=config_path,
                epochs=total_epochs,
                batch=config['batch'],
                imgsz=config['imgsz'],
                device=device.type,
                project='yolo_seg_alarm',
                name='train2_results',
                exist_ok=True,
                resume=last_epoch > 0,
                # 添加回调函数保存进度
                callbacks=[lambda epoch, model=model, start_time=start_time, stats=stats, args=args: 
                           save_model_and_progress(model, last_epoch + epoch + 1, last_epoch + total_epochs, start_time, stats) 
                           if (epoch + 1) % args.save_interval == 0 else None]
            )
            pbar.update(total_epochs)
        
        # 保存最终模型
        save_model_and_progress(model, last_epoch + total_epochs, last_epoch + total_epochs, start_time, stats, is_final=True)
        
        # 清理临时文件
        import shutil
        for split in ['train', 'val']:
            label_dir = base_path / split / 'labels'
            if label_dir.exists():
                shutil.rmtree(label_dir)
        logging.info("训练完成，模型已保存并清理临时文件")
        
        # 训练完成后重置进度文件
        save_progress(0, None, {'epochs_completed': 0, 'time': 0, 'best_metrics': None})
    
    except KeyboardInterrupt:
        logging.info("检测到Ctrl+C中断，正在保存进度...")
        try:
            # 尝试保存当前模型和进度
            if 'model' in locals() and 'last_epoch' in locals() and 'total_epochs' in locals() and 'start_time' in locals() and 'stats' in locals():
                current_epoch = last_epoch + (time.time() - start_time) / (stats.get('time', time.time() - start_time) / max(1, stats.get('epochs_completed', 1)))
                current_epoch = int(round(current_epoch))
                save_model_and_progress(model, current_epoch, last_epoch + total_epochs, start_time, stats)
        except Exception as e:
            logging.error(f"保存进度时出错: {e}")
        logging.info("进度已保存，程序已终止")
    
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        try:
            # 尝试保存当前进度
            if 'model' in locals() and 'last_epoch' in locals() and 'stats' in locals():
                save_progress(last_epoch + stats.get('epochs_completed', 0), saved_model_path, stats)
        except Exception as save_e:
            logging.error(f"保存进度时出错: {save_e}")
        raise
    
    print("训练完成，模型已保存并清理临时文件")

# 确保导入time模块
import time


# 确保导入time模块
import time


if __name__ == '__main__':
    main()
