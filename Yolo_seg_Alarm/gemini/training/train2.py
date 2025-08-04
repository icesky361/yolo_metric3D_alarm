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
# 设置matplotlib字体支持中文
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，避免在无GUI环境下报错
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
# 设置中文字体,已配置为用户提供的正确路径
font_path = 'G:/soft/soft/anaconda/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
else:
    logging.warning(f"字体文件未找到于 {font_path}，中文可能无法正确显示。")
    # 退而求其次，使用系统可用的中文字体
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]

# 禁用matplotlib字体警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
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
    return progress_file

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
def save_model_and_progress(model, epoch, total_epochs, start_time, stats, is_final=False):
    # 定义模型保存目录
    model_dir = Path(__file__).parent.parent / 'models' / 'weights'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义训练结果目录
    results_dir = Path('yolo_seg_alarm') / 'train2_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存当前模型
    if is_final:
        model_path = model_dir / 'last.pt'
    else:
        model_path = results_dir / f"intermediate_model_epoch_{epoch}.pt"
    
    model.save(str(model_path))
    logging.info(f"已保存{'最终' if is_final else '中间'}模型: {model_path}")
    
    # 如果是最终模型，同时保存最佳模型和指定名称模型
    if is_final:
        # 保存为指定名称的最终模型
        final_model_path = model_dir / 'yolo_seg_alarm_train2.pt'
        model.save(str(final_model_path))
        logging.info(f"已保存最终模型: {final_model_path}")
        
        # 如果有最佳模型，保存最佳模型
        if hasattr(model, 'best') and model.best:
            best_model_path = model_dir / 'best.pt'
            model.save(str(best_model_path))
            logging.info(f"已保存最佳模型: {best_model_path}")
    
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
    save_progress(epoch, str(model_path), stats)
    
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
        original_epochs = config.get('epochs', 10)
        if saved_model_path and Path(saved_model_path).exists() and last_epoch > 0 and args.resume:
            logging.info(f"发现上次训练进度，从epoch {last_epoch+1} 开始")
            model = YOLO(saved_model_path, task='detect')
            # 设置从上次中断的epoch继续训练
            remaining_epochs = original_epochs - last_epoch
            config['epochs'] = remaining_epochs
            logging.info(f"总训练epoch数: {original_epochs}")
            logging.info(f"已完成epoch数: {last_epoch}")
            logging.info(f"剩余训练epoch数: {remaining_epochs}")
        else:
            model_path = Path(config['model_path']) / config['weights']
            # 显式指定检测任务类型加载模型
            model = YOLO(str(model_path), task='detect')
            last_epoch = 0
            stats = {'epochs_completed': 0, 'time': 0, 'best_metrics': None}
            logging.info("开始新的训练")
        
        # 确保从配置文件中正确读取epochs值
        config_epochs = config.get('epochs', 10)
        # 如果是继续训练，已经在上面调整过epochs值
        if last_epoch > 0 and args.resume:
            total_epochs = config['epochs']
        else:
            total_epochs = config_epochs
        
        # 确保epochs是整数且大于0
        if not isinstance(total_epochs, int) or total_epochs <= 0:
            total_epochs = 10  # 默认值
            logging.warning(f"配置文件中epochs值无效，使用默认值: {total_epochs}")
        
        # 记录最终使用的epochs值
        logging.info(f"最终训练总epoch数: {total_epochs}")
        start_time = time.time() - stats.get('time', 0)
        
        # 验证配置文件中的epochs值
        config_epochs_value = config.get('epochs')
        logging.info(f"从配置文件读取的epochs值: {config_epochs_value}")
        logging.info(f"配置文件路径: {config_path}")
        
        # 训练模型
        logging.info(f"开始训练，共 {total_epochs} 个epoch")
        if last_epoch > 0 and args.resume:
            logging.info(f"训练范围: 从epoch {last_epoch+1} 到 epoch {last_epoch + total_epochs}")
        
        # 配置训练回调函数用于进度保存
        class ProgressCallback:
            def __init__(self, model, total_epochs, last_epoch, start_time, stats, save_interval):
                self.model = model
                self.total_epochs = total_epochs
                self.last_epoch = last_epoch
                self.start_time = start_time
                self.stats = stats
                self.save_interval = save_interval
                
            def on_epoch_end(self, epoch):
                current_epoch = self.last_epoch + epoch + 1
                # 定期保存进度和模型
                if (epoch + 1) % self.save_interval == 0 or epoch == self.total_epochs - 1:
                    save_model_and_progress(self.model, current_epoch, self.last_epoch + self.total_epochs, self.start_time, self.stats)
                
        # 创建回调实例
        callback = ProgressCallback(model, total_epochs, last_epoch, start_time, stats, args.save_interval)
        # 注册回调函数
        model.add_callback('on_epoch_end', callback.on_epoch_end)
        
        try:
            # 使用YOLO内置的多epoch训练功能
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
                    plots=True,
                    save_json=True
                )
        except Exception as e:
            # 获取当前训练到的epoch数
            current_epoch = last_epoch + getattr(model, 'epoch', 0) + 1
            logging.error(f"训练epoch {current_epoch}时出错: {e}")
            # 保存进度
            save_model_and_progress(model, current_epoch, last_epoch + total_epochs, start_time, stats)
            
        # 保存最终模型
        save_model_and_progress(model, last_epoch + total_epochs, last_epoch + total_epochs, start_time, stats, is_final=True)
        
        # 清理临时文件
        import shutil
        for split in ['train', 'val']:
            label_dir = base_path / split / 'labels'
            if label_dir.exists():
                shutil.rmtree(label_dir)
        logging.info("训练完成，模型已保存并清理临时文件")
        
        # 训练完成后保留最终进度
        save_progress(last_epoch + total_epochs, None, stats)
    
    except KeyboardInterrupt:
        logging.info("检测到Ctrl+C中断，正在保存进度...")
        try:
            # 尝试保存当前模型和进度
            if 'model' in locals() and 'last_epoch' in locals() and 'total_epochs' in locals() and 'start_time' in locals() and 'stats' in locals():
                  # 使用已完成的epoch数作为当前epoch
                  current_epoch = last_epoch + stats.get('epochs_completed', 0)
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

# 已在文件开头导入time模块，此处无需重复导入


if __name__ == '__main__':
    main()
