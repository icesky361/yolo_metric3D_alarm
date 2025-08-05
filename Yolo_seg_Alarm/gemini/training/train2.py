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
def get_writable_dir(path):
    from pathlib import Path
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return str(dir_path)

def get_progress_file_path():
    current_dir = Path(__file__).parent
    # 使用可写的进度目录
    progress_dir = Path(get_writable_dir(str(current_dir / 'progress')))
    logging.info(f"使用进度文件目录: {progress_dir}")
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
def save_progress(last_epoch, model_path, stats, original_epochs, best_model_path=None):
    # 确保使用可写目录
    progress_file = Path(get_writable_dir(str(Path(__file__).parent / 'progress'))) / 'train_progress.json'
    logging.debug(f"保存进度到: {progress_file}")
    logging.debug(f"保存进度: last_epoch={last_epoch}, model_path={model_path}, best_model_path={best_model_path}, stats={stats}")
    progress_file = get_progress_file_path()
    progress_data = {
        'last_epoch': last_epoch,
        'model_path': model_path,
        'best_model_path': best_model_path,
        'original_epochs': original_epochs,
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
def save_model_and_progress(model, epoch, total_epochs, start_time, stats, original_epochs, save_intermediate=False, is_final=False):
    # 定义模型保存目录
    # 使用可写的模型保存目录
    model_dir = Path(get_writable_dir(str(Path(__file__).parent.parent / 'models' / 'weights')))
    logging.info(f"使用模型保存目录: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义训练结果目录
    results_dir = Path('yolo_seg_alarm') / 'train2_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 每个epoch都保存last.pt
    last_model_path = model_dir / 'last.pt'
    model.save(str(last_model_path))
    logging.info(f"已保存最新模型: {last_model_path}")
    
    # 保存最佳模型
    # 确保至少保存一次最佳模型
    if hasattr(model, 'best') and model.best:
        best_model_path = model_dir / 'best.pt'
        model.save(str(best_model_path))
        logging.info(f"已更新最佳模型: {best_model_path}")
    # 第一轮训练强制保存最佳模型
    elif epoch == 1 and not (model_dir / 'best.pt').exists():
        best_model_path = model_dir / 'best.pt'
        model.save(str(best_model_path))
        logging.info(f"第一轮训练完成,保存初始最佳模型: {best_model_path}")
    
    # 保存中间模型（如果启用）
    intermediate_model_path = None
    if save_intermediate:
        intermediate_model_path = results_dir / f"intermediate_model_epoch_{epoch}.pt"
        model.save(str(intermediate_model_path))
        logging.info(f"已保存中间模型: {intermediate_model_path}")
    
    # 如果是最终模型，保存为指定名称的最终模型
    if is_final:
        final_model_path = model_dir / 'yolo_seg_alarm_train2.pt'
        model.save(str(final_model_path))
        logging.info(f"已保存最终模型: {final_model_path}")
    
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
    save_progress(epoch, str(last_model_path), stats, original_epochs, str(best_model_path) if best_model_path else None)
    
    return last_model_path

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
        parser.add_argument('--resume', action='store_true', help='从上次进度继续训练')
        parser.add_argument('--no_resume', action='store_true', help='不使用上次训练进度，强制从头开始')
        parser.add_argument('--save_intermediate', action='store_true', help='是否保存中间模型')
        args = parser.parse_args()
        
        # 在模型加载代码前强制设置任务类型
        args.task = 'detect'
        
        # 加载进度
        progress = load_progress()
        last_epoch = progress['last_epoch']
        saved_model_path = progress['model_path']
        stats = progress['stats']
    
        # 调试日志：显示进度文件内容
        logging.debug(f"进度文件内容: last_epoch={last_epoch}, saved_model_path={saved_model_path}, stats={stats}")
    
        # 验证模型路径
        model_exists = False
        if saved_model_path:
            saved_model_path = Path(saved_model_path)
            # 如果是相对路径，转换为绝对路径
            if not saved_model_path.is_absolute():
                saved_model_path = Path(__file__).parent / saved_model_path
            model_exists = saved_model_path.exists() and saved_model_path.stat().st_size > 0
            logging.debug(f"解析后的模型路径: {saved_model_path}, 存在: {model_exists}")
        else:
            logging.debug("进度文件中未找到模型路径")
    
        # 自动恢复逻辑
        resume_training = not args.no_resume and model_exists and last_epoch > 0
        original_epochs = progress.get('original_epochs', config.get('epochs', 10))

        # 询问用户是否继续上次训练
        if resume_training:
            logging.info(f"发现上次训练进度，从epoch {last_epoch+1} 开始")
            try:
                model = YOLO(str(saved_model_path), task='detect')
                # 使用进度文件中的原始epoch数
                logging.info(f"总训练epoch数: {original_epochs}")
                logging.info(f"已完成epoch数: {last_epoch}")
                remaining_epochs = original_epochs - last_epoch
                logging.info(f"剩余训练epoch数: {remaining_epochs}")
                # 确保剩余epoch数至少为1
                if remaining_epochs <= 0:
                    logging.warning(f"已完成所有训练epoch ({last_epoch}/{original_epochs})，将开始新的{original_epochs}轮训练")
                    resume_training = False
                else:
                    config['epochs'] = remaining_epochs
            except Exception as e:
                logging.error(f"加载保存的模型失败: {e}")
                resume_training = False
    
        if not resume_training:
            model_path = Path(config['model_path']) / config['weights']
            # 显式指定检测任务类型加载模型
            model = YOLO(str(model_path), task='detect')
            last_epoch = 0
            stats = {'epochs_completed': 0, 'time': 0, 'best_metrics': None}
            logging.info("开始新的训练")
        
        # 确保从配置文件中正确读取epochs值
        config_epochs = config.get('epochs', 10)
        # 计算训练epoch数
        original_epochs = config.get('epochs', 10)
        if last_epoch > 0 and args.resume:
            # 继续训练时，设置为剩余epoch数
            remaining_epochs = original_epochs - last_epoch
            # 检查是否还有剩余epoch
            if remaining_epochs <= 0:
                logging.error(f"训练已完成，无法继续。总epoch数: {original_epochs}, 已完成: {last_epoch}")
                return
            total_epochs = remaining_epochs
            logging.info(f"继续训练，剩余epoch数: {total_epochs}")
        else:
            total_epochs = original_epochs
        
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
            def __init__(self, model, total_epochs, last_epoch, start_time, stats, save_interval, save_intermediate):
                self.model = model
                self.total_epochs = total_epochs
                self.last_epoch = last_epoch
                self.start_time = start_time
                self.stats = stats
                self.save_interval = save_interval
                self.save_intermediate = save_intermediate
                self.best_metrics = stats.get('best_metrics', None)

            def on_epoch_end(self, epoch):
                logging.info(f"完成第 {epoch + 1} 轮训练，正在评估模型...")
                # 显式运行验证以更新最佳指标
                metrics = self.model.val()
                logging.info(f"本轮验证指标: {metrics}")
                
                # 更新最佳指标
                current_map = metrics.box.map50 if hasattr(metrics, 'box') else metrics.map50
                if self.best_metrics is None or current_map > self.best_metrics:
                    self.model.best = metrics
                    self.best_metrics = current_map
                    self.stats['best_metrics'] = current_map
                    logging.info(f"更新最佳指标: {current_map}")
                
                logging.info(f"正在保存进度...")
                current_epoch = self.last_epoch + epoch + 1
                # 每个epoch都保存模型
                save_model_and_progress(self.model, current_epoch, self.last_epoch + self.total_epochs, self.start_time, self.stats, save_intermediate=self.save_intermediate)
                
        # 创建回调实例
        callback = ProgressCallback(model, total_epochs, last_epoch, start_time, stats, args.save_interval, args.save_intermediate)
        # 注册回调函数
        model.add_callback('on_epoch_end', callback.on_epoch_end)
        
        # 检查并设置可写的项目目录
        def get_writable_dir(base_dir, fallback_dir=None):
            if not fallback_dir:
                fallback_dir = Path.home() / 'yolo_seg_alarm_train'
            fallback_dir.mkdir(parents=True, exist_ok=True)
            try:
                test_file = Path(base_dir) / 'test_write_permission.tmp'
                test_file.write_text('test')
                test_file.unlink()
                return base_dir
            except PermissionError:
                logging.warning(f"目录 {base_dir} 无写入权限，使用 fallback 目录: {fallback_dir}")
                return str(fallback_dir)

        project_dir = get_writable_dir('yolo_seg_alarm')
        logging.info(f"使用训练结果目录: {project_dir}")

        try:
            # 使用YOLO内置的多epoch训练功能
            logging.debug(f"调用model.train()参数: epochs={total_epochs}, resume={last_epoch > 0}")
            results = model.train(
                    task=args.task,
                    data=config_path,
                    epochs=total_epochs,
                    batch=config['batch'],
                    imgsz=config['imgsz'],
                    device=device.type,
                    project=project_dir,
                    name='train2_results',
                    exist_ok=True,
                    resume=last_epoch > 0,
                    plots=True,
                    save_json=True,
                    val=True
                )
            logging.debug(f"model.train()返回结果: {results}")
        except Exception as e:
            # 获取当前训练到的epoch数
            current_epoch = last_epoch + getattr(model, 'epoch', 0) + 1
            logging.error(f"训练epoch {current_epoch}时出错: {e}")
            # 保存进度
            save_model_and_progress(model, current_epoch, last_epoch + total_epochs, start_time, stats, original_epochs, save_intermediate=self.save_intermediate)
            logging.error(f"训练中断于epoch {current_epoch}，错误类型: {type(e).__name__}，错误信息: {str(e)}")
            # 判断是否为可恢复错误
            # 扩展可恢复错误类型，包括文件写入错误
            if isinstance(e, (PermissionError, OSError)) and 'Permission denied' not in str(e):
                logging.info(f"尝试从epoch {current_epoch + 1} 继续训练...")
                remaining_epochs = total_epochs - current_epoch
                if remaining_epochs > 0:
                    # 确保所有后续文件操作使用可写目录
                    progress_dir = Path(get_writable_dir(str(current_dir / 'progress')))
                    model_dir = Path(get_writable_dir(str(Path(__file__).parent.parent / 'models' / 'weights')))
                    logging.info(f"更新可写目录 - 进度: {progress_dir}, 模型: {model_dir}")
                    logging.debug(f"恢复训练调用参数: remaining_epochs={remaining_epochs}")
                    results = model.train(
                        task=args.task,
                        data=config_path,
                        epochs=remaining_epochs,
                          batch=config['batch'],
                          imgsz=config['imgsz'],
                          device=device.type,
                          project=project_dir,
                          name='train2_results',
                          exist_ok=True,
                          resume=True,
                          plots=True,
                          save_json=True
                      )
                logging.debug(f"恢复训练返回结果: {results}")
            else:
                logging.error("遇到不可恢复的错误，终止训练")
                return  # 仅在严重错误时退出
        else:
              # 只有训练成功完成时才保存最终模型和进度
              save_model_and_progress(model, last_epoch + total_epochs, last_epoch + total_epochs, start_time, stats, original_epochs, is_final=True, save_intermediate=args.save_intermediate)
              # 训练完成后保留最终进度
              save_progress(last_epoch + total_epochs, None, stats)
              logging.info(f"训练完成！总训练轮次: {total_epochs}，实际完成轮次: {last_epoch + total_epochs}")
        
        # 清理临时文件
        import shutil
        for split in ['train', 'val']:
            label_dir = base_path / split / 'labels'
            if label_dir.exists():
                shutil.rmtree(label_dir)
        logging.info("训练完成，模型已保存并清理临时文件")
        
        # 已在训练成功分支中保存最终进度，此处无需重复
    
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
                current_epoch = last_epoch + getattr(model, 'epoch', 0) + 1
            stats['epochs_completed'] = current_epoch  # 更新实际完成轮次
            save_progress(current_epoch, saved_model_path, stats)
        except Exception as save_e:
            logging.error(f"保存进度时出错: {save_e}")
        raise
    
    print("训练完成，模型已保存并清理临时文件")

if __name__ == '__main__':
    main()

def save_model_and_progress(model, epoch, total_epochs, start_time, stats, is_final=False):
    # 定义模型保存目录
    # 使用可写的模型保存目录
    model_dir = Path(get_writable_dir(str(Path(__file__).parent.parent / 'models' / 'weights')))
    logging.info(f"使用模型保存目录: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义训练结果目录
    results_dir = Path('yolo_seg_alarm') / 'train2_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 每个epoch都保存last.pt
    last_model_path = model_dir / 'last.pt'
    model.save(str(last_model_path))
    logging.info(f"已保存最新模型: {last_model_path}")
    
    # 保存最佳模型
    # 保存最佳模型
    # 获取当前验证指标
    current_map = stats.get('best_metrics', 0)
    try:
        if hasattr(model, 'metrics'):
            current_map = model.metrics.box.map50 if hasattr(model.metrics, 'box') else model.metrics.map50
            logging.info(f"当前验证mAP50: {current_map}")
    except Exception as e:
        logging.warning(f"获取验证指标失败: {e}")
    
    # 确保第一轮训练后一定保存最佳模型
    best_model_path = None
    if not (model_dir / 'best.pt').exists():
        best_model_path = model_dir / 'best.pt'
        model.save(str(best_model_path))
        stats['best_metrics'] = current_map
        stats['best_model_path'] = str(best_model_path)
        logging.info(f"第一轮训练完成，保存初始最佳模型: {best_model_path}, mAP50: {current_map}")
    # 后续轮次根据指标更新
    elif current_map > stats.get('best_metrics', 0):
        best_model_path = model_dir / 'best.pt'
        model.save(str(best_model_path))
        stats['best_metrics'] = current_map
        stats['best_model_path'] = str(best_model_path)
        logging.info(f"更新最佳模型: {best_model_path}, mAP50: {current_map}")
    
    # 保存中间模型
    intermediate_model_path = results_dir / f"intermediate_model_epoch_{epoch}.pt"
    model.save(str(intermediate_model_path))
    logging.info(f"已保存中间模型: {intermediate_model_path}")
    
    # 如果是最终模型，保存为指定名称的最终模型
    if is_final:
        final_model_path = model_dir / 'yolo_seg_alarm_train2.pt'
        model.save(str(final_model_path))
        logging.info(f"已保存最终模型: {final_model_path}")
    
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
    save_progress(epoch, str(last_model_path), stats)
    
    return last_model_path
