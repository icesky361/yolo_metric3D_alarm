import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import os
import sys
import argparse
import shutil

sys.path.append(str(Path(__file__).resolve().parents[1]))
# 尝试导入环境工具，如果不存在则使用默认设备
try:
    from utils.environment import get_device_and_adjust_config
    has_env_utils = True
except ImportError:
    has_env_utils = False
    print("警告: 未找到utils.environment模块，将使用默认设备配置")

# 强制添加命令行参数确保任务类型为检测
if '--task' not in sys.argv:
    sys.argv.extend(['--task', 'detect'])

# 1. 加载YAML配置文件 (保留配置加载功能，但可灵活使用)
def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 2. 准备数据集路径
def prepare_data_paths():
    # 使用用户提供的绝对路径
    train_path = Path(r'D:\program\python\PythonProject\Yolo_seg_Alarm\Data\raw\train\images')
    val_path = Path(r'D:\program\python\PythonProject\Yolo_seg_Alarm\Data\raw\val\images')
    
    print(f"最终训练集路径: {train_path}")
    print(f"最终验证集路径: {val_path}")
    
    # 验证路径是否存在
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {train_path} 或 {val_path}")
    
    return train_path, val_path

# 主训练函数
def main():
    # 准备数据集路径
    train_path, val_path = prepare_data_paths()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='detect', help='任务类型')
    args = parser.parse_args()
    
    # 在模型加载代码前强制设置任务类型
    args.task = 'detect'
    
    # 加载模型 (使用用户指定的模型路径)
    model_path = Path(r'models\yolov8n.pt')
    if not model_path.exists():
        # 尝试绝对路径
        model_path = Path(r'D:\program\python\PythonProject\Yolo_seg_Alarm\models\yolov8n.pt')
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型: {model_path}")
    # 显式指定检测任务类型加载模型，并禁用自动下载
    model = YOLO(str(model_path), task='detect', verbose=False)
    
    # 确保使用本地模型，不自动下载
    model.overrides['model'] = str(model_path)
    
    # 创建临时数据配置YAML文件
temp_yaml_path = Path('temp_data_config.yaml')

# 定义数据集路径
# 修正路径，确保指向正确的数据集目录
train_path = Path('datasets/train')
val_path = Path('datasets/val')

# 确保中文正常显示
class MyDumper(yaml.SafeDumper):
    def represent_dict_preserve_order(self, data):
        return self.represent_dict(data.items())

data_config = {
    'train': str(train_path),
    'val': str(val_path),
    'nc': 5,
    'names': {
        0: '挖掘机',
        1: '打桩机',
        2: '拉管机',
        3: '烟雾',
        4: '火'
    }
}

# 写入YAML文件
with open(temp_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_config, f, Dumper=MyDumper, allow_unicode=True)
    
    # 处理设备配置
    device = '0' if torch.cuda.is_available() else 'cpu'
    batch_size = 8  # 默认批次大小
    epochs = 10  # 默认训练轮次
    
    if has_env_utils:
        # 尝试使用环境工具获取设备和调整配置
        config = {'device': 'auto', 'batch': batch_size, 'epochs': epochs}
        device, config = get_device_and_adjust_config(config)
        batch_size = config['batch']
        epochs = config['epochs']
    
    # 检测到低端GPU，强制减小批次大小
    if gpu_memory < 4.0:
        config['batch'] = 1
        print(f"检测到低端GPU (显存{gpu_memory}GB)，使用{config['weights']}模型和batch_size={config['batch']}")
    
    # 配置训练参数
    train_params = {
        'task': 'detect',
        'data': str(temp_yaml_path),  # 传递YAML文件路径
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': 640,
        'project': 'yolo_seg_alarm',
        'name': 'train_note_results',
        'exist_ok': True,
        'device': device,
        'verbose': False,
        'save_period': 1,
        'amp': False  # 禁用AMP，避免下载yolo11n.pt
    }
    
    # 确保device是字符串类型
    if hasattr(device, 'type'):
        device_str = device.type
    else:
        device_str = str(device)
    train_params['device'] = device_str

    # 添加环境变量以跳过AMP检查
    os.environ['YOLO_TESTS_RUNNING'] = '1'

    # 加载YOLO模型
    from ultralytics import YOLO
    model = YOLO('yolov8n-seg.pt')

    # 开始训练
    print(f"使用设备: {device_str}")
    print(f"开始训练，参数: {train_params}")
    results = model.train(**train_params)
    
    # 保存训练后的模型
    model.save('trained_model.pt')
    print("模型已保存为: trained_model.pt")
    
    # 清理临时文件
    if temp_yaml_path.exists():
        temp_yaml_path.unlink()
        print(f"已删除临时配置文件: {temp_yaml_path}")

if __name__ == '__main__':
    main()