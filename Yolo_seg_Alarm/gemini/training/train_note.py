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
            raise FileNotFoundError(f"模型文件不存在: models\\yolov8n.pt")
    
    print(f"加载模型: {model_path}")
    # 显式指定检测任务类型加载模型
    model = YOLO(str(model_path), task='detect')
    
    # 配置训练参数
    train_params = {
        'task': args.task,
        'data': {
            'train': str(train_path),
            'val': str(val_path),
            'nc': 1,  # 默认类别数，根据实际情况修改
            'names': ['object']  # 默认类别名称，根据实际情况修改
        },
        'epochs': 100,  # 训练轮次
        'batch': 8,     # 批次大小
        'imgsz': 640,   # 图像大小
        'project': 'yolo_seg_alarm',
        'name': 'train_note_results',
        'exist_ok': True
    }
    
    # 处理设备配置
    if has_env_utils:
        # 尝试使用环境工具获取设备
        config = {'device': 'auto'}
        device, config = get_device_and_adjust_config(config)
        train_params['device'] = device.type
    else:
        # 自动选择设备
        train_params['device'] = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {train_params['device']}")
    print(f"开始训练，参数: {train_params}")
    
    # 训练模型
    results = model.train(**train_params)
    
    # 保存训练后的模型
    model.save('yolo_seg_alarm_train_note.pt')
    print("训练完成，模型已保存为 yolo_seg_alarm_train_note.pt")

if __name__ == '__main__':
    main()