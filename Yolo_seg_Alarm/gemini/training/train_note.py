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
    data_config = {
        'train': str(train_path),
        'val': str(val_path),
        'nc': 1,  # 类别数，根据实际情况修改
        'names': ['object']  # 类别名称，根据实际情况修改
    }
    
    # 写入YAML文件
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f)
    
    # 配置训练参数
    train_params = {
        'task': args.task,
        'data': str(temp_yaml_path),  # 传递YAML文件路径
        'epochs': 10,  # 训练轮次
        'batch': 8,     # 批次大小，根据显存调整
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
    
    # 设置训练参数
    train_params = {
        'task': 'detect',
        'data': temp_yaml_path,
        'epochs': 5,
        'batch': config['batch'],
        'imgsz': 640,
        'project': 'yolo_seg_alarm',
        'name': 'train_note_results',
        'exist_ok': True,
        'device': device,
        'verbose': False,
        'save_period': 1,
        'amp': False  # 禁用AMP，避免下载yolo11n.pt
    }
    
    # 添加环境变量以跳过AMP检查
    import os
    os.environ['YOLO_TESTS_RUNNING'] = '1'

    # 开始训练
    print(f"使用设备: {device}")
    print(f"开始训练，参数: {train_params}")
    results = model.train(**train_params)
    
    # 保存训练后的模型
    try:
        # 这里应该是保存模型的代码
        model.save('trained_model.pt')
    finally:
        # 清理临时文件
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)
            print(f"已删除临时配置文件: {temp_yaml_path}")

if __name__ == '__main__':
    main()