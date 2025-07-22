import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import os
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.environment import get_device_and_adjust_config

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

# 主训练函数
def main():
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
    
    # 加载模型
    model_path = Path(config['model_path']) / config['weights']
    model = YOLO(str(model_path))
    # 训练模型
    results = model.train(
        data=config_path,
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        device=device.type,
        project='yolo_seg_alarm',
        name='train2_results',
        exist_ok=True
    )
    
    # 保存训练后的模型
    model.save('yolo_seg_alarm_train2.pt')
    import shutil
    for split in ['train', 'val']:
        label_dir = base_path / split / 'labels'
        if label_dir.exists():
            shutil.rmtree(label_dir)
    print("训练完成，模型已保存并清理临时文件")

if __name__ == '__main__':
    main()
