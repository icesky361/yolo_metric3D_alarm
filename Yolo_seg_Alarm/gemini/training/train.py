# -*- coding: utf-8 -*-
"""
================================================================================
@ 脚本名: train.py
@ 功能: YOLOv11 实例分割模型训练脚本
@ 作者: TraeAI
@ 创建时间: 2024-07-28
--------------------------------------------------------------------------------
@ 脚本逻辑:
1.  **初始化与配置加载**: 
    - 导入所需库 (argparse, yaml, logging, ultralytics等)。
    - 添加项目根目录到系统路径，以正确导入自定义的`utils`模块。
    - 配置日志记录器，用于输出训练过程中的信息、警告和错误。
    - 定义`train`函数，接收一个配置文件路径作为参数。

2.  **环境检测与参数动态调整**:
    - 调用`utils.environment.get_device_and_adjust_config`函数。
    - 该函数会自动检测是否存在可用的NVIDIA GPU。
    - 如果有GPU，根据GPU型号（高端、中端、低端）动态调整训练参数，如`batch_size`。
    - 如果没有GPU，则切换到CPU模式，并使用保守的参数配置，以防止内存溢出。

3.  **模型加载**:
    - 使用`ultralytics.YOLO`加载一个预训练的YOLOv11分割模型（例如`yolov11n-seg.pt`）。
    - 这个预训练模型为我们的训练提供了一个良好的起点，可以加快收敛速度。
    - 将模型移动到前面步骤检测到的设备（GPU或CPU）上。

4.  **启动训练**:
    - 调用`model.train()`方法启动训练流程。
    - `ultralytics`库的`train`方法是一个高度集成的函数，它会自动处理：
        - 数据加载（根据`.yaml`配置文件中的路径）。
        - 数据增强（根据`.yaml`文件中的配置）。
        - 模型评估（在每个epoch后计算mAP、分割IoU等指标）。
        - 保存检查点（保存训练过程中的模型权重）。
        - 保存最佳模型（在`models/yolo_seg_experiment/weights/best.pt`）。
    - 训练完成后，记录日志，提示用户最佳模型已保存的位置。

5.  **主程序入口**:
    - 使用`argparse`库设置命令行参数，允许用户通过`--config`指定配置文件路径。
    - 如果用户未指定，则默认使用`configs/yolov11_seg.yaml`。
    - 解析命令行参数并调用`train`函数，启动整个训练过程。

@ 使用方法:
-   直接在命令行中运行此脚本:
    python training/train.py
-   或指定一个不同的配置文件:
    python training/train.py --config /path/to/your/config.yaml
================================================================================
"""

import argparse
import yaml
from pathlib import Path
import logging
from ultralytics import YOLO

# 将项目根目录添加到系统路径中，以便能够正确导入 'utils' 模块
# Path(__file__).resolve() 获取当前文件的绝对路径
# .parents[1] 获取上级目录，即项目根目录 'gemini'
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.environment import get_device_and_adjust_config

# 配置日志记录，用于在控制台输出格式化的时间、日志级别和消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(config_path: str):
    """
    运行YOLOv11分割模型训练的主函数。

    Args:
        config_path (str): 指向YAML配置文件的路径。
    """
    # --- 1. 加载配置文件 ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"成功从 {config_path} 加载配置")
    except FileNotFoundError:
        logging.error(f"配置文件未找到: {config_path}")
        return
    except Exception as e:
        logging.error(f"加载或解析YAML文件时出错: {e}")
        return
    
    # 动态调整数据集配置，指定标签路径与图像路径相同
    data_config = config
    
    # --- 2. 设置环境并动态调整配置 ---
    # 这一步会根据检测到的硬件（CPU/GPU）动态设置设备和调整超参数（如批量大小）。
    device, updated_config = get_device_and_adjust_config(config)
    
    # --- 3. 加载YOLO模型 ---
    # 我们使用 'yolov11-n-seg.pt' 作为一个起点（这是一个小而快的模型）。
    # 你可以根据需要更换为其他模型，如 'yolov11s-seg.pt', 'yolov11m-seg.pt' 等。
    try:
        model = YOLO(updated_config['weights'])  # 从配置加载一个预训练的分割模型
        model.to(device)  # 将模型移动到指定的设备上
        logging.info(f"YOLOv11分割模型 '{updated_config['weights']}' 加载成功。")
    except Exception as e:
        logging.error(f"加载YOLO模型失败。请确保 '{updated_config['weights']}' 可访问或网络连接正常。错误: {e}")
        return

    # --- 4. 开始训练 ---
    logging.info("开始模型训练...")
    logging.info(f"数据集路径: {data_config['path']}")
    logging.info(f"训练轮数 (Epochs): {updated_config['epochs']}, 批量大小 (Batch size): {updated_config['batch']}, 图像尺寸 (Image size): {updated_config['imgsz']}")

    try:
        # ultralytics的train方法负责整个训练循环，
        # 包括数据加载、数据增强、评估和保存检查点。
        # 注意: ultralytics的验证器默认会计算mAP和分割mAP(IoU)。
        # 如果需要计算Dice系数等其他指标，可以考虑使用回调函数在验证结束时进行计算。
        results = model.train(
            data=config_path,  # 直接传递配置文件路径，YOLO会自动解析
            epochs=updated_config['epochs'],
            batch=updated_config['batch'],
            imgsz=updated_config['imgsz'],
            optimizer=updated_config.get('optimizer', 'AdamW'), # 优化器，默认为AdamW
            lr0=updated_config.get('lr0', 0.01),  # 初始学习率
            lrf=updated_config.get('lrf', 0.01),  # 最终学习率
            workers=4,  # 根据CPU核心数调整，加速数据加载
            # 数据增强参数由训练器直接从yaml文件中读取
            project='models',  # 将结果保存到 'models' 目录下
            name='yolo_seg_experiment'  # 为本次特定的运行创建一个子目录
        )
        logging.info("训练成功完成。")
        # 清理GPU内存，释放显存资源
        torch.cuda.empty_cache()
        # 训练完成后，results对象包含了所有训练结果的路径信息
        logging.info(f"模型和结果已保存至: {results.save_dir}")
        logging.info(f"要运行推理，请使用最佳模型: {results.save_dir}/weights/best.pt")

    except Exception as e:
        # 如果训练过程中发生任何异常，记录详细的错误信息
        logging.error(f"训练过程中发生错误: {e}", exc_info=True)

if __name__ == '__main__':
    # --- 命令行参数解析 ---
    # 创建一个参数解析器，用于处理从命令行传递的参数
    parser = argparse.ArgumentParser(description="训练一个YOLOv8分割模型。")
    
    # 添加 '--config' 参数，用于指定训练配置文件的路径
    parser.add_argument('--config', type=str, default='d:\\program\\python\\PythonProject\\Yolo_seg_Alarm\\Yolo_seg_Alarm\\gemini\\configs\\yolov11_seg.yaml', help='Path to the configuration YAML file')
    
    # 解析命令行传入的参数
    args = parser.parse_args()

    # 加载配置文件
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 动态调整训练参数
    # 动态调整训练参数
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from gemini.utils.environment import get_device_and_adjust_config
    device, config = get_device_and_adjust_config(config)

    # 初始化YOLO模型
    from ultralytics import YOLO
    model = YOLO(config['model_path'])

    # 使用解析到的配置路径调用train函数
    # 准备数据集配置
    data_dict = {
        'path': config['path'],
        'train': config['train'],
        'val': config['val'],
        'nc': config['nc'],
        'names': config['names']
    }
    
    # 创建临时YAML文件以适应YOLO的数据加载要求
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(data_dict, f)
        temp_yaml_path = f.name

    # 开始训练
    model.train(data=temp_yaml_path, epochs=config['epochs'], batch=config['batch'], imgsz=config['img_size'], workers=0)
    
    # 清理临时文件
    import os
    os.remove(temp_yaml_path)