# -*- coding: utf-8 -*-
"""
================================================================================
@ 脚本名: environment.py
@ 功能: 环境检测与训练参数动态调整工具
@ 作者: TraeAI
@ 创建时间: 2024-07-28
--------------------------------------------------------------------------------
@ 脚本逻辑:
1.  **导入与配置**: 
    - 导入`torch`库用于检测GPU，导入`logging`库用于信息输出。
    - 配置日志记录器。

2.  **定义`get_device_and_adjust_config`函数**: 
    - **功能**: 该函数的核心目标是让训练脚本能够自适应不同的硬件环境，特别是不同性能的电脑（从高性能工作站到普通笔记本）。
    - **GPU检测**: 
        - 使用`torch.cuda.is_available()`检查系统中是否存在NVIDIA GPU以及CUDA环境是否配置正确。
    - **GPU路径**: 
        - 如果检测到GPU，则将设备设置为`cuda:0`。
        - 使用`torch.cuda.get_device_name(0)`获取GPU的具体型号名称（例如 "NVIDIA GeForce RTX 3080"）。
        - **动态参数调整**: 根据GPU型号名称中的关键字（如 'A4500', 'RTX 40', 'GTX 16'），将GPU分为“高端”、“中端”、“低端”三类。
            - **高端GPU (如 A4500, RTX 30/40系列)**: 分配一个较大的`batch_size`（例如32），因为这类GPU有足够的显存来处理更多的图片，从而可能加快训练速度。
            - **中端GPU (如 GTX 16, RTX 20系列)**: 分配一个中等的`batch_size`（例如16）。
            - **低端或未知GPU**: 分配一个保守的`batch_size`（例如8），以确保不会因为显存不足而导致程序崩溃。
    - **CPU路径**: 
        - 如果未检测到GPU，则将设备设置为`cpu`。
        - 输出警告信息，提示用户在CPU上训练会非常慢。
        - 设置一个非常小的`batch_size`（例如2），以避免占用过多内存导致系统卡顿。
    - **返回结果**: 
        - 函数最后返回两个值：
            1.  `device`: 一个`torch.device`对象，告诉训练脚本应该在哪个设备上运行模型。
            2.  `config`: 一个更新后的配置字典，其中`batch_size`已经被动态调整。

@ 使用方法:
-   此脚本通常不直接运行，而是被其他脚本（如`training/train.py`）导入和调用。
-   在训练脚本中，只需调用 `get_device_and_adjust_config(config)` 即可获得适合当前硬件的设备和配置。
================================================================================
"""

import torch
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device_and_adjust_config(config: dict) -> (torch.device, dict):
    """
    检测可用的硬件（GPU/CPU）并动态调整训练参数。

    此函数检查是否存在支持CUDA的GPU。如果找到，它会识别GPU类型
    并调整诸如批量大小之类的超参数以获得最佳性能。如果没有找到GPU，
    它会默认使用CPU，并采用安全的低内存配置。

    Args:
        config (dict): 从YAML文件加载的原始配置字典。

    Returns:
        tuple: 一个包含以下内容的元组：
            - torch.device: 所选设备 (例如, torch.device('cuda:0'))。
            - dict: 更新后的配置字典。
    """
    if torch.cuda.is_available():
        # 如果有可用的CUDA设备（NVIDIA GPU）
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"检测到支持CUDA的GPU: {gpu_name}")

        # 获取GPU显存大小 (GB)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU显存: {gpu_memory:.1f}GB")

        # 根据显存大小动态选择模型和批量大小
        if gpu_memory <= 2:
            # 2GB及以下显存（如MX350）
            config['weights'] = 'yolov8n-seg.pt'  # 使用nano模型
            config['batch'] = 1
            logging.info(f"检测到低端GPU (显存{gpu_memory:.1f}GB)，使用{config['weights']}模型和batch_size={config['batch']}")
        elif gpu_memory <= 4:
            # 2GB到4GB显存
            config['weights'] = 'yolov8s-seg.pt'  # 使用small模型
            config['batch'] = 8
            logging.info(f"检测到中端GPU (显存{gpu_memory:.1f}GB)，使用{config['weights']}模型和batch_size={config['batch']}")
        elif gpu_memory <= 8:
            # 4GB到8GB显存
            config['weights'] = 'yolov8m-seg.pt'  # 使用medium模型
            config['batch'] = 16
            logging.info(f"检测到中高端GPU (显存{gpu_memory:.1f}GB)，使用{config['weights']}模型和batch_size={config['batch']}")
        else:
            # 8GB以上显存
            config['weights'] = 'yolov8l-seg.pt'  # 使用large模型
            config['batch'] = 32
            logging.info(f"检测到高端GPU (显存{gpu_memory:.1f}GB)，使用{config['weights']}模型和batch_size={config['batch']}")
        # --- 根据GPU型号动态调整参数 --- #
        # 针对特定显卡型号的优化配置
        if 'MX350' in gpu_name:
            # MX350特殊处理：仅支持YOLOv8-nano
            config['weights'] = 'yolov8n-seg.pt'
            config['batch'] = 1
            logging.info("检测到MX350显卡，强制使用{config['weights']}模型和batch_size={config['batch']}")
        elif 'A4500' in gpu_name:
            # A4500高端显卡：使用YOLOv11-large
            config['weights'] = 'yolo11l.pt'
            config['batch'] = 32  # 20GB显存优化配置
            logging.info("检测到A4500显卡，使用{config['weights']}模型和batch_size={config['batch']}")
        elif 'RTX 40' in gpu_name or 'RTX 30' in gpu_name:
            # 其他高端GPU：使用YOLOv11-large
            config['weights'] = 'yolov11l-seg.pt'
            config['batch'] = 32
            logging.info("检测到高端GPU，使用{config['weights']}模型和batch_size={config['batch']}")
        elif 'GTX 16' in gpu_name or 'RTX 20' in gpu_name:
            # 中端GPU：保持YOLOv8-small
            config['batch'] = 16
            logging.info("检测到中端GPU，使用{config['weights']}模型和batch_size={config['batch']}")
        else:
            # 低端或未知GPU：保持保守配置
            config['batch'] = 1
            logging.info("检测到低端或未知GPU，使用{config['weights']}模型和batch_size={config['batch']}")
    else:
        # 如果没有可用的GPU，则使用CPU
        device = torch.device("cpu")
        logging.info("未找到支持CUDA的GPU。将在CPU上运行。")
        logging.warning("在CPU上训练会非常慢。使用最低配置。")
        config['batch'] = 2  # 为CPU设置最小的批量大小以避免内存问题
        # config['workers'] = 0 # 在Windows上，CPU模式下workers设为0通常是最佳选择

    logging.info(f"使用设备: {device}")
    logging.info(f"最终批量大小: {config['batch']}")

    return device, config