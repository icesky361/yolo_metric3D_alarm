# -*- coding: utf-8 -*-
"""
================================================================================
@ 脚本名: infer2.py
@ 功能: YOLOv11 目标检测模型流式推理脚本
@ 作者: TraeAI
@ 创建时间: 2024-07-28
--------------------------------------------------------------------------------
@ 脚本逻辑: 实现双层批次架构的流式推理
1. 路径批次生成器: 流式加载图像路径，避免一次性加载过多路径占用内存
2. 推理子批次: 将路径批次进一步拆分，适应GPU显存限制
3. 主动内存管理: 每批处理后显式清理内存
4. 增量结果保存: 批次处理完立即保存结果，降低数据丢失风险
================================================================================
"""

import argparse
import pandas as pd
import openpyxl
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import torch
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import logging
import gc
import time
import psutil


# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_memory_usage(stage: str, is_cleanup: bool = False):
    """记录当前GPU和系统内存使用情况"""
    # 记录系统内存使用
    mem = psutil.virtual_memory()
    mem_used_gb = mem.used / (1024 **3)
    mem_total_gb = mem.total / (1024** 3)
    mem_percent = mem.percent
    
    # 记录GPU内存使用情况
    gpu_mem_info = ""
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 **3)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024** 3)
        gpu_mem_percent = (gpu_mem_reserved / 20) * 100  # 20GB是A4500的总显存
        gpu_mem_info = f"系统内存使用: {mem_used_gb:.2f}/{mem_total_gb:.2f}GB ({mem_percent}%), GPU已分配: {gpu_mem_allocated:.2f}GB, GPU已保留: {gpu_mem_reserved:.2f}GB (利用率: {gpu_mem_percent:.1f}%)，"
    
    logger.info(f"【{stage}】{gpu_mem_info}")


def get_device():
    """检测系统可用硬件，优先使用GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"检测到支持CUDA的GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.info("未找到支持CUDA的GPU。将在CPU上运行。")
    return device

def image_batch_generator(source_path, valid_extensions, batch_size=672):
    """生成器函数，流式分批加载图像路径
    
    Args:
        source_path: 图像文件所在目录
        valid_extensions: 有效的图像文件扩展名集合
        batch_size: 每批加载的图像路径数量
    
    Yields:
        list: 一批图像路径
    """
    batch = []
    for file in source_path.rglob('*.*'):
        if file.suffix.lower() in valid_extensions and file.is_file():
            try:
                # 验证文件是否可打开
                with Image.open(file):
                    batch.append(file)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            except UnidentifiedImageError:
                logger.warning(f"跳过无效图像文件: {file}")
    # 处理最后一批
    if batch:
        yield batch


def run_inference(weights_path: str, source_dir: str, output_excel_path: str):
    """
    对一个目录中的图像进行流式推理，并将结果保存到Excel文件中。

    Args:
        weights_path (str): 训练好的模型权重（.pt文件）的路径。
        source_dir (str): 包含测试图像的目录的路径。
        output_excel_path (str): 保存结果的Excel文件的路径。
    """
    # --- 1. 初始化设置 ---    
    device = get_device()
    source_path = Path(source_dir) / 'images'  # 图片文件夹路径
    if not source_path.exists():
        logger.error(f"图片文件夹不存在: {source_path}")
        return

    # 确保输出目录存在
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    output_images_path = output_dir / 'images'
    output_images_path.mkdir(exist_ok=True)

    # 设置输出Excel文件路径
    output_excel_path = Path(output_excel_path)
    output_excel_path.parent.mkdir(exist_ok=True)

    # --- 2. 加载模型 ---    
    log_memory_usage("模型加载前")
    try:
        model = YOLO(weights_path, task='detect')
        model.to(device)
        model.eval()
        if hasattr(model, 'mode'):
            model.mode = 'predict'
        logger.info(f'模型已加载至{device}，推理模式确认完成')
        logger.info(f"成功从 {weights_path} 加载模型")
        log_memory_usage("模型加载后")
    except Exception as e:
        logger.error(f"加载模型失败。错误: {e}")
        return

    # --- 3. 初始化结果数据结构 ---    
    results_data = {
        'original_image_name': [],
        'annotated_image_name': [],
        'pred_class': [],
        'confidence': [],
        'bbox_xyxy': []
    }

    # --- 4. 流式批次推理 ---    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 快速统计总图片数量（带进度反馈）
    # 与用户交互获取预估总图片数
    while True:
        try:
            total_images_input = input("请输入待处理图片的预估总数: ")
            total_images = int(total_images_input)
            if total_images > 0:
                break
            print("请输入大于0的有效数字")
        except ValueError:
            print("输入无效，请输入一个整数")
    logger.info(f"用户指定的预估总图片数: {total_images} 张")
    valid_extensions_set = {ext.lower() for ext in valid_extensions}
    # 使用tqdm添加计数进度条
    # 使用特定扩展名glob模式加速搜索
    ext_pattern = '**/*.{{{}}}'.format(','.join(ext[1:] for ext in valid_extensions))
    with tqdm(desc="图片计数中", unit="个文件") as count_pbar:
        for file in source_path.rglob(ext_pattern):
            count_pbar.update(1)
            if file.is_file():
                total_images += 1
    logger.info(f"发现 {total_images} 张图片")
    
    image_generator = image_batch_generator(source_path, valid_extensions, batch_size=672)  # 流式生成器
    
    # 初始化进度跟踪
    progress_bar = tqdm(total=total_images, desc="总体推理进度", unit="张", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} 张 ({percentage:.1f}%) [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    processed_images = 0
    processed_batches = 0
    
    batch_idx = 0
    start_time = time.time()

    # 预热模型
    warmup_done = False

    try:
        # 第一层循环：处理路径批次
        for path_batch in image_generator:
            batch_idx += 1
            current_batch_size = len(path_batch)
            logger.info(f'开始处理路径批次 {batch_idx}，共 {current_batch_size} 张图像')
            batch_start_time = time.time()

            # 第二层循环：将路径批次拆分为推理子批次
            inference_batch_size = 64  # 推理子批次大小，从112调整为64以降低显存占用
            total_sub_batches = (current_batch_size + inference_batch_size - 1) // inference_batch_size
            logger.info(f'路径批次 {batch_idx} 将拆分为 {total_sub_batches} 个推理子批次')

            # 处理推理子批次
            for sub_batch_idx in range(total_sub_batches):
                sub_start = sub_batch_idx * inference_batch_size
                sub_end = min(sub_start + inference_batch_size, current_batch_size)
                sub_batch_files = path_batch[sub_start:sub_end]
                sub_batch_paths = [str(img_path) for img_path in sub_batch_files]

                logger.info(f'处理推理子批次 {sub_batch_idx+1}/{total_sub_batches}，共 {len(sub_batch_paths)} 张图像')

                # 预热模型（仅第一次推理前执行）
                if not warmup_done and len(sub_batch_paths) > 0:
                    with torch.no_grad():
                        model.predict(source=sub_batch_paths[0], device=device, task='detect', batch=1, verbose=False)
                    logger.info('模型预热完成')
                    warmup_done = True

                # 执行推理
                with torch.no_grad():
                    sub_results = model.predict(source=sub_batch_paths, device=device, task='detect', batch=inference_batch_size, half=True)

                # 推理后内存状态
                log_memory_usage(f"子批次 {sub_batch_idx+1}")

                # 处理推理结果
                for i, res in enumerate(sub_results):
                    img_path = sub_batch_files[i]
                    names = res.names

                    # 解析检测结果
                    if res.boxes is not None and len(res.boxes) > 0:
                        # 检查是否有置信度>=0.3的检测框
                        has_high_confidence = any(box.conf >= 0.3 for box in res.boxes)
                        
                        # 生成标注图像
                        original_image = Image.open(img_path).convert('RGB')
                        original_array = np.array(original_image)
                        annotated_image = res.plot(img=original_array)
                        
                        # 保存标注图像（仅当有高置信度检测框时）
                        new_filename = ''
                        if has_high_confidence:
                            filename = img_path.stem
                            extension = img_path.suffix
                            new_filename = f"{filename}_tl{extension}"
                            Image.fromarray(annotated_image).save(output_images_path / new_filename)
                        
                        for box in res.boxes:
                            class_id = int(box.cls)
                            confidence = float(box.conf)
                            bbox_coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        
                            results_data['original_image_name'].append(img_path.name)
                            results_data['annotated_image_name'].append(new_filename)
                            results_data['pred_class'].append(names[class_id])
                            results_data['confidence'].append(round(confidence, 4))
                            results_data['bbox_xyxy'].append(",".join(map(str, bbox_coords)))
                    else:
                        results_data['original_image_name'].append(img_path.name)
                        results_data['annotated_image_name'].append('')
                        results_data['pred_class'].append('未检测到')
                        results_data['confidence'].append(0.0)
                        results_data['bbox_xyxy'].append('')

                # 子批次处理后立即清理内存
                del sub_results, sub_batch_paths
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

                # 清理后内存状态
                log_memory_usage(f"子批次 {sub_batch_idx+1}", is_cleanup=True)

                logger.info(f'子批次 {sub_batch_idx+1}/{total_sub_batches} 处理完成，内存已清理')
                processed_images += len(sub_batch_paths)
            # 批次处理完成后保存当前结果
            # 在run_inference函数顶部添加
            output_csv_path = output_excel_path.with_suffix('.csv')
            
            # 替换原来的Excel保存代码
            results_df = pd.DataFrame(results_data)
            if not output_csv_path.exists():
                results_df.to_csv(output_csv_path, index=False, mode='w', header=True)
            else:
                results_df.to_csv(output_csv_path, index=False, mode='a', header=False)
            
            logger.info(f'路径批次 {batch_idx} 中间结果已追加至 {output_csv_path.resolve()}')
            # 每10个批次输出一次进度信息
            processed_batches += 1
            progress_bar.update(current_batch_size)
            logger.info(f"进度更新: 已完成 {processed_batches} 个路径批次，处理了 {processed_images} 张图片")
            # 删除旧的进度更新行

            # 计算处理时间和性能指标
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            
            if batch_duration > 0:
                images_per_minute = (current_batch_size / batch_duration) * 60
                seconds_per_1k_images = (batch_duration / current_batch_size) * 1000
                logger.info(f"批次 {batch_idx} 性能指标: 平均每分钟处理 {images_per_minute:.2f} 张图片, 平均每1000张图片耗时 {seconds_per_1k_images:.2f} 秒")
            else:
                logger.info(f"批次 {batch_idx} 处理完成，时间过短无法计算性能指标")

            # 重置结果数据结构，只保留当前批次数据用于聚合
            results_data = {
                'original_image_name': [],
                'annotated_image_name': [],
                'pred_class': [],
                'confidence': [],
                'bbox_xyxy': []
            }

        # 所有批次处理完成
        log_memory_usage("所有推理完成后")
        logger.info(f'所有批次推理完成，总耗时 {time.time() - start_time:.2f} 秒')
        progress_bar.close()

    except Exception as e:
        logger.error(f"推理过程中发生错误: {e}", exc_info=True)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="运行YOLOv11分割模型流式推理。")
    # 默认路径配置
    default_weights = 'G:/soft/soft/python project/Yolo_metric_alarm/Yolo_seg_Alarm/gemini/yolo_seg_alarm/train2_results/weights/best.pt'
    default_source = 'D:/20250804'
    default_excel = 'G:/soft/soft/python project/Yolo_metric_alarm/Data/test/your_excel_file.xlsx'
    
    parser.add_argument('--weights', type=str, default=default_weights, help='模型权重文件路径 (默认: %(default)s)')
    parser.add_argument('--source', type=str, default=default_source, help='测试图片所在文件夹路径 (默认: %(default)s)')
    parser.add_argument('--output_excel', type=str, default=default_excel, help='结果Excel文件路径 (默认: %(default)s)')

    args = parser.parse_args()
    run_inference(args.weights, args.source, args.output_excel)