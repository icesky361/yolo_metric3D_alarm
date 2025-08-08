# -*- coding: utf-8 -*-
"""
================================================================================
@ 脚本名: infer.py
@ 功能: YOLOv11 目标检测模型推理脚本
@ 作者: TraeAI
@ 创建时间: 2024-07-28
--------------------------------------------------------------------------------
@ 脚本逻辑:
1.  **初始化与配置**: 
    - 导入所需库 (argparse, pandas, logging, ultralytics, torch等)。
    - 配置日志记录器，用于输出推理过程中的信息。




# 其他导入语句

# 2.  **设备检测 (get_device)**: 
#    - 检查系统是否支持CUDA(即有无NVIDIA GPU)。
#    - 如果支持，则设置设备为`cuda:0`并打印GPU型号。
#    - 如果不支持，则设置设备为`cpu`，并提示用户将在CPU上运行。

# 3.  **推理主函数 (run_inference)**: 
#    - **a. 设置**: 
#        - 调用`get_device()`确定运行设备。
#        - 定义输入图片目录和输出Excel文件的路径。
#        - 自动创建`results`文件夹用于存放结果。
#    - **b. 加载模型**: 
#        - 使用`ultralytics.YOLO`加载用户指定的、已经训练好的模型权重(.pt文件)。
#        - 将模型移动到检测到的设备上（GPU或CPU）。
#    - **c. 加载原始Excel数据**: 
#        - 使用`pandas.read_excel`读取用户提供的原始Excel文件，该文件包含图片名称等信息。
#    - **d. 准备结果容器**: 
#        - 创建一个字典`results_data`，用于高效地收集每张图片的检测结果(类别、置信度、边界框、分割坐标)。
#    - **e. 遍历图片并执行推理**: 
#        - 遍历指定文件夹中的所有图片文件。
#        - 使用`tqdm`显示处理进度条。
#        - 对每张图片，调用`model.predict()`方法进行推理。
#        - `predict`方法会返回一个包含所有检测结果的列表。
#    - **f. 解析并保存结果**: 
#        - 遍历推理结果，提取每个检测到的对象的：
        - 类别名称 (`pred_class`)
        - 置信度 (`confidence`)
        - 边界框坐标 (`bbox_xyxy`)
        
        - 将提取的信息存入`results_data`字典。
    - **g. 结果整合与保存**: 
        - 将收集到的`results_data`字典转换为Pandas DataFrame。
        - **处理多目标检测**: 由于一张图片可能检测到多个目标，需要对结果进行聚合。按图片名称分组，将同一张图片的多个检测结果用分号`;`连接成一个字符串。
        - **合并数据**: 使用`pandas.merge`将聚合后的检测结果与原始Excel数据进行左连接（left merge），确保原始表格中的所有图片都被保留。
        - 对于没有检测到任何目标的图片，在结果列中填充`No Detection`。
        - 将最终合并后的DataFrame保存为一个新的Excel文件，存放在`results`目录下。

4.  **主程序入口**:
    - 使用`argparse`设置命令行参数，要求用户必须提供：
        - `--weights`: 模型权重文件路径。
        - `--source`: 测试图片所在的文件夹路径。
        - `--excel_file`: 对应的原始Excel文件路径。
    - 解析参数并调用`run_inference`函数启动推理。

@ 使用方法:
-   在命令行中运行此脚本，并提供必要的参数:
    python inference/infer.py --weights "G:/soft/soft/python project/Yolo_metric_alarm/Yolo_seg_Alarm/gemini/yolo_seg_alarm/train2_results/weights/best.pt" --source "G:/soft/soft/python project/Yolo_metric_alarm/Data/test" --excel_file Data/test/your_excel_file.xlsx
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
from PIL import Image
import logging
import numpy as np
import psutil
import gc
import os
from ultralytics.engine.results import Results

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
def get_device():
    """检测可用的硬件（GPU/CPU）。"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"检测到支持CUDA的GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.info("未找到支持CUDA的GPU。将在CPU上运行。")
    return device

def run_inference(weights_path: str, source_dir: str, output_excel_path: str):
    """
    对一个目录中的图像进行推理，并将结果保存到Excel文件中。

    Args:
        weights_path (str): 训练好的模型权重（.pt文件）的路径。
        source_dir (str): 包含测试图像的目录的路径。
        output_excel_path (str): 保存结果的Excel文件的路径。
    """
    # --- 1. 初始化设置 ---
    device = get_device()
    source_path = Path(source_dir) / 'images' # 修复路径：指向用户实际图片目录images子文件夹
    # 确保输出目录存在
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    output_images_path = output_dir / 'images'
    output_images_path.mkdir(exist_ok=True)
    # 设置输出Excel文件路径
    output_excel_path = Path(output_excel_path)
    # 确保输出Excel文件的父目录存在
    output_excel_path.parent.mkdir(exist_ok=True)

    # --- 2. 加载模型 ---
    try:
        # 极简推理模式配置 (Ultralytics官方推荐标准)
        # 1. 仅在模型初始化时指定检测任务
        # 根据train2.py中的工作配置，仅指定检测任务
        # 双重保障：初始化时指定任务 + 加载后强制设置
        model = YOLO(weights_path, task='detect') # 初始化指定检测任务 
        model.to(device)  # 移动到目标设备
        model.eval()      # 启用PyTorch评估模式
        # 兼容旧版本: 显式设置推理模式
        if hasattr(model, 'mode'):
            model.mode = 'predict'
        # 验证推理配置
        assert model.task == 'detect', f'模型任务错误: 预期detect，实际{model.task}'
        assert hasattr(model, 'predict'), f'模型没有predict方法'
        logger.info(f'模型已加载至{device}，推理模式确认完成')
        # 确保使用predict方法进行推理，显式指定任务类型
        logger.info(f"成功从 {weights_path} 加载模型")
    except Exception as e:
        logger.error(f"加载模型失败。错误: {e}")
        return

    # --- 3. 创建新的Excel文件用于存储结果 ---
    try:
        # 创建新的DataFrame用于存储结果
        df = pd.DataFrame(columns=['original_image_name', 'annotated_image_name', 'pred_class', 'confidence', 'bbox_xyxy'])
        logger.info("已初始化结果数据结构")
    except Exception as e:
        logger.error(f'初始化数据结构时出错: {str(e)}')
        return

    # --- 4. 准备用于存储结果的数据结构 ---
    # 使用列表来收集数据比直接向DataFrame中追加行更高效
    results_data = {
        'original_image_name': [],
        'annotated_image_name': [],
        'pred_class': [],
        'confidence': [],
        'bbox_xyxy': []
    }

    # --- 5. 对所有图像进行推理（包括子文件夹） ---
    # 定义有效的图像文件扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    def image_batch_generator(source_path, valid_extensions, batch_size=1000):
        """
        生成器函数：流式分批加载图像路径，避免一次性加载所有路径导致内存溢出

        Args:
            source_path (Path): 图像根目录路径
            valid_extensions (set): 有效图像扩展名集合（如{'.jpg', '.png'}）
            batch_size (int): 每批加载的图像路径数量，默认1000

        Yields:
            list: 包含当前批次图像路径的列表
        """
        batch = []
        # 递归遍历所有子目录，收集符合扩展名的图像文件路径
        for file in source_path.rglob('*.*'):
            # 过滤有效扩展名且是文件的路径
            if file.suffix.lower() in valid_extensions and file.is_file():
                batch.append(file)
                # 达到批次大小时返回当前批次并重置
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        # 处理最后一批不足batch_size的剩余路径
        if batch:  # 处理最后一批
            yield batch

    # 使用生成器分批获取图像路径
    batch_size = 16  # 进一步减小推理批次大小至8（从16调整）
    logger.info(f"开始流式推理，每批预处理5000张图像路径，推理批次大小{batch_size}")

    try:
            import time
            # 添加进度跟踪变量
            processed_images = 0  # 初始化已处理图像计数器
            start_time = time.time()
            total_images = 0
            batch_num = 0
            processed_images = 0  # 跟踪已处理图像总数
            current_image_batch = []  # 初始化图像批次变量

            # 修复进度计算逻辑
            processed_images += len(current_image_batch)
            if processed_images % 100 == 0 and processed_images > 0:
                elapsed = time.time() - start_time
                # 已通过processed_images += len(current_image_batch)更新，此处无需重复累加
                total_to_process = total_images
                remaining = (elapsed / processed_images) * (total_to_process - processed_images)
                logger.info(f"已处理 {processed_images}/{total_to_process} 张图像...")

            # 预热模型，减少首次推理开销
            warmup_done = False

            # 创建图像生成器
            image_generator = image_batch_generator(source_path, valid_extensions, batch_size=250)

            # 流式处理图像批次
            # 每个批次独立存储图像路径，避免累积
            # 获取当前进程ID
            process = psutil.Process(os.getpid())
            # 确保在每个批次开始时重新获取进程信息以避免缓存问题
            for image_batch in image_generator:
                batch_num += 1
                current_batch_size = len(image_batch)
                total_images += current_batch_size
                # 记录批次开始时的内存使用
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"处理路径批次 {batch_num}，包含 {current_batch_size} 张图像，累计 {total_images} 张，内存使用: {mem_before:.2f}MB")
                # 将当前批次图像路径转换为字符串列表
                batch_paths = [str(img_path) for img_path in image_batch]
                # 其他循环体代码...

                pass  # 已整合到主异常处理结构中
            
            # 预热模型（仅在第一批图像时执行）
            if not warmup_done and current_batch_size > 0:
                warmup_img = str(image_batch[0])
                with torch.no_grad():
                    model.predict(source=warmup_img, device=device, task='detect', batch=1, half=True, verbose=False)
                logger.info('模型预热完成')
                warmup_done = True
            
            # 每个批次独立初始化结果列表
            results = []
            # 只处理当前批次的图像，不添加到总列表中
            total_sub_batches = (current_batch_size + batch_size - 1) // batch_size

            # 分推理批次处理当前路径批次（关键分批次逻辑）
            # 由于单批次图像过多可能导致显存不足，进一步拆分为更小的推理子批次
            for sub_batch_idx in range(total_sub_batches):
                sub_start = sub_batch_idx * batch_size  # 子批次起始索引
                sub_end = min(sub_start + batch_size, current_batch_size)  # 子批次结束索引（防越界）
                sub_batch_paths = batch_paths[sub_start:sub_end]  # 当前子批次图像路径列表
                # 计算全局子批次编号（用于日志追踪）
                sub_batch_num = (batch_num - 1) * total_sub_batches + sub_batch_idx + 1

                # 记录子批次开始时间
                sub_start_time = time.time()

                # 对当前子批次进行推理
                with torch.no_grad():
                    logger.info(f'处理推理子批次 {sub_batch_num}，共 {len(sub_batch_paths)} 张图像')
                    sub_results = model.predict(source=sub_batch_paths, device=device, task='detect', batch=batch_size, half=True, verbose=False)
                    results.extend(sub_results)

                # 记录子批次结束时间
                sub_elapsed = time.time() - sub_start_time
                logger.info(f'推理子批次 {sub_batch_num} 处理完成，耗时 {sub_elapsed:.2f} 秒，每秒处理 {len(sub_batch_paths)/sub_elapsed:.2f} 张图像')
                # 子批次处理后立即清理中间变量
                del sub_results
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                gc.collect()

            # 记录批次处理完成时的内存使用（CPU内存）
            mem_after_process = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f'路径批次 {batch_num} 处理完成（未清理前），内存使用: {mem_after_process:.2f}MB，内存变化: {mem_after_process - mem_before:.2f}MB')
            
            # 保存操作完成后执行内存清理（优化调整）
            # 记录清理前内存使用（CPU内存）
            mem_before_clean = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f'路径批次 {batch_num} 保存完成，开始清理内存，清理前内存使用: {mem_before_clean:.2f}MB')

            # 彻底清理内存（关键性能优化点）
            del batch_paths  # 释放批次路径列表内存，保留image_batch用于后续结果处理
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # 释放未使用的CUDA缓存（PyTorch内部管理的显存）
                torch.cuda.ipc_collect()  # 收集并释放跨进程通信（IPC）使用的共享内存
            # 强制Python垃圾回收机制运行，释放未引用的对象内存
            gc.collect()

            # 记录清理后内存使用
            mem_after_clean = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f'路径批次 {batch_num} 内存清理完成，清理后内存使用: {mem_after_clean:.2f}MB，内存减少: {mem_before_clean - mem_after_clean:.2f}MB')

            # 初始化批次独立的结果数据
            current_results_data = {
                'original_image_name': [],
                'annotated_image_name': [],
                'pred_class': [],
                'confidence': [],
                'bbox_xyxy': []
            }
            # 处理当前路径批次的推理结果
            logger.info(f'路径批次 {batch_num} 推理完成，开始处理结果')
            
            # 传递当前批次图像路径到结果处理
            current_image_batch = image_batch.copy()  # 创建副本保留路径信息
            if not current_image_batch:
                logger.error('当前批次图像路径为空')
                return
                
            # 只处理当前批次的推理结果
            batch_results = results[-len(current_image_batch):]
            # 清理原results列表，释放内存
            del results
            
            for i, res in enumerate(tqdm(batch_results, desc="正在处理推理结果")):
                img_path = current_image_batch[i]
                # 每处理100张图像更新一次进度时间信息
                processed_images += len(current_image_batch)
            if processed_images % 100 == 0 and processed_images > 0:
                    elapsed = time.time() - start_time
                    # 使用processed_images跟踪总进度，而非results列表
                    # 已通过processed_images += len(current_image_batch)更新，此处无需重复累加
                    total_to_process = total_images
                    remaining = (elapsed / processed_images) * (total_to_process - processed_images)
                    logger.info(f"已处理 {processed_images}/{total_to_process} 张图像，耗时 {elapsed:.2f} 秒，预计剩余 {remaining:.2f} 秒")





                # 验证结果类型并获取类别名称
                if not isinstance(res, Results):
                    logger.error(f"无效的结果类型: 预期Results对象，实际得到{type(res)}，索引{i}")
                    continue
                names = res.names

            # 读取原始图像，保持色彩信息
            with Image.open(img_path).convert('RGB') as original_image:  # 使用with语句确保图像文件正确关闭
                    original_array = np.array(original_image)

                # 生成标注图像
            annotated_image = res.plot(img=original_array)

                # 获取文件名和扩展名，添加_tl后缀
            filename = img_path.stem
            extension = img_path.suffix
            new_filename = f"{filename}_tl{extension}"

            # 保存标注图像
            Image.fromarray(annotated_image).save(output_images_path / new_filename)

            # 检查是否有检测到的边界框
            if res.boxes is not None and len(res.boxes) > 0:
                # 遍历每个检测到的边界框
                for box in res.boxes:
                    class_id = int(box.cls) # 类别ID
                    confidence = float(box.conf) # 置信度
                    bbox_coords = box.xyxy[0].cpu().numpy().astype(int).tolist() # 边界框坐标 [x1, y1, x2, y2]
                        
                    # 将结果添加到数据容器中
                    current_results_data['original_image_name'].append(img_path.name)
                    current_results_data['annotated_image_name'].append(new_filename)
                    current_results_data['pred_class'].append(names[class_id])
                    current_results_data['confidence'].append(round(confidence, 4))
                    current_results_data['bbox_xyxy'].append(",".join(map(str, bbox_coords)))
            else:
                # 如果没有检测到边界框，也添加一条记录，但填充空值
                current_results_data['original_image_name'].append(img_path.name)
                current_results_data['annotated_image_name'].append(new_filename)
                current_results_data['pred_class'].append('未检测到')
                current_results_data['confidence'].append(0.0)
                current_results_data['bbox_xyxy'].append('')

            # --- 6. 将结果保存到Excel ---
            if not current_results_data['original_image_name']:
                logger.warning(f"路径批次 {batch_num} 未检测到任何目标，继续处理下一批次")
            else:
                # 从当前批次收集的结果创建DataFrame
                batch_results_df = pd.DataFrame(current_results_data)
                # 删除不再需要的变量
            del current_image_batch
            del batch_results
            
            # 聚合当前批次结果
            agg_functions = {
                'pred_class': lambda x: '; '.join(x),
                'confidence': lambda x: '; '.join(map(str, x)),
                'bbox_xyxy': lambda x: '; '.join(x)
            }
            batch_agg_df = batch_results_df.groupby(['original_image_name', 'annotated_image_name']).agg(agg_functions).reset_index()
            
            # 删除不再需要的DataFrame
            del batch_results_df

            # 填充空值
            # 批次处理主逻辑（包含所有可能抛出异常的操作）
            try:
                batch_agg_df[['pred_class', 'confidence', 'bbox_xyxy']] = batch_agg_df[['pred_class', 'confidence', 'bbox_xyxy']].fillna('No Detection')

                # 保存批次结果到Excel
                if batch_num == 1:
                    with pd.ExcelWriter(output_excel_path, engine='openpyxl', mode='w') as writer:
                        batch_agg_df.to_excel(writer, index=False, sheet_name=f'Batch_{batch_num}')
                else:
                    with pd.ExcelWriter(output_excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                        batch_agg_df.to_excel(writer, index=False, sheet_name=f'Batch_{batch_num}')
                logger.info(f'路径批次 {batch_num} 结果已保存至: {output_excel_path}')

                # 清理操作
                del batch_agg_df
                current_results_data.clear()
                torch.cuda.empty_cache() if device.type == 'cuda' else None
            except Exception as e:
                logger.error(f"批量推理过程中发生错误: {e}", exc_info=True)
    finally:
        # 无论是否发生异常都执行的清理操作
        if 'current_results_data' in locals():
            del current_results_data
        if 'results' in locals():
            del results
        gc.collect()
        if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

if __name__ == '__main__':
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="运行YOLOv11分割模型推理。")
    # 默认路径配置（根据项目结构自动设置）
    default_weights = 'G:/soft/soft/python project/Yolo_metric_alarm/Yolo_seg_Alarm/gemini/yolo_seg_alarm/train2_results/weights/best.pt'
    default_source = 'D:/20250804'
    default_excel = 'G:/soft/soft/python project/Yolo_metric_alarm/Data/test/your_excel_file.xlsx'
    
    parser.add_argument('--weights', type=str, default=default_weights, help='模型权重文件路径 (默认: %(default)s)')
    parser.add_argument('--source', type=str, default=default_source, help='测试图片所在文件夹路径 (默认: %(default)s)')
    parser.add_argument('--output_excel', type=str, default=default_excel, help='结果Excel文件路径 (默认: %(default)s)')

    args = parser.parse_args()

    run_inference(args.weights, args.source, args.output_excel)