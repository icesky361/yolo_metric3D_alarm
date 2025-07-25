# -*- coding: utf-8 -*-
"""
================================================================================
@ 脚本名: infer.py
@ 功能: YOLOv11 实例分割模型推理脚本
@ 作者: TraeAI
@ 创建时间: 2024-07-28
--------------------------------------------------------------------------------
@ 脚本逻辑:
1.  **初始化与配置**: 
    - 导入所需库 (argparse, pandas, logging, ultralytics, torch等)。
    - 配置日志记录器，用于输出推理过程中的信息。

2.  **设备检测 (`get_device`)**:
    - 检查系统是否支持CUDA（即有无NVIDIA GPU）。
    - 如果支持，则设置设备为`cuda:0`并打印GPU型号。
    - 如果不支持，则设置设备为`cpu`，并提示用户将在CPU上运行。

3.  **推理主函数 (`run_inference`)**:
    - **a. 设置**: 
        - 调用`get_device()`确定运行设备。
        - 定义输入图片目录和输出Excel文件的路径。
        - 自动创建`results`文件夹用于存放结果。
    - **b. 加载模型**: 
        - 使用`ultralytics.YOLO`加载用户指定的、已经训练好的模型权重（`.pt`文件）。
        - 将模型移动到检测到的设备上（GPU或CPU）。
    - **c. 加载原始Excel数据**: 
        - 使用`pandas.read_excel`读取用户提供的原始Excel文件，该文件包含图片名称等信息。
    - **d. 准备结果容器**: 
        - 创建一个字典`results_data`，用于高效地收集每张图片的检测结果（类别、置信度、边界框、分割坐标）。
    - **e. 遍历图片并执行推理**: 
        - 遍历指定文件夹中的所有图片文件。
        - 使用`tqdm`显示处理进度条。
        - 对每张图片，调用`model.predict()`方法进行推理。
        - `predict`方法会返回一个包含所有检测结果的列表。
    - **f. 解析并保存结果**: 
        - 遍历推理结果，提取每个检测到的对象的：
            - 类别名称 (`pred_class`)
            - 置信度 (`confidence`)
            - 边界框坐标 (`bbox_xyxy`)
            - 实例分割掩码的多边形坐标 (`segmentation_xy`)
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
    python inference/infer.py --weights models/yolo_seg_experiment/weights/best.pt --source Data/test --excel_file Data/test/your_excel_file.xlsx
================================================================================
"""

import argparse
import pandas as pd
from pathlib import Path
import logging
from ultralytics import YOLO
from tqdm import tqdm
import torch
import cv2
import numpy as np

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device():
    """检测可用的硬件（GPU/CPU）。"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"检测到支持CUDA的GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logging.info("未找到支持CUDA的GPU。将在CPU上运行。")
    return device

def run_inference(weights_path: str, source_dir: str, excel_path: str):
    """
    对一个目录中的图像进行推理，并将结果保存到Excel文件中。

    Args:
        weights_path (str): 训练好的模型权重（.pt文件）的路径。
        source_dir (str): 包含测试图像的目录的路径。
        excel_path (str): 需要更新结果的Excel文件的路径。
    """
    # --- 1. 初始化设置 ---
    device = get_device()
    source_path = Path(source_dir) / 'images' # 图片文件夹路径
    # 定义输出Excel文件的路径，并确保其父目录存在
    output_excel_path = Path('results') / f"output_{Path(excel_path).name}"
    output_excel_path.parent.mkdir(exist_ok=True)

    # --- 2. 加载模型 ---
    try:
        model = YOLO(weights_path) # 从指定的权重文件加载模型
        model.to(device) # 将模型移动到检测到的设备上
        model.eval()  # 设置为评估模式，减少内存使用并提高推理速度
        logging.info(f"成功从 {weights_path} 加载模型")
    except Exception as e:
        logging.error(f"加载模型失败。错误: {e}")
        return

    # --- 3. 加载Excel数据 ---
    try:
        df = pd.read_excel(excel_path)
        logging.info(f"成功从 {excel_path} 加载Excel文件")
    except FileNotFoundError:
        logging.error(f"Excel文件未找到: {excel_path}")
        return
    except Exception as e:
        logging.error(f"读取Excel文件时出错: {e}")
        return

    # --- 4. 准备用于存储结果的数据结构 ---
    # 使用列表来收集数据比直接向DataFrame中追加行更高效
    results_data = {
        'image_name': [],
        'pred_class': [],
        'confidence': [],
        'bbox_xyxy': [],
        'segmentation_xy': []
    }

    # --- 5. 对所有图像进行推理 ---
    image_files = list(source_path.glob('*.*')) # 获取所有图片文件
    logging.info(f"找到 {len(image_files)} 张图片待处理。")

    # 使用tqdm创建进度条
    for img_path in tqdm(image_files, desc="正在进行推理"):
        try:
            # model.predict方法处理所有事情：图像预处理、推理、后处理
            with torch.no_grad():  # 禁用梯度计算，减少内存占用
                results = model.predict(source=str(img_path), device=device)

            # 处理单张图片的所有检测结果
            for res in results:
                # 如果没有检测到任何边界框或掩码，则跳过
                if res.masks is None and res.boxes is None:
                    continue

                # 从模型结果中获取类别名称
                names = res.names

                # 遍历每个检测到的边界框
                for i, box in enumerate(res.boxes):
                    class_id = int(box.cls) # 类别ID
                    confidence = float(box.conf) # 置信度
                    bbox_coords = box.xyxy[0].cpu().numpy().astype(int).tolist() # 边界框坐标 [x1, y1, x2, y2]
                    
                    # 获取当前对象的分割掩码坐标
                    if res.masks is not None and len(res.masks.xy) > i:
                        seg_coords = res.masks.xy[i].astype(int).tolist() # 分割多边形坐标 [[x1,y1],[x2,y2],...]
                        # 将坐标展平并用逗号连接成字符串
                        seg_str = ",".join(map(str, np.array(seg_coords).flatten()))
                    else:
                        seg_str = "N/A" # 如果没有分割信息

                    # 将结果添加到数据容器中
                    results_data['image_name'].append(img_path.name)
                    results_data['pred_class'].append(names[class_id])
                    results_data['confidence'].append(round(confidence, 4))
                    results_data['bbox_xyxy'].append(",".join(map(str, bbox_coords)))
                    results_data['segmentation_xy'].append(seg_str)

        except Exception as e:
            logging.error(f"处理图片 {img_path.name} 时发生错误: {e}", exc_info=True)
            torch.cuda.empty_cache()  # 清理GPU内存，防止内存溢出

    # --- 6. 将结果保存到Excel ---
    if not results_data['image_name']:
        logging.warning("在任何图片中都未检测到目标。")
        return

    # 从收集的结果创建一个新的DataFrame
    results_df = pd.DataFrame(results_data)

    # 为了与原始DataFrame合并，我们需要一个共同的键。这里我们使用图片名称。
    # 首先，重命名结果列以匹配原始Excel中的列名 '图片名称'。
    results_df.rename(columns={'image_name': '图片名称'}, inplace=True)

    # 由于一张图片可能检测到多个目标，我们需要对结果进行聚合
    # 我们将按图片名称分组，并用分号连接结果
    agg_functions = {
        'pred_class': lambda x: '; '.join(x),
        'confidence': lambda x: '; '.join(map(str, x)),
        'bbox_xyxy': lambda x: '; '.join(x),
        'segmentation_xy': lambda x: '; '.join(x)
    }
    results_agg_df = results_df.groupby('图片名称').agg(agg_functions).reset_index()

    # 将聚合后的结果合并回原始的DataFrame中
    # 我们使用左连接（left merge）来确保所有原始行都被保留
    final_df = pd.merge(df, results_agg_df, on='图片名称', how='left')

    # 为没有检测到目标的图片填充'No Detection'
    final_df[['pred_class', 'confidence', 'bbox_xyxy', 'segmentation_xy']] = final_df[['pred_class', 'confidence', 'bbox_xyxy', 'segmentation_xy']].fillna('No Detection')

    try:
        final_df.to_excel(output_excel_path, index=False)
        logging.info(f"推理完成。结果已保存至: {output_excel_path.resolve()}")
    except Exception as e:
        logging.error(f"保存Excel文件失败: {e}")

if __name__ == '__main__':
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="运行YOLOv11分割模型推理。")
    parser.add_argument('--weights', type=str, required=True, help='训练好的.pt模型文件的路径。')
    parser.add_argument('--source', type=str, required=True, help='源目录的路径 (例如, ../Data/test)。')
    parser.add_argument('--excel_file', type=str, required=True, help='对应的Excel文件的路径。')

    args = parser.parse_args()

    run_inference(args.weights, args.source, args.excel_file)