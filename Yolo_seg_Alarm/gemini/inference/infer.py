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
from PIL import Image

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
    source_path = Path(source_dir) / 'images' # 图片文件夹路径
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
        model = YOLO(weights_path, task='detect') # 初始化指定检测任务 (移除不支持的mode参数)
        model.to(device)  # 移动到目标设备
        model.eval()      # 启用PyTorch评估模式
        # 兼容旧版本: 显式设置推理模式
        if hasattr(model, 'mode'):
            model.mode = 'predict'
        # 验证推理配置
        assert model.task == 'detect', f'模型任务错误: 预期detect，实际{model.task}'
        assert hasattr(model, 'predict'), '模型没有predict方法'
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
    # 只获取有效的图像文件
    image_files = [
        file for file in source_path.rglob('*.*') 
        if file.suffix.lower() in valid_extensions and file.is_file()
    ]
    
    logger.info(f"找到 {len(image_files)} 张有效图像文件待处理（包括子文件夹）。")
    if len(image_files) == 0:
        logger.warning(f"在路径 {source_path} 及其子文件夹中未找到任何有效图像文件。")
        logger.warning("请确保该路径包含图像文件（.jpg, .jpeg, .png, .bmp等），或使用 --source 参数指定包含图像的目录。")
        return

    # 设置模型模式并只记录一次
    model.mode = 'predict'
    logger.info(f'推理前模型模式: {model.mode}')
    
    # 使用tqdm创建进度条
    for img_path in tqdm(image_files, desc="正在进行推理"):
        try:
            # 确保文件存在
            if not img_path.exists():
                logger.error(f"文件不存在: {img_path}")
                continue
            
            # model.predict方法处理所有事情：图像预处理、推理、后处理
            with torch.no_grad():  # 禁用梯度计算，减少内存占用
                # 降低日志级别为DEBUG，避免大量冗余输出
                logger.debug(f'正在处理图像: {img_path.name}')
                results = model.predict(source=str(img_path), device=device, task='detect', batch=64,half=True, verbose=False)

            # 处理单张图片的所有检测结果
            for res in results:
                # 如果没有检测到任何边界框或掩码，则跳过
                if res.masks is None and res.boxes is None:
                    continue

                # 从模型结果中获取类别名称
                names = res.names

            # 读取原始图像，保持色彩信息
            original_image = Image.open(img_path).convert('RGB')
            original_array = np.array(original_image)

            # 生成标注图像
            annotated_image = results[0].plot(img=original_array)

            # 获取文件名和扩展名，添加_tl后缀
            filename = img_path.stem
            extension = img_path.suffix
            new_filename = f"{filename}_tl{extension}"

            # 保存标注图像
            Image.fromarray(annotated_image).save(output_images_path / new_filename)

            # 检查是否有检测到的边界框
            if len(res.boxes) > 0:
                # 遍历每个检测到的边界框
                for i, box in enumerate(res.boxes):
                    class_id = int(box.cls) # 类别ID
                    confidence = float(box.conf) # 置信度
                    bbox_coords = box.xyxy[0].cpu().numpy().astype(int).tolist() # 边界框坐标 [x1, y1, x2, y2]
                    
                    # 将结果添加到数据容器中
                    results_data['original_image_name'].append(img_path.name)
                    results_data['annotated_image_name'].append(new_filename)
                    results_data['pred_class'].append(names[class_id])
                    results_data['confidence'].append(round(confidence, 4))
                    results_data['bbox_xyxy'].append(",".join(map(str, bbox_coords)))
            else:
                # 如果没有检测到边界框，也添加一条记录，但填充空值
                results_data['original_image_name'].append(img_path.name)
                results_data['annotated_image_name'].append(new_filename)
                results_data['pred_class'].append('未检测到')
                results_data['confidence'].append(0.0)
                results_data['bbox_xyxy'].append('')

        except Exception as e:
            logger.error(f"处理图片 {img_path.name} 时发生错误: {e}", exc_info=True)
            torch.cuda.empty_cache()  # 清理GPU内存，防止内存溢出

    # --- 6. 将结果保存到Excel ---
    if not results_data['original_image_name']:
        logger.warning("在任何图片中都未检测到目标。")
        return

    # 从收集的结果创建一个新的DataFrame
    results_df = pd.DataFrame(results_data)

    # 由于一张图片可能检测到多个目标，我们需要对结果进行聚合
    # 我们将按原始图片名称和标注图片名称分组，并用分号连接结果
    agg_functions = {
        'pred_class': lambda x: '; '.join(x),
        'confidence': lambda x: '; '.join(map(str, x)),
        'bbox_xyxy': lambda x: '; '.join(x)
    }
    results_agg_df = results_df.groupby(['original_image_name', 'annotated_image_name']).agg(agg_functions).reset_index()

    # 为没有检测到目标的图片填充'No Detection'
    results_agg_df[['pred_class', 'confidence', 'bbox_xyxy']] = results_agg_df[['pred_class', 'confidence', 'bbox_xyxy']].fillna('No Detection')

    # 使用结果数据作为最终DataFrame
    final_df = results_agg_df

    try:
        final_df.to_excel(output_excel_path, index=False)
        logging.info(f"推理完成。结果已保存至: {output_excel_path.resolve()}")
    except Exception as e:
        logging.error(f"保存Excel文件失败: {e}")

if __name__ == '__main__':
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="运行YOLOv11分割模型推理。")
    # 默认路径配置（根据项目结构自动设置）
    default_weights = 'G:/soft/soft/python project/Yolo_metric_alarm/Yolo_seg_Alarm/gemini/yolo_seg_alarm/train2_results/weights/best.pt'
    default_source = 'G:/soft/soft/python project/Yolo_metric_alarm/Data/test'
    default_excel = 'G:/soft/soft/python project/Yolo_metric_alarm/Data/test/your_excel_file.xlsx'
    
    parser.add_argument('--weights', type=str, default=default_weights, help='模型权重文件路径 (默认: %(default)s)')
    parser.add_argument('--source', type=str, default=default_source, help='测试图片所在文件夹路径 (默认: %(default)s)')
    parser.add_argument('--output_excel', type=str, default=default_excel, help='结果Excel文件路径 (默认: %(default)s)')

    args = parser.parse_args()

    run_inference(args.weights, args.source, args.output_excel)