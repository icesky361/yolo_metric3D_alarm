# -*- coding: utf-8 -*-
"""
==========================================================================================
脚本描述:
    本脚本用于将Excel格式的标注数据转换为YOLOv8实例分割任务所需的.txt标签格式。
    它会读取包含图片名称、类别和坐标的Excel文件，然后为每张图片生成一个对应的.txt文件。
    同时，它还会将原始图片复制或创建符号链接到YOLO格式的数据集目录中，构建起符合YOLO训练要求的
    `images` 和 `labels` 文件夹结构。

核心逻辑:
1.  **读取配置**: 从 `configs/yolov11_seg.yaml` 文件中加载类别名称和索引的映射关系。
2.  **用户交互**: 弹出文件选择对话框，让用户选择包含训练集和验证集标注的Excel文件。
3.  **数据转换**: 
    - 遍历Excel中的每一行标注。
    - 读取图片尺寸以进行坐标归一化。
    - 将左上角和右下角的像素坐标 (x1, y1, x2, y2) 转换为YOLO格式的中心点坐标和宽高 (x_center, y_center, width, height)，并进行归一化。
    - 如果是分割任务，它会处理多边形点，并同样进行归一化。
    - 将 `class_id` 和归一化后的坐标写入到与图片同名的.txt文件中。
4.  **文件整理**: 将原始图片文件复制到目标数据集的 `images` 文件夹下，将生成的.txt标签文件保存到 `labels` 文件夹下，从而形成完整的YOLO数据集。

使用方法:
    直接在项目根目录（gemini/）下运行此脚本:
    `python data_preparation/excel_to_yolo.py`
    根据提示分别为训练集和验证集选择对应的Excel标注文件。
==========================================================================================
"""

import pandas as pd
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import yaml
from tqdm import tqdm
import logging
import shutil
import cv2
from concurrent.futures import ThreadPoolExecutor

# --- 日志配置 --- 
# 配置日志记录器，同时输出到控制台和文件
import time
from datetime import datetime
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"excel_to_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info(f"日志文件已保存至: {log_file}")

def select_excel_file(title="请选择Excel标注文件"):
    """
    功能: 弹出一个文件选择对话框，让用户选择一个Excel文件。
    参数:
        title (str): 对话框窗口的标题。
    返回:
        str: 用户选择的文件的完整路径。如果用户取消选择，则返回None。
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏Tkinter的根窗口
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))
    )
    return file_path

def get_class_mapping(config_path):
    """
    功能: 从YAML配置文件中读取类别名称，并返回一个从类别名到类别索引的映射字典。
    参数:
        config_path (str or Path): YAML配置文件的路径。
    返回:
        dict: 格式为 {'类别名': 索引, ...} 的字典。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 创建一个从名称到索引的反向映射
    return {name: index for index, name in config['names'].items()}

def build_image_cache(base_dir):
    cache = {}
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for path in base_dir.rglob(f'*{ext}'):
            cache[path.name] = path
    return cache
# 修改函数参数，将image_base_dir改为image_cache
def convert_to_yolo_format(df, class_mapping, image_cache, label_output_dir):
    success_count = 0
    missing_images = []
    error_count = 0
    start_time = time.time()
    """
    功能: 将DataFrame中的标注转换为YOLO .txt文件，并直接保存在图片所在目录。
    参数:
        df (pd.DataFrame): 包含标注信息的DataFrame。
        class_mapping (dict): 类别名到类别ID的映射字典。
        image_cache (dict): 图片名称到路径的缓存字典
        label_output_dir (Path): 存放标签文件的根目录
    """
    logging.info(f"正在处理 {len(df)} 条标注数据...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="转换标注为YOLO格式"):
        image_name = row['图片名称']
        # 处理多个类别（逗号分隔）
        class_names = [cn.strip() for cn in row['告警事由'].split(',')]
        # 处理多个坐标（分号分隔）
        coords_list = [coord.strip() for coord in str(row['坐标']).split(';')]

        # --- 步骤1: 使用缓存查找图片 ---
        image_path = image_cache.get(image_name)
        if not image_path:
            logging.warning(f"在缓存中未找到图片，跳过: {image_name}")
            missing_images.append(image_name)
            continue

        # --- 步骤2: 获取图片尺寸 --- 
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logging.error(f"无法读取图片: {image_path}")
                continue
            img_h, img_w, _ = img.shape
        except Exception as e:
            logging.error(f"使用OpenCV读取图片 {image_path} 时出错: {e}")
            error_count += 1
            continue

        # --- 步骤3: 验证类别和坐标数量匹配 --- 
        if len(class_names) != len(coords_list):
            logging.error(f"图片 {image_name} 的类别数量与坐标数量不匹配，跳过。")
            error_count += 1
            continue

        # --- 步骤4: 处理所有类别和坐标并写入文件 --- 
        label_path = image_path.with_suffix('.txt')
        success = True
        # 添加文件写入重试机制以处理文件锁定问题
        retries = 5  # 增加重试次数
        success_write = False
        for attempt in range(retries):
            try:
                # 使用临时文件写入策略避免锁定问题
                temp_path = label_path.with_suffix('.txt.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    for class_name, coord_str in zip(class_names, coords_list):
                        # --- 步骤5: 解析并转换坐标 --- 
                        try:
                            # 移除可能的方括号并分割坐标
                            coord_str_clean = coord_str.strip('[]')
                            coords = [float(p) for p in coord_str_clean.split(',')]
                            if len(coords) != 4:
                                logging.error(f"图片 {image_name} 的坐标 '{coord_str}' 格式无效，需要4个值，跳过。")
                                success = False
                                break

                            # 归一化坐标 (x_center, y_center, width, height)
                            x1, y1, w, h = coords
                            x_center = (x1 + w/2) / img_w
                            y_center = (y1 + h/2) / img_h
                            width = w / img_w
                            height = h / img_h
                            normalized_coords = [x_center, y_center, width, height]
                        except ValueError:
                            logging.error(f"图片 {image_name} 的坐标格式无效: '{coord_str}'，跳过。")
                            success = False
                            break

                        # --- 步骤6: 获取类别ID --- 
                        if class_name not in class_mapping:
                            logging.warning(f"类别 '{class_name}' 不在配置文件中，跳过。")
                            success = False
                            break
                        class_id = class_mapping[class_name]

                        # --- 步骤7: 生成YOLO格式字符串并写入 --- 
                        yolo_line = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_coords]) + "\n"
                        f.write(yolo_line)
                # 原子操作替换目标文件并处理可能的异常
                if temp_path.exists():
                    try:
                        os.replace(temp_path, label_path)
                    except PermissionError as e:
                        if attempt < retries - 1:
                            time.sleep(0.5)
                            logging.warning(f"文件替换失败，将重试: {e}")
                            continue
                        else:
                            if temp_path.exists():
                                temp_path.unlink()
                            logging.error(f"所有重试均失败，无法写入文件: {label_path}")
                            raise
                success_write = True
                break
            except PermissionError:
                if attempt < retries - 1:
                    time.sleep(1.0)  # 延长等待时间至1秒
                    logging.warning(f"文件 {label_path} 被占用，正在重试...(尝试 {attempt+1}/{retries})")
                else:
                    logging.error(f"无法写入文件 {label_path}，可能被其他程序占用")
        if success and success_write:
            success_count += 1
        else:
            error_count += 1
    end_time = time.time()
    total_time = end_time - start_time
    return success_count, missing_images, error_count, total_time

def main():
    """
    主函数，用于编排整个数据转换流程。
    新逻辑: 直接在原始图片目录下生成标签文件，不再复制图片。
    """
    CONFIG_PATH = Path('configs/yolov11_seg.yaml')
    # 修正数据根目录路径：从gemini目录上移两级到项目根目录
    DATA_ROOT = Path('../../Data/raw')

    if not CONFIG_PATH.exists():
        logging.error(f"配置文件未找到: {CONFIG_PATH}")
        return
    class_mapping = get_class_mapping(CONFIG_PATH)
    logging.info(f"已加载类别映射: {class_mapping}")

    # 初始化统计变量
    stats = {
        'train': {'success': 0, 'missing': [], 'error': 0, 'time': 0, 'count': 0},
        'val': {'success': 0, 'missing': [], 'error': 0, 'time': 0, 'count': 0}
    }
    total_start_time = time.time()

    for data_split in ['train', 'val']:
        logging.info(f"\n----- 正在处理 '{data_split}' 数据集 -----")
        # --- 步骤1: 让用户选择标注文件 ---
        excel_path = select_excel_file(title=f"请为【{data_split}】数据集选择Excel标注文件")
        if not excel_path:
            logging.warning(f"未给 '{data_split}' 数据集选择Excel文件，跳过。")
            continue
        df_labels = pd.read_excel(excel_path)
        stats[data_split]['count'] = len(df_labels)

        # --- 步骤2: 定义图片根目录 --- 
        image_base_dir = DATA_ROOT / data_split / 'images'
        if not image_base_dir.exists():
            logging.error(f"图片根目录不存在，跳过: {image_base_dir}")
            continue

        # --- 新增: 构建图片缓存 --- 
        logging.info(f"正在构建{data_split}数据集图片路径缓存...")
        image_cache = build_image_cache(image_base_dir)
        logging.info(f"共缓存 {len(image_cache)} 张图片路径")

        # --- 步骤3: 直接覆盖现有标签文件 --- 
        logging.info(f"将直接覆盖 '{data_split}' 数据集中已存在的.txt标签文件...")

        # --- 步骤4: 转换标注文件 --- 
        # 修改: 传入image_cache代替image_base_dir
        success, missing, errors, process_time = convert_to_yolo_format(
            df_labels, class_mapping, image_cache, image_base_dir
        )
        stats[data_split]['success'] = success
        stats[data_split]['missing'] = missing
        stats[data_split]['error'] = errors
        stats[data_split]['time'] = process_time

    # 计算总统计
    total_time = time.time() - total_start_time
    total_processed = stats['train']['count'] + stats['val']['count']
    total_success = stats['train']['success'] + stats['val']['success']
    total_missing = len(stats['train']['missing']) + len(stats['val']['missing'])
    total_error = stats['train']['error'] + stats['val']['error']

    # 计算每百条数据平均处理时长
    if total_processed > 0:
        avg_per_hundred = (total_time / total_processed) * 100
    else:
        avg_per_hundred = 0

    # 输出统计日志
    logging.info("\n===== 数据处理统计报告 ======")
    logging.info(f"总处理时间: {total_time:.2f}秒")
    logging.info(f"总处理图片数量: {total_processed}张")
    logging.info(f"成功生成标签文件: {total_success}个")
    logging.info(f"未找到图片: {total_missing}张")
    logging.info(f"处理出错: {total_error}条")
    logging.info(f"每百张图片平均处理时长: {avg_per_hundred:.2f}秒")
    
    # 新增: 输出各数据集详细统计
    for split in ['train', 'val']:
        if stats[split]['count'] > 0:
            logging.info(f"\n{split}数据集统计:")
            logging.info(f"  处理图片数量: {stats[split]['count']}张")
            logging.info(f"  成功生成标签: {stats[split]['success']}个")
            logging.info(f"  处理时间: {stats[split]['time']:.2f}秒")
            logging.info(f"  处理速度: {stats[split]['count']/stats[split]['time']:.2f}张/秒")

    # 导出未找到图片列表到Excel
    report_data = []
    for split in ['train', 'val']:
        for img in stats[split]['missing']:
            report_data.append({'数据集': split, '未找到图片名称': img})

    if report_data:
        report_df = pd.DataFrame(report_data)
        report_dir = Path(__file__).parent.parent / 'reports'
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"missing_images_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        report_df.to_excel(report_path, index=False)
        logging.info(f"未找到图片报告已生成: {report_path}")
    else:
        logging.info("所有图片均已成功找到，无需生成报告")

    logging.info("\n数据准备完成!")
    logging.info(f"YOLO格式的标签文件已直接在原始图片目录中生成: {DATA_ROOT.resolve()}")

if __name__ == '__main__':
    main()

def process_row(row, class_mapping, image_cache):
    # 提取单条数据处理逻辑
    ...

# 使用线程池并行处理
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(tqdm(executor.map(process_row, df.itertuples()), total=len(df)))
