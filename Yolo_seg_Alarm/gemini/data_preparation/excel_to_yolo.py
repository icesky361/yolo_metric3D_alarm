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
import json  # 确保已导入json模块
from datetime import datetime
import threading
from tkinter import messagebox  #
import time
from datetime import datetime
import argparse
from functools import partial
import csv

progress_lock = threading.Lock()
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
# 添加进度文件路径函数（插入在此处）
def get_progress_file_path(data_split):
    # 获取当前脚本所在目录（data_preparation）
    current_dir = Path(__file__).parent
    progress_dir = current_dir / 'progress'
    progress_dir.mkdir(exist_ok=True)
    return progress_dir / f'{data_split}_progress.json'
#  添加进度加载函数
def load_progress(progress_file):
    """加载已处理的图片名称列表"""
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"进度文件 {progress_file} 读取失败，将重新开始处理: {e}")
    return []
#  添加进度保存函数
def save_progress(progress_file, processed_images, start_time):
    """保存已处理的图片名称列表和处理进度"""
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(processed_images, f, ensure_ascii=False, indent=2)
        # 记录进度保存日志
        elapsed = time.time() - start_time
        logging.info(f"已保存 {len(processed_images)} 个处理项，耗时 {elapsed:.2f} 秒")
    except IOError as e:
        logging.error(f"进度文件 {progress_file} 写入失败: {e}")

# 修改函数定义，添加processed_images和progress_file参数
def convert_to_yolo_format(df, class_mapping, image_cache, label_output_dir, processed_images, progress_file):
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
        processed_images (list): 已处理图片名称列表
        progress_file (Path): 进度文件路径
    """
    # 筛选未处理的行
    df_filtered = df[~df['图片名称'].isin(processed_images)]
    total = len(df_filtered)
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
                            # 处理分号分隔的多个坐标组
                            # 正确处理1:N的类别-坐标组关系
                            class_names = [cn.strip() for cn in row['告警事由'].split(',')]
                            # 坐标组仅做一次分号分割
                            coords_list = [coord.strip() for coord in str(row['坐标']).split(';') if coord.strip()]
                            
                            # 验证类别和坐标组数量匹配（1:N关系）
                            if not class_names or len(coords_list) == 0:
                                logging.error(f"图片 {image_name} 缺少类别或坐标信息，跳过。")
                                error_count += 1
                                continue
                            
                            # 移除内部坐标组分号分割逻辑
                            # coord_groups = [g.strip() for g in coord_str.split(';') if g.strip()]
                            coord_groups = [coord_str.strip()]
                            for coord_group in coord_groups:
                                coord_group = coord_group.strip()
                                if not coord_group:
                                    continue
                                
                                # 新增：验证方括号完整性
                                if not (coord_group.startswith('[') and coord_group.endswith(']')):
                                    logging.error(f"图片 {image_name} 的坐标组 '{coord_group}' 方括号不完整，跳过。")
                                    success = False
                                    break
                                
                                # 移除方括号并清理空格
                                coord_str_clean = coord_group.strip('[]').strip()
                                coords = [float(p.strip()) for p in coord_str_clean.split(',')]
                                # 新增坐标范围验证
                                if any(coord < 0 for coord in coords):
                                    logging.error(f"图片 {image_name} 的坐标组 '{coord_group}' 包含负值，跳过。")
                                    success = False
                                    break
                                if len(coords) != 4:
                                    logging.error(f"图片 {image_name} 的坐标组 '{coord_group}' 格式无效，需要4个值，实际{len(coords)}个，跳过。")
                                    success = False
                                    break
                                
                                # 归一化坐标 (x_center, y_center, width, height)
                                x1, y1, w, h = coords
                                x_center = (x1 + w/2) / img_w
                                y_center = (y1 + h/2) / img_h
                                width = w / img_w
                                height = h / img_h
                                normalized_coords = [x_center, y_center, width, height]

                            # 1. 坐标预处理 - 仅移除首尾空白和括号（兼容有/无括号两种格式）
                            coord_str_clean = coord_str.strip().strip('[](){}')
                            if not coord_str_clean:
                                raise ValueError("坐标字符串为空")

                            # 2. 坐标分割 - 仅使用逗号分割（与旧旧.py保持一致）
                            # 支持中英文逗号和可能的空格（如"100, 200, 300, 400"）
                            coord_values = re.split(r'[,，\s]+', coord_str_clean)
                            coords = [float(coord.strip()) for coord in coord_values if coord.strip()]

                            # 3. 坐标验证 - 支持矩形(4值)和多边形(N值)两种格式
                            if len(coords) < 4:
                                raise ValueError(f"坐标数量不足，至少需要4个值，实际获得{len(coords)}个")

                            # 4. 针对矩形坐标的归一化处理（与旧旧.py逻辑完全一致）
                            if len(coords) == 4:
                                x1, y1, w, h = coords
                                # 添加坐标有效性检查（非负性验证）
                                if x1 < 0 or y1 < 0 or w <= 0 or h <= 0:
                                    raise ValueError(f"坐标值无效，包含负数或零: x1={x1}, y1={y1}, w={w}, h={h}")
                                x_center = (x1 + w/2) / img_w
                                y_center = (y1 + h/2) / img_h
                                width = w / img_w
                                height = h / img_h
                            else:
                                # 5. 多边形坐标处理（新增功能，旧旧.py未实现）
                                normalized_coords = [coord / img_w if i % 2 == 0 else coord / img_h for i, coord in enumerate(coords)]
                                x_center, y_center, width, height = calculate_bbox_from_polygon(normalized_coords)

                            normalized_coords = [x_center, y_center, width, height]

                            # 如果所有验证通过，处理标注文件
                            class_id = class_mapping.get(cls, -1)
                            if class_id == -1:
                                continue

                            with open(label_path, 'a', encoding='utf-8') as f:
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            result['success'] = True
                            
                        except ValueError as e:
                            # 增强错误日志，输出原始坐标字符串便于调试
                            logging.error(f"图片 {image_name} 的坐标 '{coord_str}' 解析失败: {str(e)}")
                            success = False
                            break
                        except PermissionError as e:
                            logging.error(f"文件 {label_path} 写入失败: {str(e)}")
                            continue
                        
                # 保存进度（使用线程锁）
                if result['success'] :
                    with progress_lock:
                        processed_images.append(image_name)
                        if len(processed_images) % 10 == 0:
                            save_progress(progress_file, processed_images, start_time)
                    result['success'] = True
                    
            except ValueError as e:
                logging.error(f"处理图片 {image_name} 时发生验证错误: {str(e)}")
            except Exception as e:
                logging.error(f"处理图片 {image_name} 时发生未知错误: {str(e)}")
                result['error'] = str(e)  # 记录错误信息
            # 计算处理时间
            result['time'] = time.time() - process_start_time
            return result  # 返回统一的result字典
            return {
                'success': success,
                'image_name': image_name,
                'missing': image_name if not image_path else None,
                'error': error_occurred
            }
def main():
    # 添加参数解析器
    parser = argparse.ArgumentParser(description='Excel转换为YOLO格式标注工具')
    parser.add_argument('--threads', type=int, default=4, help='线程数量（默认：4）')
    args = parser.parse_args()
    
    # 配置日志
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"excel_to_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 定义错误日志路径
    error_log_path = log_path  # 可以根据需要更改为不同的文件路径
    
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
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
    # 错误日志初始化代码
        # ===== 1. 定义输出根目录（参考log_dir的设计）=====
    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_root, exist_ok=True)
        # ===== 2. 数据集处理循环 =====
    for data_split in ['train', 'val']:
        # 为每个数据集创建输出目录
        output_dir = os.path.join(output_root, data_split)
        os.makedirs(output_dir, exist_ok=True)
    # 3. 初始化错误日志（使用当前数据集的output_dir）
    error_log_path = os.path.join(output_dir, 'error_records.csv')
    with open(error_log_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['图片名称', '行索引', '错误类型', '原告警事由', '原始坐标', '错误详情'])

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
        image_cache = build_image_cache(image_base_dir)
        logging.info(f"已构建图片缓存，包含 {len(image_cache)} 张图片")

        # --- 步骤3: 断点续跑逻辑 ---
        progress_file = get_progress_file_path(data_split)
        processed_images = load_progress(progress_file)
        start_time = time.time()  # 记录当前数据集开始时间

        # --- 步骤4: 筛选未处理数据 ---
        df_filtered = df_labels[~df_labels['图片名称'].isin(processed_images)]
        total_to_process = len(df_filtered)
        if total_to_process == 0:
            logging.info(f"所有 {data_split} 数据均已处理完成")
            progress_file.unlink(missing_ok=True)
            continue
        logging.info(f"发现 {total_to_process} 个未处理项目，开始处理...")

        # --- 关键修改：初始化成功计数器 --- 
        success_count = 0

        # --- 关键修改：使用线程池处理行数据 --- 
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            # 准备任务参数
            # 在main函数的任务创建处添加start_time参数
            tasks = [(row, class_mapping, image_cache, processed_images, progress_file, progress_lock, start_time, error_log_path) 
            for _, row in df_filtered.iterrows()]
           # 使用partial固定参数，提高可读性
            process_func = partial(process_single_row, class_mapping=class_mapping, image_cache=image_cache, processed_images=processed_images, 
                           progress_file=progress_file, progress_lock=progress_lock, start_time=start_time, error_log_path=error_log_path)
            # 执行并行处理
            results = list(tqdm(
                executor.map(lambda p: process_single_row(*p), tasks),
                total=len(tasks),
                desc=f"处理{data_split} 数据集", unit="项",leave=True))

        # --- 关键修改：统计结果 --- 
        for result in results:
            try:
                if result['success']:
                    success_count += 1
                    processed_images.append(result['image_name'])
                elif result['missing']:
                    stats[data_split]['missing'].append(result['missing'])
            except Exception as e:
                logging.error(f"处理结果时发生错误: {str(e)}, 结果数据: {result}")
                error_count += 1

        # --- 关键修改：更新统计信息 --- 
        stats[data_split]['success'] = success_count
        stats[data_split]['error'] = total_to_process - success_count
        stats[data_split]['time'] = time.time() - start_time

        # --- 关键修改：进度处理 --- 
        if success_count == total_to_process:
            progress_file.unlink(missing_ok=True)
            logging.info(f"{data_split} 数据集全部处理完成，已删除进度文件")
        else:
            save_progress(progress_file, processed_images, start_time)
            logging.info(f"{data_split} 数据集部分处理完成，共处理 {len(processed_images)} 张图片,进度已保存")

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