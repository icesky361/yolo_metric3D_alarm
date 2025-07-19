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
# 修改函数定义，添加processed_images和progress_file参数
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


def process_single_row(row_tuple, class_mapping, image_cache, error_log_path):
    """处理单行数据，返回处理结果字典

    Args:
        row_tuple (tuple): 包含行索引和行数据的元组.
        class_mapping (dict): 类别映射字典.
        image_cache (dict): 图片缓存字典.
        error_log_path (str): 错误日志文件路径.

    Returns:
        dict: 包含处理结果的字典.
    """
    index, row = row_tuple
    image_name = row.get('图片名称', '')

    result = {
        'success': False,
        'image_name': image_name,
        'missing': None,
        'error': None,
        'time': 0
    }

    if not image_name:
        result['error'] = '列 "图片名称" 为空或不存在'
        return result

    process_start_time = time.time()

    try:
        # 1. 查找图片
        image_path = image_cache.get(image_name)
        if not image_path:
            result['missing'] = image_name
            return result

        # 2. 读取图片尺寸
        img = cv2.imread(str(image_path))
        if img is None:
            result['error'] = f'无法读取图片: {image_path}'
            return result
        img_h, img_w, _ = img.shape

        # 3. 处理类别和坐标
        class_names_str = row.get('告警事由', '')
        coords_str = str(row.get('坐标', ''))
        
        if not class_names_str or not coords_str:
            result['error'] = '缺少告警事由或坐标信息'
            return result

        class_names = [cn.strip() for cn in class_names_str.split(',')]
        coords_list = [coord.strip() for coord in coords_str.split(';') if coord.strip()]

        if len(class_names) != len(coords_list):
            result['error'] = f'类别和坐标数量不匹配 ({len(class_names)} vs {len(coords_list)})'
            return result

        # 4. 创建标签文件
        label_path = image_path.with_suffix('.txt')
        # 使用临时文件写入，避免多线程冲突
        temp_label_path = label_path.with_suffix('.txt.tmp')
        
        with open(temp_label_path, 'w', encoding='utf-8') as f:
            for class_name, coord_str in zip(class_names, coords_list):
                if not (coord_str.startswith('[') and coord_str.endswith(']')):
                    raise ValueError(f'坐标格式错误: {coord_str}')

                coord_str_clean = coord_str.strip('[]').strip()
                coords = [float(p.strip()) for p in coord_str_clean.split(',')]
                
                if len(coords) != 4 or any(c < 0 for c in coords):
                    raise ValueError(f'坐标值无效: {coords}')

                x1, y1, w, h = coords
                x_center = (x1 + w / 2) / img_w
                y_center = (y1 + h / 2) / img_h
                width = w / img_w
                height = h / img_h

                class_id = class_mapping.get(class_name)
                if class_id is not None:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                else:
                    raise ValueError(f"未知的类别: {class_name}")
        
        # 写入成功后，重命名临时文件
        shutil.move(temp_label_path, label_path)
        result['success'] = True

    except Exception as e:
        result['error'] = f'处理异常: {str(e)}'
        # 如果临时文件存在，则删除
        if os.path.exists(temp_label_path):
            os.remove(temp_label_path)

    finally:
        result['time'] = time.time() - process_start_time
        if not result['success']:
            with progress_lock:
                with open(error_log_path, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        image_name, 
                        index, 
                        '处理失败',
                        row.get('告警事由', ''), 
                        row.get('坐标', ''),
                        result.get('error', '未知错误')
                    ])
    return result


            
def main():
    parser = argparse.ArgumentParser(description='Excel转换为YOLO格式标注工具')
    parser.add_argument('--threads', type=int, default=os.cpu_count() or 4, help='线程数量 (默认: CPU核心数或4)')
    args = parser.parse_args()

    CONFIG_PATH = Path('configs/yolov11_seg.yaml')
    DATA_ROOT = Path('../../Data/raw')

    if not CONFIG_PATH.exists():
        logging.error(f"配置文件未找到: {CONFIG_PATH}")
        messagebox.showerror("错误", f"配置文件未找到: {CONFIG_PATH}")
        return

    class_mapping = get_class_mapping(CONFIG_PATH)
    logging.info(f"已加载类别映射: {class_mapping}")

    stats = {
        'train': {'success': 0, 'missing': set(), 'error': 0, 'total': 0, 'time': 0},
        'val': {'success': 0, 'missing': set(), 'error': 0, 'total': 0, 'time': 0}
    }
    total_start_time = time.time()

    for data_split in ['train', 'val']:
        logging.info(f"\n----- 正在处理 '{data_split}' 数据集 -----")
        excel_path = select_excel_file(title=f"请为【{data_split}】数据集选择Excel标注文件")
        if not excel_path:
            logging.warning(f"未给 '{data_split}' 数据集选择Excel文件，跳过。")
            continue
        
        try:
            df_labels = pd.read_excel(excel_path)
        except Exception as e:
            logging.error(f"读取Excel文件失败: {excel_path}, 错误: {e}")
            messagebox.showerror("错误", f"无法读取Excel文件:\n{excel_path}\n\n{e}")
            continue

        stats[data_split]['total'] = len(df_labels)
        image_base_dir = DATA_ROOT / data_split / 'images'
        if not image_base_dir.exists():
            logging.error(f"图片根目录不存在，跳过: {image_base_dir}")
            continue

        image_cache = build_image_cache(image_base_dir)
        logging.info(f"已为 '{data_split}' 构建图片缓存，包含 {len(image_cache)} 张图片")

        progress_file = get_progress_file_path(data_split)
        processed_images = set(load_progress(progress_file))
        
        # 筛选未处理数据，并检查 '图片名称' 列是否存在
        if '图片名称' not in df_labels.columns:
            logging.error(f"Excel文件 {excel_path} 中缺少 '图片名称' 列，跳过此文件。")
            messagebox.showerror("列缺失", f"Excel文件缺少 '图片名称' 列:\n{excel_path}")
            continue
            
        df_filtered = df_labels[~df_labels['图片名称'].isin(processed_images)]
        total_to_process = len(df_filtered)

        if total_to_process == 0:
            logging.info(f"'{data_split}' 数据集中的所有图片均已处理。")
            if progress_file.exists():
                progress_file.unlink()
            continue
        
        logging.info(f"发现 {total_to_process} 个新项目，开始处理...")
        split_start_time = time.time()

        error_log_path = log_dir / f'{data_split}_error_records.csv'
        with open(error_log_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['图片名称', '行索引', '错误类型', '原告警事由', '原始坐标', '错误详情'])

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            process_func = partial(process_single_row, class_mapping=class_mapping, image_cache=image_cache, error_log_path=error_log_path)
            tasks = df_filtered.iterrows()
            
            results = list(tqdm(
                executor.map(process_func, tasks),
                total=total_to_process,
                desc=f"处理 {data_split} 数据集",
                unit="项"
            ))

        for result in results:
            if result['success']:
                stats[data_split]['success'] += 1
                processed_images.add(result['image_name'])
            elif result['missing']:
                stats[data_split]['missing'].add(result['missing'])
            else:
                stats[data_split]['error'] += 1

        stats[data_split]['time'] = time.time() - split_start_time
        save_progress(progress_file, list(processed_images), total_start_time)

    # Final Report
    total_time = time.time() - total_start_time
    total_success = stats['train']['success'] + stats['val']['success']
    total_missing = len(stats['train']['missing']) + len(stats['val']['missing'])
    total_error = stats['train']['error'] + stats['val']['error']
    total_items = stats['train']['total'] + stats['val']['total']

    logging.info("\n===== 数据处理统计报告 ======")
    logging.info(f"总处理时间: {total_time:.2f}秒")
    logging.info(f"总处理条目: {total_items}")
    logging.info(f"  - 成功: {total_success}")
    logging.info(f"  - 图片未找到: {total_missing}")
    logging.info(f"  - 处理错误: {total_error}")

    for split in ['train', 'val']:
        if stats[split]['total'] > 0:
            logging.info(f"\n--- {split.upper()} 数据集 ---")
            logging.info(f"  处理耗时: {stats[split]['time']:.2f}秒")
            logging.info(f"  总条目: {stats[split]['total']}")
            logging.info(f"  成功: {stats[split]['success']}")
            logging.info(f"  图片未找到: {len(stats[split]['missing'])}")
            logging.info(f"  处理错误: {stats[split]['error']}")

    missing_report_data = []
    for split in ['train', 'val']:
        for img_name in stats[split]['missing']:
            missing_report_data.append({'数据集': split, '未找到图片名称': img_name})

    if missing_report_data:
        report_df = pd.DataFrame(missing_report_data)
        report_dir = Path(__file__).parent.parent / 'reports'
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"missing_images_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        report_df.to_excel(report_path, index=False)
        logging.info(f"未找到图片报告已生成: {report_path}")
    else:
        logging.info("所有图片均已成功找到，无需生成报告。")

    logging.info("\n数据准备完成!")
    messagebox.showinfo("完成", "数据准备流程已全部完成！")

if __name__ == '__main__':
    main()