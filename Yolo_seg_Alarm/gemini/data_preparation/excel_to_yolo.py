# -*- coding: utf-8 -*-
"""
==========================================================================================
## 脚本概述

本脚本是一个为YOLOv8实例分割任务设计的数据预处理工具。它负责将人工标注的Excel数据自动化地转换为YOLOv8训练所需的标准.txt标签格式，并构建完整的数据集目录结构。

### 主要功能点:

1.  **自动化数据转换**: 读取Excel文件中的标注信息（图片名、类别、坐标），并为每张图片生成对应的YOLO格式标签文件（.txt）。
2.  **坐标归一化**: 自动处理矩形（bounding box）坐标，将其从像素值转换为YOLO所需的相对于图片尺寸的归一化坐标（中心点x, 中心点y, 宽度, 高度）。
3.  **多线程处理**: 利用线程池并发处理数据行，显著提升在大型数据集上的处理效率。
4.  **断点续传**: 自动记录和加载处理进度，如果脚本中断，下次运行时可以跳过已完成的部分，从上次中断的地方继续。
5.  **健壮的错误处理**: 详细记录处理过程中遇到的所有错误（如图片找不到、坐标格式错误、类别不匹配等），并生成独立的错误日志文件（.csv），便于问题排查。
6.  **清晰的统计报告**: 脚本执行完毕后，在控制台输出详细的统计报告，包括成功、失败、缺失的条目数量和处理耗时。同时，会生成一份未找到图片的清单（.xlsx）。
7.  **用户友好的交互**: 通过图形化界面（Tkinter）引导用户选择需要处理的Excel文件，操作直观方便。

### 脚本架构与逻辑:

本脚本的架构围绕着一个清晰、模块化的处理流程，主要由以下几个核心函数协同工作：

1.  **`main()` - 总指挥**: 
    - 作为程序的入口和总控制器，负责协调整个数据处理流程。
    - 解析命令行参数（如线程数）。
    - 初始化配置，加载类别映射。
    - 循环处理训练集（'train'）和验证集（'val'）。
    - 调用 `select_excel_file()` 让用户选择文件。
    - 调用 `build_image_cache()` 快速查找图片路径。
    - 实现断点续传逻辑，过滤已处理的数据。
    - 创建并管理 `ThreadPoolExecutor` 线程池，将数据处理任务（`process_single_row`）分发给子线程。
    - 收集并汇总处理结果，调用 `save_progress()` 保存进度。
    - 最后，生成并打印最终的统计报告。

2.  **`process_single_row()` - 核心处理器**: 
    - 这是数据转换的核心执行单元，负责处理Excel中的单行数据。
    - 安全地从数据行中提取图片名称、类别和坐标。
    - 在图片缓存中查找对应的图片路径，并使用OpenCV获取图片尺寸。
    - 验证类别和坐标的有效性与匹配性。
    - 进行坐标归一化计算。
    - 将结果写入临时的.txt文件，成功后再重命名，以防多线程写入冲突。
    - 捕获所有潜在异常，并返回一个包含详细处理结果（成功、失败、错误信息）的字典。

3.  **辅助函数 - 工具集**:
    - `get_class_mapping()`: 从 `yolov11_seg.yaml` 配置文件中加载类别ID映射。
    - `build_image_cache()`: 预先扫描图片目录，构建一个从图片名到完整路径的映射（缓存），避免在循环中反复搜索文件，大幅提升效率。
    - `load_progress()` / `save_progress()`: 负责读写进度文件（.json），实现断点续传功能。
    - `select_excel_file()`: 提供一个简单的GUI窗口供用户选择文件。

### 函数关系:

`main()` 函数是流程的起点和终点。它首先准备好必要的配置（类别映射、图片缓存），然后将从Excel读取的每一行数据（通过 `df_filtered.iterrows()`）打包，交给 `ThreadPoolExecutor`。线程池中的每个线程都运行一个 `process_single_row()` 实例来独立地处理一行数据。`process_single_row()` 在执行过程中，会利用 `class_mapping` 和 `image_cache` 来完成自己的任务。处理完成后，`main()` 函数会收集所有 `process_single_row()` 返回的结果，进行统计和保存，最终完成整个数据集的准备工作。

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

# 创建一个线程锁，用于在多线程环境下安全地写入错误日志，防止数据竞争。
progress_lock = threading.Lock()

# --- 1. 配置智能日志记录 ---
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

# 检查上一次的日志文件状态
today_str = datetime.now().strftime('%Y%m%d')
# 寻找昨天的或更早的日志文件，这里简化为查找最新的一个日志文件
log_files = sorted(log_dir.glob('excel_to_yolo_*.log'), key=os.path.getmtime, reverse=True)
if log_files:
    last_log_file = log_files[0]
    try:
        with open(last_log_file, 'r', encoding='utf-8') as f:
            # 读取最后几行来判断是否正常结束
            lines = f.readlines()
            last_lines = "".join(lines[-5:]) # 查看最后5行
            if "数据准备流程结束" not in last_lines and not last_log_file.name.endswith('_con.log'):
                # 如果没有正常结束，并且没有被标记过，则重命名
                new_name = last_log_file.stem + '_con.log'
                new_path = last_log_file.with_name(new_name)
                shutil.move(last_log_file, new_path)
                print(f"检测到上次运行未正常结束，已将日志文件重命名为: {new_path.name}")
    except Exception as e:
        print(f"检查旧日志文件时出错: {e}")

# 为本次运行创建新的日志文件
log_file_path = log_dir / f"excel_to_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info(f"日志文件已保存至: {log_file_path}")

def select_excel_file(title="请选择Excel标注文件"):
    """
    弹出一个图形化文件选择对话框，引导用户选择一个Excel文件。

    Args:
        title (str): 对话框窗口的标题。

    Returns:
        str: 用户选择的文件的完整路径。如果用户取消选择，则返回None。
    """
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
    从YOLO的YAML配置文件中读取类别名称，并创建一个从类别名到类别索引的映射字典。

    Args:
        config_path (str or Path): YAML配置文件的路径。

    Returns:
        dict: 格式为 {'类别名': 索引, ...} 的字典，便于后续通过名称查找ID。
    """
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
    """
    递归扫描指定的图片基础目录，为所有支持的图片格式（.jpg, .jpeg, .png, .bmp）
    创建一个缓存字典，将图片文件名映射到其完整的Path对象。
    这样做可以极大地加速后续通过文件名查找图片路径的过程，避免重复的文件系统I/O操作。

    Args:
        base_dir (Path): 要扫描的图片根目录。

    Returns:
        dict: 一个形如 {'图片A.jpg': Path('/path/to/imageA.jpg'), ...} 的缓存字典。
    """
    cache = {}
    # 支持多种常见图片格式
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # `rglob` 会递归地查找所有匹配的文件
        for path in base_dir.rglob(f'*{ext}'):
            cache[path.name] = path
    return cache
# 修改函数参数，将image_base_dir改为image_cache
# 添加进度文件路径函数（插入在此处）
def get_progress_file_path(data_split):
    """
    为指定的数据集（如 'train' 或 'val'）生成并返回一个标准的进度文件路径。
    进度文件用于实现断点续传功能。

    Args:
        data_split (str): 数据集的名称（例如 'train'）。

    Returns:
        Path: 指向进度文件（.json）的Path对象。
    """
    # 获取当前脚本所在目录（data_preparation）
    current_dir = Path(__file__).parent
    # 在当前目录下创建一个 'progress' 子目录用于存放进度文件
    progress_dir = current_dir / 'progress'
    progress_dir.mkdir(exist_ok=True)  # 确保目录存在
    # 返回特定数据集的进度文件路径
    return progress_dir / f'{data_split}_progress.json'
#  添加进度加载函数
def load_progress(progress_file):
    """
    从指定的进度文件（JSON格式）中加载进度，包括已处理图片和历史统计数据。

    Args:
        progress_file (Path): 进度文件的路径。

    Returns:
        tuple: 包含 (已处理图片名称集合, 历史统计数据字典)。
    """
    # 定义一个空的默认进度结构
    default_progress = {
        "processed_images": [],
        "stats": {"success": 0, "missing": set(), "error": 0, "time": 0}
    }
    if progress_file.exists() and progress_file.stat().st_size > 0:
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 检查加载的数据是字典还是列表，以兼容旧格式
                if isinstance(data, dict):
                    # 新格式：数据是字典
                    processed_images = set(data.get("processed_images", []))
                    # 加载的stats中的missing可能是一个列表，需要转为set
                    loaded_stats = data.get("stats", default_progress['stats'])
                elif isinstance(data, list):
                    # 兼容旧格式：数据是列表，只包含图片
                    processed_images = set(data)
                    loaded_stats = default_progress['stats']
                else:
                    # 未知格式，使用默认值
                    logging.warning(f"进度文件 {progress_file} 格式未知，将创建新的进度。")
                    return set(), default_progress['stats']

                # 确保 loaded_stats['missing'] 是一个集合
                loaded_stats['missing'] = set(loaded_stats.get('missing', []))
                return processed_images, loaded_stats
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"进度文件 {progress_file} 读取或解析失败，将创建新的进度: {e}")
    # 如果文件不存在、为空或读取失败，返回默认的空进度
    return set(), default_progress['stats']
#  添加进度保存函数
def save_progress(progress_file, all_processed_images, stats_to_save):
    """
    将当前所有已处理的图片名称集合和累计的统计数据安全地保存到指定的进度文件中。

    Args:
        progress_file (Path): 进度文件的路径。
        all_processed_images (set): 包含所有已成功处理的图片名称的集合。
        stats_to_save (dict): 需要保存的累计统计数据。
    """
    # 创建一个 stats_to_save 的副本以避免修改原始字典
    stats_copy = stats_to_save.copy()

    # 为了能被JSON序列化，需要将set转换为list
    if isinstance(stats_copy.get('missing'), set):
        stats_copy['missing'] = list(stats_copy['missing'])

    progress_data = {
        'processed_images': list(all_processed_images),
        'stats': stats_copy
    }
    try:
        with progress_lock:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        logging.info(f"进度已更新，已处理 {len(all_processed_images)} 个项目。")
    except IOError as e:
        logging.error(f"进度文件 {progress_file} 写入失败: {e}")


def process_single_row(row_tuple, class_mapping, image_cache, error_log_path):
    """
    处理Excel表格中的单行数据，这是整个数据转换流程的核心执行单元。
    该函数被设计为在多线程环境中独立运行。

    Args:
        row_tuple (tuple): 一个元组，包含 (行索引, 行数据Series)。Pandas的 `iterrows()` 会生成这种格式。
        class_mapping (dict): 类别名称到类别ID的映射字典。
        image_cache (dict): 图片文件名到完整路径的缓存字典。
        error_log_path (str): 错误日志文件的路径，用于记录处理失败的行。

    Returns:
        dict: 一个包含详细处理结果的字典，用于后续的统计。格式如下：
              {
                  'success': bool,      # 是否成功
                  'image_name': str,    # 处理的图片名
                  'missing': str|None,  # 如果图片未找到，则为图片名
                  'error': str|None,    # 如果发生其他错误，则为错误信息
                  'time': float         # 处理该行所花费的时间
              }
    """
    index, row = row_tuple
    # 使用 .get() 方法安全地访问列，如果 '图片名称' 列不存在或值为空，则返回空字符串。
    image_name = row.get('图片名称', '')

    # 初始化返回结果字典
    result = {
        'success': False,
        'image_name': image_name,
        'missing': None,  # 用于标记未找到的图片
        'error': None,    # 用于记录其他类型的错误
        'time': 0
    }

    # 如果图片名称为空，则直接返回错误，无需继续处理
    if not image_name:
        result['error'] = '列 "图片名称" 为空或不存在'
        return result

    process_start_time = time.time()  # 记录单行处理的开始时间

    try:
        # 步骤 1: 在图片缓存中查找图片路径。这是性能优化的关键点。
        image_path = image_cache.get(image_name)
        if not image_path:
            result['missing'] = image_name  # 标记为“未找到”
            return result

        # 步骤 2: 使用OpenCV读取图片以获取其宽度和高度，用于后续的坐标归一化。
        img = cv2.imread(str(image_path))
        if img is None:
            result['error'] = f'无法读取图片: {image_path}'
            return result
        img_h, img_w, _ = img.shape

        # 步骤 3: 从行数据中提取类别和坐标信息。
        class_names_str = row.get('告警事由', '')
        coords_str = str(row.get('坐标', ''))
        
        # 如果类别或坐标为空，则记录错误并返回。
        if not class_names_str or not coords_str:
            result['error'] = '缺少告警事由或坐标信息'
            return result

        # 解析可能包含多个类别的字符串（以逗号分隔）
        class_names = [cn.strip() for cn in class_names_str.split(',')]
        # 解析可能包含多个坐标集的字符串（以分号分隔）
        coords_list = [coord.strip() for coord in coords_str.split(';') if coord.strip()]

        # 校验类别和坐标的数量是否一致，这是数据有效性的一个重要检查。
        if len(class_names) != len(coords_list):
            result['error'] = f'类别和坐标数量不匹配 ({len(class_names)} vs {len(coords_list)})'
            return result

        # 步骤 4: 创建YOLO格式的.txt标签文件。
        # 首先确定最终的标签文件路径。
        label_path = image_path.with_suffix('.txt')
        # 为了防止多线程同时写入同一个文件导致内容错乱，这里采用“临时文件”策略。
        # 先将内容写入一个临时文件，操作成功完成后再将其重命名为最终文件名。这是一个原子操作，能保证线程安全。
        temp_label_path = label_path.with_suffix('.txt.tmp')
        
        with open(temp_label_path, 'w', encoding='utf-8') as f:
            # 遍历每一个标注（一个类别对应一组坐标）
            for class_name, coord_str in zip(class_names, coords_list):
                # 校验坐标格式是否为 '[x, y, w, h]'
                if not (coord_str.startswith('[') and coord_str.endswith(']')):
                    raise ValueError(f'坐标格式错误: {coord_str}')

                # 清理并解析坐标字符串
                coord_str_clean = coord_str.strip('[]').strip()
                coords = [float(p.strip()) for p in coord_str_clean.split(',')]                
                # 校验坐标是否为4个非负数
                if len(coords) != 4 or any(c < 0 for c in coords):
                    raise ValueError(f'坐标值无效: {coords}')

                # 提取矩形框的左上角坐标(x1, y1)和宽高(w, h)
                x1, y1, w, h = coords
                # --- 坐标归一化 ---
                # 计算中心点x坐标并归一化
                x_center = (x1 + w / 2) / img_w
                # 计算中心点y坐标并归一化
                y_center = (y1 + h / 2) / img_h
                # 计算宽度并归一化
                width = w / img_w
                # 计算高度并归一化
                height = h / img_h

                # 从类别映射中获取类别ID
                class_id = class_mapping.get(class_name)
                if class_id is not None:
                    # 将类别ID和归一化后的坐标写入文件，坐标保留6位小数
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                else:
                    # 如果在配置文件中找不到该类别，则抛出错误
                    raise ValueError(f"未知的类别: {class_name}")
        
        # 如果循环正常结束（没有抛出异常），说明所有标注都已成功写入临时文件。
        # 现在可以将临时文件重命名为正式的标签文件。
        shutil.move(temp_label_path, label_path)
        result['success'] = True  # 标记该行处理成功

    except Exception as e:
        # 捕获在处理过程中可能发生的任何异常（如ValueError, IOError等）
        result['error'] = f'处理异常: {str(e)}'
        # 如果发生错误，需要清理掉可能已创建的临时文件，防止留下垃圾文件。
        if 'temp_label_path' in locals() and os.path.exists(temp_label_path):
            os.remove(temp_label_path)

    finally:
        # `finally` 块确保无论成功还是失败，都会执行以下代码。
        result['time'] = time.time() - process_start_time  # 计算并记录处理耗时
        
        # 如果处理失败，则将详细错误信息写入CSV日志文件。
        if not result['success']:
            # 使用线程锁确保在多线程环境下对错误日志文件的写入是安全的。
            with progress_lock:
                with open(error_log_path, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    # 写入详细的错误信息，便于追溯问题。
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
    # 定义一个在退出前强制保存进度的函数
    def final_save(data_split, processed_set, new_set, stats_to_save):
        progress_file = get_progress_file_path(data_split)
        if new_set:
            all_processed = processed_set.union(new_set)
            save_progress(progress_file, all_processed, stats_to_save)
            logging.info(f"在退出前，为'{data_split}'强制保存了 {len(all_processed)} 条进度。")

    """
    程序的主入口函数，负责编排整个数据转换流程。
    """
    # --- 1. 初始化与配置加载 ---
    # 使用argparse来处理命令行参数，这里允许用户通过 --threads 自定义线程数。
    parser = argparse.ArgumentParser(description='Excel转换为YOLO格式标注工具')
    parser.add_argument('--threads', type=int, default=8, help='线程数量 (默认: 8)')
    args = parser.parse_args()

    # 定义关键路径
    CONFIG_PATH = Path('configs/yolov11_seg.yaml')  # YOLO配置文件路径
    DATA_ROOT = Path('../../Data/raw')  # 存放 train/val 数据集的根目录

    # 检查配置文件是否存在，如果不存在则无法继续。
    if not CONFIG_PATH.exists():
        logging.error(f"配置文件未找到: {CONFIG_PATH}")
        messagebox.showerror("错误", f"配置文件未找到: {CONFIG_PATH}")
        return

    # 加载类别到ID的映射
    class_mapping = get_class_mapping(CONFIG_PATH)
    logging.info(f"已加载类别映射: {class_mapping}")

    # 初始化一个字典来存储和跟踪处理过程中的统计数据。
    # 这个stats将只包含本次运行的数据，用于累加到从进度文件中加载的历史数据中。
    current_run_stats = {
        'train': {'success': 0, 'missing': set(), 'error': 0, 'time': 0},
        'val': {'success': 0, 'missing': set(), 'error': 0, 'time': 0}
    }
    # 这个字典将持有每个数据集的总统计信息（包括历史和当前）
    total_stats = {
        'train': {'success': 0, 'missing': set(), 'error': 0, 'total': 0, 'time': 0},
        'val': {'success': 0, 'missing': set(), 'error': 0, 'total': 0, 'time': 0}
    }
    total_start_time = time.time()  # 记录整个流程的开始时间

    # --- 2. 一次性选择所有Excel文件 ---
    excel_paths = {}
    for data_split in ['train', 'val']:
        path = select_excel_file(title=f"请为【{data_split}】数据集选择Excel标注文件 (可跳过)")
        if path:
            excel_paths[data_split] = path
            logging.info(f"已为 '{data_split}' 数据集选择Excel文件: {path}")
        else:
            logging.warning(f"用户未为 '{data_split}' 数据集选择Excel文件，将跳过处理。")

    # --- 3. 循环处理数据集（训练集和验证集） ---
    current_data_split = None
    processed_images_set = set()
    newly_processed_images = set()

    try:
        for data_split, excel_path in excel_paths.items():
            current_data_split = data_split
            processed_images_set = set() # 为每个数据集重置
            newly_processed_images = set() # 为每个数据集重置
            logging.info(f"\n----- 正在处理 '{data_split}' 数据集 ----- (文件: {excel_path})")

            # --- 4. 进度管理与断点续传 ---
            # 首先确定进度文件路径
            progress_file = get_progress_file_path(data_split)
            # 加载或初始化进度
            processed_images_set, historical_stats = load_progress(progress_file)

            # 进度管理：检查是否存在旧进度，并让用户选择是否继续。
            if processed_images_set:
                user_choice = messagebox.askyesno(
                    title="发现旧进度",
                    message=f"检测到 '{data_split}' 数据集存在未完成的进度（已处理 {len(processed_images_set)} 项）。\n\n是否要继续上次的任务？\n\n- 选择【是】将从上次中断的地方继续。\n- 选择【否】将开始一个全新的任务，并清空旧进度。"
                )
                if not user_choice:
                    logging.info("用户选择开始新任务，正在清空所有数据集进度...")
                    # 同时清空train和val的进度文件
                    for ds in ['train', 'val']:
                        ds_progress_file = get_progress_file_path(ds)
                        if ds_progress_file.exists():
                            save_progress(ds_progress_file, set(), {'success': 0, 'missing': set(), 'error': 0, 'time': 0})
                    # 重置当前数据集的处理状态
                    processed_images_set.clear()
                    historical_stats = {'success': 0, 'missing': set(), 'error': 0, 'time': 0}
            
            df_labels = pd.read_excel(excel_path)
            total_items_in_excel = len(df_labels)
            total_stats[data_split]['total'] = total_items_in_excel # 记录总条目数
            # 安全地合并历史统计数据
            for key, value in historical_stats.items():
                if key == 'missing':
                    # 确保missing字段是集合，并用集合的方式合并
                    total_stats[data_split]['missing'].update(value)
                elif key in total_stats[data_split]:
                    # 对于数值类型的统计，进行累加
                    total_stats[data_split][key] += value
                else:
                    # 其他新出现的键，直接赋值
                    total_stats[data_split][key] = value

            image_base_dir = DATA_ROOT / data_split / 'images'
            if not image_base_dir.exists():
                logging.error(f"图片根目录不存在，跳过: {image_base_dir}")
                continue

            image_cache = build_image_cache(image_base_dir)
            
            if '图片名称' not in df_labels.columns:
                logging.error(f"Excel文件 {excel_path} 中缺少 '图片名称' 列，跳过此文件。")
                continue
                
            df_filtered = df_labels[~df_labels['图片名称'].isin(processed_images_set)]
            total_to_process = len(df_filtered)

            if total_to_process == 0:
                logging.info(f"'{data_split}' 数据集中的所有图片均已处理。总计 {total_stats[data_split]['total']} 项。")
                continue
            
            logging.info(f"数据集 '{data_split}' 总共有 {total_items_in_excel} 个项目，其中 {len(processed_images_set)} 个已处理。")
            logging.info(f"本次需要处理 {total_to_process} 个新项目，开始处理...")
            split_start_time = time.time()

            # --- 5. 并行处理 ---
            # 初始化该数据集的错误日志文件，并写入表头
            error_log_path = log_dir / f'{data_split}_error_records.csv'
            with open(error_log_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['图片名称', '行索引', '错误类型', '原告警事由', '原始坐标', '错误详情'])

                        # 创建一个线程池来并行处理数据
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                process_func = partial(process_single_row, class_mapping=class_mapping, image_cache=image_cache, error_log_path=error_log_path)
                tasks = df_filtered.iterrows()
                save_interval_count = 100
                
                results_iterator = executor.map(process_func, tasks)
                
                # --- 优化tqdm进度条 ---
                progress_bar = tqdm(results_iterator, 
                                    total=total_items_in_excel,  # 总数是Excel中的总行数
                                    initial=len(processed_images_set), # 初始值是已处理的数量
                                    desc=f"当前处理 {data_split} 数据集总计", 
                                    unit="项")

                for i, result in enumerate(progress_bar, 1):
                    if result['success']:
                        current_run_stats[data_split]['success'] += 1
                        newly_processed_images.add(result['image_name'])
                    elif result['missing']:
                        current_run_stats[data_split]['missing'].add(result['missing'])
                    else:
                        current_run_stats[data_split]['error'] += 1

                    if i % save_interval_count == 0 and newly_processed_images:
                        all_processed_so_far = processed_images_set.union(newly_processed_images)
                        # 更新总统计数据
                        total_stats[data_split]['success'] += current_run_stats[data_split]['success']
                        total_stats[data_split]['error'] += current_run_stats[data_split]['error']
                        total_stats[data_split]['missing'].update(current_run_stats[data_split]['missing'])
                        # 保存进度
                        save_progress(progress_file, all_processed_so_far, total_stats[data_split])
                        # 更新基准集合
                        processed_images_set = all_processed_so_far
                        # 清空当前批次的集合和统计
                        newly_processed_images.clear()
                        current_run_stats[data_split] = {'success': 0, 'missing': set(), 'error': 0, 'time': 0}
                        
            
            # --- 循环结束后，最终保存一次以确保所有进度都被记录 ---
            if newly_processed_images:
                all_processed_images = processed_images_set.union(newly_processed_images)
                # 更新最终统计
                total_stats[data_split]['success'] += current_run_stats[data_split]['success']
                total_stats[data_split]['error'] += current_run_stats[data_split]['error']
                total_stats[data_split]['missing'].update(current_run_stats[data_split]['missing'])
                total_stats[data_split]['time'] += time.time() - split_start_time
                save_progress(progress_file, all_processed_images, total_stats[data_split])
            else:
                # 如果没有新处理的，也更新一下总时间
                total_stats[data_split]['time'] += time.time() - split_start_time
                save_progress(progress_file, processed_images_set, total_stats[data_split])

        # --- 6. 生成最终统计报告 (移入try块内) ---
        overall_total_time = time.time() - total_start_time
        final_total_success = total_stats['train']['success'] + total_stats['val']['success']
        final_total_missing_set = total_stats['train']['missing'].union(total_stats['val']['missing'])
        final_total_missing = len(final_total_missing_set)
        final_total_error = total_stats['train']['error'] + total_stats['val']['error']
        final_total_items = total_stats['train']['total'] + total_stats['val']['total']

        # 在控制台打印详细的报告
        logging.info("\n" + "="*25 + " 最终数据处理统计报告 " + "="*25)
        logging.info(f"总运行时间: {overall_total_time:.2f}秒 (包含所有续跑)")
        logging.info(f"总处理条目: {final_total_items}")
        logging.info(f"  - 累计成功: {final_total_success}")
        logging.info(f"  - 累计图片未找到 (共 {final_total_missing} 张): {final_total_missing}")
        logging.info(f"  - 累计处理错误: {final_total_error}")

        # 分别打印 train 和 val 的详细统计
        for split in ['train', 'val']:
            if total_stats[split]['total'] > 0:
                logging.info(f"\n--- {split.upper()} 数据集 --- ")
                logging.info(f"  累计处理耗时: {total_stats[split]['time']:.2f}秒")
                logging.info(f"  总条目: {total_stats[split]['total']}")
                logging.info(f"  成功: {total_stats[split]['success']}")
                logging.info(f"  图片未找到: {len(total_stats[split]['missing'])}")
                logging.info(f"  处理错误: {total_stats[split]['error']}")

        all_missing_images = []
        for split in ['train', 'val']:
            for img_name in total_stats[split]['missing']:
                all_missing_images.append({'数据集': split, '未找到图片名称': img_name})

        if all_missing_images:
            report_df = pd.DataFrame(all_missing_images)
            report_dir = Path(__file__).parent.parent / 'reports'
            report_dir.mkdir(exist_ok=True)
            report_path = report_dir / f"missing_images_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            report_df.to_excel(report_path, index=False)
            logging.info(f"未找到图片报告已生成: {report_path}")
        else:
            logging.info("所有图片均已成功找到，无需生成报告。")

        logging.info("\n数据准备完成!")
        messagebox.showinfo("完成", "数据准备流程已全部完成！")

    except KeyboardInterrupt:
        logging.warning("\n检测到用户中断 (Ctrl+C)，正在尝试保存当前进度...")
        if current_data_split:
            final_save(current_data_split, processed_images_set, newly_processed_images, total_stats[current_data_split])
        logging.info("进度已保存，程序即将退出。")
    except Exception as e:
        logging.error(f"发生意外错误: {e}", exc_info=True)
        if current_data_split:
            final_save(current_data_split, processed_images_set, newly_processed_images, total_stats[current_data_split])
    finally:
        logging.info("\n数据准备流程结束。")

# Python的入口点，当脚本被直接执行时，调用main()函数。
if __name__ == '__main__':
    main()