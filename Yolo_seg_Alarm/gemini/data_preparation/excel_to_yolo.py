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

# --- 日志配置 ---
# 定义日志文件存放目录，位于项目根目录下的 'logs' 文件夹中。
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)  # 如果目录不存在，则创建它。

# 定义日志文件名，包含时间戳以确保每次运行的日志都是唯一的。
log_file = log_dir / f"excel_to_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 配置logging模块，使其同时向文件和控制台输出日志。
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO，记录所有INFO、WARNING、ERROR级别的日志。
    format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式。
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # 将日志写入文件。
        logging.StreamHandler()  # 将日志输出到控制台。
    ]
)
logging.info(f"日志文件已保存至: {log_file}")

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
    从指定的进度文件（JSON格式）中加载已经处理过的图片名称集合。
    使用集合（set）可以提供更快的查找性能（O(1)），这在处理大量数据时非常重要。

    Args:
        progress_file (Path): 进度文件的路径。

    Returns:
        set: 已处理图片名称的集合。
    """
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                # 将加载的列表转换为集合，以便快速查找
                return set(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"进度文件 {progress_file} 读取失败，将重新开始处理: {e}")
    return set()  # 如果文件不存在或读取失败，返回一个空集合
#  添加进度保存函数
def save_progress(progress_file, all_processed_images):
    """
    将当前所有已处理的图片名称集合安全地保存到指定的进度文件中。

    Args:
        progress_file (Path): 进度文件的路径。
        all_processed_images (set): 包含所有已成功处理的图片名称的集合。
    """
    try:
        # 使用线程锁确保文件写入操作的原子性，防止在多线程或并发场景下出现问题。
        with progress_lock:
            with open(progress_file, 'w', encoding='utf-8') as f:
                # 将集合转换为列表以便JSON序列化，并写入文件。
                json.dump(list(all_processed_images), f, ensure_ascii=False, indent=2)
        logging.info(f"进度已成功保存至 {progress_file}。当前总计已处理 {len(all_processed_images)} 个项目。")
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
    """
    程序的主入口函数，负责编排整个数据转换流程。
    """
    # --- 1. 初始化与配置加载 ---
    # 使用argparse来处理命令行参数，这里允许用户通过 --threads 自定义线程数。
    parser = argparse.ArgumentParser(description='Excel转换为YOLO格式标注工具')
    parser.add_argument('--threads', type=int, default=os.cpu_count() or 4, help='线程数量 (默认: CPU核心数或4)')
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
    stats = {
        'train': {'success': 0, 'missing': set(), 'error': 0, 'total': 0, 'time': 0},
        'val': {'success': 0, 'missing': set(), 'error': 0, 'total': 0, 'time': 0}
    }
    total_start_time = time.time()  # 记录整个流程的开始时间

    # --- 2. 循环处理数据集（训练集和验证集） ---
    for data_split in ['train', 'val']:
        logging.info(f"\n----- 正在处理 '{data_split}' 数据集 -----")
        
        # 弹出文件选择框让用户选择Excel文件
        excel_path = select_excel_file(title=f"请为【{data_split}】数据集选择Excel标注文件")
        if not excel_path:
            logging.warning(f"未给 '{data_split}' 数据集选择Excel文件，跳过。")
            continue

        # --- 3. 进度管理与断点续传 ---
        # 首先确定进度文件路径
        progress_file = get_progress_file_path(data_split)

        # 进度管理：检查是否存在旧进度，并让用户选择是否继续。
        if progress_file.exists() and progress_file.stat().st_size > 0:
            # 弹出对话框询问用户
            user_choice = messagebox.askyesno(
                title="发现旧进度",
                message=f"检测到 '{data_split}' 数据集存在未完成的进度。\n\n是否要继续上次的任务？\n\n- 选择【是】将从上次中断的地方继续。\n- 选择【否】将开始一个全新的任务，并清空旧进度。"
            )
            
            if not user_choice: # 如果用户选择“否”（No），则开始新任务
                logging.info("用户选择开始新任务，正在初始化进度文件...")
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump([], f) # 清空文件内容
                    logging.info(f"进度文件 {progress_file} 已成功初始化。")
                except IOError as e:
                    logging.error(f"无法初始化进度文件 {progress_file}: {e}")
            else:
                logging.info("用户选择继续上次的任务。")
        else:
            logging.info("未发现有效进度文件，将开始新任务。")
        
        # 使用pandas读取Excel文件内容到DataFrame
        try:
            df_labels = pd.read_excel(excel_path)
        except Exception as e:
            logging.error(f"读取Excel文件失败: {excel_path}, 错误: {e}")
            messagebox.showerror("错误", f"无法读取Excel文件:\n{excel_path}\n\n{e}")
            continue

        # 更新总条目数统计
        stats[data_split]['total'] = len(df_labels)
        
        # 构建该数据集的图片存放目录路径
        image_base_dir = DATA_ROOT / data_split / 'images'
        if not image_base_dir.exists():
            logging.error(f"图片根目录不存在，跳过: {image_base_dir}")
            continue

        # 构建图片缓存以加速查找
        image_cache = build_image_cache(image_base_dir)
        logging.info(f"已为 '{data_split}' 构建图片缓存，包含 {len(image_cache)} 张图片")

        # 加载进度（此时应为空，因为我们刚删除了文件）
        processed_images_set = load_progress(progress_file) # 加载已处理过的图片名集合
        
        # 检查关键列 '图片名称' 是否存在
        if '图片名称' not in df_labels.columns:
            logging.error(f"Excel文件 {excel_path} 中缺少 '图片名称' 列，跳过此文件。")
            messagebox.showerror("列缺失", f"Excel文件缺少 '图片名称' 列:\n{excel_path}")
            continue
            
        # 从DataFrame中筛选出尚未处理的行
        df_filtered = df_labels[~df_labels['图片名称'].isin(processed_images_set)]
        total_to_process = len(df_filtered)

        if total_to_process == 0:
            logging.info(f"'{data_split}' 数据集中的所有图片均已处理。")
            continue
        
        logging.info(f"发现 {total_to_process} 个新项目，开始处理...")
        split_start_time = time.time()

        # --- 4. 并行处理 ---
        # 初始化该数据集的错误日志文件，并写入表头
        error_log_path = log_dir / f'{data_split}_error_records.csv'
        with open(error_log_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['图片名称', '行索引', '错误类型', '原告警事由', '原始坐标', '错误详情'])

        # 创建一个线程池来并行处理数据
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            # 使用 functools.partial 来“预设” process_single_row 函数的参数，
            # 这样 map 函数只需要传入变化的 `row_tuple` 即可。
            process_func = partial(process_single_row, class_mapping=class_mapping, image_cache=image_cache, error_log_path=error_log_path)
            
            # `df_filtered.iterrows()` 是要处理的任务迭代器
            tasks = df_filtered.iterrows()
            
            # `executor.map` 会将 `tasks` 中的每一项传递给 `process_func`，并返回一个结果的迭代器。
            # 使用 `tqdm` 来创建一个可视化的进度条。
            results = list(tqdm(
                executor.map(process_func, tasks),
                total=total_to_process,
                desc=f"处理 {data_split} 数据集",
                unit="项"
            ))

        # --- 5. 结果汇总与定期进度保存 ---
        newly_processed_images = set()
        save_interval_seconds = 60  # 每60秒保存一次进度
        last_save_time = time.time()

        # 遍历所有子线程返回的结果，进行分类统计和定期保存
        for i, result in enumerate(results):
            if result['success']:
                stats[data_split]['success'] += 1
                newly_processed_images.add(result['image_name'])
            elif result['missing']:
                stats[data_split]['missing'].add(result['missing'])
            else:
                stats[data_split]['error'] += 1

            # 检查是否达到了保存进度的时间间隔
            current_time = time.time()
            if current_time - last_save_time >= save_interval_seconds:
                if newly_processed_images:
                    all_processed_images = processed_images_set.union(newly_processed_images)
                    save_progress(progress_file, all_processed_images)
                    last_save_time = current_time
        
        # 记录该数据集的处理总耗时
        stats[data_split]['time'] = time.time() - split_start_time

        # --- 循环结束后，最终保存一次以确保所有进度都被记录 ---
        if newly_processed_images:
            all_processed_images = processed_images_set.union(newly_processed_images)
            save_progress(progress_file, all_processed_images)

    # --- 6. 生成最终统计报告 ---
    total_time = time.time() - total_start_time
    total_success = stats['train']['success'] + stats['val']['success']
    total_missing = len(stats['train']['missing'] | stats['val']['missing']) # 使用集合并集去重
    total_error = stats['train']['error'] + stats['val']['error']
    total_items = stats['train']['total'] + stats['val']['total']

    # 在控制台打印详细的报告
    logging.info("\n" + "="*25 + " 数据处理统计报告 " + "="*25)
    logging.info(f"总处理时间: {total_time:.2f}秒")
    logging.info(f"总处理条目: {total_items}")
    logging.info(f"  - 成功: {total_success}")
    logging.info(f"  - 图片未找到 (共 {total_missing} 张): {total_missing}")
    logging.info(f"  - 处理错误: {total_error}")

    # 分别打印 train 和 val 的详细统计
    for split in ['train', 'val']:
        if stats[split]['total'] > 0:
            logging.info(f"\n--- {split.upper()} 数据集 ---")
            logging.info(f"  处理耗时: {stats[split]['time']:.2f}秒")
            logging.info(f"  总条目: {stats[split]['total']}")
            logging.info(f"  成功: {stats[split]['success']}")
            logging.info(f"  图片未找到: {len(stats[split]['missing'])}")
            logging.info(f"  处理错误: {stats[split]['error']}")

    # --- 7. 生成未找到图片的Excel报告 ---
    # 合并 train 和 val 中所有未找到的图片名
    all_missing_images = []
    for split in ['train', 'val']:
        for img_name in stats[split]['missing']:
            all_missing_images.append({'数据集': split, '未找到图片名称': img_name})

    if all_missing_images:
        report_df = pd.DataFrame(all_missing_images)
        # 定义报告存放目录
        report_dir = Path(__file__).parent.parent / 'reports'
        report_dir.mkdir(exist_ok=True)
        # 定义报告文件名
        report_path = report_dir / f"missing_images_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        # 将DataFrame导出为Excel文件
        report_df.to_excel(report_path, index=False)
        logging.info(f"未找到图片报告已生成: {report_path}")
    else:
        logging.info("所有图片均已成功找到，无需生成报告。")

    logging.info("\n数据准备完成!")
    messagebox.showinfo("完成", "数据准备流程已全部完成！")

# Python的入口点，当脚本被直接执行时，调用main()函数。
if __name__ == '__main__':
    main()