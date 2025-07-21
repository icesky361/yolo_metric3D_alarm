# -*- coding: utf-8 -*-
"""
================================================================================
@ 脚本名: train.py
@ 功能: YOLOv11 实例分割模型训练脚本
@ 作者: TraeAI
@ 创建时间: 2024-07-28
--------------------------------------------------------------------------------
@ 脚本逻辑:
1.  **初始化与配置加载**: 
    - 导入所需库 (argparse, yaml, logging, ultralytics等)。
    - 添加项目根目录到系统路径，以正确导入自定义的`utils`模块。
    - 配置日志记录器，用于输出训练过程中的信息、警告和错误。
    - 定义`train`函数，接收一个配置文件路径作为参数。

2.  **环境检测与参数动态调整**:
    - 调用`utils.environment.get_device_and_adjust_config`函数。
    - 该函数会自动检测是否存在可用的NVIDIA GPU。
    - 如果有GPU，根据GPU型号（高端、中端、低端）动态调整训练参数，如`batch_size`。
    - 如果没有GPU，则切换到CPU模式，并使用保守的参数配置，以防止内存溢出。

3.  **模型加载**:
    - 使用`ultralytics.YOLO`加载一个预训练的YOLOv11分割模型（例如`yolov11n-seg.pt`）。
    - 这个预训练模型为我们的训练提供了一个良好的起点，可以加快收敛速度。
    - 将模型移动到前面步骤检测到的设备（GPU或CPU）上。

4.  **启动训练**:
    - 调用`model.train()`方法启动训练流程。
    - `ultralytics`库的`train`方法是一个高度集成的函数，它会自动处理：
        - 数据加载（根据`.yaml`配置文件中的路径）。
        - 数据增强（根据`.yaml`文件中的配置）。
        - 模型评估（在每个epoch后计算mAP、分割IoU等指标）。
        - 保存检查点（保存训练过程中的模型权重）。
        - 保存最佳模型（在`models/yolo_seg_experiment/weights/best.pt`）。
    - 训练完成后，记录日志，提示用户最佳模型已保存的位置。

5.  **主程序入口**:
    - 使用`argparse`库设置命令行参数，允许用户通过`--config`指定配置文件路径。
    - 如果用户未指定，则默认使用`configs/yolov11_seg.yaml`。
    - 解析命令行参数并调用`train`函数，启动整个训练过程。

@ 使用方法:
-   直接在命令行中运行此脚本:
    python training/train.py
-   或指定一个不同的配置文件:
    python training/train.py --config /path/to/your/config.yaml
================================================================================
"""

import argparse
import yaml
from pathlib import Path
import logging
import tempfile  # 添加tempfile模块导入
from ultralytics import YOLO
from pathlib import Path

# 将项目根目录添加到系统路径中，以便能够正确导入 'utils' 模块
# Path(__file__).resolve() 获取当前文件的绝对路径
# .parents[1] 获取上级目录，即项目根目录 'gemini'
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.environment import get_device_and_adjust_config

# 配置日志记录，用于在控制台输出格式化的时间、日志级别和消息
import os
from datetime import datetime

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

# 生成带日期的日志文件名
log_filename = f"data_prep_{datetime.now().strftime('%Y%m%d')}.log"
log_filepath = os.path.join(log_dir, log_filename)

# 配置日志同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

def train(config_path: str):
    """
    运行YOLOv11分割模型训练的主函数。

    Args:
        config_path (str): 指向YAML配置文件的路径。
    """
    # --- 1. 加载配置文件 ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"成功从 {config_path} 加载配置")
    except FileNotFoundError:
        logging.error(f"配置文件未找到: {config_path}")
        return
    except Exception as e:
        logging.error(f"加载或解析YAML文件时出错: {e}")
        return
    
    # 动态调整数据集配置，指定标签路径与图像路径相同
    data_config = config
    train_data_path = Path(config['path']) / config['train']
    val_data_path = Path(config['path']) / config['val']   # 验证路径存在性
    if not train_data_path.exists():
        logging.error(f"训练数据路径不存在: {train_data_path.resolve()}")
        return
    # --- 2. 设置环境并动态调整配置 ---
    # 这一步会根据检测到的硬件（CPU/GPU）动态设置设备和调整超参数（如批量大小）。
    device, updated_config = get_device_and_adjust_config(config)
    
    # --- 3. 加载YOLO模型 ---
    # 我们使用 'yolov11-n-seg.pt' 作为一个起点（这是一个小而快的模型）。
    # 你可以根据需要更换为其他模型，如 'yolov11s-seg.pt', 'yolov11m-seg.pt' 等。
    try:
        model = YOLO(updated_config['weights'])  # 从配置加载一个预训练的分割模型
        model.to(device)  # 将模型移动到指定的设备上
        logging.info(f"YOLOv11分割模型 '{updated_config['weights']}' 加载成功。")
    except Exception as e:
        logging.error(f"加载YOLO模型失败。请确保 '{updated_config['weights']}' 可访问或网络连接正常。错误: {e}")
        return

    # --- 4. 开始训练 ---
    logging.info("开始模型训练...")
    logging.info(f"数据集路径: {data_config['path']}")
    logging.info(f"训练轮数 (Epochs): {updated_config['epochs']}, 批量大小 (Batch size): {updated_config['batch']}, 图像尺寸 (Image size): {updated_config['imgsz']}")

    try:
        # ultralytics的train方法负责整个训练循环，
        # 包括数据加载、数据增强、评估和保存检查点。
        # 注意: ultralytics的验证器默认会计算mAP和分割mAP(IoU)。
        # 如果需要计算Dice系数等其他指标，可以考虑使用回调函数在验证结束时进行计算。
        results = model.train(
            data=str(train_data_path),  # 传递拼接后的绝对路径，YOLO会自动解析
            epochs=updated_config['epochs'],
            batch=updated_config['batch'],
            imgsz=updated_config['imgsz'],
            optimizer=updated_config.get('optimizer', 'AdamW'), # 优化器，默认为AdamW
            lr0=updated_config.get('lr0', 0.01),  # 初始学习率
            lrf=updated_config.get('lrf', 0.01),  # 最终学习率
            workers=4,  # 根据CPU核心数调整，加速数据加载
            # 数据增强参数由训练器直接从yaml文件中读取
            project='models',  # 将结果保存到 'models' 目录下
            name='yolo_seg_experiment'  # 为本次特定的运行创建一个子目录
        )
        logging.info("训练成功完成。")
        # 清理GPU内存，释放显存资源
        torch.cuda.empty_cache()
        # 训练完成后，results对象包含了所有训练结果的路径信息
        logging.info(f"模型和结果已保存至: {results.save_dir}")
        logging.info(f"要运行推理，请使用最佳模型: {results.save_dir}/weights/best.pt")

    except Exception as e:
        # 如果训练过程中发生任何异常，记录详细的错误信息
        logging.error(f"训练过程中发生错误: {e}", exc_info=True)

if __name__ == '__main__':
    # --- 命令行参数解析 ---
    # 创建一个参数解析器，用于处理从命令行传递的参数
    parser = argparse.ArgumentParser(description="训练一个YOLOv8分割模型。")
    
    # 添加 '--config' 参数，用于指定训练配置文件的路径
    # 使用动态路径构造，避免硬编码绝对路径
    default_config = Path(__file__).resolve().parents[1] / 'configs' / 'yolov11_seg.yaml'
    parser.add_argument('--config', type=str, default=str(default_config), help='Path to the configuration YAML file')
    
    # 解析命令行传入的参数
    args = parser.parse_args()

    # 加载配置文件
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 动态调整训练参数
    # 动态调整训练参数
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from gemini.utils.environment import get_device_and_adjust_config
    device, config = get_device_and_adjust_config(config)

    # 初始化YOLO模型
    from ultralytics import YOLO
    model = YOLO(config['model_path'])

    # 使用解析到的配置路径调用train函数
    # 准备数据集配置
    # 修复路径大小写问题并确保正确指向标签目录
    # 构建并验证数据集路径
    # 直接使用包含图片和标签的目录（同一文件夹）
    train_labels_path = os.path.abspath(os.path.join(config['path'], config['train']))
    val_labels_path = os.path.abspath(os.path.join(config['path'], config['val']))
    
    # 递归查找所有图像文件并生成文件列表
    def get_image_paths(directory):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    # 检查对应标签文件是否存在
                    label_path = os.path.splitext(os.path.join(root, file))[0] + '.txt'
                    if os.path.exists(label_path):
                        # 检查标签文件是否非空且包含有效内容
                        if os.path.getsize(label_path) > 0:
                            with open(label_path, 'r') as f:
                                label_content = f.read().strip()
                            if label_content:
                                # 验证标签文件格式
                                valid = True
                                for line_num, line in enumerate(label_content.split('\n')):
                                    line = line.strip()
                                    if not line:  # 跳过空行
                                        continue
                                    parts = line.split()
                                    if len(parts) < 5:
                                        logging.warning(f"标签文件格式错误 {label_path} (行 {line_num+1}): 至少需要5个值")
                                        valid = False
                                        break
                                    try:
                                        # 验证类别ID和边界框坐标
                                        cls_id = int(parts[0])
                                        coords = list(map(float, parts[1:5]))
                                        if not all(0 <= c <= 1 for c in coords):
                                            logging.warning(f"标签文件坐标错误 {label_path} (行 {line_num+1}): 坐标应在0-1范围内")
                                            valid = False
                                            break
                                    except ValueError:
                                        logging.warning(f"标签文件数值错误 {label_path} (行 {line_num+1}): {line}")
                                        valid = False
                                        break
                                if valid:
                                    image_paths.append(os.path.join(root, file))
                                else:
                                    logging.warning(f"跳过格式错误的标签文件: {label_path}")
                            else:
                                logging.warning(f"跳过空白标签文件: {label_path}")
                        else:
                            logging.warning(f"跳过空标签文件: {label_path}")
                    else:
                        logging.warning(f"跳过无标签图像: {os.path.join(root, file)}")
        return image_paths
    
    # 获取数据集根目录
    root_data_path = config['path']
    
    # 获取训练和验证集图像的相对路径
    train_images = [os.path.relpath(path, root_data_path) for path in get_image_paths(train_labels_path)]
    val_images = [os.path.relpath(path, root_data_path) for path in get_image_paths(val_labels_path)]
    
    # 记录有效图像数量和样本路径
    logging.info(f"找到 {len(train_images)} 个训练图像和标签对")
    logging.info(f"找到 {len(val_images)} 个验证图像和标签对")
    if train_images:
        logging.info(f"训练样本示例: {train_images[:3]}")
        # 验证第一个训练标签文件内容
        first_train_label = os.path.splitext(train_images[0])[0] + '.txt'
        first_train_label_path = os.path.join(root_data_path, first_train_label)
        if os.path.exists(first_train_label_path):
            with open(first_train_label_path, 'r') as f:
                label_sample = f.read(200)  # 读取前200字符
            logging.info(f"第一个训练标签内容示例: {label_sample}")
        else:
            logging.error(f"第一个训练标签文件不存在: {first_train_label_path}")
    if val_images:
        logging.info(f"验证样本示例: {val_images[:3]}")
    
    # 记录数据集根路径
    logging.info(f"数据集根路径: {root_data_path}")
    
    # 导入tempfile模块
    import tempfile
    
    # 创建临时文件列表
    train_txt = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False).name
    val_txt = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False).name
    
    # 验证临时文件路径
    logging.info(f"训练文件列表路径: {train_txt}")
    logging.info(f"验证文件列表路径: {val_txt}")
    
    # 写入图像路径
    # 将Windows路径分隔符替换为正斜杠以兼容YOLO
    with open(train_txt, 'w') as f:
        f.write('\n'.join([path.replace('\\', '/') for path in train_images]))
    with open(val_txt, 'w') as f:
        f.write('\n'.join([path.replace('\\', '/') for path in val_images]))
    
    # 更新数据配置使用文件列表
    data_dict = {
          'path': config['path'],
          'train': train_txt,
          'val': val_txt,
          'label_dir': 'images',  # 指定标签文件与图像在同一目录
        'nc': config['nc'],
        'names': config['names']
    }
    
    # 创建临时YAML文件以适应YOLO的数据加载要求
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(data_dict, f)
        temp_yaml_path = f.name

    # 开始训练
    # 添加异常捕获以确保日志完整记录
try:
    model.train(data=temp_yaml_path, epochs=config['epochs'], batch=config['batch'], imgsz=config['img_size'], workers=0)
except Exception as e:
    logging.error(f"训练过程中发生致命错误: {str(e)}", exc_info=True)
    raise  # 重新抛出异常以便终端显示
    
    # 清理所有临时文件
    import os
    os.remove(temp_yaml_path)
    os.remove(train_txt)
    os.remove(val_txt)
