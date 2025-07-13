# 户外工程机械安全距离检测系统

## 1. 项目概述

本项目旨在构建一个实时的户外施工现场安全监控系统。它利用计算机视觉模型进行目标检测、实例分割和深度估计，以计算与挖掘机、打桩机、拉管机等关键机械的3D距离。主要目标是通过及时发出警告来预防事故。

本MVP（最小可行产品）阶段专注于使用**YOLOv11**模型实现**目标检测和实例分割**功能。

## 2. 关键技术

- **目标检测/实例分割**: `YOLO (You Only Look Once，意为你只看一次)` 通过 `ultralytics` 库实现。它在速度和精度之间提供了极佳的平衡，非常适合实时应用。
- **数据处理**: `Pandas` 用于Excel文件的读写，`OpenCV` 用于图像处理。
- **深度学习框架**: `PyTorch`。
- **项目配置**: 使用 `YAML` 文件来管理所有参数（模型、数据、训练等），以提高灵活性和可复现性。

## 3. 架构设计思想

本项目在设计上遵循了多个核心原则，以确保代码的健壮性、可维护性和可扩展性。

-   **模块化与高内聚低耦合**: 
    -   项目被划分为独立的功能模块（如 `data_preparation`, `training`, `inference`, `utils`）。每个模块都专注于一个特定的任务（高内聚）。
    -   模块之间的依赖关系被最小化（低耦合），例如，训练脚本不直接依赖于数据准备的内部实现，而是通过标准化的文件格式（YOLO .txt）进行交互。这使得任何模块都可以被独立修改或替换，而不会影响到其他部分。

-   **配置文件驱动 (Configuration-Driven)**: 
    -   所有重要的参数，如模型类型、学习率、批量大小、数据集路径和数据增强选项，都集中在 `configs/yolov11_seg.yaml` 文件中进行管理。
    -   这种方法将“配置”与“代码”分离，使得非开发人员（如数据科学家）也能轻松调整实验参数，而无需修改Python代码。它也极大地提高了实验的可复现性。

-   **可扩展性设计 (Scalability)**:
    -   **模型可替换**: 由于训练和推理脚本从配置文件中读取模型权重，更换不同的YOLO模型（如从`yolov11n-seg.pt`升级到`yolov11x-seg.pt`）或甚至其他模型架构，只需更新配置文件即可。
    -   **功能可扩展**: 如果需要添加新的功能，如姿态估计或深度估计，可以创建新的模块（例如 `pose_estimation/`），并将其集成到主工作流程中，而无需大规模重构现有代码。

-   **数据处理流水线化 (Data Pipelining)**:
    -   数据处理流程被设计成一个清晰的流水线：`原始数据 (Excel + Images)` -> `数据预处理 (excel_to_yolo.py)` -> `标准化格式 (YOLO .txt)` -> `模型消费`。
    -   这种清晰的界限使得每个阶段都可以被独立验证和调试。例如，在开始训练之前，可以先检查生成的 `.txt` 标签文件是否正确。

-   **日志与监控标准化 (Standardized Logging)**:
    -   所有脚本都使用Python的 `logging` 模块来记录关键信息、警告和错误。日志输出格式统一，包含了时间戳和日志级别，便于问题追踪和调试。
    -   在训练过程中，`ultralytics` 库会自动记录详细的性能指标（如mAP, loss），并将它们保存到 `runs/` 目录中，便于后续分析和可视化。

## 5. 目录结构

```
/gemini
├── configs/                  # 所有YAML配置文件
│   └── yolov11_seg.yaml      # YOLOv11分割模型的配置
├── data_preparation/         # 数据处理和转换脚本
│   └── excel_to_yolo.py      # 将Excel标注转换为YOLO .txt格式的脚本
├── datasets/                 # (此项目已不再使用) 原用于存放处理后的数据集
├── docs/                     # 项目文档
├── inference/                # 用于模型推理的脚本
│   └── infer.py              # 主推理脚本
├── logs/                     # (不应提交到Git) 日志文件
├── models/                   # (不应提交到Git) 训练好的模型权重
├── notebooks/                # 用于实验和分析的Jupyter Notebook
├── results/                  # (不应提交到Git) 推理结果
│   └── test_output.xlsx      # 示例输出文件
├── training/                 # 用于模型训练的脚本
│   └── train.py              # 主训练脚本
├── utils/                    # 工具函数 (例如硬件检测、日志记录)
│   └── environment.py        # GPU/CPU检测和设置
│   └── general.py            # 通用辅助函数
├── README.md                 # 本文件
└── requirements.txt          # Python依赖项
```

### 核心模块交互流程

1.  **数据准备**: 用户运行 `data_preparation/excel_to_yolo.py`。该脚本会:
    -   提示用户为训练集和验证集分别选择Excel标注文件。
    -   读取Excel中的标注信息。
    -   在原始图片目录 (`Data/raw/train/images/` 和 `Data/raw/val/images/`) 中，直接在每张图片旁边生成对应的 `.txt` 标签文件。
    -   **重要**: 此过程不会复制任何图片，以节省磁盘空间。每次运行时，它会先自动删除旧的 `.txt` 文件，确保标签总是最新的。
2.  **配置**: 训练前，用户检查并调整 `configs/yolov11_seg.yaml`。该文件已配置为直接从 `Data/raw/` 目录读取数据进行训练。
3.  **训练**: 用户运行 `training/train.py`。
    -   它首先调用 `utils/environment.py` 来检测可用的硬件（GPU/CPU），并设置适当的设备和参数（例如，为A4500 GPU设置更大的批量大小）。
    -   然后从 `.yaml` 文件加载配置。
    -   启动 `ultralytics` 训练流程，该流程自动处理数据加载、增强、模型训练和评估。
    -   评估指标（mAP50, mAP50-95, 精确率, 召回率）会被打印到控制台并保存。
    -   训练出的最佳模型权重保存在 `models/` 目录下。
4.  **推理**: 用户运行 `inference/infer.py`。
    -   它同样使用 `utils/environment.py` 来设置设备。
    -   从 `models/` 加载训练好的模型权重。
    -   处理 `Data/test/images` 文件夹中的图像。
    -   对于每个检测结果，提取类别、置信度、边界框和分割掩码坐标。
    -   读取原始Excel文件，添加新列以保存结果，并将更新后的数据保存到 `results/test_output.xlsx`。

## 6. 安装与设置

**第1步：创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # 在Windows上, 使用 `venv\Scripts\activate`
```

**第2步：安装PyTorch**
使用 `nvidia-smi` 命令检查您的NVIDIA GPU的CUDA版本。然后，访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取适合您系统的正确安装命令。**这是至关重要的一步。**

**第3步：安装依赖**
```bash
pip install -r requirements.txt
```

**第4步：准备数据**
根据以下结构放置您的原始数据，其中 `Yolo_seg_Alarm` 是项目根目录。**注意**：此处的 `Data` 目录应重命名为 `Data/raw` 以匹配脚本配置。
```
/Yolo_seg_Alarm
├── Data/
│   └── raw/
│       ├── train/
│       │   ├── images/       # 包含所有训练图片，可分子文件夹
│       │   │   └── *.jpg
│       │   └── train_annotations.xlsx # 训练集标注文件
│       ├── val/
│       │   ├── images/       # 包含所有验证图片，可分子文件夹
│       │   │   └── *.jpg
│       │   └── val_annotations.xlsx   # 验证集标注文件
│       └── test/             # 测试集（当前未使用）
│           ├── images/
│           │   └── *.jpg
│           └── test_annotations.xlsx
└── gemini/ 
    └── ... (项目文件)
```

## 7. 如何运行

**1. 生成标签文件**
运行转换脚本。系统将分别为训练集和验证集提示您选择对应的Excel文件。
```bash
python data_preparation/excel_to_yolo.py
```

**2. 训练模型**
如果需要，修改 `configs/yolov11_seg.yaml`，然后开始训练：
```bash
python training/train.py --config configs/yolov11_seg.yaml
```

**3. 运行推理**
在测试集上运行推理。结果将保存在 `results` 文件夹中。
```bash
python inference/infer.py --weights models/best.pt --source ../Data/test/ --excel_file ../Data/test/annotations.xlsx
```

## 8. 评估指标

训练脚本将输出以下关键指标：
-   **精确率 (Precision)**: 在所有做出的正向预测中，有多少比例是正确的？
-   **召回率 (Recall)**: 在所有实际的正向实例中，模型正确识别了多少比例？
-   **mAP50**: 在IoU（交并比）阈值为0.5时计算的平均精度均值。这是衡量目标检测质量的标准指标。
-   **mAP50-95**: 在一系列IoU阈值（从0.5到0.95，步长为0.05）上计算的mAP分数的平均值。这为定位精度提供了更全面的评估。
-   **分割IoU (Intersection over Union)**: 也称为Jaccard指数，用于衡量预测分割掩码与真实掩码之间的重叠程度。计算公式为 `(A ∩ B) / (A ∪ B)`。
-   **Dice系数 (Dice Coefficient)**: 与IoU类似，也是衡量分割效果的指标，对小目标的检测更为敏感。计算公式为 `2 * |A ∩ B| / (|A| + |B|)`。

## 9. 后续步骤与待办事项

-   [ ] **实现数据转换脚本**: 创建 `data_preparation/excel_to_yolo.py`。
-   [ ] **实现训练脚本**: 创建带有动态硬件检测功能的 `training/train.py`。
-   [ ] **实现推理脚本**: 创建能将结果输出到Excel的 `inference/infer.py`。
-   [ ] **创建配置文件**: 定义 `configs/yolov11_seg.yaml` 中的默认配置。
-   [ ] **实现环境工具**: 编写 `utils/environment.py` 用于硬件检测。
-   [ ] **开始数据标注**: 为训练数据标注分割掩码。推荐使用CVAT或LabelMe等工具。
-   [ ] **集成深度估计模型**: 在分割模型稳定后，开始集成Metric3D V2。