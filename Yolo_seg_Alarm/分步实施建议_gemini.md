# 基于YOLOv11-seg与Metric3D V2的重型机械安全距离预警项目实施建议

## 1. 项目概述与目标

本项目旨在构建一个端到端的智能视觉监控系统，用于实时检测户外场景（如农田、施工场地）中的特定重型机械（挖掘机、打桩机、拉管机），并精确计算其与预设安全参考线之间的三维空间距离，实现安全预警功能。

**核心挑战:**
- **数据层面**: 缺乏实例分割和深度信息的标注数据。
- **算法层面**: 需要高召回率、高鲁棒性，以应对户外复杂场景（小目标、遮挡、光照变化、图像模糊）。
- **工程层面**: 平衡模型精度与推理速度，实现准实时处理。

## 2. 技术选型与分析

为了在精度、速度和部署难度之间取得最佳平衡，我们对关键模块的技术方案进行如下对比分析。

| 模块 | 推荐方案 | 备选方案 1 | 备选方案 2 | 综合评估 |
| :--- | :--- | :--- | :--- | :--- |
| **实例分割** | **YOLOv9-seg / YOLOv8-seg** | Mask R-CNN | RT-DETR | **推荐YOLOv9/v8-seg**。YOLOv11目前并非官方或广泛认可的版本，其稳定性和社区支持存疑。YOLOv9/v8在精度和速度上取得了极佳的平衡，拥有活跃的社区和丰富的部署工具链，更适合作为项目起点。Mask R-CNN精度高但速度慢，不适合实时应用。 |
| **深度估计** | **Metric3D V2** | MiDaS v3.1 | ZoeDepth | **推荐Metric3D V2**。其在零样本室外场景的度量恢复上表现出色。MiDaS作为备选，提供了不同规模的模型，灵活性高。ZoeDepth同样是优秀的SOTA模型。三者都可以作为初始方案进行集成测试。 |
| **三维距离计算** | **OpenCV + Numpy** | Open3D | - | **推荐OpenCV**。其功能全面，计算高效，足以满足本项目从2D像素+深度到3D坐标转换和距离计算的需求。Open3D更适用于复杂的点云后处理，本项目中略显重型。 |

**最终技术栈建议:**
- **实例分割**: **YOLOv8-seg** (社区支持最好，文档最全，性能优异)
- **深度估计**: **Metric3D V2** (针对室外场景的零样本度量恢复能力)
- **核心库**: **PyTorch, OpenCV, Ultralytics, Timm**

## 3. 系统架构与数据流设计

采用模块化的级联架构，解耦各个功能，便于独立开发、测试和优化。

**数据流:**

`原始图像` → `1. 实例分割模块` → `2. 深度估计模块` → `3. 坐标转换与距离计算模块` → `4. 结果输出/预警模块`

1.  **实例分割模块 (YOLOv8-seg):**
    - **输入**: 原始图像 (H, W, 3)
    - **输出**: 检测框 `[x,y,w,h]`, 类别 `ID`, 置信度 `score`, 实例掩码 `(h, w)`
2.  **深度估计模块 (Metric3D V2):**
    - **输入**: 原始图像 (H, W, 3)
    - **输出**: 深度图 `(H, W)`，每个像素值为估计的米制距离。
3.  **坐标转换与距离计算模块 (OpenCV):**
    - **输入**: 实例掩码, 深度图, 相机内参 `K`, 参考线定义。
    - **处理**: 提取掩码区域内的深度值，将像素坐标 `(u,v)` 和深度 `d` 转换为相机坐标系下的三维点 `(X,Y,Z)`，计算该点云与参考线的最短距离。
    - **输出**: 每个目标的最小三维距离。
4.  **结果输出/预警模块:**
    - **输入**: 原始图像, 检测结果, 分割结果, 计算出的距离。
    - **输出**: 可视化的结果图像（绘制检测框、掩码、距离文本），或触发预警信号（如API调用、声音提示）。

## 4. 关键实现步骤 (Step-by-Step)

### **Phase 1: 环境搭建与项目初始化**

1.  **创建Conda虚拟环境:**
    ```bash
    conda create -n alarm_system python=3.9
    conda activate alarm_system
    ```
2.  **安装核心依赖:**
    ```bash
    # 安装PyTorch (请根据您的CUDA版本从官网选择命令)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # 安装Ultralytics (YOLOv8)
    pip install ultralytics

    # 安装其他库
    pip install opencv-python numpy pandas matplotlib tqdm timm
    ```
3.  **项目结构初始化:**
    ```
    /Yolo_seg_Alarm
    ├── configs/             # 配置文件 (如yolo_config.yaml)
    ├── data/                # 数据集
    │   ├── images/
    │   └── labels/
    ├── notebooks/           # 实验和探索性的Jupyter Notebook
    ├── scripts/             # 数据处理、训练、评估等脚本
    │   ├── 1_excel_to_yolo.py
    │   ├── 2_generate_masks.py
    │   ├── 3_train.py
    │   └── 4_inference.py
    ├── weights/             # 训练好的模型权重
    ├── requirements.txt     # 项目依赖
    └── README.md
    ```

### **Phase 2: 数据准备与标注**

这是项目中最关键且耗时的一步。由于没有分割标注，我们需要“无中生有”。

1.  **转换检测框标注**: 
    - 编写 `scripts/1_excel_to_yolo.py` 脚本，读取您提供的Excel文件，将其中的COCO检测框 `[xmin,ymin,w,h]` 转换为YOLO的 `.txt` 格式（`class_id x_center y_center width height`，均为归一化值）。

2.  **半自动生成分割掩码 (核心步骤)**:
    - **策略**: 利用 **Segment Anything Model (SAM)** 的强大零样本分割能力，结合已有的检测框，快速生成初始分割掩码。
    - **工具**: 推荐使用 `Label Studio` 或 `CVAT`，它们集成了SAM模型，可以实现“点击即分割”或“框内分割”。
    - **流程**: 
        a. 导入所有图片和YOLO格式的检测框到 `Label Studio`。
        b. 对每个检测框，使用SAM自动生成掩码。
        c. **人工审核与修正**: 这是保证质量的关键。快速检查并修正SAM生成的掩码，特别是边缘和被遮挡的部分。
        d. 导出为YOLO分割格式（每个目标一个txt文件，内容为 `class_id norm_x1 norm_y1 norm_x2 norm_y2 ...`）。

3.  **创建`data.yaml`文件**: 
    在 `configs` 目录下创建 `data.yaml`，定义训练集、验证集的路径和类别信息。
    ```yaml
    train: ../data/images/train/
    val: ../data/images/val/

    # Classes
    nc: 3  # 类别数量
    names: ['excavator', 'piling_machine', 'pipe_pulling_machine'] # 挖掘机, 打桩机, 拉管机
    ```

### **Phase 3: 模型训练 (YOLOv8-seg)**

1.  **编写训练脚本 `scripts/3_train.py`:**
    ```python
    from ultralytics import YOLO

    # 加载预训练的YOLOv8分割模型
    model = YOLO('yolov8n-seg.pt') # 从小模型开始实验，快速迭代

    # 训练模型
    results = model.train(
        data='../configs/data.yaml', 
        epochs=150, 
        imgsz=640, # 考虑到显存和速度，可从640开始
        batch=8,   # A4500显存较大，可以尝试8或16
        patience=30, # 30个epoch没有提升则早停
        device=0,
        project='../runs/train',
        name='exp_yolov8n_seg',
        # --- 重点调优参数 ---
        # 提高召回率，可以适当降低box损失的权重，或调整iou阈值
        # box=7.5, cls=0.5, dfl=1.5, 
        # --- 数据增强 ---
        augment=True, # 开启内置数据增强
        mosaic=1.0,   # Mosaic对小目标检测有益
        mixup=0.1,    # Mixup
    )
    ```

### **Phase 4: 推理与集成**

1.  **编写推理脚本 `scripts/4_inference.py`**: 该脚本将整合所有模块。

    ```python
    import cv2
    import torch
    from ultralytics import YOLO
    # 假设Metric3D模型封装在一个类中
    # from metric3d_model import Metric3D

    # --- 1. 初始化模型 ---
    seg_model = YOLO('../weights/best.pt')
    # depth_model = Metric3D(model_path='path/to/metric3d/weights')

    # --- 2. 加载图像和相机参数 ---
    # !!! 关键：必须要有相机内参矩阵K !!!
    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    image = cv2.imread('path/to/image.jpg')

    # --- 3. 执行推理 ---
    seg_results = seg_model(image)[0]
    # depth_map = depth_model.inference(image)

    # --- 4. 处理每个检测到的目标 ---
    if seg_results.masks is not None:
        for i, mask_data in enumerate(seg_results.masks.data):
            # a. 获取掩码和检测框
            mask = mask_data.cpu().numpy()
            box = seg_results.boxes.xyxy[i].cpu().numpy()

            # b. 提取目标区域的深度值
            # object_depths = depth_map[mask > 0]
            # representative_depth = np.median(object_depths) # 使用中位数更鲁棒

            # c. 计算三维坐标 (以目标中心点为例)
            # cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            # Z = representative_depth
            # X = (cx - K[0, 2]) * Z / K[0, 0]
            # Y = (cy - K[1, 2]) * Z / K[1, 1]

            # d. 计算与参考线的距离 (此处为伪代码)
            # distance = calculate_distance_to_ref_line((X,Y,Z), ref_line_params)

            # e. 可视化
            # cv2.putText(image, f'{distance:.2f}m', ...)

    # --- 5. 显示/保存结果 ---
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    ```

## 5. 模型评估指标

为了实现您的目标（高召回，平衡精确率），需要关注以下指标：

- **mAP (mean Average Precision)**: @IoU=0.5:0.95 (综合评估标准), @IoU=0.5 (宽松评估标准)。同时要看`mAP_mask`和`mAP_box`。
- **Precision (精确率)**: `P = TP / (TP + FP)`。在所有预测为正的样本中，真正为正的比例。越高，误报越少。
- **Recall (召回率)**: `R = TP / (TP + FN)`。在所有真正为正的样本中，被成功预测为正的比例。**这是您需要重点提高的指标，越高，漏报越少。**
- **F1-Score**: `2 * (P * R) / (P + R)`。精确率和召回率的调和平均数，用于综合评估。

**如何平衡?**
在YOLO的验证和推理中，可以调整`conf`（置信度阈值）和`iou`（IoU阈值）。
- **降低`conf`阈值**: 会检测出更多目标，**提高召回率**，但可能引入更多误报（降低精确率）。
- **调整`iou`阈值**: 影响NMS（非极大值抑制）过程，对重叠目标的处理有影响。

## 6. 关键问题与解决方案

- **问题1: 如何获取相机内参?**
  - **最佳方案**: 使用棋盘格或圆形标定板对每个摄像头进行**相机标定**。这是一个标准的OpenCV流程，可以获得精确的内参 `K` 和畸变系数。
  - **次优方案**: 如果无法标定，可以根据相机型号查询其数据手册，获取近似的焦距和传感器尺寸来估算。**但这会严重影响最终距离计算的精度。**

- **问题2: 如何定义参考线?**
  - **方案A (图像空间定义)**: 用户在图像上交互式地画一条线。需要将这条2D线反投影到三维空间，这要求地面是平的，或者有额外的高度信息。
  - **方案B (三维空间定义)**: 在相机的三维坐标系中定义一条线（例如，平行于X轴，距离原点5米）。这种方式更稳定，但需要对场景的几何结构有先验知识。

## 7. 部署与优化建议

1.  **模型导出**: 将训练好的PyTorch (`.pt`) 模型导出为更高效的格式。
    ```bash
    yolo export model=best.pt format=onnx imgsz=640
    # 或导出为TensorRT
    yolo export model=best.pt format=engine device=0
    ```
2.  **推理后端**: 使用 `ONNX Runtime` 或 `NVIDIA TensorRT` 进行推理，可以获得数倍的性能提升。
3.  **代码优化**: 批量推理（batch inference）、异步处理、减少CPU和GPU之间的数据拷贝。

**强烈建议您先从解决“相机内参”和“参考线定义”这两个问题开始，它们是后续所有三维计算的基础。**