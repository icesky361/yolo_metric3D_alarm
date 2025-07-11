import torch
print(torch.cuda.is_available())  # 应输出True
print(torch.cuda.get_device_name(0))  # 应显示GeForce MX350

# 检查PyTorch和CUDA版本
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")

# 测试YOLOv8推理
from ultralytics import YOLO
import os
try:
    # 加载轻量级模型以适应2GB显存
    model = YOLO(r"D:\program\python\PythonProject\Yolo_seg_Alarm\yolov8n-seg.pt") 
    print("YOLOv8模型加载成功")
    print('模型类别:', list(model.names.values())[:5])
    
    # 创建测试图像（1x3x640x640随机张量）
    test_image = torch.randn(1, 3, 640, 640).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 执行推理
    results = model(test_image)
    print("YOLOv8推理成功")
    print(f"推理结果: {results}")

except Exception as e:
    print(f"测试失败: {str(e)}")