import torch
from PIL import Image
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
    
    # 加载测试图片（请将路径替换为实际图片路径）
    image_path = r"D:\program\python\PythonProject\Yolo_seg_Alarm\Data\test_image.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"测试图片不存在: {image_path}")
    image = Image.open(image_path)
    
    # 执行推理
    results = model(image)
    print("YOLOv8推理成功")
    
    # 生成并显示标注图像
    annotated_image = results[0].plot()
    Image.fromarray(annotated_image).show()
    print("推理结果图像已显示")

except Exception as e:
    print(f"测试失败: {str(e)}")
