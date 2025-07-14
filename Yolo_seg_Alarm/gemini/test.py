import torch
from PIL import Image
import os
from utils.environment import get_device_and_adjust_config

# 初始化配置
config = {}

# 检测显卡并获取动态配置
device, config = get_device_and_adjust_config(config)

# 打印系统信息
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# 测试YOLO推理
from ultralytics import YOLO
try:
    # 动态加载模型
    model_name = config['weights']
    model = YOLO(model_name)
    print(f"模型加载成功: {model_name}")
    print('模型类别:', list(model.names.values())[:5])
    
    # 加载测试图片（请将路径替换为实际图片路径）
    image_path = r"D:\program\python\PythonProject\Yolo_seg_Alarm\Data\test_image.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"测试图片不存在: {image_path}")
    image = Image.open(image_path)
    
    # 执行推理
    results = model(image)
    print(f"{model_name}推理成功")
    
    # 生成并显示标注图像
    annotated_image = results[0].plot()
    Image.fromarray(annotated_image).show()
    print("推理结果图像已显示")

except Exception as e:
    print(f"测试失败: {str(e)}")
