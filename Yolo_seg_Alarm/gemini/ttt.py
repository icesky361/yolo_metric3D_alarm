import os
# 打印详细路径调试信息
current_file = __file__
print(f"当前文件路径: {current_file}")
parent_dir = os.path.dirname(current_file)
print(f"父目录: {parent_dir}")
model_path = os.path.dirname(parent_dir)
print(f"模型路径 (上两级): {model_path}")
print(os.path.dirname(model_path))
print(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))