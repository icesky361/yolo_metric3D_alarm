from pathlib import Path
# 从train2.py位置计算项目根目录
project_root = Path(__file__).resolve().parents[3]  # 向上三级到Yolo_metric_alarm

data_path = project_root / 'Data' / 'raw'
config['path'] = str(data_path)

# 添加验证
print(f"计算后的数据集路径: {data_path}")
assert data_path.exists(), f"数据集路径不存在: {data_path}"