import torch
print(torch.cuda.is_available())  # 应输出True
print(torch.cuda.get_device_name(0))  # 应显示GeForce MX350