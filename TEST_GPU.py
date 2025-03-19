import torch
import torch
print(torch.__version__)  # PyTorch 版本
print(torch.version.cuda)  # PyTorch 关联的 CUDA 版本
print(torch.cuda.is_available())  # 检查是否检测到 GPU