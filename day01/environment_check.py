"""Day 1：检查 PyTorch 安装与运行设备。"""
import platform

import torch
import torchvision


if __name__ == "__main__":
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    x = torch.tensor([1.0, 2.0, 3.0])
    print("tensor test:", x * 2)
