"""卷积中的填充和步幅。"""
import torch
from torch import nn

x = torch.randn(1, 1, 8, 8)
same_size = nn.Conv2d(1, 1, kernel_size=3, padding=1)(x)
downsampled = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)(x)
print("input:", tuple(x.shape))
print("padding=1:", tuple(same_size.shape))
print("stride=2:", tuple(downsampled.shape))
