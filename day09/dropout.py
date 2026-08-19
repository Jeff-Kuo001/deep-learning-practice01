"""Day 9：从零实现 Dropout 并比较训练/预测模式。"""
import torch
from torch import nn


def dropout_layer(x, probability):
    if probability == 0:
        return x
    if probability == 1:
        return torch.zeros_like(x)
    mask = (torch.rand_like(x) > probability).float()
    return mask * x / (1 - probability)


x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
print("scratch dropout:\n", dropout_layer(x, 0.5))
layer = nn.Dropout(0.5)
layer.train()
print("training mode:\n", layer(x))
layer.eval()
print("evaluation mode:\n", layer(x))
