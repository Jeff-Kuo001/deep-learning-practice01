"""Day 11：自定义无参数层与带参数层。"""
import torch
from torch import nn


class CenteredLayer(nn.Module):
    def forward(self, x):
        return x - x.mean()


class MyLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        return torch.relu(x @ self.weight + self.bias)


print("centered:", CenteredLayer()(torch.tensor([1.0, 2.0, 3.0])))
print("custom linear:", MyLinear(5, 3)(torch.randn(2, 5)).shape)
