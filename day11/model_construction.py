"""Day 11：使用 Module 与 Sequential 构造模型。"""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))


network = nn.Sequential(MLP(), nn.ReLU(), nn.Linear(10, 2))
print(network)
print("output shape:", network(torch.randn(4, 20)).shape)
