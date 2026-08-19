"""Day 7：多层感知机的网络结构与前向传播。"""
import torch
from torch import nn


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
images = torch.randn(8, 1, 28, 28)
print(model)
print("output shape:", model(images).shape)
