"""Day 11：访问、初始化与共享模型参数。"""
import torch
from torch import nn


shared = nn.Linear(8, 8)
network = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared)
for module in network.modules():
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.01)
        nn.init.zeros_(module.bias)

for name, parameter in network.named_parameters():
    print(name, tuple(parameter.shape))
print("shared weight:", network[2].weight.data_ptr() == network[4].weight.data_ptr())
print("output:", network(torch.randn(2, 4)).shape)
