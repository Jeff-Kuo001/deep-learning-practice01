"""Day 10：模型初始化与激活函数。"""
import torch
from torch import nn


relu_model = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
sigmoid_model = nn.Sequential(nn.Linear(100, 100), nn.Sigmoid())
nn.init.kaiming_normal_(relu_model[0].weight, nonlinearity="relu")
nn.init.xavier_uniform_(sigmoid_model[0].weight)

x = torch.randn(64, 100)
relu_output = relu_model(x)
sigmoid_output = sigmoid_model(x)
print("ReLU mean/std:", relu_output.mean().item(), relu_output.std().item())
print("Sigmoid mean/std:", sigmoid_output.mean().item(), sigmoid_output.std().item())
