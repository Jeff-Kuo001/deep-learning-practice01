"""Day 9：权重衰减对模型参数范数的影响。"""
import torch
from torch import nn


torch.manual_seed(0)
features = torch.randn(40, 100)
labels = torch.randn(40, 1)


def train(weight_decay):
    model = nn.Linear(100, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=weight_decay)
    for _ in range(100):
        loss = nn.functional.mse_loss(model(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), model.weight.norm().item()


print("without decay:", train(0.0))
print("with decay:", train(0.01))
