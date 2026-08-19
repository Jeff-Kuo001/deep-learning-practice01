"""Day 1：用一个小型神经网络展示深度学习的基本流程。"""
import torch
from torch import nn


torch.manual_seed(0)
features = torch.randn(64, 4)
labels = (features.sum(dim=1) > 0).long()
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

for _ in range(50):
    loss = nn.functional.cross_entropy(model(features), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

accuracy = (model(features).argmax(dim=1) == labels).float().mean()
print("loss:", loss.item())
print("accuracy:", accuracy.item())
