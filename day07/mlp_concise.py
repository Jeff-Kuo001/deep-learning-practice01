"""Day 7：使用 PyTorch 简洁实现多层感知机。"""
import torch
from torch import nn


features = torch.randn(256, 20)
labels = (features[:, :3].sum(dim=1) > 0).long()
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
criterion = nn.CrossEntropyLoss()

for _ in range(100):
    loss = criterion(model(features), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

accuracy = (model(features).argmax(1) == labels).float().mean()
print("loss:", loss.item(), "accuracy:", accuracy.item())
