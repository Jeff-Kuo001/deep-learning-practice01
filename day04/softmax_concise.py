"""Day 4：使用 PyTorch 简洁实现 Softmax 回归。"""
import torch
from torch import nn


features = torch.randn(300, 2)
labels = ((features[:, 0] > 0).long() + (features[:, 1] > 0).long()).clamp(max=2)
model = nn.Linear(2, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
criterion = nn.CrossEntropyLoss()

for _ in range(80):
    loss = criterion(model(features), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

accuracy = (model(features).argmax(dim=1) == labels).float().mean()
print("loss:", loss.item(), "accuracy:", accuracy.item())
