"""Day 3：使用 PyTorch 简洁实现线性回归。"""
import torch
from torch import nn


features = torch.randn(1000, 2)
labels = features @ torch.tensor([2.0, -3.4]) + 4.2
dataset = torch.utils.data.TensorDataset(features, labels[:, None])
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
loss_fn = nn.MSELoss()

for epoch in range(3):
    for x, y in loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch + 1}: loss={loss_fn(model(features), labels[:, None]).item():.6f}")
