"""序列模型：自回归预测。"""
import torch
from torch import nn

torch.manual_seed(0)
time = torch.arange(0, 200, dtype=torch.float32)
series = torch.sin(0.05 * time) + 0.1 * torch.randn(200)
tau = 8
features = torch.stack([series[i:200-tau+i] for i in range(tau)], dim=1)
labels = series[tau:].reshape(-1, 1)

model = nn.Sequential(nn.Linear(tau, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
for _ in range(100):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(features), labels)
    loss.backward()
    optimizer.step()
print("training loss:", float(loss))
print("next value:", float(model(series[-tau:].reshape(1, -1))))
