"""图片分类竞赛：训练、验证与预测的基本流程。"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
train_x = torch.randn(128, 3, 32, 32)
train_y = torch.randint(0, 10, (128,))
loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for x, y in loader:
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
print("last batch loss:", float(loss))
print("prediction shape:", tuple(model(train_x[:4]).shape))
