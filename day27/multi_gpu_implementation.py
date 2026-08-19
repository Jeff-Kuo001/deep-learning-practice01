"""多 GPU 训练的实现：模型复制、损失反传和参数更新。"""
import torch
from torch import nn

def train_step(model, x, y, optimizer):
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    return float(loss)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
model = nn.DataParallel(base_model).to(device) if torch.cuda.device_count() > 1 else base_model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.randn(32, 10, device=device)
y = torch.randint(0, 3, (32,), device=device)
print("loss:", train_step(model, x, y, optimizer))
