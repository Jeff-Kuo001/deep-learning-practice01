"""多 GPU 训练：使用 DataParallel 切分小批量。"""
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

x = torch.randn(16, 20, device=device)
y = torch.randint(0, 5, (16,), device=device)
loss = nn.CrossEntropyLoss()(model(x), y)
loss.backward()
print("device count:", torch.cuda.device_count())
print("loss:", float(loss))
