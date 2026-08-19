"""Day 4：比较均方误差、绝对误差与交叉熵损失。"""
import torch
import torch.nn.functional as F


prediction = torch.tensor([1.0, 2.5, 4.0])
target = torch.tensor([1.5, 2.0, 3.0])
print("MSE:", F.mse_loss(prediction, target).item())
print("L1:", F.l1_loss(prediction, target).item())

logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 1.6, 0.1]])
labels = torch.tensor([0, 1])
print("cross entropy:", F.cross_entropy(logits, labels).item())
