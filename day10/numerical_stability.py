"""Day 10：数值稳定的 Softmax 与梯度裁剪。"""
import torch
from torch import nn


def stable_softmax(logits):
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=-1, keepdim=True)


logits = torch.tensor([[1000.0, 1001.0, 1002.0]])
print("stable probabilities:", stable_softmax(logits))

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
loss = model(torch.randn(32, 20)).square().mean()
loss.backward()
before = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
after = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
print("gradient norm:", before.item(), "->", after.item())
