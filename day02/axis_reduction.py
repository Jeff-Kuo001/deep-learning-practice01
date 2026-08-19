"""Day 2：按指定轴求和并保留维度。"""
import torch


x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print("input shape:", x.shape)
print("sum axis 0:", x.sum(dim=0).shape)
print("sum axis 1:", x.sum(dim=1).shape)
print("sum axes 0 and 2:", x.sum(dim=(0, 2)).shape)
kept = x.sum(dim=1, keepdim=True)
print("keepdim shape:", kept.shape)
print("broadcast normalized shape:", (x / x.sum(dim=1, keepdim=True)).shape)
