"""多输入通道和多输出通道卷积。"""
import torch

def corr2d(x, k):
    h, w = k.shape
    return torch.stack([
        torch.stack([(x[i:i+h, j:j+w] * k).sum() for j in range(x.shape[1]-w+1)])
        for i in range(x.shape[0]-h+1)
    ])

def corr2d_multi_in(x, k):
    return sum(corr2d(xi, ki) for xi, ki in zip(x, k))

def corr2d_multi_in_out(x, k):
    return torch.stack([corr2d_multi_in(x, ko) for ko in k])

x = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
k = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
print(corr2d_multi_in_out(x, k))
