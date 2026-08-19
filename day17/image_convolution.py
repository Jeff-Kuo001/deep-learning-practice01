"""图像卷积：二维互相关运算。"""
import torch

def corr2d(x, kernel):
    kh, kw = kernel.shape
    y = torch.zeros(x.shape[0] - kh + 1, x.shape[1] - kw + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i + kh, j:j + kw] * kernel).sum()
    return y

x = torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
kernel = torch.tensor([[1., 0.], [0., -1.]])
print(corr2d(x, kernel))
