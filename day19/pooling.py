"""池化层的从零实现。"""
import torch

def pool2d(x, pool_size, mode="max"):
    ph, pw = pool_size
    y = torch.zeros(x.shape[0] - ph + 1, x.shape[1] - pw + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            window = x[i:i+ph, j:j+pw]
            y[i, j] = window.max() if mode == "max" else window.mean()
    return y

x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
print("max pooling:\n", pool2d(x, (2, 2)))
print("average pooling:\n", pool2d(x, (2, 2), "avg"))
