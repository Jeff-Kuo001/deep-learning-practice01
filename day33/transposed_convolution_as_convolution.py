"""转置卷积是一种卷积：用卷积矩阵的转置完成计算。"""
import torch
from torch.nn import functional as F

kernel = torch.tensor([[1., 2.], [3., 4.]])
x_shape = (3, 3)
basis = torch.eye(9).reshape(9, 1, *x_shape)
conv_matrix = F.conv2d(basis, kernel.reshape(1, 1, 2, 2)).reshape(9, -1).T
x = torch.arange(9, dtype=torch.float32)
y = conv_matrix @ x
restored_direction = conv_matrix.T @ y
print("convolution matrix:", tuple(conv_matrix.shape))
print("forward output:", y.reshape(2, 2))
print("transpose output:", restored_direction.reshape(3, 3))
