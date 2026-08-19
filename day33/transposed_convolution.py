"""转置卷积的基本实现。"""
import torch
from torch import nn

x = torch.tensor([[[[0., 1.], [2., 3.]]]])
layer = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[[[1., 2.], [3., 4.]]]]))
print(layer(x))
print("output shape:", tuple(layer(x).shape))
