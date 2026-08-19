"""从全连接层到卷积：1×1 卷积与逐像素全连接等价。"""
import torch
from torch import nn

torch.manual_seed(0)
x = torch.randn(2, 3, 4, 5)
linear = nn.Linear(3, 6)
conv = nn.Conv2d(3, 6, kernel_size=1)
with torch.no_grad():
    conv.weight.copy_(linear.weight[:, :, None, None])
    conv.bias.copy_(linear.bias)

y_linear = linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
y_conv = conv(x)
print("output shape:", tuple(y_conv.shape))
print("maximum difference:", float((y_linear - y_conv).abs().max()))
