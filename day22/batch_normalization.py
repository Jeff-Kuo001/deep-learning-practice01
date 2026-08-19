"""Day 22：从零实现批量归一化层。"""
import torch
from torch import nn


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims, momentum=0.9, eps=1e-5):
        super().__init__()
        shape = (1, num_features) if num_dims == 2 else (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.register_buffer("moving_mean", torch.zeros(shape))
        self.register_buffer("moving_var", torch.ones(shape))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        if self.training:
            dims = (0,) if x.ndim == 2 else (0, 2, 3)
            mean = x.mean(dim=dims, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
            self.moving_mean.mul_(self.momentum).add_(mean.detach() * (1 - self.momentum))
            self.moving_var.mul_(self.momentum).add_(var.detach() * (1 - self.momentum))
        else:
            mean, var = self.moving_mean, self.moving_var
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


if __name__ == "__main__":
    layer = BatchNorm(3, num_dims=4)
    x = torch.randn(8, 3, 16, 16)
    y = layer(x)
    print("channel means:", y.mean(dim=(0, 2, 3)))
    print("channel variances:", y.var(dim=(0, 2, 3), unbiased=False))
