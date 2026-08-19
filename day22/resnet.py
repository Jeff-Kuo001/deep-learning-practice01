"""Day 22：残差网络 ResNet-18。"""
import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        )

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


def resnet18(num_classes=10):
    layers = [
        nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
    ]
    in_channels = 64
    for out_channels in (64, 128, 256, 512):
        stride = 1 if out_channels == 64 else 2
        layers += [
            Residual(in_channels, out_channels, stride),
            Residual(out_channels, out_channels),
        ]
        in_channels = out_channels
    layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, num_classes)]
    return nn.Sequential(*layers)


if __name__ == "__main__":
    print("output shape:", resnet18()(torch.randn(1, 1, 224, 224)).shape)
