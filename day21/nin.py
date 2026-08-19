"""Day 21：网络中的网络（NiN）。"""
import torch
from torch import nn


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )


def nin(num_classes=10):
    return nn.Sequential(
        nin_block(1, 96, 11, 4, 0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, 5, 1, 2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, 3, 1, 1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, num_classes, 3, 1, 1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )


if __name__ == "__main__":
    print("output shape:", nin()(torch.randn(1, 1, 224, 224)).shape)
