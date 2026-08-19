"""Day 20：使用重复卷积块构造 VGG。"""
import torch
from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(num_classes=10):
    architecture = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    blocks = []
    in_channels = 1
    for num_convs, out_channels in architecture:
        blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *blocks,
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )


if __name__ == "__main__":
    network = vgg()
    print("output shape:", network(torch.randn(1, 1, 224, 224)).shape)
