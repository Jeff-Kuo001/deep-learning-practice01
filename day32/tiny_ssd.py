"""SSD 实现：TinySSD 的多尺度预测网络。"""
import torch
from torch import nn

class DownsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.MaxPool2d(2),
        )

class TinySSD(nn.Module):
    def __init__(self, num_classes=2, anchors_per_pixel=4):
        super().__init__()
        channels = [3, 16, 32, 64]
        self.blocks = nn.ModuleList([DownsampleBlock(channels[i], channels[i+1]) for i in range(3)])
        self.class_heads = nn.ModuleList([
            nn.Conv2d(c, anchors_per_pixel * (num_classes + 1), 3, padding=1) for c in channels[1:]
        ])
        self.box_heads = nn.ModuleList([
            nn.Conv2d(c, anchors_per_pixel * 4, 3, padding=1) for c in channels[1:]
        ])

    def forward(self, x):
        class_outputs, box_outputs = [], []
        for block, cls_head, box_head in zip(self.blocks, self.class_heads, self.box_heads):
            x = block(x)
            class_outputs.append(cls_head(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 3))
            box_outputs.append(box_head(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4))
        return torch.cat(class_outputs, 1), torch.cat(box_outputs, 1)

classes, boxes = TinySSD()(torch.randn(2, 3, 64, 64))
print("classes:", tuple(classes.shape), "boxes:", tuple(boxes.shape))
