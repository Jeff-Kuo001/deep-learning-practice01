"""单发多框检测（SSD）：多尺度类别预测与边界框预测。"""
import torch
from torch import nn

def prediction_head(in_channels, num_anchors, num_classes):
    return nn.ModuleDict({
        "class": nn.Conv2d(in_channels, num_anchors * (num_classes + 1), 3, padding=1),
        "box": nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1),
    })

class SSDHeads(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()),
        ])
        self.heads = nn.ModuleList([prediction_head(32, 4, num_classes), prediction_head(64, 4, num_classes)])

    def forward(self, x):
        outputs = []
        for block, head in zip(self.features, self.heads):
            x = block(x)
            outputs.append((head["class"](x), head["box"](x)))
        return outputs

for level, (cls, box) in enumerate(SSDHeads()(torch.randn(1, 3, 64, 64))):
    print(level, tuple(cls.shape), tuple(box.shape))
