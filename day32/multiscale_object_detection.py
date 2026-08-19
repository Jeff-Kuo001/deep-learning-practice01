"""多尺度物体检测：不同分辨率特征图上的锚框数量。"""
import torch
from torch.nn import functional as F

def anchors_per_level(feature, boxes_per_pixel):
    h, w = feature.shape[-2:]
    return h * w * boxes_per_pixel

x = torch.randn(1, 64, 32, 32)
features = []
for _ in range(4):
    features.append(x)
    x = F.adaptive_max_pool2d(x, (max(1, x.shape[-2] // 2), max(1, x.shape[-1] // 2)))

for i, feature in enumerate(features):
    print(f"level {i}: shape={tuple(feature.shape)}, anchors={anchors_per_level(feature, 4)}")
