"""区域卷积神经网络（R-CNNs）：候选区域特征池化与分类回归。"""
import torch
from torch import nn
from torchvision.ops import roi_align

class FastRCNNHead(nn.Module):
    def __init__(self, in_channels=16, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, in_channels, 3, padding=1), nn.ReLU())
        self.classifier = nn.Linear(in_channels * 4 * 4, num_classes)
        self.box_regressor = nn.Linear(in_channels * 4 * 4, num_classes * 4)

    def forward(self, images, rois):
        features = self.backbone(images)
        pooled = roi_align(features, rois, output_size=(4, 4), spatial_scale=1.0)
        flat = pooled.flatten(1)
        return self.classifier(flat), self.box_regressor(flat)

model = FastRCNNHead()
images = torch.randn(1, 3, 32, 32)
rois = torch.tensor([[0., 2., 2., 20., 20.], [0., 8., 8., 28., 28.]])
classes, boxes = model(images, rois)
print("class logits:", tuple(classes.shape), "box offsets:", tuple(boxes.shape))
