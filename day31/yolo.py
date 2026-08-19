"""你只看一次（YOLO）：网格化的类别、置信度和边界框预测。"""
import torch
from torch import nn

class TinyYOLO(nn.Module):
    def __init__(self, grid_size=7, boxes_per_cell=2, num_classes=3):
        super().__init__()
        self.grid_size = grid_size
        self.output_size = boxes_per_cell * 5 + num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )
        self.predictor = nn.Conv2d(64, self.output_size, 1)

    def forward(self, x):
        return self.predictor(self.features(x)).permute(0, 2, 3, 1)

output = TinyYOLO()(torch.randn(2, 3, 224, 224))
print("B × S × S × (B*5+C):", tuple(output.shape))
