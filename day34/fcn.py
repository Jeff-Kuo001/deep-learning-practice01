"""全连接卷积神经网络（FCN）：将分类特征恢复为像素预测。"""
import torch
from torch import nn

class FCN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Conv2d(64, num_classes, 1)
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, 8, stride=4, padding=2)

    def forward(self, x):
        return self.upsample(self.classifier(self.features(x)))

output = FCN()(torch.randn(1, 3, 64, 64))
print("pixel logits:", tuple(output.shape))
