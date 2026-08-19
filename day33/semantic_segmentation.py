"""语义分割：像素级分类网络。"""
import torch
from torch import nn

class SegmentationNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

x = torch.randn(2, 3, 64, 64)
labels = torch.randint(0, 4, (2, 64, 64))
logits = SegmentationNet()(x)
print("logits:", tuple(logits.shape))
print("loss:", float(nn.CrossEntropyLoss()(logits, labels)))
