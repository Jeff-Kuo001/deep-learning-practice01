"""Day 28：替换预训练 ResNet-18 分类头进行微调。"""
import argparse

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes, pretrained=False, freeze_backbone=False):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    network = build_model(args.classes, args.pretrained, args.freeze_backbone)
    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print("output classes:", network.fc.out_features)
    print("trainable parameters:", trainable)
