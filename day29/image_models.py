"""Day 29 图像分类模型。"""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_cifar10_model():
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def build_dog_breed_model(num_classes, pretrained=False):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
