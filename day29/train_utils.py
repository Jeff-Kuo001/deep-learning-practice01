"""图像分类训练公共函数。"""
import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += len(images)

    return total_loss / samples, correct / samples


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
