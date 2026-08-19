"""Day 29：CIFAR-10 图像分类训练。"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from image_models import build_cifar10_model
from image_transforms import cifar10_transform
from train_utils import choose_device, train_one_epoch


def train_cifar10(data_dir, epochs=1, download=False):
    dataset = datasets.CIFAR10(
        data_dir,
        train=True,
        transform=cifar10_transform(),
        download=download,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    device = choose_device()
    model = build_cifar10_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        loss, accuracy = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch {epoch + 1}: loss={loss:.4f}, accuracy={accuracy:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    train_cifar10(args.data, args.epochs, args.download)


if __name__ == "__main__":
    main()
