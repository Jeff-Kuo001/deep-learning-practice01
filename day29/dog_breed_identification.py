"""Day 29：使用迁移学习完成狗品种识别。"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from image_models import build_dog_breed_model
from image_transforms import dog_breed_transform
from train_utils import choose_device, train_one_epoch


def train_dog_breed(train_dir, epochs=1, pretrained=False):
    dataset = datasets.ImageFolder(train_dir, transform=dog_breed_transform())
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    device = choose_device()
    model = build_dog_breed_model(len(dataset.classes), pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        loss, accuracy = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch {epoch + 1}: loss={loss:.4f}, accuracy={accuracy:.4f}")
    return model, dataset.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    train_dog_breed(args.train, args.epochs, args.pretrained)


if __name__ == "__main__":
    main()
