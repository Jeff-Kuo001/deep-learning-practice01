"""Day 4：读取 Fashion-MNIST 图像分类数据集。"""
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_batch(data_dir, batch_size=64, download=False):
    dataset = datasets.FashionMNIST(
        data_dir,
        train=True,
        transform=transforms.ToTensor(),
        download=download,
    )
    return next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    images, labels = load_batch(args.data, args.batch_size, args.download)
    print("images:", images.shape)
    print("labels:", labels.shape)
    print("pixel range:", images.min().item(), images.max().item())


if __name__ == "__main__":
    main()
