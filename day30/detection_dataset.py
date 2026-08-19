"""物体检测数据集：图像与变长边界框的批处理。"""
import torch
from torch.utils.data import Dataset, DataLoader

class ToyDetectionDataset(Dataset):
    def __init__(self, size=8):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.rand(3, 64, 64)
        boxes = torch.tensor([[8., 10., 36., 42.]])
        labels = torch.tensor([index % 2])
        return image, {"boxes": boxes, "labels": labels}

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

loader = DataLoader(ToyDetectionDataset(), batch_size=2, collate_fn=collate_fn)
images, targets = next(iter(loader))
print("images:", tuple(images.shape))
print("first boxes:", targets[0]["boxes"])
