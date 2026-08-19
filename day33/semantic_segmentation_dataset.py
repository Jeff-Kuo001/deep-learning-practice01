"""语义分割数据集：图像与像素标签的同步裁剪。"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

class ToySegmentationDataset(Dataset):
    def __init__(self, size=6, crop_size=(48, 48)):
        self.size = size
        self.crop_size = crop_size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.rand(3, 64, 64)
        mask = torch.randint(0, 3, (64, 64))
        top = (index * 3) % (64 - self.crop_size[0] + 1)
        left = (index * 5) % (64 - self.crop_size[1] + 1)
        image = TF.crop(image, top, left, *self.crop_size)
        mask = TF.crop(mask, top, left, *self.crop_size)
        return image, mask.long()

images, masks = next(iter(DataLoader(ToySegmentationDataset(), batch_size=2)))
print(tuple(images.shape), tuple(masks.shape), masks.dtype)
