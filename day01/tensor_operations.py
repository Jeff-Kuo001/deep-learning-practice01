"""Day 1：张量创建、索引、形状变换与广播。"""
import torch


if __name__ == "__main__":
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    y = torch.ones(3, 4)
    print("x:\n", x)
    print("shape:", x.shape, "elements:", x.numel())
    print("last row:", x[-1])
    print("elementwise sum:\n", x + y)
    print("concatenate:", torch.cat((x, y), dim=0).shape)
    print("broadcast:\n", x + torch.arange(4))
