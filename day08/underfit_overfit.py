"""Day 8：比较多项式模型的欠拟合与过拟合。"""
import math

import torch


torch.manual_seed(1)
train_x = torch.linspace(-2, 2, 20).reshape(-1, 1)
test_x = torch.linspace(-2, 2, 100).reshape(-1, 1)
train_y = 1 + 2 * train_x - 0.5 * train_x ** 2 + 0.2 * torch.randn_like(train_x)
test_y = 1 + 2 * test_x - 0.5 * test_x ** 2

for degree in (1, 2, 10):
    train_features = torch.cat(
        [train_x ** i / math.factorial(i) for i in range(degree + 1)], dim=1
    )
    test_features = torch.cat(
        [test_x ** i / math.factorial(i) for i in range(degree + 1)], dim=1
    )
    weights = torch.linalg.lstsq(train_features, train_y).solution
    train_mse = ((train_features @ weights - train_y) ** 2).mean()
    test_mse = ((test_features @ weights - test_y) ** 2).mean()
    print(f"degree={degree}: train={train_mse:.5f}, test={test_mse:.5f}")
