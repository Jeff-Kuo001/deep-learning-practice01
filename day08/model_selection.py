"""Day 8：训练集、验证集与超参数选择。"""
import torch


torch.manual_seed(0)
x = torch.linspace(-2, 2, 100).reshape(-1, 1)
y = 1 + 2 * x - 0.5 * x ** 2 + torch.randn_like(x) * 0.1
train_x, valid_x = x[:70], x[70:]
train_y, valid_y = y[:70], y[70:]


def fit_and_score(degree):
    train_features = torch.cat([train_x ** i for i in range(degree + 1)], dim=1)
    valid_features = torch.cat([valid_x ** i for i in range(degree + 1)], dim=1)
    weights = torch.linalg.lstsq(train_features, train_y).solution
    return ((valid_features @ weights - valid_y) ** 2).mean().item()


scores = {degree: fit_and_score(degree) for degree in (1, 2, 3, 5)}
print("validation MSE:", scores)
print("selected degree:", min(scores, key=scores.get))
