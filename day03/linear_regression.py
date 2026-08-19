"""Day 3：线性回归模型、均方损失与预测。"""
import torch


true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
labels = features @ true_w + true_b


def linear_model(x, w, b):
    return x @ w + b


def squared_loss(prediction, target):
    return (prediction - target) ** 2 / 2


predictions = linear_model(features, true_w, true_b)
print("predictions:", predictions)
print("labels:", labels)
print("loss:", squared_loss(predictions, labels))
