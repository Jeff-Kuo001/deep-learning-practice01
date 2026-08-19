"""房价预测模型与训练函数。"""
import torch
from torch import nn


def build_model(input_size):
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def log_rmse(model, features, labels):
    predictions = torch.clamp(model(features), min=0)
    return torch.sqrt(nn.functional.mse_loss(predictions, labels))


def train_model(model, features, labels, epochs=200, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        loss = nn.functional.mse_loss(model(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return log_rmse(model, features, labels).item()
