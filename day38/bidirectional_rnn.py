"""双向循环神经网络。"""
import torch
from torch import nn

model = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    bidirectional=True,
    batch_first=True,
)
inputs = torch.randn(4, 7, 10)
outputs, (hidden, cell) = model(inputs)
print("outputs (2 directions):", tuple(outputs.shape))
print("hidden (layers*directions):", tuple(hidden.shape))
print("cell:", tuple(cell.shape))
