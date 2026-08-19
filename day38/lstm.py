"""长短期记忆网络（LSTM）。"""
import torch
from torch import nn

lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
inputs = torch.randn(4, 10, 16)
outputs, (hidden, cell) = lstm(inputs)
print("outputs:", tuple(outputs.shape))
print("hidden state:", tuple(hidden.shape))
print("cell state:", tuple(cell.shape))
