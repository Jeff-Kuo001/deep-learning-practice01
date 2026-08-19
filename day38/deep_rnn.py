"""深层循环神经网络。"""
import torch
from torch import nn

num_layers = 3
model = nn.GRU(
    input_size=12,
    hidden_size=24,
    num_layers=num_layers,
    dropout=0.2,
    batch_first=True,
)
inputs = torch.randn(5, 8, 12)
outputs, state = model(inputs)
print("outputs:", tuple(outputs.shape))
print("one hidden state per layer:", tuple(state.shape))
