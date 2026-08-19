"""循环神经网络：隐藏状态随时间传递。"""
import torch
from torch import nn

rnn = nn.RNN(input_size=4, hidden_size=8, num_layers=1)
inputs = torch.randn(6, 3, 4)
state = torch.zeros(1, 3, 8)
outputs, final_state = rnn(inputs, state)
print("all time steps:", tuple(outputs.shape))
print("final hidden state:", tuple(final_state.shape))
print("last output equals final state:", torch.allclose(outputs[-1], final_state[0]))
