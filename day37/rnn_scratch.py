"""循环神经网络的从零开始实现。"""
import torch
from torch.nn import functional as F

def get_params(vocab_size, hidden_size):
    def normal(shape):
        return torch.randn(shape) * 0.01
    params = [
        normal((vocab_size, hidden_size)),
        normal((hidden_size, hidden_size)),
        torch.zeros(hidden_size),
        normal((hidden_size, vocab_size)),
        torch.zeros(vocab_size),
    ]
    for param in params:
        param.requires_grad_()
    return params

def init_state(batch_size, hidden_size):
    return (torch.zeros(batch_size, hidden_size),)

def rnn(inputs, state, params):
    w_xh, w_hh, b_h, w_hq, b_q = params
    h, = state
    outputs = []
    for x in inputs:
        h = torch.tanh(x @ w_xh + h @ w_hh + b_h)
        outputs.append(h @ w_hq + b_q)
    return torch.cat(outputs, dim=0), (h,)

vocab_size, hidden_size = 10, 16
indices = torch.randint(0, vocab_size, (5, 2))
inputs = F.one_hot(indices.T, vocab_size).float().permute(1, 0, 2)
outputs, state = rnn(inputs, init_state(2, hidden_size), get_params(vocab_size, hidden_size))
print("outputs:", tuple(outputs.shape), "state:", tuple(state[0].shape))
