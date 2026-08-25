import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout=0.0, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )
        encoding = torch.zeros(max_len, hidden_size)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.encoding[:, :x.size(1)])


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, x, padding_mask=None):
        return self.attention(x, x, x, key_padding_mask=padding_mask)


if __name__ == "__main__":
    x = torch.randn(2, 6, 16)
    padding_mask = torch.tensor([
        [False, False, False, False, True, True],
        [False, False, False, False, False, False],
    ])
    x = PositionalEncoding(16)(x)
    output, weights = SelfAttention(16, 4)(x, padding_mask)
    print(output.shape, weights.shape)
