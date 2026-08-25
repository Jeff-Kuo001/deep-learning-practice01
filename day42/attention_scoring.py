import math

import torch
from torch import nn


def masked_softmax(scores, valid_lens):
    if valid_lens is None:
        return torch.softmax(scores, dim=-1)
    shape = scores.shape
    if valid_lens.dim() == 1:
        valid_lens = valid_lens.repeat_interleave(shape[1])
    scores = scores.reshape(-1, shape[-1])
    mask = torch.arange(shape[-1], device=scores.device)[None, :] < valid_lens[:, None]
    scores[~mask] = -1e6
    return torch.softmax(scores.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, hidden_size, dropout=0.0):
        super().__init__()
        self.w_k = nn.Linear(key_size, hidden_size, bias=False)
        self.w_q = nn.Linear(query_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        features = torch.tanh(self.w_q(queries).unsqueeze(2) + self.w_k(keys).unsqueeze(1))
        scores = self.w_v(features).squeeze(-1)
        weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(weights), values)


class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.size(-1))
        weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(weights), values)


if __name__ == "__main__":
    queries = torch.randn(2, 3, 8)
    keys = torch.randn(2, 5, 8)
    values = torch.randn(2, 5, 6)
    valid_lens = torch.tensor([3, 5])
    print(AdditiveAttention(8, 8, 16)(queries, keys, values, valid_lens).shape)
    print(DotProductAttention()(queries, keys, values, valid_lens).shape)
