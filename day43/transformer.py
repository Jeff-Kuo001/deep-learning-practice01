import torch
from torch import nn


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.ReLU(),
            nn.Linear(ffn_size, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, ffn_size, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.addnorm1 = AddNorm(hidden_size, dropout)
        self.ffn = PositionWiseFFN(hidden_size, ffn_size)
        self.addnorm2 = AddNorm(hidden_size, dropout)

    def forward(self, x, padding_mask=None):
        attention, _ = self.attention(x, x, x, key_padding_mask=padding_mask)
        x = self.addnorm1(x, attention)
        return self.addnorm2(x, self.ffn(x))


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, ffn_size, num_heads, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_size, ffn_size, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 8, 32)
    padding_mask = torch.tensor([
        [False, False, False, False, False, True, True, True],
        [False, False, False, False, False, False, False, False],
    ])
    model = TransformerEncoder(32, 64, 4, 2)
    print(model(x, padding_mask).shape)
