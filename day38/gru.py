"""门控循环单元（GRU）。"""
import torch
from torch import nn

class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size=30, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, state=None):
        y, state = self.gru(self.embedding(tokens), state)
        return self.output(y), state

tokens = torch.randint(0, 30, (3, 7))
logits, state = GRULanguageModel()(tokens)
print("logits:", tuple(logits.shape), "state:", tuple(state.shape))
