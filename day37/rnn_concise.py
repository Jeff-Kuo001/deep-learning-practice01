"""循环神经网络的简洁实现。"""
import torch
from torch import nn

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, state=None):
        y, state = self.rnn(self.embedding(tokens), state)
        return self.output(y), state

model = RNNLanguageModel(vocab_size=20, hidden_size=32)
tokens = torch.randint(0, 20, (4, 6))
logits, state = model(tokens)
print("logits:", tuple(logits.shape), "state:", tuple(state.shape))
