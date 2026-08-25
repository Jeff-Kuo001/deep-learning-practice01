import torch
from torch import nn

from bert import BERTModel


class BERTForNLI(nn.Module):
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.bert = BERTModel(vocab_size, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, tokens, segments, valid_lens):
        encoded, _, _ = self.bert(tokens, segments, valid_lens)
        return self.output(encoded[:, 0])


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BERTForNLI(vocab_size=80)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, 80, (3, 14))
    segments = torch.tensor([[0] * 7 + [1] * 7] * 3)
    valid_lens = torch.tensor([14, 12, 10])
    labels = torch.tensor([0, 1, 2])
    logits = model(tokens, segments, valid_lens)
    loss = nn.functional.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(logits.shape, round(loss.item(), 4))
