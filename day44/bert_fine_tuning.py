import torch
from torch import nn

from bert import BERTModel


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size, num_classes):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, tokens, segments, valid_lens):
        encoded, _, _ = self.bert(tokens, segments, valid_lens)
        return self.classifier(encoded[:, 0])


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BERTClassifier(BERTModel(60, hidden_size=32), 32, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, 60, (4, 12))
    segments = torch.tensor([[0] * 6 + [1] * 6] * 4)
    labels = torch.tensor([0, 1, 0, 1])
    logits = model(tokens, segments, torch.tensor([12, 11, 10, 9]))
    loss = nn.functional.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(logits.shape, round(loss.item(), 4))
