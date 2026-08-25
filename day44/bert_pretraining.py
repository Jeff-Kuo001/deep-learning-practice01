import torch
from torch import nn

from bert import BERTModel


def pretraining_loss(mlm_logits, mlm_labels, mlm_weights, nsp_logits, nsp_labels):
    vocab_size = mlm_logits.size(-1)
    mlm = nn.functional.cross_entropy(
        mlm_logits.reshape(-1, vocab_size), mlm_labels.reshape(-1), reduction="none"
    )
    mlm = (mlm * mlm_weights.reshape(-1)).sum() / mlm_weights.sum()
    nsp = nn.functional.cross_entropy(nsp_logits, nsp_labels)
    return mlm, nsp


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BERTModel(vocab_size=40, hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, 40, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * 2)
    positions = torch.tensor([[1, 5], [2, 6]])
    mlm_labels = torch.randint(0, 40, (2, 2))
    mlm_weights = torch.ones(2, 2)
    nsp_labels = torch.tensor([1, 0])
    _, mlm_logits, nsp_logits = model(
        tokens, segments, torch.tensor([8, 7]), positions
    )
    mlm_loss, nsp_loss = pretraining_loss(
        mlm_logits, mlm_labels, mlm_weights, nsp_logits, nsp_labels
    )
    optimizer.zero_grad()
    (mlm_loss + nsp_loss).backward()
    optimizer.step()
    print("MLM loss:", round(mlm_loss.item(), 4))
    print("NSP loss:", round(nsp_loss.item(), 4))
