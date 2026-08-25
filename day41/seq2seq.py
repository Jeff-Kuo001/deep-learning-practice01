import torch
from torch import nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        return self.rnn(x)


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state):
        x = self.embedding(x).permute(1, 0, 2)
        context = state[-1].repeat(x.shape[0], 1, 1)
        output, state = self.rnn(torch.cat((x, context), dim=2), state)
        return self.output(output).permute(1, 0, 2), state


def sequence_mask(x, valid_len, value=0):
    max_len = x.size(1)
    mask = torch.arange(max_len, device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, prediction, label, valid_len):
        weights = torch.ones_like(label, dtype=torch.float32)
        weights = sequence_mask(weights, valid_len)
        loss = nn.functional.cross_entropy(
            prediction.permute(0, 2, 1), label, reduction="none"
        )
        return (loss * weights).sum(dim=1) / valid_len


if __name__ == "__main__":
    encoder = Seq2SeqEncoder(30, 12, 16)
    decoder = Seq2SeqDecoder(35, 12, 16)
    source = torch.randint(0, 30, (4, 8))
    target_input = torch.randint(0, 35, (4, 7))
    target_label = torch.randint(0, 35, (4, 7))
    _, state = encoder(source)
    prediction, _ = decoder(target_input, state)
    loss = MaskedSoftmaxCELoss()(prediction, target_label, torch.tensor([7, 6, 5, 4]))
    loss.mean().backward()
    print(prediction.shape, loss)
