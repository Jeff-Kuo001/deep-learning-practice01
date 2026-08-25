import torch
from torch import nn


def masked_softmax(scores, valid_lens):
    shape = scores.shape
    valid_lens = valid_lens.repeat_interleave(shape[1])
    scores = scores.reshape(-1, shape[-1])
    mask = torch.arange(shape[-1], device=scores.device)[None, :] < valid_lens[:, None]
    scores[~mask] = -1e6
    return torch.softmax(scores.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, valid_lens):
        features = torch.tanh(self.w_q(query).unsqueeze(2) + self.w_k(keys).unsqueeze(1))
        weights = masked_softmax(self.w_v(features).squeeze(-1), valid_lens)
        return torch.bmm(weights, keys), weights


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)

    def forward(self, x):
        output, state = self.rnn(self.embedding(x).permute(1, 0, 2))
        return output.permute(1, 0, 2), state


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = AdditiveAttention(hidden_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state, encoder_output, valid_lens):
        embeddings = self.embedding(x).permute(1, 0, 2)
        outputs, all_weights = [], []
        for embedding in embeddings:
            query = state[-1].unsqueeze(1)
            context, weights = self.attention(query, encoder_output, valid_lens)
            rnn_input = torch.cat((context.squeeze(1), embedding), dim=1).unsqueeze(0)
            output, state = self.rnn(rnn_input, state)
            outputs.append(self.output(output.squeeze(0)).unsqueeze(1))
            all_weights.append(weights)
        return torch.cat(outputs, dim=1), state, torch.cat(all_weights, dim=1)


if __name__ == "__main__":
    encoder = Seq2SeqEncoder(30, 12, 16)
    decoder = AttentionDecoder(35, 12, 16)
    source = torch.randint(0, 30, (4, 8))
    target = torch.randint(0, 35, (4, 7))
    encoder_output, state = encoder(source)
    output, _, attention_weights = decoder(
        target, state, encoder_output, torch.tensor([8, 7, 6, 5])
    )
    print(output.shape, attention_weights.shape)
