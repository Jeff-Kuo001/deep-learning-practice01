import collections

import torch
from torch.utils.data import DataLoader, Dataset


class Vocab:
    def __init__(self, sentences):
        counter = collections.Counter(token for sentence in sentences for token in sentence)
        self.idx_to_token = ["<pad>", "<cls>", "<sep>", "<unk>"] + list(counter)
        self.token_to_idx = {token: i for i, token in enumerate(self.idx_to_token)}

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx["<unk>"])


def encode_pair(premise, hypothesis, vocab, max_len):
    tokens = ["<cls>"] + premise + ["<sep>"] + hypothesis + ["<sep>"]
    segments = [0] * (len(premise) + 2) + [1] * (len(hypothesis) + 1)
    valid_len = min(len(tokens), max_len)
    tokens = tokens[:max_len] + ["<pad>"] * max(0, max_len - len(tokens))
    segments = segments[:max_len] + [0] * max(0, max_len - len(segments))
    return [vocab[token] for token in tokens], segments, valid_len


class NLIDataset(Dataset):
    def __init__(self, examples, vocab, max_len=12):
        self.examples = [
            (*encode_pair(premise, hypothesis, vocab, max_len), label)
            for premise, hypothesis, label in examples
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        tokens, segments, valid_len, label = self.examples[index]
        return (
            torch.tensor(tokens), torch.tensor(segments),
            torch.tensor(valid_len), torch.tensor(label)
        )


if __name__ == "__main__":
    examples = [
        (["a", "person", "runs"], ["someone", "moves"], 0),
        (["a", "dog", "sleeps"], ["no", "animal", "rests"], 1),
        (["two", "people", "talk"], ["they", "are", "friends"], 2),
    ]
    vocab = Vocab([part for example in examples for part in example[:2]])
    batch = next(iter(DataLoader(NLIDataset(examples, vocab), batch_size=3)))
    tokens, segments, valid_lens, labels = batch
    print(tokens.shape, segments.shape)
    print(valid_lens, labels)
