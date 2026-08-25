import collections
import re

import torch


def preprocess(text):
    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()
    return re.sub(r"([,.!?])", r" \1", text)


def tokenize(lines):
    source, target = [], []
    for line in lines:
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split())
            target.append(parts[1].split())
    return source, target


class Vocab:
    def __init__(self, tokens, min_freq=1, reserved_tokens=None):
        counter = collections.Counter(token for line in tokens for token in line)
        self.idx_to_token = ["<unk>"] + (reserved_tokens or [])
        self.idx_to_token += [
            token for token, freq in counter.items()
            if freq >= min_freq and token not in self.idx_to_token
        ]
        self.token_to_idx = {token: i for i, token in enumerate(self.idx_to_token)}

    def __getitem__(self, tokens):
        if isinstance(tokens, str):
            return self.token_to_idx.get(tokens, 0)
        return [self[token] for token in tokens]

    def __len__(self):
        return len(self.idx_to_token)


def truncate_pad(tokens, num_steps, padding_token):
    if len(tokens) > num_steps:
        return tokens[:num_steps]
    return tokens + [padding_token] * (num_steps - len(tokens))


def build_array(lines, vocab, num_steps):
    lines = [vocab[line] + [vocab["<eos>"]] for line in lines]
    array = torch.tensor([
        truncate_pad(line, num_steps, vocab["<pad>"]) for line in lines
    ])
    valid_len = (array != vocab["<pad>"]).sum(dim=1)
    return array, valid_len


if __name__ == "__main__":
    raw_text = "go .\tva !\ni lost .\tj'ai perdu .\nhe is calm .\til est calme ."
    source, target = tokenize(preprocess(raw_text).splitlines())
    reserved = ["<pad>", "<bos>", "<eos>"]
    source_vocab = Vocab(source, reserved_tokens=reserved)
    target_vocab = Vocab(target, reserved_tokens=reserved)
    source_array, source_valid_len = build_array(source, source_vocab, 8)
    target_array, target_valid_len = build_array(target, target_vocab, 8)
    print(source_array)
    print(source_valid_len)
    print(target_array.shape, target_valid_len)
