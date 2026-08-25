import collections
import random

import torch


class Vocab:
    def __init__(self, paragraphs):
        counter = collections.Counter(
            token for paragraph in paragraphs for sentence in paragraph for token in sentence
        )
        self.idx_to_token = ["<pad>", "<mask>", "<cls>", "<sep>", "<unk>"]
        self.idx_to_token += [token for token in counter if token not in self.idx_to_token]
        self.token_to_idx = {token: i for i, token in enumerate(self.idx_to_token)}

    def __getitem__(self, tokens):
        if isinstance(tokens, str):
            return self.token_to_idx.get(tokens, self.token_to_idx["<unk>"])
        return [self[token] for token in tokens]

    def __len__(self):
        return len(self.idx_to_token)


def tokens_and_segments(sentence_a, sentence_b):
    tokens = ["<cls>"] + sentence_a + ["<sep>"] + sentence_b + ["<sep>"]
    segments = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
    return tokens, segments


def next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        return sentence, next_sentence, True
    random_paragraph = random.choice(paragraphs)
    return sentence, random.choice(random_paragraph), False


def replace_mlm_tokens(tokens, vocab):
    candidates = [i for i, token in enumerate(tokens) if token not in ("<cls>", "<sep>")]
    random.shuffle(candidates)
    prediction_positions = sorted(candidates[:max(1, round(len(tokens) * 0.15))])
    labels = [tokens[position] for position in prediction_positions]
    output = tokens.copy()
    for position in prediction_positions:
        draw = random.random()
        if draw < 0.8:
            output[position] = "<mask>"
        elif draw < 0.9:
            output[position] = random.choice(vocab.idx_to_token)
    return output, prediction_positions, labels


def build_example(paragraphs, vocab, max_len=16):
    sentence_a, sentence_b, is_next = next_sentence(
        paragraphs[0][0], paragraphs[0][1], paragraphs
    )
    tokens, segments = tokens_and_segments(sentence_a, sentence_b)
    tokens, positions, labels = replace_mlm_tokens(tokens[:max_len], vocab)
    valid_len = len(tokens)
    padding = max_len - valid_len
    token_ids = vocab[tokens] + [vocab["<pad>"]] * padding
    segments += [0] * padding
    return {
        "tokens": torch.tensor(token_ids),
        "segments": torch.tensor(segments[:max_len]),
        "valid_len": torch.tensor(valid_len),
        "pred_positions": torch.tensor(positions),
        "mlm_labels": torch.tensor(vocab[labels]),
        "nsp_label": torch.tensor(int(is_next)),
    }


if __name__ == "__main__":
    random.seed(0)
    paragraphs = [
        [["a", "crane", "is", "flying"], ["the", "sky", "is", "blue"]],
        [["deep", "learning", "uses", "data"], ["models", "learn", "patterns"]],
    ]
    vocab = Vocab(paragraphs)
    example = build_example(paragraphs, vocab)
    for name, value in example.items():
        print(name, value)
