"""文本预处理：分词、词表和数字化。"""
import collections
import re

text = """Time flies like an arrow.
Fruit flies like a banana."""
lines = [re.sub(r"[^A-Za-z]+", " ", line).strip().lower() for line in text.splitlines()]
tokens = [line.split() for line in lines]
counter = collections.Counter(token for line in tokens for token in line)
idx_to_token = ["<unk>"] + sorted(counter, key=lambda token: (-counter[token], token))
token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
corpus = [[token_to_idx.get(token, 0) for token in line] for line in tokens]
print("tokens:", tokens)
print("vocabulary:", token_to_idx)
print("corpus:", corpus)
