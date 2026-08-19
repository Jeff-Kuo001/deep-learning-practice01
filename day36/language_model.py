"""语言模型和数据集：相邻序列的小批量采样。"""
import torch

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = 0
    num_tokens = ((len(corpus) - 1) // batch_size) * batch_size
    x_tokens = torch.tensor(corpus[offset:offset + num_tokens])
    y_tokens = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    x_tokens = x_tokens.reshape(batch_size, -1)
    y_tokens = y_tokens.reshape(batch_size, -1)
    num_batches = x_tokens.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        yield x_tokens[:, i:i+num_steps], y_tokens[:, i:i+num_steps]

corpus = list(range(30))
for x, y in seq_data_iter_sequential(corpus, batch_size=2, num_steps=5):
    print("X:\n", x)
    print("Y:\n", y)
    break
