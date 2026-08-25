import torch


def masked_softmax(scores, valid_lens):
    if valid_lens is None:
        return torch.softmax(scores, dim=-1)
    shape = scores.shape
    if valid_lens.dim() == 1:
        valid_lens = valid_lens.repeat_interleave(shape[1])
    scores = scores.reshape(-1, shape[-1])
    mask = torch.arange(shape[-1], device=scores.device)[None, :] < valid_lens[:, None]
    scores[~mask] = -1e6
    return torch.softmax(scores.reshape(shape), dim=-1)


def attention_pool(query, keys, values, valid_lens=None):
    scores = torch.bmm(query, keys.transpose(1, 2)) / query.size(-1) ** 0.5
    weights = masked_softmax(scores, valid_lens)
    return torch.bmm(weights, values), weights


if __name__ == "__main__":
    torch.manual_seed(0)
    query = torch.randn(2, 1, 4)
    keys = torch.randn(2, 5, 4)
    values = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    context, weights = attention_pool(query, keys, values, torch.tensor([3, 5]))
    print(context.shape)
    print(weights)
