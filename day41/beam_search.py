import math

import torch


def beam_search(step, bos, eos, beam_size=3, max_steps=8, alpha=0.75):
    beams = [([bos], 0.0)]
    finished = []
    for _ in range(max_steps):
        candidates = []
        for tokens, score in beams:
            log_probs = step(tokens)
            values, indices = torch.topk(log_probs, beam_size)
            for value, index in zip(values.tolist(), indices.tolist()):
                new_tokens = tokens + [index]
                item = (new_tokens, score + value)
                if index == eos:
                    finished.append(item)
                else:
                    candidates.append(item)
        if not candidates:
            break
        beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]

    choices = finished or beams
    length_penalty = lambda item: item[1] / math.pow(len(item[0]), alpha)
    return max(choices, key=length_penalty)


def next_token_log_probs(tokens):
    transition = torch.tensor([
        [0.01, 0.44, 0.35, 0.10, 0.10],
        [0.01, 0.05, 0.49, 0.35, 0.10],
        [0.01, 0.08, 0.06, 0.40, 0.45],
        [0.01, 0.05, 0.05, 0.09, 0.80],
        [0.01, 0.01, 0.01, 0.01, 0.96],
    ])
    return torch.log(transition[tokens[-1]])


if __name__ == "__main__":
    sequence, score = beam_search(next_token_log_probs, bos=0, eos=4)
    print("token sequence:", sequence)
    print("log probability:", round(score, 4))
