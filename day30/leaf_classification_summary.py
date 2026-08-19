"""树叶分类竞赛总结：验证准确率与概率融合。"""
import torch

def accuracy(logits, labels):
    return float((logits.argmax(dim=1) == labels).float().mean())

def probability_ensemble(logit_list, weights=None):
    probs = torch.stack([x.softmax(dim=1) for x in logit_list])
    if weights is None:
        weights = torch.ones(len(logit_list)) / len(logit_list)
    weights = torch.as_tensor(weights, dtype=probs.dtype)
    weights = weights / weights.sum()
    return (probs * weights[:, None, None]).sum(dim=0)

torch.manual_seed(0)
labels = torch.tensor([0, 1, 2, 1])
logits1 = torch.randn(4, 3)
logits2 = torch.randn(4, 3)
merged = probability_ensemble([logits1, logits2], [0.6, 0.4])
print("ensemble accuracy:", accuracy(merged, labels))
