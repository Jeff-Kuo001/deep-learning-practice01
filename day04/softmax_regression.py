"""Day 4：Softmax 回归的概率计算。"""
import torch


logits = torch.tensor([[1.2, 0.3, -0.4], [0.1, 1.5, 0.2]])
probabilities = torch.softmax(logits, dim=1)
print("probabilities:\n", probabilities)
print("row sums:", probabilities.sum(dim=1))
print("predicted classes:", probabilities.argmax(dim=1))
