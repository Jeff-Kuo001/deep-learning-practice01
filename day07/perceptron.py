"""Day 7：感知机二分类算法。"""
import torch


features = torch.tensor([[2., 1.], [1., 2.], [-1., -2.], [-2., -1.]])
labels = torch.tensor([1., 1., -1., -1.])
weights = torch.zeros(2)
bias = torch.tensor(0.)

for epoch in range(10):
    mistakes = 0
    for x, y in zip(features, labels):
        if y * (x @ weights + bias) <= 0:
            weights += y * x
            bias += y
            mistakes += 1
    if mistakes == 0:
        break

predictions = torch.sign(features @ weights + bias)
print("weights:", weights, "bias:", bias.item())
print("predictions:", predictions)
