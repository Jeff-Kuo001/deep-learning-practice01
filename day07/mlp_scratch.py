"""Day 7：从零开始实现多层感知机。"""
import torch


torch.manual_seed(0)
features = torch.randn(256, 20)
labels = (features[:, :3].sum(dim=1) > 0).long()
w1 = (torch.randn(20, 64) * 0.01).requires_grad_()
b1 = torch.zeros(64, requires_grad=True)
w2 = (torch.randn(64, 2) * 0.01).requires_grad_()
b2 = torch.zeros(2, requires_grad=True)
params = [w1, b1, w2, b2]

for _ in range(100):
    hidden = torch.relu(features @ w1 + b1)
    logits = hidden @ w2 + b2
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    with torch.no_grad():
        for parameter in params:
            parameter -= 0.2 * parameter.grad
            parameter.grad.zero_()

accuracy = ((torch.relu(features @ w1 + b1) @ w2 + b2).argmax(1) == labels).float().mean()
print("loss:", loss.item(), "accuracy:", accuracy.item())
