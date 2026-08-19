"""Softmax 回归的从零开始实现。"""
import torch

def softmax(x):
    exp = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return exp / exp.sum(dim=1, keepdim=True)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y].clamp_min(1e-12)).mean()

torch.manual_seed(0)
x = torch.randn(200, 4)
y = torch.randint(0, 3, (200,))
w = (torch.randn(4, 3) * 0.01).requires_grad_()
b = torch.zeros(3, requires_grad=True)

for epoch in range(30):
    loss = cross_entropy(softmax(x @ w + b), y)
    loss.backward()
    with torch.no_grad():
        w -= 0.2 * w.grad
        b -= 0.2 * b.grad
        w.grad.zero_()
        b.grad.zero_()

print("loss:", float(loss))
