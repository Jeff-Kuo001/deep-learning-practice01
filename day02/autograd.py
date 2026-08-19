"""Day 2：自动求导、梯度清零与分离计算图。"""
import torch


x = torch.arange(4.0, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print("gradient of sum(x^2):", x.grad)

x.grad.zero_()
z = (x * x.detach()).sum()
z.backward()
print("gradient after detach:", x.grad)
print("detached requires_grad:", x.detach().requires_grad)
