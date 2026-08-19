"""Day 2：矩阵计算中的二次型与梯度。"""
import torch


torch.manual_seed(0)
matrix = torch.randn(4, 3)
x = torch.randn(3, requires_grad=True)
target = torch.randn(4)
loss = ((matrix @ x - target) ** 2).sum()
loss.backward()
analytic = 2 * matrix.T @ (matrix @ x.detach() - target)
print("autograd gradient:", x.grad)
print("analytic gradient:", analytic)
print("same:", torch.allclose(x.grad, analytic, atol=1e-6))
