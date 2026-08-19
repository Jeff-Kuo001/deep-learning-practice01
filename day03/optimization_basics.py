"""Day 3：用随机梯度下降最小化二次函数。"""
import torch


x = torch.tensor([8.0], requires_grad=True)
learning_rate = 0.1
for step in range(30):
    objective = (x - 3) ** 2
    objective.backward()
    with torch.no_grad():
        x -= learning_rate * x.grad
        x.grad.zero_()
print("minimum point:", x.item())
print("objective:", ((x - 3) ** 2).item())
