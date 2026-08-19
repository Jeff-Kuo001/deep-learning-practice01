"""Day 3：从零开始实现线性回归。"""
import torch


torch.manual_seed(0)
features = torch.randn(1000, 2)
labels = features @ torch.tensor([2.0, -3.4]) + 4.2
labels += torch.randn(1000) * 0.01
weights = torch.randn(2, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

for epoch in range(10):
    prediction = features @ weights + bias
    loss = ((prediction - labels) ** 2).mean() / 2
    loss.backward()
    with torch.no_grad():
        weights -= 0.1 * weights.grad
        bias -= 0.1 * bias.grad
        weights.grad.zero_()
        bias.grad.zero_()
    print(f"epoch {epoch + 1}: loss={loss.item():.6f}")

print("weights:", weights.detach(), "bias:", bias.item())
