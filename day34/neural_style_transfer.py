"""样式迁移：内容损失、样式损失和 Gram 矩阵。"""
import torch
from torch.nn import functional as F

def gram_matrix(x):
    n, c, h, w = x.shape
    features = x.reshape(n, c, h * w)
    return features @ features.transpose(1, 2) / (c * h * w)

def content_loss(generated, content):
    return F.mse_loss(generated, content.detach())

def style_loss(generated, style):
    return F.mse_loss(gram_matrix(generated), gram_matrix(style).detach())

generated = torch.randn(1, 8, 16, 16, requires_grad=True)
content = torch.randn_like(generated)
style = torch.randn_like(generated)
loss = content_loss(generated, content) + 10 * style_loss(generated, style)
loss.backward()
print("total loss:", float(loss))
