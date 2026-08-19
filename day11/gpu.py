"""Day 11：张量与模型在 GPU 上的使用。"""
import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.ones(4, device=device)
model = nn.Linear(4, 2).to(device)
output = model(tensor)
print("selected device:", device)
print("tensor device:", tensor.device)
print("model output device:", output.device)
print("CUDA device count:", torch.cuda.device_count())
