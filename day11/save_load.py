"""Day 11：保存和加载张量与模型参数。"""
from pathlib import Path

import torch
from torch import nn


path = Path("checkpoints/day11_model.pt")
path.parent.mkdir(parents=True, exist_ok=True)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
sample = torch.randn(1, 4)
expected = model(sample)
torch.save({"model": model.state_dict(), "sample": sample}, path)

checkpoint = torch.load(path, weights_only=True)
restored = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
restored.load_state_dict(checkpoint["model"])
print("outputs match:", torch.allclose(expected, restored(checkpoint["sample"])))
