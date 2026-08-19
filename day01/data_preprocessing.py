"""Day 1：缺失值处理与类别特征编码。"""
import pandas as pd
import torch


data = pd.DataFrame({
    "rooms": [2.0, None, 4.0, 3.0],
    "area": [60.0, 85.0, None, 72.0],
    "district": ["east", "west", "east", None],
})
numeric = data.select_dtypes(include="number")
data[numeric.columns] = numeric.fillna(numeric.mean())
features = pd.get_dummies(data, dummy_na=True, dtype=float)
tensor = torch.tensor(features.values, dtype=torch.float32)
print(features)
print("tensor shape:", tensor.shape)
