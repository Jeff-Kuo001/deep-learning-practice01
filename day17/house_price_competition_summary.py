"""预测房价竞赛总结：对数均方根误差与模型融合。"""
import torch

def log_rmse(pred, label):
    pred = torch.clamp(pred, min=1.0)
    return torch.sqrt(torch.mean((torch.log(pred) - torch.log(label)) ** 2))

def blend_predictions(predictions, weights=None):
    stacked = torch.stack(predictions)
    if weights is None:
        weights = torch.ones(len(predictions)) / len(predictions)
    weights = torch.as_tensor(weights, dtype=stacked.dtype)
    weights = weights / weights.sum()
    return (stacked * weights[:, None]).sum(dim=0)

labels = torch.tensor([120000.0, 180000.0, 250000.0])
p1 = torch.tensor([125000.0, 175000.0, 240000.0])
p2 = torch.tensor([118000.0, 185000.0, 255000.0])
merged = blend_predictions([p1, p2], [0.5, 0.5])
print("log RMSE:", float(log_rmse(merged, labels)))
