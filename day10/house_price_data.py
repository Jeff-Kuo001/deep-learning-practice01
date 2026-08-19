"""房价数据读取与预处理。"""
from pathlib import Path

import pandas as pd
import torch


def load_house_price_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    test_ids = test["Id"].copy()

    labels = torch.log1p(
        torch.tensor(train.pop("SalePrice").to_numpy(), dtype=torch.float32)
    ).reshape(-1, 1)

    features = pd.concat((train, test), axis=0)
    features = features.drop(columns=["Id"], errors="ignore")
    numeric_columns = features.select_dtypes(include="number").columns
    numeric = features[numeric_columns]
    features[numeric_columns] = (
        (numeric - numeric.mean()) / numeric.std().replace(0, 1)
    ).fillna(0)
    features = pd.get_dummies(features, dummy_na=True, dtype=float)

    values = torch.tensor(features.to_numpy(), dtype=torch.float32)
    train_features = values[: len(train)]
    test_features = values[len(train) :]
    return train_features, test_features, labels, test_ids
