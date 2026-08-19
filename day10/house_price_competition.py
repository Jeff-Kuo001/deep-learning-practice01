"""Day 10：训练房价模型并生成 submission.csv。"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from house_price_data import load_house_price_data
from house_price_model import build_model, train_model


def make_submission(data_dir, output_path, epochs=300):
    train_x, test_x, labels, test_ids = load_house_price_data(data_dir)
    model = build_model(train_x.shape[1])
    score = train_model(model, train_x, labels, epochs=epochs)

    with torch.no_grad():
        predictions = torch.expm1(model(test_x)).clamp(min=0).squeeze(1).numpy()
    pd.DataFrame({"Id": test_ids, "SalePrice": predictions}).to_csv(
        output_path, index=False
    )
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    score = make_submission(args.data, args.output, args.epochs)
    print("training log RMSE:", score)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
