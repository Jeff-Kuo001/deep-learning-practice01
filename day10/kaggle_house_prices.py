"""Day 10：Kaggle 房价预测训练基线。"""
import argparse
from pathlib import Path

from house_price_data import load_house_price_data
from house_price_model import build_model, train_model


def run_training(data_dir, epochs=200):
    train_x, _, labels, _ = load_house_price_data(data_dir)
    model = build_model(train_x.shape[1])
    score = train_model(model, train_x, labels, epochs=epochs)
    return model, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    _, score = run_training(args.data, args.epochs)
    print("training log RMSE:", score)


if __name__ == "__main__":
    main()
