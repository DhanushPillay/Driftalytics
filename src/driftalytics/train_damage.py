"""Training pipeline for the damage predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .core import CorruptionVector
from .damage import DamagePredictor


def load_training_data(path: str):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"training data not found: {source}")
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    required = list(CorruptionVector.feature_names) + ["damage"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"training data is missing columns: {', '.join(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("training data contains missing values")
    features = frame[list(CorruptionVector.feature_names)].to_numpy(dtype=float)
    return features, frame["damage"].to_numpy(dtype=float)


def train_and_save(out_path: str, data_path: str, test_size: float = 0.2) -> DamagePredictor:
    if not 0.0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5")
    features, target = load_training_data(data_path)
    if len(features) < 3:
        raise ValueError("need at least three training rows")
    n_test = max(1, int(test_size * len(features)))
    n_validation = max(1, int(test_size * len(features)))
    n_train = len(features) - n_test - n_validation
    if n_train < 1:
        raise ValueError("test_size leaves no training rows")

    train = slice(0, n_train)
    validation = slice(n_train, n_train + n_validation)
    test = slice(n_train + n_validation, len(features))

    model = DamagePredictor()
    model.fit(features[train], target[train])
    model.validation_metrics_ = model.evaluate(features[validation], target[validation])
    model.test_metrics_ = model.evaluate(features[test], target[test])

    print(f"train rows: {n_train}, validation rows: {n_validation}, test rows: {n_test}")
    for name, metrics in (
        ("train", model.training_metrics_),
        ("validation", model.validation_metrics_),
        ("test", model.test_metrics_),
    ):
        print(f"{name}: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}")
    model.save(out_path)
    print(f"saved damage predictor to {out_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/damage_predictor.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    train_and_save(args.out, args.data, args.test_size)


if __name__ == "__main__":
    main()
