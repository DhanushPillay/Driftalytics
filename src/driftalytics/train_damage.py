"""Training pipeline for the damage predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .core import CorruptionVector
from .damage import DamagePredictor


def make_synthetic(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    drift = rng.uniform(0, 2, n)
    missing = rng.uniform(0, 0.3, n)
    cardinality = rng.uniform(0, 1, n)
    schema = rng.uniform(0, 0.5, n)
    volume = rng.uniform(0, 0.5, n)
    features = np.column_stack([drift, missing, cardinality, schema, volume])
    target = (
        0.30 * drift
        + 0.20 * np.sqrt(np.clip(missing * 3, 0, 1))
        + 0.15 * cardinality
        + 0.20 * schema
        + 0.10 * volume
        + 0.10 * drift * schema
    )
    target = np.clip(target + rng.normal(0, 0.03, n), 0, 1)
    return features, target


def load_training_data(path: str):
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)
    required = list(CorruptionVector.feature_names) + ["damage"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"training data is missing columns: {', '.join(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("training data contains missing values")
    features = frame[list(CorruptionVector.feature_names)].to_numpy(dtype=float)
    return features, frame["damage"].to_numpy(dtype=float)


def train_and_save(
    out_path: str = "models/damage_predictor.json",
    n: int = 2000,
    data_path: str | None = None,
    seed: int = 0,
    test_size: float = 0.2,
) -> DamagePredictor:
    features, target = load_training_data(data_path) if data_path else make_synthetic(n, seed)
    if not 0.0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5")
    if len(features) < 3:
        raise ValueError("need at least three training rows")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    n_test = max(1, int(test_size * len(features)))
    n_validation = max(1, int(test_size * len(features)))
    n_train = len(features) - n_test - n_validation
    if n_train < 1:
        raise ValueError("test_size leaves no training rows")
    train_idx = order[:n_train]
    validation_idx = order[n_train : n_train + n_validation]
    test_idx = order[n_train + n_validation :]

    model = DamagePredictor()
    model.fit(features[train_idx], target[train_idx])
    model.validation_metrics_ = model.evaluate(features[validation_idx], target[validation_idx])
    model.test_metrics_ = model.evaluate(features[test_idx], target[test_idx])
    for label, metrics in (
        ("validation", model.validation_metrics_),
        ("test", model.test_metrics_),
    ):
        print(
            f"{label}: rows={len(features)} MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
        )
    model.save(out_path)
    print(
        f"saved to {out_path} (train={len(train_idx)}, "
        f"validation={len(validation_idx)}, test={len(test_idx)})"
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/damage_predictor.json")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--data", help="CSV or parquet with CorruptionVector columns and damage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    train_and_save(args.out, args.n, args.data, args.seed, args.test_size)


if __name__ == "__main__":
    main()
