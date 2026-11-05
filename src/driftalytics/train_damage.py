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
    frame = frame[required].dropna()
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
    if len(features) < 2:
        raise ValueError("need at least two training rows")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    split = int((1.0 - test_size) * len(features))
    split = max(1, min(split, len(features) - 1))
    train_idx, test_idx = order[:split], order[split:]

    model = DamagePredictor()
    model.fit(features[train_idx], target[train_idx])
    metrics = model.evaluate(features[test_idx], target[test_idx])
    print(
        f"trained damage predictor: rows={len(features)} train={len(train_idx)} "
        f"test={len(test_idx)} MAE={metrics['mae']:.4f} "
        f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
    )
    model.save(out_path)
    print(f"saved to {out_path}")
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
