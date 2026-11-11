"""Detection and damage-prediction benchmark on NYC TLC data.

The script trains a small linear fare model on clean trips, injects controlled
feature corruptions into held-out trips, and measures:

- clean vs. corrupted detection rates per monitor
- false-positive rate on clean windows
- damage-predictor MAE/R2 on held-out corruption scenarios

Expects data/nyc_tlc/yellow_tripdata_2024-01.parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from driftalytics import DamagePredictor, SlidingWindowRunner, compute_corruption_vector
from driftalytics.core import CorruptionVector

DATA = Path("data/nyc_tlc/yellow_tripdata_2024-01.parquet")
COLUMNS = [
    "trip_distance",
    "fare_amount",
    "passenger_count",
    "payment_type",
    "PULocationID",
    "DOLocationID",
]
FEATURES = ["trip_distance", "passenger_count", "payment_type", "PULocationID", "DOLocationID"]
MONITORS = ["psi", "ks", "wasserstein", "missing_rate", "cardinality", "drift_magnitude"]


def load(n: int) -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Download from "
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
        )
    frame = pd.read_parquet(DATA)
    return frame[COLUMNS].dropna().head(n).reset_index(drop=True)


def fit_linear(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features))])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return solution


def predict_linear(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features))])
    return design @ coefficients


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    raw = frame[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    distance = raw[:, 0]
    return np.column_stack([distance, distance**2, raw[:, 1:]])


def corruptions(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frame = frame.astype(
        {
            "PULocationID": "int64",
            "DOLocationID": "int64",
            "payment_type": "int64",
            "passenger_count": "float64",
        }
    )
    scenarios: list[tuple[str, pd.DataFrame]] = []

    for factor in (1.1, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0):
        changed = frame.copy()
        changed["trip_distance"] = changed["trip_distance"] * factor
        scenarios.append((f"distance_scale_{factor}", changed))

    for sigma in (0.5, 1.0, 2.0, 4.0):
        rng = np.random.default_rng(int(sigma * 10))
        changed = frame.copy()
        changed["trip_distance"] = changed["trip_distance"] + rng.normal(0, sigma, len(changed))
        scenarios.append((f"distance_noise_{sigma}", changed))

    for rate in (0.05, 0.1, 0.2, 0.3, 0.5):
        rng = np.random.default_rng(int(rate * 100))
        changed = frame.copy()
        changed.loc[rng.random(len(changed)) < rate, "passenger_count"] = np.nan
        scenarios.append((f"passenger_missing_{rate}", changed))

    for rate in (0.1, 0.25, 0.5):
        rng = np.random.default_rng(int(rate * 100) + 1)
        changed = frame.copy()
        mask = rng.random(len(changed)) < rate
        changed.loc[mask, "PULocationID"] = rng.integers(1, 266, int(mask.sum()))
        scenarios.append((f"pickup_location_shift_{rate}", changed))

    for rate in (0.1, 0.3):
        rng = np.random.default_rng(int(rate * 100) + 2)
        changed = frame.copy()
        mask = rng.random(len(changed)) < rate
        changed.loc[mask, "payment_type"] = rng.integers(1, 5, int(mask.sum()))
        scenarios.append((f"payment_type_shift_{rate}", changed))

    combined = frame.copy()
    combined["trip_distance"] = combined["trip_distance"] * 1.5
    combined.loc[combined.sample(frac=0.15, random_state=1).index, "passenger_count"] = np.nan
    scenarios.append(("combined_scale_missing", combined))

    combined = frame.copy()
    combined["trip_distance"] = combined["trip_distance"] * 2.0
    rng = np.random.default_rng(3)
    mask = rng.random(len(combined)) < 0.25
    combined.loc[mask, "PULocationID"] = rng.integers(1, 266, int(mask.sum()))
    scenarios.append(("combined_scale_location", combined))

    combined = frame.copy()
    combined["trip_distance"] = combined["trip_distance"] * 1.3
    rng = np.random.default_rng(4)
    combined["trip_distance"] += rng.normal(0, 1.0, len(combined))
    combined.loc[rng.random(len(combined)) < 0.1, "passenger_count"] = np.nan
    scenarios.append(("combined_scale_noise_missing", combined))

    combined = frame.copy()
    rng = np.random.default_rng(5)
    mask = rng.random(len(combined)) < 0.5
    combined.loc[mask, "PULocationID"] = rng.integers(1, 266, int(mask.sum()))
    mask = rng.random(len(combined)) < 0.3
    combined.loc[mask, "payment_type"] = rng.integers(1, 5, int(mask.sum()))
    scenarios.append(("combined_location_payment", combined))

    combined = frame.copy()
    combined["trip_distance"] = combined["trip_distance"] * 2.5
    combined.loc[combined.sample(frac=0.3, random_state=2).index, "passenger_count"] = np.nan
    rng = np.random.default_rng(6)
    mask = rng.random(len(combined)) < 0.25
    combined.loc[mask, "PULocationID"] = rng.integers(1, 266, int(mask.sum()))
    scenarios.append(("combined_scale_missing_location", combined))

    return scenarios


def scenario_family(name: str) -> str:
    """Keep related corruption families in the same split."""
    if name.startswith(("distance_scale", "distance_noise")):
        return "train"
    if name.startswith("passenger_missing"):
        return "validation"
    return "test"


def detect(frame: pd.DataFrame, reference: pd.DataFrame, window_size: int) -> dict:
    runner = SlidingWindowRunner(
        reference=reference,
        monitors=MONITORS,
        window_size=window_size,
        sliding=True,
        stride=window_size,
        include_partial=False,
    )
    report = runner.run(frame)
    windows = report.window_summaries
    total = max(len(windows), 1)
    per_monitor = {
        monitor: sum(
            1
            for window in windows
            if any(result.name == monitor and result.alert for result in window["results"])
        )
        / total
        for monitor in MONITORS
    }
    return {
        "windows": len(windows),
        "any_alert_rate": sum(
            1 for window in windows if any(result.alert for result in window["results"])
        )
        / total,
        "per_monitor": per_monitor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--window-size", type=int, default=5000)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    frame = load(args.n * 2).sample(frac=1.0, random_state=0).reset_index(drop=True)
    reference = frame.iloc[: args.n].reset_index(drop=True)
    test = frame.iloc[args.n :].reset_index(drop=True)

    target = test["fare_amount"].to_numpy(float)
    coefficients = fit_linear(feature_matrix(reference), reference["fare_amount"].to_numpy(float))
    clean_prediction = predict_linear(feature_matrix(test), coefficients)
    clean_metrics = regression_metrics(target, clean_prediction)

    rows = []
    for name, changed in corruptions(test):
        prediction = predict_linear(feature_matrix(changed), coefficients)
        metrics = regression_metrics(target, prediction)
        damage = max(0.0, (metrics["rmse"] - clean_metrics["rmse"]) / clean_metrics["rmse"])
        vector = compute_corruption_vector(reference, changed)
        rows.append(
            {
                "scenario": name,
                "family": scenario_family(name),
                "damage": min(1.0, damage),
                **vector.as_dict(),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
            }
        )

    scenario_frame = pd.DataFrame(rows)
    features = scenario_frame[list(CorruptionVector.feature_names)].to_numpy(float)
    damage = scenario_frame["damage"].to_numpy(float)
    masks = {
        family: scenario_frame["family"] == family for family in ("train", "validation", "test")
    }
    train_idx = scenario_frame.index[masks["train"]].to_numpy()
    validation_idx = scenario_frame.index[masks["validation"]].to_numpy()
    test_idx = scenario_frame.index[masks["test"]].to_numpy()

    predictor = DamagePredictor().fit(features[train_idx], damage[train_idx])
    predictor.validation_metrics_ = predictor.evaluate(
        features[validation_idx], damage[validation_idx]
    )
    predictor.test_metrics_ = predictor.evaluate(features[test_idx], damage[test_idx])
    heuristic = DamagePredictor()
    heuristic_validation = heuristic.evaluate(features[validation_idx], damage[validation_idx])
    heuristic_test = heuristic.evaluate(features[test_idx], damage[test_idx])

    clean_detection = detect(test, reference, args.window_size)
    corrupted = test.copy()
    corrupted["trip_distance"] = corrupted["trip_distance"] * 1.6
    corrupted.loc[corrupted.sample(frac=0.15, random_state=8).index, "passenger_count"] = np.nan
    corrupted_detection = detect(corrupted, reference, args.window_size)

    summary = {
        "rows": len(test),
        "clean_model": clean_metrics,
        "detection": {
            "clean_false_positive_rate": clean_detection["any_alert_rate"],
            "corrupted_detection_rate": corrupted_detection["any_alert_rate"],
            "clean_per_monitor": clean_detection["per_monitor"],
            "corrupted_per_monitor": corrupted_detection["per_monitor"],
        },
        "damage_predictor": {
            "scenarios": len(scenario_frame),
            "split": (
                "family holdout: train distance, validation missingness, "
                "test location/payment/combined"
            ),
            "train_rows": len(train_idx),
            "validation_rows": len(validation_idx),
            "test_rows": len(test_idx),
            "validation_damage_std": float(np.std(damage[validation_idx])),
            "test_damage_std": float(np.std(damage[test_idx])),
            "train_metrics": predictor.training_metrics_,
            "validation_metrics": predictor.validation_metrics_,
            "test_metrics": predictor.test_metrics_,
            "heuristic_validation_metrics": heuristic_validation,
            "heuristic_test_metrics": heuristic_test,
            "caveat": "Small scenario counts and a narrow corruption family; treat as directional.",
        },
    }

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    scenario_frame.to_csv(output / "real_damage_scenarios.csv", index=False)
    predictor.save(str(output / "damage_predictor_real.json"))
    (output / "real_benchmark.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
