"""Detection and damage-prediction benchmark on the Insects real dataset.

A logistic-regression classifier is trained on the first 20,000 rows and
evaluated on each 500-row window of the remaining stream. Damage is the real
increase in classification error over the reference period, not a synthetic
corruption label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from driftalytics import DamagePredictor, SlidingWindowRunner, compute_corruption_vector
from driftalytics.core import CorruptionVector
from driftalytics.datasets import DATASET_NAME, FEATURES, load_insects

MONITORS = ["psi", "ks", "wasserstein", "missing_rate", "cardinality", "drift_magnitude"]
REFERENCE_ROWS = 20_000
WINDOW_SIZE = 500
SEVERE_DAMAGE = 0.5


def _fit_classifier(features: np.ndarray, target: np.ndarray):
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(max_iter=3000).fit(scaler.transform(features), target)
    return scaler, model


def _classification_metrics(model, scaler, features, target, labels) -> dict:
    transformed = scaler.transform(features)
    predictions = model.predict(transformed)
    probabilities = model.predict_proba(transformed)
    accuracy = float(accuracy_score(target, predictions))
    return {
        "accuracy": round(accuracy, 6),
        "error": round(1.0 - accuracy, 6),
        "log_loss": round(float(log_loss(target, probabilities, labels=labels)), 6),
    }


def _fit_linear(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def _predict_linear(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features))])
    return np.clip(design @ coefficients, 0.0, 1.0)


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    residuals = target - prediction
    variance = float(np.var(target))
    return {
        "mae": round(float(np.mean(np.abs(residuals))), 6),
        "rmse": round(float(np.sqrt(np.mean(residuals**2))), 6),
        "r2": 0.0 if variance == 0.0 else round(float(1.0 - np.var(residuals) / variance), 6),
    }


def _evaluate_windows(frame: pd.DataFrame, reference: pd.DataFrame, scaler, model, labels):
    live = frame.iloc[REFERENCE_ROWS:].reset_index(drop=True)
    runner = SlidingWindowRunner(
        reference=reference[FEATURES],
        monitors=MONITORS,
        window_size=WINDOW_SIZE,
        stride=WINDOW_SIZE,
        sliding=True,
        include_partial=False,
    )
    report = runner.run(live[FEATURES])
    reference_metrics = _classification_metrics(
        model,
        scaler,
        reference[FEATURES].to_numpy(dtype=float),
        reference["class"].to_numpy(dtype=int),
        labels,
    )
    rows = []
    for window in report.window_summaries:
        relative_start = int(window["start"])
        relative_end = int(window["end"])
        chunk = live.iloc[relative_start:relative_end]
        metrics = _classification_metrics(
            model,
            scaler,
            chunk[FEATURES].to_numpy(dtype=float),
            chunk["class"].to_numpy(dtype=int),
            labels,
        )
        corruption = compute_corruption_vector(
            reference[FEATURES],
            chunk[FEATURES],
            expected_size=len(chunk),
        )
        damage = max(0.0, min(1.0, metrics["error"] - reference_metrics["error"]))
        damage_relative = max(
            0.0,
            min(
                1.0,
                (metrics["error"] - reference_metrics["error"])
                / max(reference_metrics["error"], 1e-9),
            ),
        )
        damage_log_loss = max(
            0.0,
            min(
                1.0,
                (metrics["log_loss"] - reference_metrics["log_loss"])
                / max(reference_metrics["log_loss"], 1e-9),
            ),
        )
        row = {
            "window": int(window["window"]),
            "start": REFERENCE_ROWS + relative_start,
            "end": REFERENCE_ROWS + relative_end,
            "n_rows": int(window["n_rows"]),
            "error": metrics["error"],
            "accuracy": metrics["accuracy"],
            "log_loss": metrics["log_loss"],
            "damage": round(damage, 6),
            "damage_relative": round(damage_relative, 6),
            "damage_log_loss": round(damage_log_loss, 6),
            "alerts": ",".join(sorted({alert.monitor for alert in window["alerts"]})),
        }
        row.update({name: round(float(value), 6) for name, value in corruption.as_dict().items()})
        rows.append(row)
    return report, reference_metrics, pd.DataFrame(rows)


def _detection_metrics(report, windows: pd.DataFrame) -> dict:
    severe = windows["damage"].to_numpy(dtype=float) > SEVERE_DAMAGE
    severe_windows = int(severe.sum())
    per_monitor = {}
    for name in MONITORS:
        alerted = np.array(
            [
                any(alert.monitor == name for alert in window["alerts"])
                for window in report.window_summaries
            ]
        )
        alert_count = int(alerted.sum())
        true_positive = int((alerted & severe).sum())
        per_monitor[name] = {
            "alerts": alert_count,
            "alert_rate": round(alert_count / max(len(windows), 1), 4),
            "precision": round(true_positive / max(alert_count, 1), 4),
            "recall": round(true_positive / max(severe_windows, 1), 4),
        }
    return {
        "severe_damage_threshold": SEVERE_DAMAGE,
        "severe_windows": severe_windows,
        "per_monitor": per_monitor,
    }


def _damage_model(windows: pd.DataFrame):
    feature_names = list(CorruptionVector.feature_names)
    features = windows[feature_names].to_numpy(dtype=float)
    target = windows["damage"].to_numpy(dtype=float)
    n_rows = len(features)
    n_test = max(1, int(n_rows * 0.2))
    n_validation = max(1, int(n_rows * 0.2))
    n_train = n_rows - n_test - n_validation
    if n_train < 1:
        raise ValueError("not enough windows for a chronological split")

    train = slice(0, n_train)
    validation = slice(n_train, n_train + n_validation)
    test = slice(n_train + n_validation, n_rows)

    model = DamagePredictor()
    model.fit(features[train], target[train])
    model.validation_metrics_ = model.evaluate(features[validation], target[validation])
    model.test_metrics_ = model.evaluate(features[test], target[test])

    heuristic = DamagePredictor()
    heuristic_validation = _regression_metrics(
        target[validation],
        np.array([heuristic.predict(row.tolist()) for row in features[validation]]),
    )
    heuristic_test = _regression_metrics(
        target[test],
        np.array([heuristic.predict(row.tolist()) for row in features[test]]),
    )

    drift_train = features[train][:, [0]]
    drift_coefficients = _fit_linear(drift_train, target[train])
    drift_validation = _regression_metrics(
        target[validation],
        _predict_linear(drift_coefficients, features[validation][:, [0]]),
    )
    drift_test = _regression_metrics(
        target[test],
        _predict_linear(drift_coefficients, features[test][:, [0]]),
    )

    summary = {
        "target": "absolute classification-error increase over the reference period",
        "split": "chronological 60/20/20",
        "train_rows": n_train,
        "validation_rows": n_validation,
        "test_rows": n_test,
        "train_metrics": model.training_metrics_,
        "validation_metrics": model.validation_metrics_,
        "test_metrics": model.test_metrics_,
        "heuristic_validation_metrics": heuristic_validation,
        "heuristic_test_metrics": heuristic_test,
        "drift_only_validation_metrics": drift_validation,
        "drift_only_test_metrics": drift_test,
        "coefficients": {
            name: round(float(value), 6)
            for name, value in zip(feature_names, model.coef_, strict=True)
        },
        "intercept": round(float(model.intercept_), 6),
        "caveat": (
            "Damage labels are observed error increases on a curated real stream. Only "
            "drift_magnitude and cardinality_shift vary in this dataset; missingness, schema, "
            "and volume dimensions are constant. The model is linear and should be retrained "
            "on deployment-specific labels."
        ),
    }
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    frame = load_insects(args.path)
    reference = frame.iloc[:REFERENCE_ROWS]
    if len(reference) < 2 or len(frame) - len(reference) < WINDOW_SIZE:
        raise ValueError("dataset is too small for the configured split")

    labels = np.sort(frame["class"].unique())
    scaler, classifier = _fit_classifier(
        reference[FEATURES].to_numpy(dtype=float),
        reference["class"].to_numpy(dtype=int),
    )
    report, reference_metrics, windows = _evaluate_windows(
        frame, reference, scaler, classifier, labels
    )
    detection = _detection_metrics(report, windows)
    damage_model, damage_summary = _damage_model(windows)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output / "insects_damage_windows.csv", index=False)
    damage_model.save(str(output / "damage_predictor_real.json"))

    payload = {
        "dataset": DATASET_NAME,
        "total_rows": len(frame),
        "reference_rows": len(reference),
        "live_rows": len(frame) - len(reference),
        "window_size": WINDOW_SIZE,
        "windows": len(windows),
        "classifier": "StandardScaler + LogisticRegression(max_iter=3000)",
        "reference_model": reference_metrics,
        "detection": detection,
        "damage_predictor": damage_summary,
    }
    (output / "real_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
