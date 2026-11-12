"""Detection and damage-prediction benchmark on the CICIDS2017 dataset.

The classifier is trained on benign traffic from 3 July and attacks from
4 July, evaluated against a held-out benign control from 3 July, then streamed
over 5-7 July. Damage is the observed increase in classification error over
the held-out control, not a synthetic corruption label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from driftalytics import DamagePredictor, compute_corruption_vector
from driftalytics.core import CorruptionVector
from driftalytics.datasets import DATASET_NAME, FEATURES, load_cicids

REFERENCE_DAY = "2017-07-03"
ATTACK_DAY = "2017-07-04"
LIVE_DAYS = ("2017-07-05", "2017-07-06", "2017-07-07")
BENIGN_TRAIN_ROWS = 50_000
CONTROL_ROWS = 50_000
WINDOW_SIZE = 5_000
SEED = 0
BOOTSTRAP_ROUNDS = 1_000
TRAIN_FRACTION = 0.6
VALIDATION_FRACTION = 0.2


def _fit_classifier(features: np.ndarray, target: np.ndarray):
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        n_jobs=-1,
    ).fit(scaler.transform(features), target)
    return scaler, model


def _classification_metrics(model, scaler, features, target) -> dict:
    transformed = scaler.transform(features)
    predictions = model.predict(transformed)
    probabilities = model.predict_proba(transformed)
    accuracy = float(accuracy_score(target, predictions))
    error = 1.0 - accuracy
    if len(np.unique(target)) > 1:
        balanced_error = 1.0 - float(balanced_accuracy_score(target, predictions))
    else:
        balanced_error = error
    return {
        "n_rows": int(len(target)),
        "accuracy": round(accuracy, 6),
        "error": round(error, 6),
        "balanced_error": round(balanced_error, 6),
        "log_loss": round(float(log_loss(target, probabilities, labels=model.classes_)), 6),
        "attack_share": round(float(np.mean(target)), 6),
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


def _bootstrap_metrics(target: np.ndarray, prediction: np.ndarray, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    maes: list[float] = []
    r2s: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        index = rng.integers(0, len(target), len(target))
        metrics = _regression_metrics(target[index], prediction[index])
        maes.append(metrics["mae"])
        r2s.append(metrics["r2"])
    return {
        "rounds": BOOTSTRAP_ROUNDS,
        "mae_95ci": [
            round(float(np.percentile(maes, 2.5)), 6),
            round(float(np.percentile(maes, 97.5)), 6),
        ],
        "r2_95ci": [
            round(float(np.percentile(r2s, 2.5)), 6),
            round(float(np.percentile(r2s, 97.5)), 6),
        ],
    }


def _windows(frame: pd.DataFrame, reference: pd.DataFrame, scaler, model, control: dict):
    rows = []
    for start in range(0, len(frame) - WINDOW_SIZE + 1, WINDOW_SIZE):
        end = start + WINDOW_SIZE
        chunk = frame.iloc[start:end]
        metrics = _classification_metrics(
            model,
            scaler,
            chunk[FEATURES].to_numpy(dtype=np.float64),
            chunk["label"].to_numpy(dtype=int),
        )
        corruption = compute_corruption_vector(
            reference[FEATURES],
            chunk[FEATURES],
            expected_size=WINDOW_SIZE,
        )
        damage = max(0.0, min(1.0, metrics["error"] - control["error"]))
        balanced_damage = max(0.0, min(1.0, metrics["balanced_error"] - control["balanced_error"]))
        labels = chunk["attack_label"].value_counts()
        rows.append(
            {
                "window": len(rows),
                "day": str(chunk["timestamp"].iloc[0].date()),
                "start": start,
                "end": end,
                "start_time": str(chunk["timestamp"].iloc[0]),
                "end_time": str(chunk["timestamp"].iloc[-1]),
                "n_rows": int(len(chunk)),
                "error": metrics["error"],
                "balanced_error": metrics["balanced_error"],
                "log_loss": metrics["log_loss"],
                "attack_share": metrics["attack_share"],
                "damage": round(damage, 6),
                "balanced_damage": round(balanced_damage, 6),
                "dominant_label": str(labels.index[0]),
                **{name: round(float(value), 6) for name, value in corruption.as_dict().items()},
            }
        )
    return pd.DataFrame(rows)


def _chronological_split(n_rows: int) -> tuple[slice, slice, slice]:
    n_test = max(1, int(n_rows * (1.0 - TRAIN_FRACTION - VALIDATION_FRACTION)))
    n_validation = max(1, int(n_rows * VALIDATION_FRACTION))
    n_train = n_rows - n_test - n_validation
    if n_train < 1:
        raise ValueError("not enough windows for a chronological split")
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_validation),
        slice(n_train + n_validation, n_rows),
    )


def _damage_models(windows: pd.DataFrame) -> dict:
    feature_names = list(CorruptionVector.feature_names)
    features = windows[feature_names].to_numpy(dtype=float)
    target = windows["damage"].to_numpy(dtype=float)
    balanced_target = windows["balanced_damage"].to_numpy(dtype=float)
    train, validation, test = _chronological_split(len(windows))

    model = DamagePredictor().fit(features[train], target[train])
    model.validation_metrics_ = model.evaluate(features[validation], target[validation])
    model.test_metrics_ = model.evaluate(features[test], target[test])
    test_prediction = np.array([model.predict(row.tolist()) for row in features[test]])
    model_ci = _bootstrap_metrics(target[test], test_prediction)

    balanced_model = DamagePredictor().fit(features[train], balanced_target[train])
    balanced_model.validation_metrics_ = balanced_model.evaluate(
        features[validation], balanced_target[validation]
    )
    balanced_model.test_metrics_ = balanced_model.evaluate(features[test], balanced_target[test])
    heuristic = DamagePredictor()
    heuristic_prediction = np.array([heuristic.predict(row.tolist()) for row in features[test]])

    prior_features = windows[["attack_share"]].to_numpy(dtype=float)
    prior_coefficients = _fit_linear(prior_features[train], target[train])
    prior_prediction = _predict_linear(prior_coefficients, prior_features[test])

    drift_features = windows[["drift_magnitude"]].to_numpy(dtype=float)
    drift_coefficients = _fit_linear(drift_features[train], target[train])
    drift_prediction = _predict_linear(drift_coefficients, drift_features[test])

    return {
        "model": model,
        "balanced_model": balanced_model,
        "summary": {
            "target": "absolute classification-error increase over the held-out control",
            "split": "chronological 60/20/20",
            "train_rows": len(range(*train.indices(len(windows)))),
            "validation_rows": len(range(*validation.indices(len(windows)))),
            "test_rows": len(range(*test.indices(len(windows)))),
            "train_metrics": model.training_metrics_,
            "validation_metrics": model.validation_metrics_,
            "test_metrics": model.test_metrics_,
            "test_metrics_95ci": model_ci,
            "balanced_target_test_metrics": balanced_model.test_metrics_,
            "heuristic_test_metrics": _regression_metrics(target[test], heuristic_prediction),
            "prior_only_test_metrics": _regression_metrics(target[test], prior_prediction),
            "drift_only_test_metrics": _regression_metrics(target[test], drift_prediction),
            "coefficients": {
                name: round(float(value), 6)
                for name, value in zip(feature_names, model.coef_, strict=True)
            },
            "intercept": round(float(model.intercept_), 6),
            "caveat": (
                "Each day has a different attack mix and benign traffic changes across days, "
                "so damage combines concept drift with class-prior change. Compare raw error "
                "with the balanced-error target and the prior-only baseline."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    frame = load_cicids(args.path)
    days = frame["timestamp"].dt.strftime("%Y-%m-%d")
    reference_day = frame[days == REFERENCE_DAY].reset_index(drop=True)
    attack_day = frame[days == ATTACK_DAY].reset_index(drop=True)
    live = frame[days.isin(LIVE_DAYS)].reset_index(drop=True)

    benign = reference_day[reference_day["label"] == 0]
    if len(benign) < BENIGN_TRAIN_ROWS + CONTROL_ROWS:
        raise ValueError("reference day does not have enough benign rows")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(benign))
    train_benign = benign.iloc[order[:BENIGN_TRAIN_ROWS]]
    control = benign.iloc[order[BENIGN_TRAIN_ROWS : BENIGN_TRAIN_ROWS + CONTROL_ROWS]]
    attacks = attack_day[attack_day["label"] == 1]
    if attacks.empty:
        raise ValueError("attack day does not contain attacks")
    train = pd.concat([train_benign, attacks], ignore_index=True).sample(
        frac=1.0, random_state=SEED
    )

    scaler, classifier = _fit_classifier(
        train[FEATURES].to_numpy(dtype=np.float64),
        train["label"].to_numpy(dtype=int),
    )
    control_metrics = _classification_metrics(
        classifier,
        scaler,
        control[FEATURES].to_numpy(dtype=np.float64),
        control["label"].to_numpy(dtype=int),
    )
    windows = _windows(live, train_benign, scaler, classifier, control_metrics)
    damage = _damage_models(windows)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output / "cicids_damage_windows.csv", index=False)
    damage["model"].save(str(output / "damage_predictor_real.json"))

    payload = {
        "dataset": DATASET_NAME,
        "total_rows": len(frame),
        "reference_day": REFERENCE_DAY,
        "attack_day": ATTACK_DAY,
        "live_days": list(LIVE_DAYS),
        "train_rows": len(train),
        "train_benign_rows": len(train_benign),
        "train_attack_rows": len(attacks),
        "control_rows": len(control),
        "live_rows": len(live),
        "window_size": WINDOW_SIZE,
        "windows": len(windows),
        "classifier": "StandardScaler + LogisticRegression(class_weight=balanced)",
        "control_metrics": control_metrics,
        "live_attack_counts": {
            str(name): int(value) for name, value in live["attack_label"].value_counts().items()
        },
        "damage_summary": {
            "damage_mean": round(float(windows["damage"].mean()), 6),
            "damage_std": round(float(windows["damage"].std(ddof=0)), 6),
            "damage_min": round(float(windows["damage"].min()), 6),
            "damage_max": round(float(windows["damage"].max()), 6),
            "balanced_damage_mean": round(float(windows["balanced_damage"].mean()), 6),
            "balanced_damage_std": round(float(windows["balanced_damage"].std(ddof=0)), 6),
            "windows": int(len(windows)),
            "windows_zero_damage": int((windows["damage"] == 0.0).sum()),
            "windows_above_0_5": int((windows["damage"] > 0.5).sum()),
        },
        "damage_by_day": {
            str(day): {
                "windows": int(len(group)),
                "damage_mean": round(float(group["damage"].mean()), 6),
                "damage_median": round(float(group["damage"].median()), 6),
                "damage_max": round(float(group["damage"].max()), 6),
                "balanced_damage_mean": round(float(group["balanced_damage"].mean()), 6),
                "attack_share_mean": round(float(group["attack_share"].mean()), 6),
                "accuracy_error_mean": round(float(group["error"].mean()), 6),
                "drift_magnitude_mean": round(float(group["drift_magnitude"].mean()), 6),
                "cardinality_shift_mean": round(float(group["cardinality_shift"].mean()), 6),
            }
            for day, group in windows.groupby("day", sort=True)
        },
        "damage_predictor": damage["summary"],
    }
    (output / "real_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "live_attack_counts"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
