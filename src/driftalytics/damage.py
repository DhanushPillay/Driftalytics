"""Damage predictor: corruption vector -> estimated performance drop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import CorruptionVector

FORMAT = "driftalytics-damage-v1"


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _read_metrics(payload: dict[str, Any], key: str) -> dict[str, float]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return {str(name): _finite_float(metric, f"{key}.{name}") for name, metric in value.items()}


class DamagePredictor:
    """Linear damage predictor trained on (corruption vector, performance drop) pairs."""

    def __init__(self) -> None:
        self.coef_: list[float] | None = None
        self.intercept_: float = 0.0
        self.metrics_: dict[str, float] = {}
        self.training_metrics_: dict[str, float] = {}
        self.validation_metrics_: dict[str, float] = {}
        self.test_metrics_: dict[str, float] = {}
        self.feature_names: tuple[str, ...] = CorruptionVector.feature_names

    @property
    def is_trained(self) -> bool:
        return self.coef_ is not None

    def fit(self, X, y) -> DamagePredictor:
        values = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        if values.ndim != 2:
            raise ValueError("X must be a 2D array")
        if target.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(values) != len(target):
            raise ValueError("X and y must have the same number of rows")
        if len(values) == 0:
            raise ValueError("training data must not be empty")
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError("training data must be finite")

        design = np.hstack([values, np.ones((len(values), 1))])
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        self.coef_ = solution[:-1].tolist()
        self.intercept_ = float(solution[-1])
        self.training_metrics_ = self.evaluate(values, target)
        self.metrics_ = self.training_metrics_
        return self

    def evaluate(self, X, y) -> dict[str, float]:
        values = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        if values.ndim != 2:
            raise ValueError("X must be a 2D array")
        if target.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(values) != len(target):
            raise ValueError("X and y must have the same number of rows")
        if len(values) == 0:
            raise ValueError("evaluation data must not be empty")
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError("evaluation data must be finite")
        predictions = np.array([self.predict(row) for row in values])
        residual = predictions - target
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        total = float(np.sum((target - target.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 0.0
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def predict(self, vector: Any) -> float:
        values = vector.to_list() if isinstance(vector, CorruptionVector) else list(vector)
        if len(values) != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {len(values)}")
        values = [float(v) for v in values]
        if not all(np.isfinite(v) for v in values):
            raise ValueError("corruption vector must be finite")

        if self.coef_ is None:
            score = (
                0.35 * values[0]
                + 0.25 * values[1]
                + 0.15 * values[2]
                + 0.15 * values[3]
                + 0.10 * values[4]
            )
        else:
            score = float(np.dot(self.coef_, values) + self.intercept_)
        return float(np.clip(score, 0.0, 1.0))

    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT,
            "feature_names": list(self.feature_names),
            "coef": self.coef_,
            "intercept": self.intercept_,
            "metrics": self.metrics_,
            "training_metrics": self.training_metrics_,
            "validation_metrics": self.validation_metrics_,
            "test_metrics": self.test_metrics_,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> DamagePredictor:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("damage predictor payload must be a mapping")
        if data.get("format") != FORMAT:
            raise ValueError(f"unsupported damage predictor format: {data.get('format')!r}")
        feature_names = tuple(data.get("feature_names", ()))
        if feature_names != CorruptionVector.feature_names:
            raise ValueError("damage predictor feature order does not match CorruptionVector")

        model = cls()
        coef = data.get("coef")
        if coef is not None:
            if not isinstance(coef, list):
                raise ValueError("damage predictor coefficients must be a list")
            coef = [_finite_float(value, "coefficient") for value in coef]
            if len(coef) != len(feature_names):
                raise ValueError("damage predictor coefficient count is invalid")
        model.coef_ = coef
        model.intercept_ = _finite_float(data.get("intercept", 0.0), "intercept")
        model.training_metrics_ = _read_metrics(data, "training_metrics") or _read_metrics(
            data, "metrics"
        )
        model.validation_metrics_ = _read_metrics(data, "validation_metrics")
        model.test_metrics_ = _read_metrics(data, "test_metrics")
        model.metrics_ = model.training_metrics_
        return model
