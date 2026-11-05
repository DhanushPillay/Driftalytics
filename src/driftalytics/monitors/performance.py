"""Performance monitors. These require labels in the batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import Monitor, MonitorResult

CLASSIFICATION_METRICS = {"accuracy", "precision", "recall", "f1"}
REGRESSION_METRICS = {"rmse", "mae", "r2"}


def _classification_metrics(y_true, y_pred) -> dict[str, float]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(true) == 0:
        raise ValueError("no rows to score")

    labels = np.unique(np.concatenate([true, pred]))
    accuracy = float(np.mean(true == pred))
    precision_values = []
    recall_values = []
    f1_values = []
    supports = []
    for label in labels:
        true_positive = float(np.sum((true == label) & (pred == label)))
        false_positive = float(np.sum((true != label) & (pred == label)))
        false_negative = float(np.sum((true == label) & (pred != label)))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        supports.append(float(np.sum(true == label)))

    support = float(np.sum(supports))
    if support:
        weights = np.asarray(supports) / support
        weighted_precision = float(np.dot(precision_values, weights))
        weighted_recall = float(np.dot(recall_values, weights))
        weighted_f1 = float(np.dot(f1_values, weights))
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    return {
        "accuracy": accuracy,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
    }


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    mask = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[mask], pred[mask]
    if len(true) == 0:
        raise ValueError("no finite rows to score")

    residual = pred - true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


class PerfMonitor(Monitor):
    def __init__(
        self,
        task: str = "classification",
        metric: str | None = None,
        threshold: float = 0.05,
        reference_score: float | None = None,
    ):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.metric = metric or ("f1" if task == "classification" else "rmse")
        allowed = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
        if self.metric not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)} for {task}")
        self.threshold = threshold
        self.reference_score = reference_score
        self.name = "perf_" + task

    def on_batch(self, reference, batch) -> MonitorResult:
        assert self.threshold is not None
        for column in ("y_true", "y_pred"):
            if column not in batch.columns:
                return MonitorResult(
                    name=self.name,
                    score=0.0,
                    alert=False,
                    threshold=self.threshold,
                    status="error",
                    error=f"missing required column: {column}",
                )
        if batch.empty:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=self.threshold,
                status="skipped",
                details={"reason": "empty batch"},
            )
        if self.reference_score is None:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=self.threshold,
                status="skipped",
                details={"reason": "reference_score is required to measure degradation"},
            )

        try:
            if self.task == "classification":
                mask = ~(pd.isna(batch["y_true"]) | pd.isna(batch["y_pred"]))
                metrics = _classification_metrics(
                    batch.loc[mask, "y_true"].to_numpy(),
                    batch.loc[mask, "y_pred"].to_numpy(),
                )
                value = metrics[self.metric]
                drop = max(0.0, self.reference_score - value)
            else:
                metrics = _regression_metrics(
                    batch["y_true"].to_numpy(),
                    batch["y_pred"].to_numpy(),
                )
                value = metrics[self.metric]
                if self.metric == "r2":
                    drop = max(0.0, self.reference_score - value)
                else:
                    drop = max(0.0, value - self.reference_score)
        except (TypeError, ValueError) as exc:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=self.threshold,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        return MonitorResult(
            name=self.name,
            score=float(drop),
            alert=drop > self.threshold,
            threshold=self.threshold,
            details={
                "metric": self.metric,
                "value": value,
                "reference_score": self.reference_score,
                **metrics,
            },
        )


class PerfClassificationMonitor(PerfMonitor):
    def __init__(self, **kwargs):
        super().__init__(task="classification", **kwargs)


class PerfRegressionMonitor(PerfMonitor):
    def __init__(self, **kwargs):
        super().__init__(task="regression", **kwargs)
