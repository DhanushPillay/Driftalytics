"""Tabular drift monitors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..core import (
    CorruptionVector,
    Monitor,
    MonitorResult,
    compute_corruption_vector,
    expected_cardinality,
)


def _common_columns(reference, batch) -> list[str]:
    return [c for c in reference.columns if c in batch.columns]


def _numeric_columns(reference, batch) -> list[str]:
    return [
        c
        for c in _common_columns(reference, batch)
        if pd.api.types.is_numeric_dtype(reference[c]) and pd.api.types.is_numeric_dtype(batch[c])
    ]


def _float_values(series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _safe_nunique(series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except (TypeError, ValueError):
        return 0


def _score(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _threshold(value: float | None) -> float:
    if value is None:
        raise ValueError("threshold must be configured")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("threshold must be a finite non-negative number")
    return value


def _expected_size(value: int | None) -> int | None:
    if value is not None and value < 0:
        raise ValueError("expected_size must be non-negative")
    return value


def _details(values: dict[str, float], **extra) -> dict:
    return {
        "per_column": values,
        "mean": float(np.mean(list(values.values()))) if values else 0.0,
        "max": _score(list(values.values())),
        **extra,
    }


def _bh_reject(p_values: list[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg correction for multiple column tests."""
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.asarray(p_values, dtype=float)[order] * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    rejected_sorted = ranked <= alpha
    rejected = np.zeros(count, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected.tolist()


def _psi_col(reference, batch, n_bins: int) -> float:
    ref_values = _float_values(reference)
    batch_values = _float_values(batch)
    if not len(ref_values) or not len(batch_values):
        return 0.0

    quantiles = np.unique(np.quantile(ref_values, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 2:
        edges = np.array([-np.inf, float(ref_values[0]), np.inf])
    else:
        interior = quantiles[1:-1]
        edges = np.concatenate(([-np.inf], interior, [np.inf]))

    ref_counts, _ = np.histogram(ref_values, bins=edges)
    batch_counts, _ = np.histogram(batch_values, bins=edges)
    ref_share = np.clip(ref_counts / max(len(ref_values), 1), 1e-6, None)
    batch_share = np.clip(batch_counts / max(len(batch_values), 1), 1e-6, None)
    return float(np.sum((batch_share - ref_share) * np.log(batch_share / ref_share)))


class PSIMonitor(Monitor):
    name = "psi"

    def __init__(self, threshold: float = 0.2, n_bins: int = 10):
        self.threshold = _threshold(threshold)
        if n_bins <= 0:
            raise ValueError("n_bins must be greater than 0")
        self.n_bins = n_bins

    def on_batch(self, reference, batch) -> MonitorResult:
        threshold = _threshold(self.threshold)
        columns = _numeric_columns(reference, batch)
        values = {c: _psi_col(reference[c], batch[c], self.n_bins) for c in columns}
        if not values:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=threshold,
                status="skipped",
                details={"reason": "no common numeric columns"},
            )
        score = _score(list(values.values()))
        return MonitorResult(
            name=self.name,
            score=score,
            alert=score > threshold,
            threshold=threshold,
            details=_details(values),
        )


class KSMonitor(Monitor):
    name = "ks"

    def __init__(self, alpha: float = 0.05, min_effect: float = 0.1):
        self.alpha = _threshold(alpha)
        self.threshold = self.alpha
        self.min_effect = _threshold(min_effect)

    def on_batch(self, reference, batch) -> MonitorResult:
        threshold = _threshold(self.threshold)
        statistics: dict[str, float] = {}
        p_values: dict[str, float] = {}
        for column in _numeric_columns(reference, batch):
            ref_values = _float_values(reference[column])
            batch_values = _float_values(batch[column])
            if not len(ref_values) or not len(batch_values):
                continue
            result = stats.ks_2samp(ref_values, batch_values)
            statistics[column] = float(result.statistic)
            p_values[column] = float(result.pvalue)

        if not statistics:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=threshold,
                status="skipped",
                details={"reason": "no common numeric columns"},
            )

        rejected = _bh_reject(list(p_values.values()), self.alpha)
        columns = list(statistics)
        effective = {
            column: bool(rejected[index]) and statistics[column] >= self.min_effect
            for index, column in enumerate(columns)
        }
        score = _score(list(statistics.values()))
        return MonitorResult(
            name=self.name,
            score=score,
            alert=any(effective.values()),
            threshold=threshold,
            details={
                "statistics": statistics,
                "p_values": p_values,
                "rejected": effective,
                "min_effect": self.min_effect,
            },
        )


class WassersteinMonitor(Monitor):
    name = "wasserstein"

    def __init__(self, threshold: float = 0.1):
        self.threshold = _threshold(threshold)

    def on_batch(self, reference, batch) -> MonitorResult:
        threshold = _threshold(self.threshold)
        values: dict[str, float] = {}
        for column in _numeric_columns(reference, batch):
            ref_values = _float_values(reference[column])
            batch_values = _float_values(batch[column])
            if not len(ref_values) or not len(batch_values):
                continue
            scale = float(np.std(ref_values))
            if not np.isfinite(scale) or scale == 0.0:
                scale = 1.0
            values[column] = float(stats.wasserstein_distance(ref_values, batch_values) / scale)
        if not values:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=threshold,
                status="skipped",
                details={"reason": "no common numeric columns"},
            )
        score = _score(list(values.values()))
        return MonitorResult(
            name=self.name,
            score=score,
            alert=score > threshold,
            threshold=threshold,
            details=_details(values),
        )


class MissingRateMonitor(Monitor):
    name = "missing_rate"

    def __init__(self, threshold: float = 0.05):
        self.threshold = _threshold(threshold)

    def on_batch(self, reference, batch) -> MonitorResult:
        threshold = _threshold(self.threshold)
        columns = _common_columns(reference, batch)
        values = {
            c: abs(float(batch[c].isna().mean()) - float(reference[c].isna().mean()))
            for c in columns
        }
        if not values:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=threshold,
                status="skipped",
                details={"reason": "no common columns"},
            )
        score = _score(list(values.values()))
        return MonitorResult(
            name=self.name,
            score=score,
            alert=score > threshold,
            threshold=threshold,
            details=_details(values),
        )


class CardinalityMonitor(Monitor):
    name = "cardinality"

    def __init__(self, threshold: float = 0.1, expected_size: int | None = None):
        self.threshold = _threshold(threshold)
        self.expected_size = _expected_size(expected_size)

    def on_batch(self, reference, batch) -> MonitorResult:
        threshold = _threshold(self.threshold)
        sample_size = self.expected_size if self.expected_size is not None else len(batch)
        values: dict[str, float] = {}
        for column in _common_columns(reference, batch):
            ref_card = max(expected_cardinality(reference[column], sample_size), 1.0)
            values[column] = abs(_safe_nunique(batch[column]) - ref_card) / ref_card
        if not values:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=threshold,
                status="skipped",
                details={"reason": "no common columns"},
            )
        score = _score(list(values.values()))
        return MonitorResult(
            name=self.name,
            score=score,
            alert=score > threshold,
            threshold=threshold,
            details=_details(values),
        )


class ChiSquareMonitor(Monitor):
    name = "chi_square"

    def __init__(self, alpha: float = 0.05, min_effect: float = 0.1, max_categories: int = 50):
        self.alpha = _threshold(alpha)
        self.threshold = self.alpha
        self.min_effect = _threshold(min_effect)
        if max_categories <= 0:
            raise ValueError("max_categories must be greater than 0")
        self.max_categories = max_categories

    def on_batch(self, reference, batch) -> MonitorResult:
        threshold = _threshold(self.threshold)
        statistics: dict[str, float] = {}
        p_values: dict[str, float] = {}
        skipped: dict[str, str] = {}
        for column in _common_columns(reference, batch):
            if pd.api.types.is_numeric_dtype(reference[column]):
                continue
            categories = sorted(
                set(reference[column].dropna().unique()) | set(batch[column].dropna().unique()),
                key=str,
            )
            if not categories:
                continue
            if len(categories) > self.max_categories:
                skipped[column] = f"cardinality {len(categories)} exceeds {self.max_categories}"
                continue

            ref_counts = reference[column].value_counts().reindex(categories, fill_value=0)
            batch_counts = batch[column].value_counts().reindex(categories, fill_value=0)
            table = [ref_counts.to_numpy() + 1, batch_counts.to_numpy() + 1]
            chi2, p_value, _, _ = stats.chi2_contingency(table)
            total = int(np.sum(table))
            cramers_v = float(np.sqrt(chi2 / total)) if total else 0.0
            statistics[column] = cramers_v
            p_values[column] = float(p_value)

        if not statistics:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=threshold,
                status="skipped",
                details={"reason": "no eligible categorical columns", "skipped": skipped},
            )

        rejected = _bh_reject(list(p_values.values()), self.alpha)
        columns = list(statistics)
        effective = {
            column: bool(rejected[index]) and statistics[column] >= self.min_effect
            for index, column in enumerate(columns)
        }
        score = _score(list(statistics.values()))
        return MonitorResult(
            name=self.name,
            score=score,
            alert=any(effective.values()),
            threshold=threshold,
            details={
                "statistics": statistics,
                "p_values": p_values,
                "rejected": effective,
                "skipped": skipped,
                "min_effect": self.min_effect,
            },
        )


class VolumeMonitor(Monitor):
    name = "volume"

    def __init__(self, threshold: float = 0.2, expected_size: int | None = None):
        self.threshold = _threshold(threshold)
        self.expected_size = _expected_size(expected_size)

    def on_batch(self, reference, batch) -> MonitorResult:
        threshold = _threshold(self.threshold)
        expected = self.expected_size if self.expected_size is not None else len(reference)
        score = abs(len(batch) - expected) / max(expected, 1)
        return MonitorResult(
            name=self.name,
            score=float(score),
            alert=score > threshold,
            threshold=threshold,
            details={"expected_size": expected, "batch_size": len(batch)},
        )


class SchemaChangeMonitor(Monitor):
    name = "schema_change"

    def __init__(self, threshold: float = 0.0):
        self.threshold = _threshold(threshold)

    def on_batch(self, reference, batch) -> MonitorResult:
        threshold = _threshold(self.threshold)
        columns = set(reference.columns) | set(batch.columns)
        changes: dict[str, str] = {}
        for column in reference.columns:
            if column not in batch.columns:
                changes[column] = "missing"
            elif str(reference[column].dtype) != str(batch[column].dtype):
                changes[column] = f"{reference[column].dtype} -> {batch[column].dtype}"
        for column in batch.columns:
            if column not in reference.columns:
                changes[column] = "unexpected"
        score = len(changes) / max(len(columns), 1)
        return MonitorResult(
            name=self.name,
            score=float(score),
            alert=score > threshold,
            threshold=threshold,
            details={"changes": changes},
        )


class DriftMagnitudeMonitor(Monitor):
    name = "drift_magnitude"

    def __init__(self, threshold: float = 0.5, expected_size: int | None = None):
        self.threshold = _threshold(threshold)
        self.expected_size = _expected_size(expected_size)

    def result_from_corruption(self, corruption: CorruptionVector) -> MonitorResult:
        threshold = _threshold(self.threshold)
        score = corruption.drift_magnitude
        return MonitorResult(
            name=self.name,
            score=float(score),
            alert=score > threshold,
            threshold=threshold,
            details={"corruption": corruption.as_dict()},
        )

    def on_batch(self, reference, batch) -> MonitorResult:
        corruption = compute_corruption_vector(
            reference,
            batch,
            expected_size=self.expected_size,
        )
        return self.result_from_corruption(corruption)
