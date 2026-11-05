"""Core types, monitor registry, and corruption-vector computation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class MonitorResult:
    name: str
    score: float
    alert: bool
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    threshold: float | None = None
    status: str = "ok"
    error: str | None = None


@dataclass
class CorruptionVector:
    """Low-dimensional displacement summary for a batch vs. reference."""

    drift_magnitude: float = 0.0
    missingness_delta: float = 0.0
    cardinality_shift: float = 0.0
    schema_change: float = 0.0
    volume_delta: float = 0.0

    feature_names: ClassVar[tuple[str, ...]] = (
        "drift_magnitude",
        "missingness_delta",
        "cardinality_shift",
        "schema_change",
        "volume_delta",
    )

    def to_list(self) -> list[float]:
        return [
            self.drift_magnitude,
            self.missingness_delta,
            self.cardinality_shift,
            self.schema_change,
            self.volume_delta,
        ]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.to_list(), strict=True))


class Monitor:
    """Base class for every monitor."""

    name: str = "monitor"
    threshold: float | None = None

    def on_batch(self, reference, batch) -> MonitorResult:
        raise NotImplementedError


class MonitorRegistry:
    _registry: ClassVar[dict[str, type[Monitor]]] = {}

    @classmethod
    def register(cls, name: str, monitor_cls: type[Monitor]) -> None:
        cls._registry[name] = monitor_cls

    @classmethod
    def get(cls, name: str) -> type[Monitor]:
        if name not in cls._registry:
            raise KeyError(f"Unknown monitor: {name!r}. Registered: {sorted(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry)


def register_monitor(name: str, monitor_cls: type[Monitor]) -> None:
    MonitorRegistry.register(name, monitor_cls)


def _nunique(series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except (TypeError, ValueError):
        return 0


def expected_cardinality(series, sample_size: int) -> float:
    """Expected distinct values in a sample drawn from the reference distribution."""
    import numpy as np

    if sample_size <= 0:
        return 0.0
    shares = series.dropna().value_counts(normalize=True).to_numpy(dtype=float)
    if not len(shares):
        return 0.0
    return float(np.sum(1.0 - (1.0 - shares) ** sample_size))


def compute_corruption_vector(
    reference,
    batch,
    expected_size: int | None = None,
) -> CorruptionVector:
    """Summarize batch displacement without labels.

    ``expected_size`` is the batch size the monitor expects. It defaults to
    the reference length, which is only meaningful for whole-batch runs.
    Streaming callers should pass their window size.
    """
    import numpy as np
    import pandas as pd

    cv = CorruptionVector()
    common = [c for c in reference.columns if c in batch.columns]
    union = set(reference.columns) | set(batch.columns)

    expected = len(reference) if expected_size is None else expected_size
    cv.volume_delta = abs(len(batch) - expected) / max(expected, 1)

    missing = [
        abs(float(batch[c].isna().mean()) - float(reference[c].isna().mean())) for c in common
    ]
    cv.missingness_delta = float(np.mean(missing)) if missing else 0.0

    sample_size = expected_size if expected_size is not None else len(batch)
    cardinality = []
    for c in common:
        ref_card = max(expected_cardinality(reference[c], sample_size), 1.0)
        cardinality.append(abs(_nunique(batch[c]) - ref_card) / ref_card)
    cv.cardinality_shift = float(np.mean(cardinality)) if cardinality else 0.0

    drifts: list[float] = []
    for c in common:
        if not (
            pd.api.types.is_numeric_dtype(reference[c]) and pd.api.types.is_numeric_dtype(batch[c])
        ):
            continue
        ref_values = pd.to_numeric(reference[c], errors="coerce").dropna()
        batch_values = pd.to_numeric(batch[c], errors="coerce").dropna()
        if ref_values.empty or batch_values.empty:
            continue
        scale = float(ref_values.std(ddof=0))
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        drifts.append(abs(float(batch_values.mean()) - float(ref_values.mean())) / scale)
    cv.drift_magnitude = float(np.mean(drifts)) if drifts else 0.0

    changes = 0
    for c in reference.columns:
        if c not in batch.columns:
            changes += 1
        elif str(reference[c].dtype) != str(batch[c].dtype):
            changes += 1
    changes += sum(1 for c in batch.columns if c not in reference.columns)
    cv.schema_change = changes / max(len(union), 1)
    return cv


def _load_builtins() -> None:
    from . import monitors as _monitors

    _monitors.register_all()


_load_builtins()
