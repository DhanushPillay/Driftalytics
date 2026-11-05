"""Built-in monitors."""

from __future__ import annotations

from ..core import Monitor, MonitorRegistry
from .llm_monitors import AnswerRelevanceMonitor, FaithfulnessMonitor
from .performance import PerfClassificationMonitor, PerfRegressionMonitor
from .tabular import (
    CardinalityMonitor,
    ChiSquareMonitor,
    DriftMagnitudeMonitor,
    KSMonitor,
    MissingRateMonitor,
    PSIMonitor,
    SchemaChangeMonitor,
    VolumeMonitor,
    WassersteinMonitor,
)

BUILTIN_MONITORS: dict[str, type[Monitor]] = {
    "answer_relevance": AnswerRelevanceMonitor,
    "cardinality": CardinalityMonitor,
    "chi_square": ChiSquareMonitor,
    "drift_magnitude": DriftMagnitudeMonitor,
    "faithfulness": FaithfulnessMonitor,
    "ks": KSMonitor,
    "missing_rate": MissingRateMonitor,
    "perf_classification": PerfClassificationMonitor,
    "perf_regression": PerfRegressionMonitor,
    "psi": PSIMonitor,
    "schema_change": SchemaChangeMonitor,
    "volume": VolumeMonitor,
    "wasserstein": WassersteinMonitor,
}


def register_all() -> None:
    for name, monitor_class in BUILTIN_MONITORS.items():
        MonitorRegistry.register(name, monitor_class)
