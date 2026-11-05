"""Driftalytics: lightweight, stream-first ML/LLM evaluation and drift monitoring."""

from .core import (
    CorruptionVector,
    Monitor,
    MonitorRegistry,
    MonitorResult,
    compute_corruption_vector,
    register_monitor,
)
from .damage import DamagePredictor
from .llm import LLMJudge
from .report import Report
from .runner import Alert, SlidingWindowRunner
from .version import __version__

__all__ = [
    "Alert",
    "CorruptionVector",
    "DamagePredictor",
    "LLMJudge",
    "Monitor",
    "MonitorRegistry",
    "MonitorResult",
    "Report",
    "SlidingWindowRunner",
    "__version__",
    "compute_corruption_vector",
    "register_monitor",
]
