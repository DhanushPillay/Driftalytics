"""Sliding-window and streaming runner."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (
    CorruptionVector,
    Monitor,
    MonitorRegistry,
    MonitorResult,
    compute_corruption_vector,
)
from .damage import DamagePredictor
from .report import Report


@dataclass
class Alert:
    monitor: str
    score: float
    threshold: float | None
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    window: int | None = None


def _covered_rows(starts: list[int], window_size: int, n_rows: int) -> int:
    """Count rows covered by possibly overlapping windows."""
    intervals = sorted((start, min(start + window_size, n_rows)) for start in starts)
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def read_frame(data) -> pd.DataFrame:
    """Read a DataFrame, parquet, or CSV input."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        raise ValueError(f"Unsupported data format: {suffix or path.name}")
    raise TypeError(f"Expected DataFrame or path, got {type(data).__name__}")


class SlidingWindowRunner:
    def __init__(
        self,
        reference,
        monitors: Sequence[str | Monitor | type[Monitor]],
        window_size: int = 1000,
        alert_threshold: float | None = None,
        damage_predictor: DamagePredictor | None = None,
        stride: int | None = None,
        sliding: bool = False,
        include_partial: bool = True,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be greater than 0")

        self.reference = read_frame(reference)
        if self.reference.empty:
            raise ValueError("reference data must not be empty")

        self.monitors = [self._make_monitor(m) for m in monitors]
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.damage_predictor = damage_predictor
        self.stride = stride or window_size
        self.sliding = sliding
        self.include_partial = include_partial
        self.expected_size = window_size if sliding else len(self.reference)

        for monitor in self.monitors:
            if hasattr(monitor, "expected_size") and monitor.expected_size is None:
                monitor.expected_size = self.expected_size

        self._needs_corruption = damage_predictor is not None or any(
            getattr(m, "name", "") == "drift_magnitude" for m in self.monitors
        )
        self._stream_buffer: pd.DataFrame | None = None
        self._stream_window = 0

    @staticmethod
    def _make_monitor(item: str | Monitor | type[Monitor]) -> Monitor:
        if isinstance(item, str):
            return MonitorRegistry.get(item)()
        if isinstance(item, type) and issubclass(item, Monitor):
            return item()
        if isinstance(item, Monitor):
            return item
        raise TypeError(f"Unsupported monitor: {item!r}")

    def _run_one(
        self,
        batch_df: pd.DataFrame,
        window: int | None = None,
    ) -> tuple[list[MonitorResult], list[Alert], CorruptionVector | None, float | None, str | None]:
        if not isinstance(batch_df, pd.DataFrame):
            raise TypeError(f"batch must be a DataFrame, got {type(batch_df).__name__}")

        corruption = (
            compute_corruption_vector(
                self.reference,
                batch_df,
                expected_size=self.expected_size,
            )
            if self._needs_corruption
            else None
        )

        results: list[MonitorResult] = []
        alerts: list[Alert] = []
        for monitor in self.monitors:
            try:
                if corruption is not None and hasattr(monitor, "result_from_corruption"):
                    result = monitor.result_from_corruption(corruption)
                else:
                    result = monitor.on_batch(self.reference, batch_df)

                if not isinstance(result, MonitorResult):
                    raise TypeError(f"{monitor.name} returned {type(result).__name__}")

                if result.threshold is None:
                    result.threshold = getattr(monitor, "threshold", None)
                if result.threshold is None and self.alert_threshold is not None:
                    result.threshold = self.alert_threshold
                    result.alert = bool(result.score > self.alert_threshold)
                else:
                    result.alert = bool(result.alert)
            except Exception as exc:
                result = MonitorResult(
                    name=getattr(monitor, "name", type(monitor).__name__),
                    score=0.0,
                    alert=False,
                    threshold=getattr(monitor, "threshold", None),
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )

            results.append(result)
            if result.alert:
                alerts.append(
                    Alert(
                        monitor=result.name,
                        score=result.score,
                        threshold=result.threshold,
                        window=window,
                        message=result.error or "",
                    )
                )

        damage: float | None = None
        damage_error: str | None = None
        if self.damage_predictor is not None and corruption is not None:
            try:
                damage = float(self.damage_predictor.predict(corruption))
            except Exception as exc:
                damage_error = f"{type(exc).__name__}: {exc}"

        return results, alerts, corruption, damage, damage_error

    def _run_window(self, chunk: pd.DataFrame, start: int, end: int, window: int) -> Report:
        results, alerts, corruption, damage, damage_error = self._run_one(chunk, window)
        summary = {
            "window": window,
            "start": start,
            "end": end,
            "n_rows": len(chunk),
            "results": results,
            "alerts": alerts,
            "corruption": corruption,
            "damage_estimate": damage,
            "damage_error": damage_error,
        }
        return Report(
            results=results,
            alerts=alerts,
            n_rows=len(chunk),
            n_rows_analyzed=len(chunk),
            n_rows_skipped=0,
            damage_estimate=damage,
            damage_error=damage_error,
            corruptions=[corruption] if corruption is not None else [],
            n_windows=1,
            window_summaries=[summary],
        )

    def run(self, data) -> Report:
        df = read_frame(data)
        if df.empty:
            return Report(n_rows=0, n_rows_analyzed=0, n_rows_skipped=0, n_windows=0)

        if not self.sliding or len(df) <= self.window_size:
            return self._run_window(df, 0, len(df), 0)

        starts = list(range(0, len(df) - self.window_size + 1, self.stride))
        last_end = starts[-1] + self.window_size if starts else 0
        if self.include_partial and last_end < len(df):
            starts.append(last_end)

        reports = [
            self._run_window(
                df.iloc[start : min(start + self.window_size, len(df))],
                start,
                min(start + self.window_size, len(df)),
                idx,
            )
            for idx, start in enumerate(starts)
        ]
        covered = _covered_rows(starts, self.window_size, len(df))
        damage_values = [r.damage_estimate for r in reports if r.damage_estimate is not None]
        damage_errors = [r.damage_error for r in reports if r.damage_error]
        return Report(
            results=reports[-1].results,
            alerts=[a for report in reports for a in report.alerts],
            n_rows=len(df),
            n_rows_analyzed=covered,
            n_rows_skipped=len(df) - covered,
            damage_estimate=float(np.mean(damage_values)) if damage_values else None,
            damage_error="; ".join(dict.fromkeys(damage_errors)) or None,
            corruptions=[c for report in reports for c in report.corruptions],
            n_windows=len(reports),
            window_summaries=[summary for report in reports for summary in report.window_summaries],
        )

    def update(self, batch, final: bool = False) -> list[Report]:
        """Feed a stream one batch at a time and return completed windows."""
        frame = read_frame(batch)
        if self._stream_buffer is None:
            self._stream_buffer = frame.copy()
        else:
            self._stream_buffer = pd.concat([self._stream_buffer, frame], ignore_index=True)

        reports: list[Report] = []
        while len(self._stream_buffer) >= self.window_size:
            chunk = self._stream_buffer.iloc[: self.window_size]
            reports.append(self._run_window(chunk, 0, self.window_size, self._stream_window))
            self._stream_window += 1
            self._stream_buffer = self._stream_buffer.iloc[self.stride :]

        if final and self._stream_buffer is not None and not self._stream_buffer.empty:
            chunk = self._stream_buffer
            reports.append(self._run_window(chunk, 0, len(chunk), self._stream_window))
            self._stream_window += 1
            self._stream_buffer = None

        return reports

    def run_stream(self, batches: Iterable) -> Iterator[Report]:
        """Yield one report per completed window from an iterable of batches."""
        for batch in batches:
            yield from self.update(batch)
        yield from self.update(pd.DataFrame(), final=True)
