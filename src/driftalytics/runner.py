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
    start: int | None = None
    end: int | None = None


def _covered_rows(intervals: list[tuple[int, int]]) -> int:
    """Count rows covered by possibly overlapping windows."""
    if not intervals:
        return 0
    merged = sorted(intervals)
    total = 0
    current_start, current_end = merged[0]
    for start, end in merged[1:]:
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

        self._needs_corruption = damage_predictor is not None or any(
            getattr(m, "name", "") == "drift_magnitude" for m in self.monitors
        )
        self._stream_buffer = pd.DataFrame()
        self._stream_buffer_start = 0
        self._stream_total = 0
        self._stream_next_start = 0
        self._stream_last_end = 0
        self._stream_window = 0
        self._stream_finalized = False

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
        window: int | None,
        start: int,
        end: int,
        expected_size: int,
    ) -> tuple[list[MonitorResult], list[Alert], CorruptionVector | None, float | None, str | None]:
        if not isinstance(batch_df, pd.DataFrame):
            raise TypeError(f"batch must be a DataFrame, got {type(batch_df).__name__}")

        corruption = (
            compute_corruption_vector(
                self.reference,
                batch_df,
                expected_size=expected_size,
            )
            if self._needs_corruption
            else None
        )

        results: list[MonitorResult] = []
        alerts: list[Alert] = []
        for monitor in self.monitors:
            previous_size = getattr(monitor, "expected_size", None)
            try:
                if hasattr(monitor, "expected_size"):
                    monitor.expected_size = expected_size
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
            finally:
                if hasattr(monitor, "expected_size"):
                    monitor.expected_size = previous_size

            results.append(result)
            if result.alert:
                alerts.append(
                    Alert(
                        monitor=result.name,
                        score=result.score,
                        threshold=result.threshold,
                        window=window,
                        message=result.error or "",
                        start=start,
                        end=end,
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

    def _run_window(
        self,
        chunk: pd.DataFrame,
        start: int,
        end: int,
        window: int,
        expected_size: int,
    ) -> Report:
        results, alerts, corruption, damage, damage_error = self._run_one(
            chunk, window, start, end, expected_size
        )
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
            start=start,
            end=end,
        )

    def _combine(self, reports: list[Report], n_rows: int) -> Report:
        intervals = [(report.start, report.end) for report in reports]
        covered = _covered_rows(intervals)
        damage_values = [r.damage_estimate for r in reports if r.damage_estimate is not None]
        damage_errors = [r.damage_error for r in reports if r.damage_error]
        return Report(
            results=reports[-1].results if reports else [],
            alerts=[alert for report in reports for alert in report.alerts],
            n_rows=n_rows,
            n_rows_analyzed=covered,
            n_rows_skipped=n_rows - covered,
            damage_estimate=float(np.mean(damage_values)) if damage_values else None,
            damage_error="; ".join(dict.fromkeys(damage_errors)) or None,
            corruptions=[c for report in reports for c in report.corruptions],
            n_windows=len(reports),
            window_summaries=[summary for report in reports for summary in report.window_summaries],
            start=reports[0].start if reports else 0,
            end=reports[-1].end if reports else 0,
        )

    def run(self, data) -> Report:
        df = read_frame(data)
        if df.empty:
            return Report(n_rows=0, n_rows_analyzed=0, n_rows_skipped=0, n_windows=0)

        if not self.sliding:
            report = self._run_window(df, 0, len(df), 0, len(self.reference))
            return self._combine([report], len(df))

        reports: list[Report] = []
        if len(df) >= self.window_size:
            starts = list(range(0, len(df) - self.window_size + 1, self.stride))
            reports.extend(
                self._run_window(
                    df.iloc[start : start + self.window_size],
                    start,
                    start + self.window_size,
                    index,
                    self.window_size,
                )
                for index, start in enumerate(starts)
            )
            last_end = starts[-1] + self.window_size
            tail_start = max(last_end, starts[-1] + self.stride)
        else:
            tail_start = 0

        if self.include_partial and tail_start < len(df):
            reports.append(
                self._run_window(
                    df.iloc[tail_start:],
                    tail_start,
                    len(df),
                    len(reports),
                    len(df) - tail_start,
                )
            )

        return self._combine(reports, len(df))

    def _trim_stream(self) -> None:
        drop = self._stream_next_start - self._stream_buffer_start
        if drop <= 0 or self._stream_buffer.empty:
            return
        drop = min(drop, len(self._stream_buffer))
        self._stream_buffer = self._stream_buffer.iloc[drop:]
        self._stream_buffer_start += drop

    def _append_stream(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self._stream_buffer = pd.concat([self._stream_buffer, frame], ignore_index=True)
        self._stream_total += len(frame)

    def _emit_stream_windows(self) -> list[Report]:
        reports: list[Report] = []
        while True:
            self._trim_stream()
            if len(self._stream_buffer) < self.window_size:
                return reports
            start = self._stream_buffer_start
            chunk = self._stream_buffer.iloc[: self.window_size]
            reports.append(
                self._run_window(
                    chunk,
                    start,
                    start + self.window_size,
                    self._stream_window,
                    self.window_size,
                )
            )
            self._stream_window += 1
            self._stream_next_start = start + self.stride
            self._stream_last_end = start + self.window_size
            self._trim_stream()

    def update(self, batch, final: bool = False) -> list[Report]:
        """Feed a stream one batch at a time and return completed windows."""
        frame = read_frame(batch)
        if self._stream_finalized:
            if final and frame.empty:
                return []
            raise RuntimeError("stream already finalized")

        if not self.sliding:
            self._append_stream(frame)
            if not final:
                return []
            self._stream_finalized = True
            if self._stream_buffer.empty:
                return []
            report = self._run_window(
                self._stream_buffer,
                self._stream_buffer_start,
                self._stream_total,
                0,
                len(self.reference),
            )
            self._stream_buffer = pd.DataFrame()
            return [report]

        self._append_stream(frame)
        reports = self._emit_stream_windows()

        if final:
            self._stream_finalized = True
            partial_start = max(self._stream_last_end, self._stream_next_start)
            drop = partial_start - self._stream_buffer_start
            if drop > 0 and not self._stream_buffer.empty:
                drop = min(drop, len(self._stream_buffer))
                self._stream_buffer = self._stream_buffer.iloc[drop:]
                self._stream_buffer_start += drop
            if self.include_partial and not self._stream_buffer.empty:
                start = self._stream_buffer_start
                reports.append(
                    self._run_window(
                        self._stream_buffer,
                        start,
                        start + len(self._stream_buffer),
                        self._stream_window,
                        len(self._stream_buffer),
                    )
                )
                self._stream_window += 1
            self._stream_buffer = pd.DataFrame()

        return reports

    def run_stream(self, batches: Iterable) -> Iterator[Report]:
        """Yield one report per completed window from an iterable of batches."""
        for batch in batches:
            yield from self.update(batch)
        yield from self.update(pd.DataFrame(), final=True)
