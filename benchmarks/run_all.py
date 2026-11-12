"""Throughput benchmark on the CICIDS2017 real dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd

from driftalytics import SlidingWindowRunner
from driftalytics.datasets import DATASET_NAME, FEATURES, load_cicids

MONITORS = ["psi", "ks", "missing_rate"]
WINDOW_SIZE = 1_000
CHUNK_SIZE = 100_000
REFERENCE_ROWS = 5_000
REPEATS = 3
LARGE_ROWS = 1_000_000
SIZES = (1_000, 10_000, 100_000, LARGE_ROWS)
PER_MONITOR_ROWS = 100_000


def _batch(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    features = frame[FEATURES]
    copies = (n + len(features) - 1) // len(features)
    return pd.concat([features] * copies, ignore_index=True).iloc[:n]


def bench_throughput(frame: pd.DataFrame, n: int, monitors: list[str]) -> dict:
    reference = frame[FEATURES].iloc[:REFERENCE_ROWS]
    batch = _batch(frame, n)
    measurements = 1 if n >= LARGE_ROWS else REPEATS
    warmup = 0 if n >= LARGE_ROWS else 1
    seconds: list[float] = []
    alerts = 0
    windows = 0
    for run in range(warmup + measurements):
        runner = SlidingWindowRunner(
            reference=reference,
            monitors=monitors,
            window_size=WINDOW_SIZE,
            sliding=True,
        )
        start = time.perf_counter()
        observed_windows = 0
        observed_alerts = 0
        for offset in range(0, n, CHUNK_SIZE):
            reports = runner.update(batch.iloc[offset : offset + CHUNK_SIZE])
            observed_windows += len(reports)
            observed_alerts += sum(len(report.alerts) for report in reports)
        reports = runner.update(pd.DataFrame(), final=True)
        observed_windows += len(reports)
        observed_alerts += sum(len(report.alerts) for report in reports)
        elapsed = time.perf_counter() - start
        if run < warmup:
            continue
        seconds.append(elapsed)
        alerts = observed_alerts
        windows = observed_windows
        assert windows == (n + WINDOW_SIZE - 1) // WINDOW_SIZE
    median = statistics.median(seconds)
    return {
        "n": n,
        "seconds": round(median, 4),
        "rows_per_sec": round(n / median, 1),
        "windows": windows,
        "alerts": alerts,
        "repeats": measurements,
        "seconds_all": [round(value, 4) for value in seconds],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    frame = load_cicids(args.path)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": DATASET_NAME,
        "rows": len(frame),
        "features": len(FEATURES),
        "monitors": MONITORS,
        "window_size": WINDOW_SIZE,
        "mode": "sliding",
        "reference_rows": REFERENCE_ROWS,
        "repeat_policy": "three timed runs after one warmup for 1k-100k; single run for 1M",
        "results": [bench_throughput(frame, n, MONITORS) for n in SIZES],
        "per_monitor": {
            name: bench_throughput(frame, PER_MONITOR_ROWS, [name]) for name in MONITORS
        },
    }
    (output / "throughput.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
