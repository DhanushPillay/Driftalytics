"""Throughput benchmark on the Insects real dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd

from driftalytics import SlidingWindowRunner
from driftalytics.datasets import DATASET_NAME, FEATURES, load_insects

MONITORS = ["psi", "ks", "missing_rate"]
WINDOW_SIZE = 1_000
REFERENCE_ROWS = 5_000
REPEATS = 3
SIZES = (1_000, 10_000, 100_000)


def _batch(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    features = frame[FEATURES]
    copies = (n + len(features) - 1) // len(features)
    return pd.concat([features] * copies, ignore_index=True).iloc[:n]


def bench_throughput(frame: pd.DataFrame, n: int) -> dict:
    reference = frame[FEATURES].iloc[:REFERENCE_ROWS]
    batch = _batch(frame, n)
    seconds: list[float] = []
    alerts = 0
    for run in range(REPEATS + 1):
        runner = SlidingWindowRunner(
            reference=reference,
            monitors=MONITORS,
            window_size=WINDOW_SIZE,
            sliding=True,
        )
        start = time.perf_counter()
        report = runner.run(batch)
        elapsed = time.perf_counter() - start
        assert report.n_rows == n
        if run == 0:
            continue
        seconds.append(elapsed)
        alerts = len(report.alerts)
    median = statistics.median(seconds)
    return {
        "n": n,
        "seconds": round(median, 4),
        "rows_per_sec": round(n / median, 1),
        "alerts": alerts,
        "seconds_all": [round(value, 4) for value in seconds],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    frame = load_insects(args.path)
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
        "repeats": REPEATS,
        "results": [bench_throughput(frame, n) for n in SIZES],
    }
    (output / "throughput.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
