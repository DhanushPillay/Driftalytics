"""Drift detection on NYC TLC yellow taxi trips.

Expects data/nyc_tlc/yellow_tripdata_2024-01.parquet.
Download:
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from driftalytics import SlidingWindowRunner

DATA = Path("data/nyc_tlc/yellow_tripdata_2024-01.parquet")
COLUMNS = [
    "trip_distance",
    "fare_amount",
    "passenger_count",
    "payment_type",
    "PULocationID",
    "DOLocationID",
    "total_amount",
]


def load(n: int = 200_000) -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Download from "
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
        )
    frame = pd.read_parquet(DATA)
    return frame[COLUMNS].dropna().head(n).reset_index(drop=True)


def inject_drift(frame: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output = frame.copy()
    output["trip_distance"] = output["trip_distance"] * 1.6 + rng.normal(0, 0.5, len(output))
    output["fare_amount"] = output["fare_amount"] * 1.4
    mask = rng.random(len(output)) < 0.15
    output.loc[mask, "passenger_count"] = np.nan
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--window-size", type=int, default=5000)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    frame = load(args.n * 2)
    reference = frame.iloc[: args.n].reset_index(drop=True)
    live = inject_drift(frame.iloc[args.n : args.n * 2].reset_index(drop=True))

    runner = SlidingWindowRunner(
        reference=reference,
        monitors=["psi", "ks", "missing_rate", "cardinality", "drift_magnitude"],
        window_size=args.window_size,
        sliding=True,
        stride=args.window_size,
        include_partial=False,
    )
    start = time.perf_counter()
    report = runner.run(live)
    elapsed = time.perf_counter() - start
    summary = {
        "rows": len(live),
        "window_size": args.window_size,
        "seconds": round(elapsed, 4),
        "rows_per_sec": round(len(live) / elapsed, 1),
        "windows": report.n_windows,
        "alerts": len(report.alerts),
        "results": [
            {
                "monitor": result.name,
                "score": round(result.score, 6),
                "threshold": result.threshold,
                "alert": result.alert,
                "status": result.status,
            }
            for result in report.results
        ],
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "real_data_drift.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(report.summary())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
