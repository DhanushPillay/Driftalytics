"""Synthetic throughput benchmark."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from driftalytics import SlidingWindowRunner


def bench_throughput(n: int) -> dict:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(size=5000)})
    batch = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    runner = SlidingWindowRunner(
        reference=reference,
        monitors=["psi", "ks", "missing_rate"],
        window_size=1000,
    )
    start = time.perf_counter()
    report = runner.run(batch)
    elapsed = time.perf_counter() - start
    assert report.n_rows == n
    return {
        "n": n,
        "seconds": round(elapsed, 4),
        "rows_per_sec": round(n / elapsed, 1),
        "alerts": len(report.alerts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    results = [bench_throughput(n) for n in [1_000, 10_000, 100_000]]
    (output / "throughput.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
