"""Natural drift detection on the Insects real dataset.

The first 20,000 rows form the reference period. The remaining rows are
streamed in 500-row windows without any injected corruption.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from driftalytics import SlidingWindowRunner
from driftalytics.datasets import DATASET_NAME, FEATURES, load_insects

MONITORS = ["psi", "ks", "missing_rate", "cardinality", "drift_magnitude"]
REFERENCE_ROWS = 20_000
WINDOW_SIZE = 500


def _result_summary(result) -> dict:
    return {
        "score": None if result.score is None else round(float(result.score), 6),
        "threshold": None if result.threshold is None else round(float(result.threshold), 6),
        "alert": bool(result.alert),
        "status": result.status,
        "error": result.error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    frame = load_insects(args.path)
    if not 0 < REFERENCE_ROWS < len(frame):
        raise ValueError("reference period must leave live rows")

    reference = frame.iloc[:REFERENCE_ROWS]
    live = frame.iloc[REFERENCE_ROWS:]
    runner = SlidingWindowRunner(
        reference=reference[FEATURES],
        monitors=MONITORS,
        window_size=WINDOW_SIZE,
        stride=WINDOW_SIZE,
        sliding=True,
        include_partial=False,
    )

    start = time.perf_counter()
    report = runner.run(live[FEATURES])
    elapsed = time.perf_counter() - start

    windows = report.window_summaries
    alert_counts = Counter(alert.monitor for alert in report.alerts)
    per_monitor = {
        name: {
            "alerts": alert_counts.get(name, 0),
            "alert_rate": round(alert_counts.get(name, 0) / max(len(windows), 1), 4),
        }
        for name in MONITORS
    }
    timeline = [
        {
            "window": int(window["window"]),
            "start": REFERENCE_ROWS + int(window["start"]),
            "end": REFERENCE_ROWS + int(window["end"]),
            "n_rows": int(window["n_rows"]),
            "alerts": [alert.monitor for alert in window["alerts"]],
            "results": {result.name: _result_summary(result) for result in window["results"]},
        }
        for window in windows
    ]
    alerting_windows = sum(1 for window in windows if window["alerts"])
    payload = {
        "dataset": DATASET_NAME,
        "total_rows": len(frame),
        "reference_rows": len(reference),
        "rows": len(live),
        "window_size": WINDOW_SIZE,
        "windows": len(windows),
        "rows_skipped": report.n_rows_skipped,
        "seconds": round(elapsed, 4),
        "rows_per_sec": round(len(live) / elapsed, 1),
        "alerts": len(report.alerts),
        "alerting_windows": alerting_windows,
        "alert_rate": round(alerting_windows / max(len(windows), 1), 4),
        "first_alert_window": min((alert.window for alert in report.alerts), default=None),
        "per_monitor": per_monitor,
        "reference_class_counts": {
            str(int(key)): int(value)
            for key, value in reference["class"].value_counts().sort_index().items()
        },
        "live_class_counts": {
            str(int(key)): int(value)
            for key, value in live["class"].value_counts().sort_index().items()
        },
        "timeline": timeline,
    }

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "real_data_drift.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report.summary())
    print(json.dumps({key: value for key, value in payload.items() if key != "timeline"}, indent=2))


if __name__ == "__main__":
    main()
