"""Natural attack-window detection on the CICIDS2017 stream.

Each day gets an exchangeable benign reference and control split. The rest of
the day is streamed chronologically in fixed windows. A window is labelled as
an attack window when at least half of its rows are attacks. The benchmark
reports the default-threshold alert rate, score AUC, and recall at a 5% control
false-positive rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from driftalytics import SlidingWindowRunner
from driftalytics.datasets import DATASET_NAME, FEATURES, load_cicids

MONITORS = ["psi", "ks", "missing_rate"]
WINDOW_SIZE = 1_000
REFERENCE_ROWS = 50_000
CONTROL_ROWS = 50_000
CONTROL_FPR = 0.05
DAYS = ("2017-07-04", "2017-07-05", "2017-07-06", "2017-07-07")
SEED = 0


def _score(result) -> float:
    return float(result.score)


def _window_rows(report, live: pd.DataFrame, day: str) -> list[dict]:
    rows = []
    for window in report.window_summaries:
        start = int(window["start"])
        end = int(window["end"])
        attack_fraction = float(live["label"].iloc[start:end].mean())
        scores = {result.name: _score(result) for result in window["results"]}
        rows.append(
            {
                "day": day,
                "window": int(window["window"]),
                "start": start,
                "end": end,
                "n_rows": int(window["n_rows"]),
                "attack_fraction": round(attack_fraction, 6),
                "alerted": bool(window["alerts"]),
                "alerts": sorted({alert.monitor for alert in window["alerts"]}),
                **{
                    f"{name}_alerted": any(alert.monitor == name for alert in window["alerts"])
                    for name in MONITORS
                },
                **{f"{name}_score": round(score, 6) for name, score in scores.items()},
            }
        )
    return rows


def _summary(rows: pd.DataFrame, control: pd.DataFrame, monitors: list[str]) -> dict:
    benign = rows[rows["attack_fraction"] == 0.0]
    attack = rows[rows["attack_fraction"] >= 0.5]
    mixed = len(rows) - len(benign) - len(attack)
    control_windows = len(control)

    per_monitor = {}
    for name in monitors:
        score_column = f"{name}_score"
        control_scores = control[score_column].to_numpy(dtype=float)
        threshold = float(np.quantile(control_scores, 1.0 - CONTROL_FPR))
        attack_scores = attack[score_column].to_numpy(dtype=float)
        benign_scores = benign[score_column].to_numpy(dtype=float)
        auc = None
        if len(attack_scores) and len(benign_scores):
            auc = float(
                roc_auc_score(
                    np.concatenate([np.ones(len(attack_scores)), np.zeros(len(benign_scores))]),
                    np.concatenate([attack_scores, benign_scores]),
                )
            )
        per_monitor[name] = {
            "control_alert_rate": round(
                float(control[f"{name}_alerted"].mean()) if control_windows else 0.0, 6
            ),
            "live_benign_alert_rate": round(
                float(benign[f"{name}_alerted"].mean()) if len(benign) else 0.0, 6
            ),
            "attack_alert_rate": round(
                float(attack[f"{name}_alerted"].mean()) if len(attack) else 0.0, 6
            ),
            "auc": round(auc, 6) if auc is not None else None,
            "threshold_at_5pct_fpr": round(threshold, 6),
            "recall_at_5pct_fpr": round(
                float((attack_scores > threshold).mean()) if len(attack_scores) else 0.0, 6
            ),
        }

    return {
        "windows": len(rows),
        "attack_windows": len(attack),
        "benign_windows": len(benign),
        "mixed_windows": mixed,
        "control_windows": control_windows,
        "control_alert_rate": round(
            float(control["alerted"].mean()) if control_windows else 0.0, 6
        ),
        "live_benign_alert_rate": round(float(benign["alerted"].mean()) if len(benign) else 0.0, 6),
        "attack_alert_rate": round(float(attack["alerted"].mean()) if len(attack) else 0.0, 6),
        "per_monitor": per_monitor,
    }


def _control_rows(report, day: str) -> list[dict]:
    rows = []
    for window in report.window_summaries:
        scores = {result.name: _score(result) for result in window["results"]}
        rows.append(
            {
                "day": day,
                "window": int(window["window"]),
                "start": int(window["start"]),
                "end": int(window["end"]),
                "alerted": bool(window["alerts"]),
                "alerts": sorted({alert.monitor for alert in window["alerts"]}),
                **{
                    f"{name}_alerted": any(a.monitor == name for a in window["alerts"])
                    for name in MONITORS
                },
                **{f"{name}_score": round(score, 6) for name, score in scores.items()},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    frame = load_cicids(args.path)
    day_values = frame["timestamp"].dt.strftime("%Y-%m-%d")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    all_live: list[dict] = []
    all_control: list[dict] = []
    per_day = {}
    for index, day in enumerate(DAYS):
        day_frame = frame[day_values == day].reset_index(drop=True)
        benign = day_frame[day_frame["label"] == 0]
        if len(benign) < REFERENCE_ROWS + CONTROL_ROWS:
            raise ValueError(f"{day} has only {len(benign)} benign rows")

        rng = np.random.default_rng(SEED + index)
        order = rng.permutation(len(benign))
        reference = benign.iloc[order[:REFERENCE_ROWS]]
        control = benign.iloc[order[REFERENCE_ROWS : REFERENCE_ROWS + CONTROL_ROWS]]
        excluded = benign.index.to_numpy()[order[: REFERENCE_ROWS + CONTROL_ROWS]]
        keep = np.ones(len(day_frame), dtype=bool)
        keep[excluded] = False
        live = day_frame.iloc[keep].reset_index(drop=True)

        runner = SlidingWindowRunner(
            reference=reference[FEATURES],
            monitors=MONITORS,
            window_size=WINDOW_SIZE,
            stride=WINDOW_SIZE,
            sliding=True,
            include_partial=False,
        )
        live_report = runner.run(live[FEATURES])
        control_runner = SlidingWindowRunner(
            reference=reference[FEATURES],
            monitors=MONITORS,
            window_size=WINDOW_SIZE,
            stride=WINDOW_SIZE,
            sliding=True,
            include_partial=False,
        )
        control_report = control_runner.run(control[FEATURES])

        live_rows = _window_rows(live_report, live, day)
        control_rows = _control_rows(control_report, day)
        all_live.extend(live_rows)
        all_control.extend(control_rows)
        per_day[day] = {
            "rows": len(day_frame),
            "benign_rows": len(benign),
            "attack_rows": int(day_frame["label"].sum()),
            "attack_rate": round(float(day_frame["label"].mean()), 6),
            "live_rows": len(live),
            "live_attack_rate": round(float(live["label"].mean()), 6),
            "first_attack": str(day_frame.loc[day_frame["label"] == 1, "timestamp"].min()),
            **_summary(pd.DataFrame(live_rows), pd.DataFrame(control_rows), MONITORS),
        }
        print(day, json.dumps(per_day[day], indent=2))

    live_frame = pd.DataFrame(all_live)
    control_frame = pd.DataFrame(all_control)
    payload = {
        "dataset": DATASET_NAME,
        "total_rows": len(frame),
        "window_size": WINDOW_SIZE,
        "reference_rows_per_day": REFERENCE_ROWS,
        "control_rows_per_day": CONTROL_ROWS,
        "control_fpr": CONTROL_FPR,
        "days": list(DAYS),
        "per_day": per_day,
        "pooled": _summary(live_frame, control_frame, MONITORS),
    }
    (output / "real_data_drift.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    live_frame.to_csv(output / "cicids_detection_windows.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
