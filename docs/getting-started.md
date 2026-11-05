# Getting started

## Install

Driftalytics is not published on PyPI. Install from a source checkout:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -e ".[dev]"
```

The core runtime needs Python 3.10+, `numpy`, `pandas`, `scipy`, and `pyyaml`. There is
no separate LLM extra; the judge talks to provider HTTP APIs directly.

## Batch example

```python
import pandas as pd
from driftalytics import SlidingWindowRunner

reference = pd.read_parquet("reference.parquet")
live = pd.read_parquet("live.parquet")

runner = SlidingWindowRunner(
    reference=reference,
    monitors=["psi", "ks", "missing_rate", "cardinality"],
    window_size=1000,
)
report = runner.run(live)
print(report.summary())
report.to_html("drift_report.html")
report.to_json("drift_report.json")
```

Monitors use their own thresholds by default. A global `alert_threshold` is only applied
to monitors that have no threshold of their own.

## Sliding windows

```python
runner = SlidingWindowRunner(
    reference=reference,
    monitors=["psi", "ks", "missing_rate"],
    window_size=5000,
    stride=2500,
    sliding=True,
    include_partial=True,
)
report = runner.run(live)
print(report.n_windows, report.n_rows_analyzed, report.n_rows_skipped)
```

`stride` defaults to `window_size`. Overlapping windows are counted once when the report
computes analyzed and skipped rows. `include_partial=True` scores a final short window
instead of dropping it.

## Streaming

```python
runner = SlidingWindowRunner(
    reference=reference,
    monitors=["psi", "ks"],
    window_size=1000,
    sliding=True,
)

for batch in batches:
    for report in runner.update(batch):
        print(report.summary())

for report in runner.update(pd.DataFrame(), final=True):
    print(report.summary())
```

`run_stream(batches)` yields the same reports from an iterable. The runner buffers rows
until a full window is available.

## Custom monitor

```python
from driftalytics import Monitor, MonitorResult, register_monitor

class RowCountMonitor(Monitor):
    name = "row_count"

    def on_batch(self, reference, batch):
        delta = abs(len(batch) - len(reference)) / max(len(reference), 1)
        return MonitorResult(
            name=self.name,
            score=delta,
            alert=delta > 0.5,
            threshold=0.5,
        )

register_monitor("row_count", RowCountMonitor)
```

If `on_batch` raises, the runner records a result with `status="error"` and the exception
message. It does not abort the remaining monitors.

## Configuration file

```yaml
# config.yaml
reference: reference.parquet
data: live.parquet
monitors: [psi, ks, missing_rate, cardinality, drift_magnitude]
window_size: 1000
stride: 500
sliding: true
include_partial: true
alert_threshold: 0.1        # fallback only, for monitors without a threshold
output:
  html: drift_report.html
  json: drift_report.json
```

```bash
driftalytics run config.yaml
driftalytics monitors
```

To attach a trained damage model, add a JSON model path:

```yaml
damage_predictor: models/damage_predictor.json
```

Train one with:

```bash
python -m driftalytics.train_damage --out models/damage_predictor.json
```

## Next steps

- Read [Architecture](architecture.md) for the data flow.
- See [Monitors](monitors.md) for thresholds and per-monitor behavior.
- See [LLM evaluation](llm-eval.md) for RAG and LLM judging.
- See [Benchmarking](benchmarking.md) to reproduce the published numbers.
