# Architecture

## Data flow

```
reference data  +  live batches
        |                |
        v                v
   +---------------------------+
   |   SlidingWindowRunner     |
   |  windowing + alerting     |
   +-------------+-------------+
                 |
        +--------+--------+
        | MonitorRegistry |
        +--------+--------+
                 |
   +-------------+-------------+
   | Tabular | Performance |   |
   | LLM monitors            |
   +-------------+-------------+
                 |
        +--------+--------+
        | CorruptionVector|
        | DamagePredictor |
        +--------+--------+
                 |
        +--------+--------+
        | Report          |
        | HTML / JSON     |
        +-----------------+
```

## Components

- **`Monitor`**: base class with `on_batch(reference, batch) -> MonitorResult`. A monitor
  may expose `expected_size` when its score depends on the expected batch size.
- **`MonitorResult`**: `name`, `score`, `alert`, `details`, `timestamp`, `threshold`,
  `status`, and `error`. `status` is `ok`, `skipped`, or `error`.
- **`MonitorRegistry`**: maps names to classes. `register_all()` installs the 13 built-ins
  when the package is imported. `register_monitor(name, cls)` adds custom monitors.
- **`SlidingWindowRunner`**: reads input, splits it into windows, runs monitors, resolves
  thresholds, collects alerts, and builds a `Report`. It also supports incremental
  `update` calls and `run_stream`.
- **`CorruptionVector`**: five displacement features computed from reference and batch:
  `drift_magnitude`, `missingness_delta`, `cardinality_shift`, `schema_change`, and
  `volume_delta`.
- **`DamagePredictor`**: linear model from a corruption vector to an estimated
  performance drop. Trained models are stored as JSON.
- **`LLMJudge`**: provider-agnostic judge with cache, sampling, retries, and token budget.
- **`Report`**: aggregates results and window summaries. Renders escaped HTML and JSON.

## Window semantics

`run(data)` treats the whole input as one batch unless `sliding=True`. With sliding
enabled, windows start at multiples of `stride`, and `include_partial` controls whether a
trailing short window is scored. The report separates `n_rows`, `n_rows_analyzed`, and
`n_rows_skipped`; overlapping windows are de-duplicated when counting coverage.

`update(batch)` appends rows to an internal buffer and returns one report per completed
window. `final=True` flushes the remaining rows as a partial window. `run_stream` yields
those reports as an iterator.

## Thresholds

Each monitor carries `threshold` where the score is directly comparable to one. Test-based
monitors (`ks`, `chi_square`) use `alpha` for the statistical decision plus a minimum
effect size, so a tiny but significant difference does not raise an alert.

If a monitor has no threshold and the runner has `alert_threshold`, the runner applies the
global fallback and recomputes the alert. Otherwise the monitor's own `alert` value is
kept.

## Error handling

Monitor exceptions become `MonitorResult(status="error")` entries. Damage predictor
failures are recorded in `Report.damage_error` rather than aborting a run. LLM judge
failures return per-item statuses (`error`, `budget`, `sampled`, `cached`) and the LLM
monitor skips scoring when no valid judgments remain.

## Dependencies and storage

Core dependencies are `numpy`, `pandas`, `scipy`, and `pyyaml`. The HTML report is built
with string formatting and `html.escape`, not Jinja2. Damage models are JSON, not pickle.
There is no database or built-in log sink. Reports are written to the paths you choose.
