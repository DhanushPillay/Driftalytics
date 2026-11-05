# Changelog

## 0.2.0

### Fixed

- Per-monitor thresholds are now authoritative; the global `alert_threshold` only fills
  in for monitors without one.
- Cardinality compares against the expected distinct count for the window size, not the
  full reference count.
- KS and chi-square monitors apply Benjamini-Hochberg correction and a minimum effect
  size before alerting.
- Non-numeric and type-drifted columns no longer crash numeric monitors.
- Missing or invalid monitor inputs return `status="error"` or `status="skipped"` instead
  of fabricated scores.
- Overlapping windows no longer over-count analyzed rows in report totals.
- LLM judge parsing, budget, sampling, and provider failures are reported per item.
- HTML reports escape all dynamic values.

### Added

- `Report.n_rows_analyzed`, `n_rows_skipped`, `window_summaries`, and per-window details.
- `SlidingWindowRunner.update(batch)` and `run_stream(batches)` for streaming input.
- `stride` and `include_partial` window controls.
- `CorruptionVector.as_dict()` and expected-sample cardinality support.
- JSON damage model format (`driftalytics-damage-v1`) with validation on load.
- `DamagePredictor.evaluate()` with MAE, RMSE, and R².
- `benchmarks/real_benchmark.py` for detection and damage-prediction measurement.
- CLI `monitors` subcommand.

### Removed

- Jinja2, pickle, and scikit-learn dependencies.
- Unimplemented LLM batching and the nonexistent `[llm]` extra.
- `src/driftalytics/templates/`.

### Verification

- `pytest`: 24 passed.
- `ruff check src tests benchmarks`: passed.
- `mypy src`: passed, 13 source files.
- Synthetic throughput (3 monitors): 244,642 / 646,768 / 6,202,512 rows/sec at 1k / 10k /
  100k rows.
- Real NYC TLC detection (50k rows, 6 monitors): clean FPR 0.0, corrupted detection 1.0.
- Real damage predictor (8 held-out scenarios): MAE 0.254, RMSE 0.314, R² 0.430.

## 0.1.0

- Initial package layout, CLI, monitor registry, tabular/performance/LLM monitors.
- Sliding-window runner, corruption-vector damage predictor, LLM judge, HTML/JSON reports.
- Documentation and benchmark harness.
