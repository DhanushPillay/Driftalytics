# Changelog

## 0.3.0

### Fixed

- `run()` now scores the whole input as one window when `sliding=False`, including inputs
  shorter than `window_size`.
- Sliding runs emit full windows at absolute offsets plus at most one partial tail. The tail
  starts at the later of the last full window's end and the next scheduled stride, so gaps
  are skipped and short strides do not re-score rows.
- Partial windows use their actual row count as `expected_size`, preventing false volume and
  cardinality alerts.
- Streaming tracks absolute offsets, dropped gap rows, and coverage. `final=True` emits at
  most one allowed partial window; repeated finalization returns `[]`, and new data after
  finalization raises `RuntimeError`.
- `Report` and `Alert` carry `start`/`end` offsets. Analyzed rows use union coverage rather
  than summing overlapping windows.
- Report JSON converts non-finite values to `null` and rejects NaN; HTML renders them as
  `n/a`.
- LLM prompts serialize untrusted fields as JSON, use a separate system instruction, and
  include a prompt version in the cache key along with provider, model, and metrics.
- LLM budget is reserved before a request and charged on failure. Retries use jitter and
  honor `Retry-After`. All providers receive an output cap, and malformed provider responses
  are rejected with clear errors.
- Damage training, validation, and test metrics are stored separately. `fit()` reports
  training metrics, and loading validates coefficients, intercept, and metrics.
- `train_damage` uses train/validation/test splits and rejects missing training values
  instead of silently dropping rows.
- CLI boolean keys accept only real booleans or `true`/`false` strings. Monitor names and
  output paths are validated.
- Monitor thresholds are validated at construction instead of using `assert`.
  `expected_size=0` is treated as a real value.

### Added

- `Report.start`, `Report.end`, `Alert.start`, and `Alert.end`.
- `DamagePredictor.training_metrics_`, `validation_metrics_`, and `test_metrics_`, saved as
  additive fields in the existing `driftalytics-damage-v1` format.
- `LLMJudge.max_output_tokens`, `PROMPT_VERSION`, and provider output caps.
- Optional `parquet` extra (`pyarrow>=14`); `pyarrow` is also in `dev`.
- Python 3.13 classifier and GitHub Actions CI for Python 3.10-3.13.
- Regression tests for runner boundaries, stream finalization, LLM injection/caching/budget,
  report JSON validity, damage validation, monitor edge cases, and CLI config errors.
- A family-aware damage benchmark split with a heuristic baseline.

### Changed

- Damage benchmark reporting now uses train (distance), validation (missingness), and test
  (location/payment/combined) families instead of a random scenario split. The previous
  split leaked related corruptions across train and test.
- `benchmarks/real_benchmark.py` writes split sizes, target spread, and heuristic metrics.

### Verification

- `pytest`: 41 passed.
- `ruff check src tests benchmarks`: passed.
- `ruff format --check src tests benchmarks`: passed.
- `mypy src`: passed, 13 source files.
- Synthetic throughput (3 monitors): 233,590 / 600,010 / 6,961,994 rows/sec at 1k / 10k /
  100k rows.
- Real NYC TLC detection (50k rows, 5 monitors): 10 windows, 39 alerts, 80,866 rows/sec;
  clean FPR 0.0, corrupted detection 1.0.
- Family-holdout damage predictor: train MAE 0.1892, validation MAE 0.2643, test MAE
  0.3446. The untrained heuristic reaches validation MAE 0.0687 and test MAE 0.3087, so the
  linear model does not beat the heuristic on held-out families.

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
