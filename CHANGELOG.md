# Changelog

## 0.5.0

### Changed

- Replaced the Insects benchmark with CICIDS2017: 2,827,677 network flows with 68 numeric
  features, real timestamps, a benign-only 3 July control period, and natural attacks on
  4-7 July.
- `drift_magnitude` is now the maximum Population Stability Index across numeric columns
  instead of a mean standardized mean shift. The old metric let near-zero-variance columns
  dominate and ignored distribution shape.
- `KSMonitor` uses the asymptotic KS test when either sample exceeds 1,000 rows and caches
  reference arrays between windows. The exact test remains for small samples.
- `WassersteinMonitor` treats a reference scale below `1e-12` as degenerate instead of
  dividing by a near-zero value.
- The detection benchmark now uses per-day exchangeable benign reference/control splits plus
  chronological same-day live windows. It reports default-threshold alert rates, score AUC,
  and recall at a 5% control false-positive rate.
- The damage benchmark trains on 3 July benign traffic plus 4 July attacks, scores a
  held-out 3 July benign control, and streams 5-7 July. It reports raw and balanced damage,
  prior-only and drift-only baselines, an untrained heuristic, and bootstrap confidence
  intervals.

### Added

- `load_cicids`, which downloads the CICIDS2017 parquet, verifies its SHA-256, parses the
  mixed timestamp formats, sorts by time, and validates the schema and row counts.
- `results/cicids_detection_windows.csv` and `results/cicids_damage_windows.csv`.
- Tests for CICIDS parsing, the KS asymptotic and cache paths, near-zero-scale corruption
  and Wasserstein behavior, and `exact_threshold` validation.

### Removed

- The Insects loader, cached CSV, and `results/insects_damage_windows.csv`.
- The cached NYC TLC parquet.

### Verification

- `pytest`: 47 passed.
- `ruff check src tests benchmarks`: passed.
- `ruff format --check src tests benchmarks`: passed.
- `mypy src`: passed, 14 source files.
- CLI version: 0.5.0.
- Throughput (68 features, 3 monitors, streaming): 11,302.9 / 12,319.2 / 12,099.0 rows/sec
  at 1k / 10k / 100k rows, and 6,356.7 rows/sec at 1M rows.
- Detection: 0 alerts across 200 exchangeable control windows; pooled AUC 0.9878 (`psi`)
  and 0.9940 (`ks`); recall at 5% control FPR 1.0000 for both; live benign alert rate
  0.9412, which reflects temporal benign drift.
- Damage: held-out test MAE 0.1097 and R² 0.7503 for the five-feature model; drift-only
  MAE 0.0951 and R² 0.9252; label-derived prior oracle MAE 0.0604 and R² 0.9901;
  balanced-target model R² -0.0204.

## 0.4.0

### Changed

- All tabular benchmarks now use the real Insects `incremental_abrupt_balanced` stream
  instead of synthetic normal rows or NYC TLC rows with injected corruption.
- Damage labels are observed chronological classification-error increases from a logistic
  regression trained on the first 20,000 rows.
- The damage predictor uses a chronological 60/20/20 split instead of a random split.
- Benchmark reporting includes label-aware detection precision and recall plus a
  drift-only ablation.

### Added

- `src/driftalytics/datasets.py`, which downloads, SHA-256 verifies, caches, and validates
  the Insects CSV.
- Optional `bench` extra with `pyarrow` and `scikit-learn`. Core runtime dependencies are
  unchanged.
- `results/insects_damage_windows.csv` with the per-window corruption features and damage
  labels.

### Removed

- Synthetic throughput data and synthetic damage training (`make_synthetic`, `--n`, and
  `--seed`).
- The NYC TLC controlled-corruption benchmark and `results/real_damage_scenarios.csv`.

### Verification

- `pytest`: 43 passed.
- `ruff check src tests benchmarks`: passed.
- `ruff format --check src tests benchmarks`: passed.
- `mypy src`: passed, 14 source files.
- CLI version: 0.4.0.
- Throughput (33 features, 3 monitors): 9,495.7 / 6,087.9 / 3,544.3 rows/sec at 1k / 10k /
  100k rows.
- Natural drift (59,986 live rows, 5 monitors): 119 windows, 231 alerts, 2,806.2 rows/sec.
- Damage predictor: test MAE 0.0405 and R-squared 0.9525, against 0.1843 and 0.4198 for the
  untrained heuristic. The drift-only ablation reaches test MAE 0.0385.

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
