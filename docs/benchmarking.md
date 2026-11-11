# Benchmarking

All benchmark scripts live in `benchmarks/`. Install the package with the benchmark
dependencies:

```bash
pip install -e ".[bench]"
```

The `bench` extra adds `pyarrow` and `scikit-learn`. The `dev` extra adds the test, lint,
and type-check tools.

## Dataset

All tabular benchmarks use the Insects `incremental_abrupt_balanced` stream from the USP
Data Stream Repository. It contains real optical-sensor measurements of disease-vector
insects with 33 numeric features and six classes. The row order is curated by the dataset
authors to include incremental, abrupt, and reoccurring drift. It is not a calendar-time
stream.

| Property | Value |
|---|---|
| Rows | 79,986 |
| Columns | 34 (`f1` through `f33`, plus `class`) |
| Classes | 2, 3, 4, 5, 11, 12 |
| File size | 21,421,452 bytes |
| License | CC BY 4.0 |
| Citation | Souza et al. 2020, DOI `10.1007/s10618-020-00698-5` |
| Cache | `data/raw/insects/insects_incremental_abrupt_balanced.csv` |
| SHA-256 | `c1cd19d429844349009311fa372ee700d5bf0e21545fb61d0d3fba724123188b` |

`src/driftalytics/datasets.py` downloads the headerless CSV on demand, writes it through an
atomic `.part` file, verifies the SHA-256, and caches it under `data/raw/`. The dataset is
not committed. Pass `--path` to use an existing local copy without downloading.

## Scripts

- `benchmarks/run_all.py`: real-data throughput at 1k, 10k, and 100k rows.
- `benchmarks/real_data_drift.py`: natural drift detection on the live stream.
- `benchmarks/real_benchmark.py`: classifier damage, detection utility, and damage-predictor
  accuracy.
- `src/driftalytics/train_damage.py`: train and save a damage predictor from a labeled CSV
  or Parquet file.

## Real throughput

```bash
python benchmarks/run_all.py --out results/
```

Configuration: 5,000 reference rows, 33 numeric features, window size 1,000, sliding
windows, monitors `psi`, `ks`, and `missing_rate`. The batch is cycled from the same stream
to reach each size. The script discards one warmup run and reports the median of three timed
runs.

| Rows | Seconds | Rows/sec | Alerts |
|---|---|---|---|
| 1,000 | 0.1053 | 9,496 | 1 |
| 10,000 | 1.6426 | 6,088 | 15 |
| 100,000 | 28.2142 | 3,544 | 176 |

These numbers are much lower than the earlier synthetic two-column table. The workload now
has 33 real features and sliding windows, so the old 233k to 6.9M rows/sec figures are not
comparable.

## Natural drift detection

```bash
python benchmarks/real_data_drift.py --out results/
```

The script uses the first 20,000 rows as reference and the remaining 59,986 as live data. It
runs 500-row sliding windows with the monitors `psi`, `ks`, `missing_rate`, `cardinality`,
and `drift_magnitude`. No corruption is injected. The trailing 486 rows do not fill a window
and are skipped.

Result: 59,986 live rows, 119 windows, 21.3763 seconds, 2,806.2 rows/sec, 231 alerts, and an
alert in every window. The first alert is in window 0.

| Monitor | Alerts | Alert rate |
|---|---|---|
| `psi` | 93 | 0.7815 |
| `ks` | 119 | 1.0000 |
| `missing_rate` | 0 | 0.0000 |
| `cardinality` | 0 | 0.0000 |
| `drift_magnitude` | 19 | 0.1597 |

The first reference block is already non-stationary. There is no clean control segment, so
the benchmark cannot report a meaningful false-positive rate. Do not describe these windows
as clean.

## Detection utility

```bash
python benchmarks/real_benchmark.py --out results/
```

The script trains a logistic regression on the 20,000 reference rows, evaluates it on each
500-row live window, and measures classification error and log-loss. Damage is the absolute
error increase over the reference period, clipped to `[0, 1]`. The reference model reaches
accuracy 0.63985, error 0.36015, and log-loss 0.8880.

Severe damage means a damage above 0.5. That covers 22 of 119 windows.

| Monitor | Alert rate | Precision | Recall |
|---|---|---|---|
| `psi` | 0.7815 | 0.2366 | 1.0000 |
| `ks` | 1.0000 | 0.1849 | 1.0000 |
| `wasserstein` | 1.0000 | 0.1849 | 1.0000 |
| `missing_rate` | 0.0000 | 0.0000 | 0.0000 |
| `cardinality` | 0.0000 | 0.0000 | 0.0000 |
| `drift_magnitude` | 0.1597 | 1.0000 | 0.8636 |

`drift_magnitude` is the only high-precision monitor. PSI and KS catch every severe window
but alert on most windows. Report the precision and recall trade-off instead of a single
detection rate.

## Damage prediction

The damage target is the observed classification-error increase over the reference period.
Windows are split chronologically 60/20/20 into 73 train, 23 validation, and 23 test
windows. The five-feature predictor uses `drift_magnitude`, `missingness_delta`,
`cardinality_shift`, `schema_change`, and `volume_delta`.

| Predictor | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Five-feature model | 0.0224 | 0.0405 | 0.0536 | 0.9525 |
| Drift-only OLS | 0.0284 | 0.0385 | 0.0501 | 0.9605 |
| Untrained heuristic | 0.0410 | 0.1843 | 0.2455 | 0.4198 |

The five-feature model reaches MAE 0.0530, RMSE 0.0850, and R-squared 0.8790 on the training
windows. Validation R-squared is numerically unstable because validation damage has low
variance; compare MAE and RMSE there.

Only `drift_magnitude` and `cardinality_shift` vary in this stream. The other three
corruption features are constant zero, and their fitted coefficients are zero. The
drift-only ablation is slightly better on test (MAE 0.0385 versus 0.0405), so this benchmark
validates drift and cardinality damage, not missingness, schema, or volume. The five-feature
model still beats the untrained heuristic by a wide margin.

The script writes the per-window training data to `results/insects_damage_windows.csv`. Train
a standalone model with:

```bash
python -m driftalytics.train_damage \
  --data results/insects_damage_windows.csv \
  --out results/damage_predictor_real.json
```

`train_damage` requires the five corruption features plus a `damage` column, rejects missing
values, and uses a chronological split. It no longer generates synthetic data.

## Reporting rules

- Do not report a false-positive rate unless the benchmark has a stationary control segment.
- Report precision and recall against realized damage, not only an any-alert rate.
- Disclose that the Insects stream order is curated. Do not call it calendar-time data.
- Report the model, the untrained heuristic, and the drift-only ablation together.
- Report throughput at several sizes, name the monitor set, and state the hardware and
  Python version.
- Do not publish LLM judge-versus-human agreement unless a labeled subset exists. This
  repository does not include one.
- Do not claim comparisons against other tools unless the comparison was actually run.

## This machine

Windows 11, Python 3.13.5, consumer laptop hardware. Your numbers will differ.
