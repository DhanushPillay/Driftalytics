# Benchmarking

All benchmark scripts live in `benchmarks/`. Install the project from source first:

```bash
pip install -e ".[dev]"
```

## Scripts

- `benchmarks/run_all.py`: synthetic throughput at 1k, 10k, and 100k rows.
- `benchmarks/real_data_drift.py`: windowed drift detection on NYC TLC trips.
- `benchmarks/real_benchmark.py`: clean versus corrupted detection and damage-predictor
  accuracy on held-out corruption scenarios.

## Synthetic throughput

```bash
python benchmarks/run_all.py --out results/
```

Configuration: 5,000-row reference, 2 numeric columns, monitors `psi`, `ks`, and
`missing_rate`, default window size. The batch is scored as one window.

| Rows | Seconds | Rows/sec | Alerts |
|---|---|---|---|
| 1,000 | 0.0041 | 244,642 | 0 |
| 10,000 | 0.0155 | 646,768 | 0 |
| 100,000 | 0.0161 | 6,202,512 | 0 |

These numbers vary between runs. Fixed pandas overhead dominates at small sizes, and
larger runs benefit from warm caches and vectorized operations. The 100k result is not
extrapolated to 1M rows.

## Real data

The real dataset is the NYC TLC yellow taxi parquet for January 2024, about 47.6 MB:

```bash
mkdir data/nyc_tlc
curl -L -o data/nyc_tlc/yellow_tripdata_2024-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
```

### Windowed drift detection

```bash
python benchmarks/real_data_drift.py --n 50000
```

The script splits the loaded rows into reference and live halves, applies scale and
missingness corruption to the live half, and runs 5 monitors over 10 windows of 5,000
rows.

Result: 50,000 rows, 10 windows, 39 alerts, 0.4994 seconds, 100,118.4 rows/sec. The final
window's monitor scores were:

| Monitor | Score | Threshold | Alert |
|---|---|---|---|
| `psi` | 0.2477 | 0.2 | yes |
| `ks` | 0.2199 | 0.05 | yes |
| `missing_rate` | 0.1526 | 0.05 | yes |
| `cardinality` | 3.3755 | 0.1 | yes |
| `drift_magnitude` | 0.1624 | 0.5 | no |

The cardinality alert is driven by `trip_distance`. The reference column has many repeated
rounded values, so a 5,000-row sample contains far fewer distinct values than 5,000.
Adding Gaussian noise makes nearly every value unique, which raises the score. The score
uses expected sample cardinality, so the window size alone is not the cause.

### Detection and damage prediction

```bash
python benchmarks/real_benchmark.py --n 50000
```

Method: shuffle 100,000 rows, use 50,000 for reference and 50,000 for test. Fit a linear
fare model on the reference with `trip_distance`, `trip_distance²`, `passenger_count`,
`payment_type`, `PULocationID`, and `DOLocationID`. The clean model reaches MAE 4.385,
RMSE 11.227, and R² 0.744 on the held-out half.

Detection uses 5,000-row windows and the monitors `psi`, `ks`, `wasserstein`,
`missing_rate`, `cardinality`, and `drift_magnitude`. The corrupted case scales trip
distance by 1.6 and makes 15% of passenger counts missing.

| Case | Any-alert rate |
|---|---|
| Clean windows | 0.0 |
| Corrupted windows | 1.0 |

Per-monitor alert rates on the corrupted run: `psi` 1.0, `ks` 1.0, `wasserstein` 1.0,
`missing_rate` 1.0, `cardinality` 0.0, `drift_magnitude` 0.0. The last two did not cross
their thresholds for this particular corruption. Report this as a scenario result, not a
general detection claim.

The script builds 26 corruption scenarios across distance scaling, distance noise,
passenger-count missingness, pickup-location shifts, payment-type shifts, and combined
cases. It fits the damage predictor on 18 scenarios and evaluates on 8:

| Metric | Value |
|---|---|
| MAE | 0.2536 |
| RMSE | 0.3140 |
| R² | 0.4299 |

The target is the relative RMSE increase over the clean model, clipped to `[0, 1]`. The
predictor is linear and trained on simulated corruptions. Eight test scenarios is a small
sample; the R² is indicative, not a guarantee.

## Reporting rules

- Report detection rate and false-positive rate together.
- Report throughput at several sizes and name the monitor set.
- Report hardware and Python version.
- Do not publish LLM judge-versus-human agreement unless a labeled subset exists. This
  repository does not include one.
- Do not claim comparisons against other tools unless the comparison was actually run.

## This machine

Windows 11, Python 3.13.5, consumer laptop hardware. Your numbers will differ.
