# Benchmarking

All benchmark scripts live in `benchmarks/`. Install the package with the benchmark
dependencies:

```bash
pip install -e ".[bench]"
```

The `bench` extra adds `pyarrow` and `scikit-learn`. The `dev` extra adds the test, lint,
and type-check tools.

## Dataset

All tabular benchmarks use CICIDS2017 network-flow data. The copy is the
`rdpahalavan/CIC-IDS2017` Hugging Face packaging of the Canadian Institute for Cybersecurity
dataset. It contains real captured traffic with calendar timestamps, a benign-only first day,
and natural attacks over the following four days.

| Property | Value |
|---|---|
| Rows | 2,827,677 |
| Features | 68 numeric flow features |
| Target | Binary `label` (`0` benign, `1` attack) plus `attack_label` |
| Period | 2017-07-03 to 2017-07-07 |
| Attack rows | 556,555 |
| File size | 369,814,810 bytes |
| License | Apache-2.0 for the Hugging Face packaging |
| Citation | Sharafaldin et al. 2018, DOI `10.5220/0006639801080116` |
| Cache | `data/raw/cicids2017/CICIDS_Flow.parquet` |
| SHA-256 | `e64f6adfbcb7f96b018cf36c731be64ef4de3353ccd7340b190590734a0dda35` |

The loader drops flow identifiers, IP addresses, ports, protocol, and eight constant columns,
leaving 68 numeric features. It parses the mixed-format timestamps with `dayfirst=True`,
maps benign rows to `0` and every attack label to `1`, sorts by timestamp, converts features
to `float32`, and validates the expected row and attack counts. `src/driftalytics/datasets.py`
downloads the Parquet file on demand, writes it through an atomic `.part` file, verifies the
SHA-256, and caches it under `data/raw/`. The dataset is not committed. Pass `--path` to use
an existing local copy without downloading.

Daily composition:

| Day | Rows | Attacks | Attack rate | Attack types |
|---|---:|---:|---:|---|
| 2017-07-03 | 529,450 | 0 | 0.0000 | none |
| 2017-07-04 | 445,641 | 13,832 | 0.0310 | FTP-Patator, SSH-Patator |
| 2017-07-05 | 691,390 | 251,722 | 0.3641 | DoS Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed |
| 2017-07-06 | 458,483 | 2,216 | 0.0048 | web attacks, Infiltration |
| 2017-07-07 | 702,713 | 288,785 | 0.4110 | Bot, DDoS, PortScan |

## Scripts

- `benchmarks/run_all.py`: streaming throughput at 1k, 10k, 100k, and 1M rows.
- `benchmarks/real_data_drift.py`: attack-window detection with a per-day benign control.
- `benchmarks/real_benchmark.py`: classifier damage and damage-predictor accuracy.
- `src/driftalytics/train_damage.py`: train and save a damage predictor from a labeled CSV
  or Parquet file.

## Real throughput

```bash
python benchmarks/run_all.py --out results/
```

Configuration: 68 numeric flow features, 5,000 reference rows, window size 1,000, sliding
windows, monitors `psi`, `ks`, and `missing_rate`, and a streaming chunk of 100,000 rows.
The batch is cycled from the same data to reach each size. Runs up to 100,000 rows use one
discarded warmup followed by three timed runs; the median is reported. The 1M case uses one
timed run because a full repetition takes several minutes.

| Rows | Seconds | Rows/sec | Windows | Monitor alerts | Timed runs |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.0885 | 11,303 | 1 | 1 | 3 |
| 10,000 | 0.8117 | 12,319 | 10 | 14 | 3 |
| 100,000 | 8.2652 | 12,099 | 100 | 190 | 3 |
| 1,000,000 | 157.3149 | 6,357 | 1,000 | 1,899 | 1 |

Per-monitor throughput at 100,000 rows:

| Monitor | Seconds | Rows/sec | Alerts |
|---|---:|---:|---:|
| `psi` | 1.7685 | 56,544 | 91 |
| `ks` | 5.9476 | 16,813 | 99 |
| `missing_rate` | 0.5115 | 195,515 | 0 |

The streaming path is faster than a single `run()` call on this data. An isolated 1M-row
comparison measured 183 seconds through `update()` and 266 seconds through `run()`, so the
benchmark exercises streaming, which is also the production path.

## Attack detection

```bash
python benchmarks/real_data_drift.py --out results/
```

Each day is handled separately. The script takes 50,000 random benign rows as reference and
another 50,000 random benign rows as control. The split is exchangeable, so the control
measures false positives from sampling noise rather than from time-of-day drift. The
remaining rows are streamed chronologically in 1,000-row windows.

A window is counted as an attack window when at least half of its rows are attacks. The
script reports the default-threshold alert rate, score AUC, and recall at the threshold that
gives 5% false positives on the control split.

Per-day results:

| Day | Attack rate | Windows | Attack windows | Control alerts | Live benign alert rate | Attack alert rate |
|---|---:|---:|---:|---:|---:|---:|
| 2017-07-04 | 0.0310 | 345 | 0 | 0/50 | 0.8773 | 0.0000 |
| 2017-07-05 | 0.3641 | 591 | 260 | 0/50 | 0.9225 | 1.0000 |
| 2017-07-06 | 0.0048 | 358 | 0 | 0/50 | 1.0000 | 0.0000 |
| 2017-07-07 | 0.4110 | 602 | 297 | 0/50 | 0.9605 | 1.0000 |
| Pooled | 0.3072 | 1,896 | 557 | 0/200 | 0.9412 | 1.0000 |

Pooled monitor metrics:

| Monitor | Control alert rate | Live benign alert rate | Attack alert rate | AUC | Recall at 5% control FPR |
|---|---:|---:|---:|---:|---:|
| `psi` | 0.0000 | 0.6441 | 1.0000 | 0.9878 | 1.0000 |
| `ks` | 0.0000 | 0.9412 | 1.0000 | 0.9940 | 1.0000 |
| `missing_rate` | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |

The exchangeable control produces zero alerts in 200 windows. Live benign traffic still
moves during the day, so default thresholds alert on temporal benign drift, especially under
`ks`. A default alert therefore means the distribution changed, not that model quality
dropped. Ranking is strong: AUC is 0.9878 for `psi` and 0.9940 for `ks`, and both recover
every attack window at a 5% control false-positive threshold.

July 4 and July 6 have no majority-attack windows because their attacks are sparse and fall
inside mixed windows. Attack recall is only measurable on July 5 and July 7. `missing_rate`
never fires because CICIDS contains no missing values. The per-window output is written to
`results/cicids_detection_windows.csv`.

## Damage prediction

```bash
python benchmarks/real_benchmark.py --out results/
```

The classifier is trained on 50,000 benign July 3 flows plus all 13,832 July 4 attacks. A
separate 50,000-row benign July 3 sample is held out as control. The remaining 1,852,586
rows from July 5-7 are streamed in 5,000-row windows, giving 370 complete windows; the
trailing 2,586 rows do not fill a window and are skipped.

Control classification error is 0.04548. Raw damage is the window error minus that control
error, clipped to `[0, 1]`. Balanced damage repeats the calculation with balanced error,
which removes the class-prior component. The five-feature corruption vector is computed for
each window from the same reference.

| Day | Windows | Raw damage mean | Raw damage median | Raw damage max | Balanced damage mean | Attack share | Drift magnitude |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2017-07-05 | 139 | 0.3542 | 0.0473 | 0.9379 | 0.2782 | 0.3622 | 1.4392 |
| 2017-07-06 | 91 | 0.1526 | 0.0285 | 0.7405 | 0.2860 | 0.0049 | 0.4466 |
| 2017-07-07 | 140 | 0.1784 | 0.0072 | 0.9531 | 0.1860 | 0.4124 | 2.3073 |

Across all 370 windows, mean raw damage is 0.2381, mean balanced damage is 0.2453, 102
windows have zero damage, and 97 exceed 0.5. July 6 is the cleanest evidence that benign
drift matters: it has a 0.49% attack share, yet balanced damage averages 0.2860. July 7 has
the highest attack share and drift magnitude but a near-zero median raw damage, because its
attack families are concentrated and easy for the classifier.

The predictor is evaluated on a chronological 60/20/20 split: 222 train, 74 validation, and
74 test windows. Test metrics with 1,000-round bootstrap intervals:

| Predictor | Test MAE | Test RMSE | Test R? |
|---|---:|---:|---:|
| Five-feature model | 0.1097 | 0.1710 | 0.7503 |
| Five-feature 95% CI | 0.0833 to 0.1415 | n/a | 0.7507 to 0.8935 |
| Drift-only OLS | 0.0951 | 0.1196 | 0.9252 |
| Prior-only oracle | 0.0604 | 0.0685 | 0.9901 |
| Untrained heuristic | 0.1510 | 0.2151 | 0.7997 |
| Balanced-target model | 0.2006 | 0.2239 | -0.0204 |

Training metrics are MAE 0.0524, RMSE 0.0776, and R? 0.9536. Validation R? is -13.65
because the validation block crosses a regime change and has low target variance; MAE 0.4483
is the meaningful validation number.

The prior-only row is an oracle diagnostic, not a deployable predictor: it uses the true
window attack share, which is unavailable without labels. It shows that most raw damage here
is class-prior movement. The five-feature model beats the untrained heuristic, but
`drift_magnitude` alone is slightly better on this test slice. `missingness_delta`,
`schema_change`, and `volume_delta` are constant zero; `cardinality_shift` adds more noise
than signal. The balanced-target model fails to generalize, so CICIDS supports prior-driven
damage estimation more strongly than concept-driven damage estimation.

The script writes `results/cicids_damage_windows.csv`, `results/real_benchmark.json`, and
`results/damage_predictor_real.json`. Train a standalone model with:

```bash
python -m driftalytics.train_damage   --data results/cicids_damage_windows.csv   --out results/damage_predictor_real.json
```

## Reporting rules

- Report a false-positive rate only from a declared control split, and state whether the
  split is exchangeable or temporal.
- Report AUC and recall at a calibrated false-positive threshold alongside default alert
  rates. A default alert does not prove model damage.
- Disclose CICIDS class-prior shifts, attack families, and the fact that benign traffic also
  drifts across each day.
- Report raw damage, balanced damage, the prior-only oracle, the drift-only ablation, and the
  untrained heuristic together.
- Report throughput at several sizes, name the monitor set, and state the hardware and
  Python version.
- Do not publish LLM judge-versus-human agreement unless a labeled subset exists. This
  repository does not include one.
- Do not claim comparisons against other tools unless the comparison was actually run.

## This machine

Windows 11, Python 3.13.5, consumer laptop hardware. Your numbers will differ.
