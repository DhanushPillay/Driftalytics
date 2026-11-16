# Benchmarks

These scripts produce the numbers reported in the README and in
[`docs/benchmarking.md`](../docs/benchmarking.md). They are research scripts, not part of the
installed library.

## Prerequisites

```bash
pip install -e ".[bench]"
```

The scripts use the CICIDS2017 flow dataset through `driftalytics.datasets.load_cicids`.
The first run downloads roughly 370 MB to `data/raw/cicids2017/CICIDS_Flow.parquet` and
verifies its SHA256. Set `DRIFTALYTICS_DATA_DIR` to cache it elsewhere:

```bash
DRIFTALYTICS_DATA_DIR=/path/to/cache python benchmarks/run_all.py
```

Run every command from the repository root so the default data path and `results/` output
resolve correctly.

## Scripts

### `run_all.py`

Streaming throughput and per-monitor cost on CICIDS2017. It reports rows per second for
1k, 10k, 100k, and 1M row streams with a 5,000 row reference and 1,000 row windows. Writes
`results/throughput.json`.

```bash
python benchmarks/run_all.py
```

### `real_data_drift.py`

Detection quality per day. Each day gets an exchangeable benign reference and control, and
the remaining rows are streamed chronologically in 1,000 row windows. It reports control
false-positive rate, AUC, and recall at a 5% control false-positive rate. Writes
`results/real_data_drift.json` and `results/cicids_detection_windows.csv`.

```bash
python benchmarks/real_data_drift.py
```

### `real_benchmark.py`

Damage prediction under real drift. A logistic regression is trained on benign and attack
flows, then evaluated on later days in 5,000 row windows. Damage is the increase in error
over the control, clipped to `[0, 1]`. The script compares the five-feature corruption
vector, a balanced-target model, the untrained heuristic, a prior-only oracle, and a
drift-only OLS baseline. It reports bootstrap intervals over 1,000 rounds. Writes
`results/real_benchmark.json`, `results/cicids_damage_windows.csv`, and
`results/damage_predictor_real.json`.

```bash
python benchmarks/real_benchmark.py
```

## Notes

- Timings depend on CPU, disk, and pandas version. Compare relative numbers, not absolute
  ones.
- The CICIDS cache is ignored by git. The checked-in `results/` files are evidence from a
  specific run, not a guarantee for every machine.
- Training and evaluation are chronological. Do not shuffle windows when reproducing the
  numbers.
