# Design decisions

## Why a separate project

Evidently and similar tools are mature and broader. Driftalytics is deliberately small:
stream-friendly windowing, a stable monitor interface, and a corruption-vector estimate of
performance damage before labels arrive. No code is copied from other projects.

## Stream-first, not batch-first

Production failures are continuous. The runner supports both one-shot batch runs and
incremental windows over a stream. Batch mode is the simple case; `sliding=True` and
`update` cover ongoing traffic without retaining the full history.

## Per-monitor thresholds

A monitor's own threshold is authoritative. A global `alert_threshold` remains as a
fallback for custom monitors that do not define one. Letting a global value override tuned
thresholds would make monitor configuration misleading.

## Statistical tests need effect sizes

`ks` and `chi_square` apply Benjamini-Hochberg correction across columns to control false
positives from many tests. They also require a minimum effect size, because a large sample
makes trivial differences statistically significant. The reported score is still the
maximum statistic, but the alert reflects the corrected and effect-filtered decision.

## Expected sample cardinality

Comparing a 1,000-row window's distinct count with a 50,000-row reference always looks
like a collapse. Cardinality now compares against the number of distinct values expected
in a sample of the window size, computed from the reference frequencies. The same logic
feeds the corruption vector.

## JSON damage models, not pickle

Pickle is unsafe for untrusted files and couples models to Python versions. Damage models
are small linear models, so JSON is enough. Loading validates the format string and the
feature order. Corruption-vector prediction also validates feature count and finiteness.

## No template engine

The HTML report is a single generated document with every dynamic value passed through
`html.escape`. Removing Jinja2 dropped a dependency and removed a class of template
injection mistakes.

## LLM judge limits

Judging every request is expensive. The judge caches exact requests, samples traffic, and
enforces a token budget. An earlier batching option was removed because it was not
implemented; the library does not claim batching. Provider failures are returned as per-item
statuses instead of raising.

## Dependencies

Core runtime: `numpy`, `pandas`, `scipy`, `pyyaml`. Development dependencies are separate.
The LLM judge uses the standard library HTTP client, so `openai` and `anthropic` SDKs are
not required.

## Corruption-vector drift magnitude

`drift_magnitude` is the maximum Population Stability Index across numeric columns, not the
mean standardized difference of means. A standardized mean shift is dominated by columns with
near-zero reference variance, ignores changes in distribution shape, and saturated on
near-constant flow features. PSI compares binned distributions on reference quantiles, so it
reacts to shape and tail movement and stays finite when a column barely varies. The monitor
and the corruption vector share one implementation.

## Real-data benchmarks

All tabular benchmarks use CICIDS2017 network-flow records from the Canadian Institute for
Cybersecurity, mirrored on Hugging Face under Apache-2.0. The file has 2,827,677 flows, 68
numeric features, real timestamps, and a binary benign/attack label. July 3 is a benign-only
capture, so it supplies both the reference sample and a held-out control. July 4-7 carry
patator, DoS, web, bot, DDoS, and port-scan traffic, so they supply natural drift without
injected corruption.

The loader downloads the parquet file on demand, verifies its SHA-256, parses mixed timestamp
formats, drops constant columns, and caches the file under `data/raw/`. The dataset is not
committed. Its Apache-2.0 license is separate from the library's MIT license.

The detection benchmark draws an exchangeable reference and control split from each day's
benign traffic, then streams the rest of the day chronologically. Benign temporal drift alone
saturates alert thresholds, so the control split, not an earlier time slice, defines the
false-positive rate. Results report default-threshold alert rates, score AUC, and recall at a
5% control false-positive rate.

The damage benchmark trains on July 3 benign traffic and July 4 attacks, calibrates on a
held-out July 3 benign control, and streams July 5-7. Damage is the observed
classification-error increase over that control. Each day has a different attack mix and its
benign traffic also moves, so the benchmark reports raw and balanced damage, a prior-only
baseline, a drift-only baseline, an untrained heuristic, and bootstrap confidence intervals.
It does not claim that the unused corruption features are validated.

## Version 0.5 benchmark migration

The 0.5 release replaces the Insects stream with CICIDS2017:

- The dataset grows from 79,986 curated rows to 2,827,677 timestamped network flows with a
  benign-only control day and four days of natural attacks.
- `KSMonitor` uses the asymptotic test above 1,000 samples and caches reference arrays. The
  exact test was the throughput bottleneck on 68 columns.
- `WassersteinMonitor` treats scales below `1e-12` as degenerate instead of dividing by them.
- Detection and damage evaluation are separate scripts. Detection reports calibrated alert
  rates and AUC; damage reports observed error increase against a held-out control.
- The damage benchmark adds prior-only and drift-only baselines plus bootstrap confidence
  intervals, because attack prevalence moves from 0% to 41% across the capture.

## Version 0.2 scope

The 0.2 release fixes correctness and packaging issues found in review, adds streaming
updates, status reporting, expected-cardinality handling, BH correction, and a real-data
detection benchmark. It does not add hosted infrastructure, database logging, or a
dashboard.

## Version 0.3 review

The 0.3 release fixes defects that survived 0.2:

- **Absolute window offsets.** Reports and alerts now carry `start` and `end`, so streaming
  windows can be traced back to their input rows. Partial tails no longer reuse window-zero
  coordinates.
- **Explicit partial-window policy.** A non-sliding run scores the whole input as one window.
  Sliding tails start at the later of the last full window's end and the next scheduled
  stride, which prevents overlap when `stride < window_size` and respects gaps when
  `stride > window_size`.
- **Partial windows use their actual size.** Expected-size checks no longer compare a short
  tail against a full window and raise false volume or cardinality alerts.
- **Idempotent stream finalization.** `final=True` emits at most one allowed partial window;
  repeated finalization returns no reports, and new data after finalization is rejected.
- **Separated damage metrics.** Training, validation, and test metrics are stored separately.
  The saved `metrics` field remains the training metric for compatibility, but it is no
  longer presented as generalization.
- **Family-aware benchmark holdout.** Related corruption families stay in one split. This
  removed leakage from the previous random 70/30 split and produced a negative transfer
  result: the linear predictor does not beat the heuristic on held-out families.
- **LLM prompt and budget hardening.** Untrusted fields are JSON-encoded, the system
  instruction is separate, the prompt version is part of the cache key, output length is
  capped, failed calls consume budget, and retries use jitter and `Retry-After`.
- **Report and CLI validation.** Non-finite values become `null`/`n/a`, JSON rejects NaN,
  boolean config values are parsed explicitly, and invalid monitors or output paths fail
  early.
- **Optional Parquet and CI.** `pyarrow` is an optional extra, and CI runs lint, format,
  types, and tests on Python 3.10 through 3.13.

## Version 0.4 real-data benchmark

The 0.4 release replaces synthetic benchmark input and simulated damage training with one
public real dataset.

- All tabular benchmarks use Insects `incremental_abrupt_balanced`. ELEC2 and Bike Sharing
  were tested as alternatives, but both produced weak or negative damage predictability on
  chronological holdouts. NYC TLC has real rows but no labels for actual performance damage.
  Insects was the only candidate where the damage model beat the heuristic on future
  windows.
- The loader in `src/driftalytics/datasets.py` downloads the CSV on demand, verifies its
  SHA-256, and caches it under `data/raw/`. The dataset is not committed.
- Damage labels are observed chronological error increases from a logistic regression
  trained on the first 20,000 rows.
- The damage predictor uses a chronological 60/20/20 split. The five-feature model reaches
  test MAE 0.0405 and R-squared 0.9525, against 0.1843 and 0.4198 for the untrained
  heuristic.
- Only `drift_magnitude` and `cardinality_shift` vary in the stream. The drift-only
  ablation reaches test MAE 0.0385, so the benchmark validates those two dimensions and does
  not validate missingness, schema, or volume.
- The first reference block is already non-stationary. The benchmark reports precision and
  recall against realized damage instead of a false-positive rate.
- `make_synthetic`, `--n`, and `--seed` are removed. `train_damage` now requires `--data`
  and uses a chronological split.
- `scikit-learn` is a benchmark-only dependency in the new `bench` extra. Core runtime
  dependencies are unchanged.
