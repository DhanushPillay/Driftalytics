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

## Synthetic and real benchmarks

Throughput uses synthetic normal data for comparability across machines. Real evaluation
uses NYC TLC yellow taxi trips with a linear fare model, injected corruptions, and held-out
scenarios. The damage predictor is trained on simulated corruptions, not on production
labels, so its error numbers describe that simulation. Real deployments should retrain on
their own labeled outcomes.

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
