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

v0.2 makes the monitor's own threshold authoritative. A global `alert_threshold` remains
as a fallback for custom monitors that do not define one. The earlier behavior let a
global value override tuned thresholds, which made monitor configuration misleading.

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

Judging every request is expensive. The judge caches exact triples, samples traffic, and
enforces a token budget. An earlier batching option was removed because it was not
implemented; v0.2 does not claim batching. Provider failures are returned as per-item
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
