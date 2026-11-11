# Monitors

All built-ins are registered on import. Score direction is consistent: higher means more
drift or more degradation, so the same threshold rule applies.

## Tabular drift monitors

| Name | Score | Default alert |
|---|---|---|
| `psi` | PSI per numeric column, maximum across columns | score > `threshold` (0.2) |
| `ks` | KS statistic, maximum across numeric columns | any column passes BH correction at `alpha` (0.05) and reaches `min_effect` (0.1) |
| `wasserstein` | Earth-mover distance divided by reference standard deviation | score > `threshold` (0.1) |
| `missing_rate` | Absolute change in missing rate per column, maximum | score > `threshold` (0.05) |
| `cardinality` | Relative difference from expected sample cardinality, maximum | score > `threshold` (0.1) |
| `chi_square` | Cramer's V for categorical columns, maximum | any column passes BH correction at `alpha` (0.05) and reaches `min_effect` (0.1) |
| `volume` | Relative row-count deviation from `expected_size` | score > `threshold` (0.2) |
| `schema_change` | Fraction of union columns that are missing, new, or retyped | score > `threshold` (0.0) |
| `drift_magnitude` | Mean standardized numeric shift from the corruption vector | score > `threshold` (0.5) |

Notes:

- `psi`, `ks`, and `wasserstein` only use columns that are numeric in both frames.
  Non-numeric or unparseable values are skipped.
- `ks` and `chi_square` apply Benjamini-Hochberg correction across tested columns. The
  reported score is the maximum statistic, but the alert comes from the corrected
  per-column decisions.
- `cardinality` compares the batch's distinct-value count with the number expected in a
  sample of that size, not with the full reference count. This avoids false alerts when
  windows are smaller than the reference. For continuous columns, added noise or increased
  precision can raise the score by making values more distinct.
- `schema_change` reports a per-column change map in `details`.
- Results with no eligible columns return `status="skipped"`, not a silent zero alert.
- `details` includes `per_column` values for most monitors and `statistics`, `p_values`,
  and `rejected` for the test-based monitors.
- Monitor constructors validate thresholds as finite and non-negative. Invalid values raise
  `ValueError` at construction instead of failing later under `python -O`.
- `expected_size=0` is treated as a real expected size, not as "unset". The runner sets a
  window's expected size temporarily for each batch, including the actual size of a partial
  tail, then restores the monitor's configured value.

## Performance monitors

| Name | Task | Metrics |
|---|---|---|
| `perf_classification` | Classification | `accuracy`, `precision`, `recall`, `f1` |
| `perf_regression` | Regression | `rmse`, `mae`, `r2` |

Both require `y_true` and `y_pred` columns in the batch and a `reference_score`. The score
is the degradation relative to that reference: metric drop for accuracy, precision,
recall, F1, and R², or metric increase for RMSE and MAE. Missing columns produce
`status="error"`; a missing `reference_score` produces `status="skipped"`.

## LLM monitors

| Name | Measures |
|---|---|
| `faithfulness` | Whether the answer is supported by the retrieved context |
| `answer_relevance` | Whether the answer addresses the question |

Both require `question`, `context`, and `answer` columns and a configured `LLMJudge`. The
score is `1 - mean(metric)`, so higher means worse. When the judge returns no usable
scores because of budget, sampling, or errors, the monitor skips rather than alerting on a
fabricated value.

## Writing a custom monitor

```python
from driftalytics import Monitor, MonitorResult, register_monitor

class RowCountMonitor(Monitor):
    name = "row_count"

    def on_batch(self, reference, batch):
        delta = abs(len(batch) - len(reference)) / max(len(reference), 1)
        return MonitorResult(
            name=self.name,
            score=delta,
            alert=delta > 0.5,
            threshold=0.5,
        )

register_monitor("row_count", RowCountMonitor)
```

If you need the expected window size, accept it in `__init__` and expose it as
`expected_size`. The runner temporarily sets that attribute to the current window's
expected size, which is the configured window size for full windows and the actual row count
for partial tails, then restores it.

## Corruption vector and damage predictor

Each run that needs it computes a `CorruptionVector` with these fields:

- `drift_magnitude`: mean absolute standardized mean shift across numeric columns.
- `missingness_delta`: mean absolute missing-rate change across common columns.
- `cardinality_shift`: mean relative difference from expected sample cardinality.
- `schema_change`: fraction of union columns added, removed, or retyped.
- `volume_delta`: relative row-count difference from the expected size.

`DamagePredictor` maps that vector to an estimated performance drop in `[0, 1]`. An
untrained predictor uses fixed heuristic weights. A trained predictor loads from JSON:

```python
from driftalytics import DamagePredictor

predictor = DamagePredictor.load("models/damage_predictor.json")
estimate = predictor.predict(corruption_vector)
print(f"estimated performance drop: {estimate:.3f}")
```

The vector is model-agnostic. It measures input displacement, not the true loss of a
specific model, so calibrate the predictor on your own labeled outcomes when possible.
