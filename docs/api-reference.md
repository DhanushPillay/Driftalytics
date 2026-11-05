# API reference

This is a compact reference for the public surface. See the source under
`src/driftalytics/` for full type signatures.

## Core

### `Monitor`

Base class for monitors.

```python
class Monitor:
    name: str = "monitor"
    threshold: float | None = None

    def on_batch(self, reference, batch) -> MonitorResult: ...
```

Subclass it, set `name`, and implement `on_batch`. Raise an exception to have the runner
record an error result.

### `MonitorResult`

```python
MonitorResult(
    name: str,
    score: float,
    alert: bool,
    details: dict[str, Any] = {},
    timestamp: float = time.time(),
    threshold: float | None = None,
    status: str = "ok",
    error: str | None = None,
)
```

### `MonitorRegistry`

- `register(name, monitor_cls)`
- `get(name) -> type[Monitor]`
- `list() -> list[str]`

`register_monitor(name, cls)` is a module-level shortcut.

### `SlidingWindowRunner`

```python
SlidingWindowRunner(
    reference,
    monitors,
    window_size=1000,
    alert_threshold=None,
    damage_predictor=None,
    stride=None,
    sliding=False,
    include_partial=True,
)
```

- `reference`: DataFrame, parquet path, or CSV path.
- `monitors`: names, instances, or monitor classes.
- `stride`: defaults to `window_size`.
- `alert_threshold`: fallback for monitors without their own threshold.

Methods:

- `run(data) -> Report`
- `update(batch, final=False) -> list[Report]`
- `run_stream(batches) -> Iterator[Report]`

### `Report`

Fields include `results`, `alerts`, `n_rows`, `n_rows_analyzed`, `n_rows_skipped`,
`damage_estimate`, `damage_error`, `corruptions`, `n_windows`, and `window_summaries`.

- `summary() -> str`
- `to_dict() -> dict`
- `to_json(path) -> None`
- `to_html(path) -> None`

### `Alert`

`monitor`, `score`, `threshold`, `timestamp`, `message`, `window`.

## Corruption and damage

### `compute_corruption_vector(reference, batch, expected_size=None)`

Returns a `CorruptionVector`. Pass `expected_size` when the batch is a window whose
expected row count differs from the reference length.

### `CorruptionVector`

Fields: `drift_magnitude`, `missingness_delta`, `cardinality_shift`, `schema_change`,
`volume_delta`. Methods: `to_list()`, `as_dict()`.

### `DamagePredictor`

- `fit(X, y) -> DamagePredictor`
- `evaluate(X, y) -> dict[str, float]` with `mae`, `rmse`, `r2`
- `predict(vector) -> float`
- `save(path) -> None`
- `load(path) -> DamagePredictor` (class method)

Models are JSON with format `driftalytics-damage-v1`.

### Training helper

`driftalytics.train_damage.train_and_save(out_path, n, data_path, seed, test_size)`

The CLI wrapper is:

```bash
python -m driftalytics.train_damage --out models/damage_predictor.json
python -m driftalytics.train_damage --data scenarios.csv --out models/damage_predictor.json
```

Training data needs the five corruption-vector columns plus `damage`.

## LLM judging

### `LLMJudge`

```python
LLMJudge(
    provider="openai",          # "openai", "anthropic", "ollama", or a provider object
    model="gpt-4o-mini",
    metrics=["faithfulness", "answer_relevance"],
    budget_tokens=50_000,
    sample_rate=1.0,
    cache=True,
    api_key=None,
    host=None,
    timeout=60,
    retries=2,
    seed=0,
)
```

- `evaluate(questions, contexts, answers) -> list[dict]`

Each result contains the requested metric keys plus `_status` and `_error`. Status values
are `ok`, `cached`, `sampled`, `budget`, and `error`. Metric values are clamped to
`[0, 1]`; unavailable values are `None`.

Provider classes: `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`, and
`make_provider(provider, model, ...)`.

## CLI

```bash
driftalytics --version
driftalytics run CONFIG.yaml
driftalytics monitors
```

Config keys: `reference`, `data`, `monitors`, `window_size`, `alert_threshold`,
`damage_predictor`, `stride`, `sliding`, `include_partial`, and `output.html` /
`output.json`.
