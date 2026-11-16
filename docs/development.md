# Development

Internal setup and conventions for working on Driftalytics.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # macOS and Linux
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Make the change on a branch.
2. Write code and tests.
3. Run `pre-commit run --all-files`, which covers the lint, format, and type checks.
4. Run `pytest --cov=driftalytics`.
5. Keep documentation and benchmark numbers in sync with behavior.

CI runs the checks on Python 3.10, 3.11, 3.12, and 3.13. It also builds a wheel, installs
it into a clean environment, runs the tests against that wheel, and builds the docs with
`mkdocs build --strict`.

## Tests

Tests are grouped by area under `tests/`: core, tabular and performance monitors, LLM
judging and monitors, runner and streaming, reports, damage, CLI, datasets, and one
integration test. Shared fixtures live in `tests/helpers.py`. The suite imports the
installed package, so run `pip install -e .` before testing.

## Standards

- Type hints on public functions.
- Tests for new monitors, thresholds, and error paths.
- Keep core dependencies to `numpy`, `pandas`, `scipy`, and `pyyaml`. Parquet support is an
  optional `pyarrow` extra.
- No vendored third-party code and no copied implementations.
- Be honest in benchmarks: report detection rate, false-positive rate, throughput at
  multiple sizes, split strategy, and the hardware used.
- Do not claim features that are not implemented. If a path is removed, remove it from
  the docs and config examples too.

## Adding a monitor

1. Subclass `Monitor` in `src/driftalytics/monitors/`.
2. Implement `on_batch` and return a `MonitorResult` with a threshold.
3. Add the class to `BUILTIN_MONITORS` in `src/driftalytics/monitors/__init__.py`.
4. Add tests in `tests/`.
5. Document the score direction, default threshold, and skip conditions in
   `docs/monitors.md`.

For test-based monitors, apply multiple-testing correction and require an effect size.
For monitors that depend on batch size, expose `expected_size` so the runner can fill it.

## Adding a benchmark

Place the script in `benchmarks/`, make data paths configurable, write machine-readable
output under `results/`, and document the exact command and caveats in
`docs/benchmarking.md`. Do not add a number to the docs without the run that produced it.
