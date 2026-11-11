# Contributing

Thanks for considering a contribution.

## Dev setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -e ".[dev]"
```

The public repository is at <https://github.com/DhanushPillay/Driftalytics>.

## Workflow

1. Describe the change in an issue or note before large edits.
2. Create a feature branch if you are using version control.
3. Write code and tests.
4. Run `ruff check src tests benchmarks`, `ruff format --check src tests benchmarks`,
   `mypy src`, and `pytest`.
5. Keep documentation and benchmark numbers in sync with behavior.

CI runs the same checks on Python 3.10, 3.11, 3.12, and 3.13.

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

## Code of conduct

Be kind. Be specific. Cite sources.
