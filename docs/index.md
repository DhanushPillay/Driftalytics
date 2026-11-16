# Driftalytics

Lightweight drift monitoring for tabular machine-learning models and LLM applications.
Driftalytics scores live batches over sliding windows, raises per-monitor alerts, and
estimates expected model damage before labels arrive.

```bash
pip install -e ".[dev]"
```

The package is not published to PyPI yet. Install from a checkout.

## Start here

| Page | What it covers |
|---|---|
| [Getting started](getting-started.md) | Install, first run, config file, CLI |
| [Architecture](architecture.md) | Components, windowing, streaming, data flow |
| [Monitors](monitors.md) | All 13 monitors, thresholds, skip and error behavior |
| [LLM evaluation](llm-eval.md) | Judges, providers, budgets, caching, sampling |
| [Benchmarking](benchmarking.md) | CICIDS2017 protocol and measured results |
| [API reference](api-reference.md) | Public classes, functions, CLI, config schema |
| [Decisions](decisions.md) | Design tradeoffs and rejected alternatives |
| [Development](development.md) | Dev setup, tests, checks, adding a monitor |

The source is at
[github.com/DhanushPillay/Driftalytics](https://github.com/DhanushPillay/Driftalytics).
