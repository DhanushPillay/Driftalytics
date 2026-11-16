"""Core registry, corruption-vector, and schema-drift tests."""

import numpy as np
import pandas as pd

from driftalytics import MonitorRegistry, SlidingWindowRunner, compute_corruption_vector


def test_registry_has_13_builtins():
    names = MonitorRegistry.list()
    assert len(names) == 13
    assert "psi" in names
    assert "faithfulness" in names
    assert "perf_regression" in names


def test_type_drift_does_not_crash():
    ref = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.arange(100, dtype=float)})
    batch = pd.DataFrame({"a": ["x"] * 50, "b": np.arange(50, dtype=float)})
    runner = SlidingWindowRunner(
        reference=ref,
        monitors=["psi", "ks", "wasserstein", "schema_change", "drift_magnitude"],
        window_size=20,
    )
    report = runner.run(batch)
    assert all(result.status in {"ok", "skipped"} for result in report.results)
    schema = next(result for result in report.results if result.name == "schema_change")
    assert schema.alert is True
    vector = compute_corruption_vector(ref, batch)
    assert vector.schema_change > 0


def test_corruption_vector_handles_near_zero_scale():
    reference = pd.DataFrame(
        {
            "constant": np.full(100, 1.0),
            "tiny": np.linspace(1e-15, 2e-15, 100),
        }
    )
    batch = pd.DataFrame(
        {
            "constant": np.full(100, 2.0),
            "tiny": np.linspace(3e-15, 4e-15, 100),
        }
    )
    vector = compute_corruption_vector(reference, batch)
    assert np.isfinite(vector.drift_magnitude)
    assert vector.drift_magnitude < 100.0
