"""Tabular monitor tests."""

import numpy as np
import pandas as pd
import pytest
from helpers import data
from scipy import stats

from driftalytics.monitors.tabular import (
    KSMonitor,
    MissingRateMonitor,
    PSIMonitor,
    SchemaChangeMonitor,
    VolumeMonitor,
    WassersteinMonitor,
)


def test_psi_detects_out_of_range_values():
    ref = pd.DataFrame({"a": np.linspace(0, 1, 1000)})
    batch = pd.DataFrame({"a": np.full(1000, 10.0)})
    result = PSIMonitor().on_batch(ref, batch)
    assert result.score > 0.2
    assert result.alert is True


def test_missing_rate_catches_localized_drift():
    rng = np.random.default_rng(0)
    columns = {f"c{i}": rng.normal(size=200) for i in range(100)}
    ref = pd.DataFrame(columns)
    batch = ref.copy()
    batch["c0"] = np.nan
    result = MissingRateMonitor().on_batch(ref, batch)
    assert result.score == pytest.approx(1.0)
    assert result.alert is True
    assert result.details["per_column"]["c0"] == pytest.approx(1.0)


def test_ks_requires_effect_size():
    ref = data(2000, 0)
    small = data(2000, 1, shift=0.01)
    large = data(2000, 1, shift=2.0)
    monitor = KSMonitor(alpha=0.05, min_effect=0.5)
    assert monitor.on_batch(ref, small).alert is False
    assert monitor.on_batch(ref, large).alert is True


def test_ks_uses_asymptotic_for_large_samples(monkeypatch):
    calls = []
    real = stats.ks_2samp

    def fake(reference, batch, method="auto"):
        calls.append(method)
        return real(reference, batch, method=method)

    monkeypatch.setattr("driftalytics.monitors.tabular.stats.ks_2samp", fake)
    monitor = KSMonitor()
    monitor.on_batch(data(2000, 0), data(2000, 1, shift=1.0))
    assert calls == ["asymp", "asymp"]

    calls.clear()
    monitor = KSMonitor()
    monitor.on_batch(data(100, 0), data(100, 1, shift=1.0))
    assert calls == ["exact", "exact"]


def test_ks_caches_reference_values(monkeypatch):
    ref = data(2000, 0)
    monitor = KSMonitor()
    monitor.on_batch(ref, data(1000, 1))
    cached = monitor._cached_values
    monitor.on_batch(ref, data(1000, 2))
    assert monitor._cached_values is cached


def test_wasserstein_handles_near_zero_scale():
    reference = pd.DataFrame({"a": np.linspace(1e-15, 2e-15, 100)})
    batch = pd.DataFrame({"a": np.linspace(3e-15, 4e-15, 100)})
    result = WassersteinMonitor().on_batch(reference, batch)
    assert np.isfinite(result.score)
    assert result.score < 1.0


def test_schema_and_volume_monitors():
    ref = data(100, 0)
    batch = data(50, 1)
    changed = batch.drop(columns=["b"])
    assert VolumeMonitor(expected_size=100).on_batch(ref, batch).score == pytest.approx(0.5)
    assert SchemaChangeMonitor().on_batch(ref, changed).score > 0


def test_monitor_edge_cases_and_validation():
    ref = data(100, 0)
    for constructor in (
        lambda: PSIMonitor(threshold=-1),
        lambda: KSMonitor(alpha=float("nan")),
        lambda: KSMonitor(exact_threshold=0),
        lambda: MissingRateMonitor(threshold=float("inf")),
        lambda: VolumeMonitor(expected_size=-1),
    ):
        with pytest.raises(ValueError):
            constructor()

    volume = VolumeMonitor(expected_size=0).on_batch(ref, data(2, 1))
    assert volume.details["expected_size"] == 0
    assert volume.score == 2.0

    all_nan = pd.DataFrame({"a": [np.nan, np.nan]})
    assert WassersteinMonitor().on_batch(all_nan, all_nan).status == "skipped"
