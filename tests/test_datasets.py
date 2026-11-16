"""Dataset loader tests."""

from pathlib import Path

import pandas as pd

from driftalytics.datasets import COLUMNS, FEATURES, _default_cache_path, load_cicids


def test_load_cicids_parses_and_sorts_fixture(tmp_path):
    path = tmp_path / "cicids.parquet"
    attacks = {"attack_label": "DDoS", "Timestamp": "7/7/2017 9:59"}
    benign = {"attack_label": "BENIGN", "Timestamp": "03/07/2017 01:00:01"}
    rows = []
    for index, base in enumerate([attacks, benign]):
        row = {feature: float(index + 1) for feature in FEATURES}
        row.update(base)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path)

    frame = load_cicids(path, validate=False)
    assert list(frame.columns) == COLUMNS
    assert frame["label"].tolist() == [0, 1]
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["attack_label"].tolist() == ["BENIGN", "DDoS"]


def test_cache_path_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("DRIFTALYTICS_DATA_DIR", str(tmp_path))
    expected = Path(tmp_path) / "raw" / "cicids2017" / "CICIDS_Flow.parquet"
    assert _default_cache_path() == expected
