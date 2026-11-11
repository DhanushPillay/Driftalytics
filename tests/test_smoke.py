"""Smoke tests for Driftalytics."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from driftalytics import MonitorRegistry, Report, SlidingWindowRunner
from driftalytics.cli import main
from driftalytics.core import MonitorResult


def _data(n=500, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(loc=shift, size=n),
            "b": rng.normal(size=n),
        }
    )


def test_registry_has_13_builtins():
    names = MonitorRegistry.list()
    assert len(names) == 13
    assert "psi" in names
    assert "faithfulness" in names
    assert "perf_regression" in names


def test_runner_runs():
    ref = _data(500, 0)
    batch = _data(200, 1, shift=0.5)
    runner = SlidingWindowRunner(
        reference=ref,
        monitors=["psi", "ks", "missing_rate"],
        window_size=100,
    )
    report = runner.run(batch)
    assert report.n_rows == 200
    assert report.n_rows_analyzed == 200
    assert report.n_rows_skipped == 0
    assert len(report.results) == 3
    assert all(result.status == "ok" for result in report.results)


def test_report_render(tmp_path):
    ref = _data(300, 0)
    batch = _data(100, 1)
    runner = SlidingWindowRunner(reference=ref, monitors=["psi"])
    report = runner.run(batch)
    html = tmp_path / "r.html"
    jsn = tmp_path / "r.json"
    report.to_html(str(html))
    report.to_json(str(jsn))

    assert html.exists() and jsn.exists()
    payload = json.loads(jsn.read_text(encoding="utf-8"))
    assert payload["n_rows"] == 100
    assert payload["n_windows"] == 1
    assert payload["results"][0]["status"] == "ok"


def test_report_escapes_html(tmp_path):
    report = Report(
        results=[
            MonitorResult(
                name="<script>alert(1)</script>",
                score=0.5,
                alert=True,
                details={"note": "<b>bad</b>"},
            )
        ],
        n_rows=1,
        n_rows_analyzed=1,
        alerts=[],
    )
    path = tmp_path / "escaped.html"
    report.to_html(str(path))
    text = path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "&lt;b&gt;bad&lt;/b&gt;" in text


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "driftalytics.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "0.4.0" in result.stdout


def test_cli_run_config(tmp_path):
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    live = pd.DataFrame({"a": rng.normal(size=150), "b": rng.normal(size=150)})
    reference_path = tmp_path / "reference.parquet"
    live_path = tmp_path / "live.parquet"
    reference.to_parquet(reference_path)
    live.to_parquet(live_path)

    html_path = tmp_path / "report.html"
    json_path = tmp_path / "report.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"reference: {reference_path.as_posix()}",
                f"data: {live_path.as_posix()}",
                "monitors: [psi, missing_rate]",
                "window_size: 100",
                f"output:\n  html: {html_path.as_posix()}\n  json: {json_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["run", str(config_path)]) == 0
    assert html_path.exists() and json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_rows"] == 150
    assert len(payload["results"]) == 2
