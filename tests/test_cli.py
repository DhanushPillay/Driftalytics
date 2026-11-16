"""CLI tests."""

import json
import subprocess
import sys

import pytest
from helpers import data, write_damage_csv

from driftalytics.cli import _as_bool, main, run_config


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "driftalytics", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "0.5.0" in result.stdout


def test_cli_run_config(tmp_path):
    reference = data(300, 3)
    live = data(150, 3)
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


def test_cli_boolean_parsing_and_config_errors(tmp_path):
    assert _as_bool("false", "sliding") is False
    assert _as_bool("YES", "sliding") is True
    with pytest.raises(ValueError):
        _as_bool("maybe", "sliding")

    reference = tmp_path / "reference.parquet"
    live = tmp_path / "live.parquet"
    data(100, 0).to_parquet(reference)
    data(100, 1).to_parquet(live)
    config = tmp_path / "config.yaml"
    html = tmp_path / "report.html"
    jsn = tmp_path / "report.json"
    config.write_text(
        "\n".join(
            [
                f"reference: {reference.as_posix()}",
                f"data: {live.as_posix()}",
                "monitors: [psi]",
                "sliding: 'false'",
                "include_partial: 'false'",
                "output:",
                f"  html: {html.as_posix()}",
                f"  json: {jsn.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    assert run_config(str(config)) == 0
    assert html.exists() and jsn.exists()

    config.write_text(f"reference: {reference.as_posix()}\nmonitors: psi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="monitors"):
        run_config(str(config))

    config.write_text(f"reference: {reference.as_posix()}\nmonitors: [nope]\n", encoding="utf-8")
    with pytest.raises(KeyError):
        run_config(str(config))

    config.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        run_config(str(config))


def test_cli_train_damage(tmp_path):
    data_path = tmp_path / "damage.csv"
    write_damage_csv(data_path)
    model_path = tmp_path / "model.json"
    assert main(["train-damage", "--data", str(data_path), "--out", str(model_path)]) == 0
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["format"] == "driftalytics-damage-v1"
