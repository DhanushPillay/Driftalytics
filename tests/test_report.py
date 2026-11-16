"""Report rendering and serialization tests."""

import json

from helpers import data

from driftalytics import CorruptionVector, MonitorResult, Report


def test_report_render(tmp_path):
    from driftalytics import SlidingWindowRunner

    ref = data(300, 0)
    batch = data(100, 1)
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


def test_report_normalizes_non_finite_values(tmp_path):
    report = Report(
        results=[
            MonitorResult(
                name="bad",
                score=float("nan"),
                alert=True,
                threshold=float("inf"),
                details={"x": float("inf"), "y": float("nan")},
            )
        ],
        n_rows=1,
        n_rows_analyzed=1,
        damage_estimate=float("nan"),
        corruptions=[CorruptionVector(drift_magnitude=float("nan"))],
    )
    payload = report.to_dict()
    assert payload["damage_estimate"] is None
    assert payload["results"][0]["score"] is None
    assert payload["results"][0]["threshold"] is None
    assert payload["results"][0]["details"] == {"x": None, "y": None}
    assert payload["corruptions"][0]["drift_magnitude"] is None

    path = tmp_path / "report.json"
    report.to_json(str(path))
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text)["results"][0]["score"] is None

    html_path = tmp_path / "report.html"
    report.to_html(str(html_path))
    assert "n/a" in html_path.read_text(encoding="utf-8")


def test_report_renders_windows_and_corruptions(tmp_path):
    report = Report(
        results=[MonitorResult(name="psi", score=0.1, alert=False)],
        n_rows=10,
        n_rows_analyzed=10,
        n_windows=1,
        window_summaries=[
            {
                "window": 0,
                "start": 0,
                "end": 10,
                "n_rows": 10,
                "results": [],
                "alerts": [],
                "corruption": CorruptionVector(0.5, 0.0, 0.0, 0.0, 0.0),
                "damage_estimate": 0.25,
                "damage_error": None,
            }
        ],
        corruptions=[CorruptionVector(0.5, 0.0, 0.0, 0.0, 0.0)],
    )
    path = tmp_path / "windows.html"
    report.to_html(str(path))
    text = path.read_text(encoding="utf-8")
    assert "Windows" in text
    assert "Corruption vectors" in text
    assert "0.5000" in text
    assert "0.2500" in text
