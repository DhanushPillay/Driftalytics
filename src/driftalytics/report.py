"""Report aggregation and rendering."""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .core import CorruptionVector, MonitorResult

if TYPE_CHECKING:
    from .runner import Alert


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def _clean(value: Any) -> Any:
    """Convert values to JSON-safe primitives; non-finite floats become None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _clean(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _result_dict(result: MonitorResult) -> dict[str, Any]:
    return _clean(
        {
            "name": result.name,
            "score": result.score,
            "alert": result.alert,
            "threshold": result.threshold,
            "status": result.status,
            "error": result.error,
            "details": result.details,
            "timestamp": result.timestamp,
        }
    )


def _alert_dict(alert: Alert) -> dict[str, Any]:
    return _clean(
        {
            "monitor": alert.monitor,
            "score": alert.score,
            "threshold": alert.threshold,
            "window": alert.window,
            "start": alert.start,
            "end": alert.end,
            "message": alert.message,
            "timestamp": alert.timestamp,
        }
    )


def _corruption_dict(corruption: CorruptionVector) -> dict[str, float]:
    return _clean(corruption.as_dict())


@dataclass
class Report:
    results: list[MonitorResult] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    n_rows: int = 0
    n_rows_analyzed: int = 0
    n_rows_skipped: int = 0
    damage_estimate: float | None = None
    damage_error: str | None = None
    corruptions: list[CorruptionVector] = field(default_factory=list)
    n_windows: int = 1
    window_summaries: list[dict[str, Any]] = field(default_factory=list)
    start: int = 0
    end: int = 0

    def summary(self) -> str:
        lines = [
            f"Driftalytics report: {self.n_rows} rows, "
            f"{self.n_rows_analyzed} analyzed, {self.n_rows_skipped} skipped, "
            f"{len(self.alerts)} alert(s), {self.n_windows} window(s)"
        ]
        for result in self.results:
            suffix = f" status={result.status}"
            if result.error:
                suffix += f" error={result.error}"
            lines.append(
                f"  {result.name}: score={_fmt(result.score)} "
                f"threshold={_fmt(result.threshold)} alert={result.alert}{suffix}"
            )
        if self.damage_estimate is not None:
            lines.append(f"  estimated performance drop: {_fmt(self.damage_estimate)}")
        if self.damage_error:
            lines.append(f"  damage prediction error: {self.damage_error}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return _clean(
            {
                "n_rows": self.n_rows,
                "n_rows_analyzed": self.n_rows_analyzed,
                "n_rows_skipped": self.n_rows_skipped,
                "n_windows": self.n_windows,
                "start": self.start,
                "end": self.end,
                "damage_estimate": self.damage_estimate,
                "damage_error": self.damage_error,
                "results": [_result_dict(r) for r in self.results],
                "alerts": [_alert_dict(a) for a in self.alerts],
                "corruptions": [_corruption_dict(c) for c in self.corruptions],
                "windows": [
                    {
                        **{
                            key: value
                            for key, value in window.items()
                            if key not in {"results", "alerts", "corruption"}
                        },
                        "results": [_result_dict(r) for r in window["results"]],
                        "alerts": [_alert_dict(a) for a in window["alerts"]],
                        "corruption": (
                            _corruption_dict(window["corruption"])
                            if window.get("corruption") is not None
                            else None
                        ),
                    }
                    for window in self.window_summaries
                ],
            }
        )

    def to_json(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    def to_html(self, path: str) -> None:
        row_html = "".join(
            "<tr>"
            f"<td>{html.escape(result.name)}</td>"
            f"<td>{_fmt(result.score)}</td>"
            f"<td>{_fmt(result.threshold)}</td>"
            f"<td>{html.escape(str(result.alert))}</td>"
            f"<td>{html.escape(result.status)}</td>"
            f"<td>{html.escape(result.error or '')}</td>"
            f"<td><code>{html.escape(json.dumps(_clean(result.details), default=str))}</code></td>"
            "</tr>"
            for result in self.results
        )
        alert_html = "".join(
            "<tr>"
            f"<td>{alert.window if alert.window is not None else ''}</td>"
            f"<td>{alert.start if alert.start is not None else ''}</td>"
            f"<td>{alert.end if alert.end is not None else ''}</td>"
            f"<td>{html.escape(alert.monitor)}</td>"
            f"<td>{_fmt(alert.score)}</td>"
            f"<td>{_fmt(alert.threshold)}</td>"
            f"<td>{html.escape(alert.message)}</td>"
            "</tr>"
            for alert in self.alerts
        )
        damage = (
            f"<p>Estimated performance drop: <strong>{_fmt(self.damage_estimate)}</strong></p>"
            if self.damage_estimate is not None
            else ""
        )
        if self.damage_error:
            damage += (
                f"<p class='error'>Damage prediction failed: {html.escape(self.damage_error)}</p>"
            )

        document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Driftalytics report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
th {{ background: #f4f4f4; }}
code {{ font-size: 0.85rem; }}
.error {{ color: #a00; }}
</style>
</head>
<body>
<h1>Driftalytics report</h1>
<p>{self.n_rows} rows, {self.n_rows_analyzed} analyzed, {self.n_rows_skipped} skipped,
{len(self.alerts)} alert(s), {self.n_windows} window(s)</p>
{damage}
<h2>Monitor results</h2>
<table>
<thead><tr><th>Monitor</th><th>Score</th><th>Threshold</th><th>Alert</th><th>Status</th><th>Error</th><th>Details</th></tr></thead>
<tbody>{row_html}</tbody>
</table>
<h2>Alerts</h2>
<table>
<thead><tr><th>Window</th><th>Start</th><th>End</th><th>Monitor</th><th>Score</th><th>Threshold</th><th>Message</th></tr></thead>
<tbody>{alert_html}</tbody>
</table>
</body>
</html>
"""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(document, encoding="utf-8")
