"""Runner, windowing, threshold, and streaming tests."""

import pandas as pd
import pytest
from helpers import AlwaysAlert, data

from driftalytics import SlidingWindowRunner
from driftalytics.monitors.tabular import PSIMonitor


def test_runner_runs():
    ref = data(500, 0)
    batch = data(200, 1, shift=0.5)
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


def test_monitor_threshold_is_honored():
    ref = data(500, 0)
    batch = data(200, 1, shift=1.0)
    monitor = PSIMonitor(threshold=1000.0)
    report = SlidingWindowRunner(
        reference=ref,
        monitors=[monitor],
        alert_threshold=0.0,
    ).run(batch)
    assert report.results[0].alert is False
    assert report.alerts == []


def test_global_threshold_only_applies_without_monitor_threshold():
    ref = data(300, 0)
    batch = data(100, 1)
    report = SlidingWindowRunner(
        reference=ref,
        monitors=[AlwaysAlert()],
        alert_threshold=0.1,
    ).run(batch)
    assert report.results[0].alert is True
    assert report.results[0].threshold == 0.1


def test_sliding_windows_include_partial_tail():
    ref = data(300, 0)
    batch = data(1050, 1, shift=0.3)
    report = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=200,
        stride=200,
        sliding=True,
        include_partial=True,
    ).run(batch)
    assert report.n_windows == 6
    assert report.n_rows_analyzed == 1050
    assert report.n_rows_skipped == 0

    skipped = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=200,
        stride=200,
        sliding=True,
        include_partial=False,
    ).run(batch)
    assert skipped.n_windows == 5
    assert skipped.n_rows_skipped == 50


def test_overlapping_windows_do_not_double_count_rows():
    ref = data(300, 0)
    batch = data(500, 1, shift=0.3)
    report = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=200,
        stride=100,
        sliding=True,
        include_partial=True,
    ).run(batch)
    assert report.n_windows == 4
    assert report.n_rows_analyzed == 500
    assert report.n_rows_skipped == 0
    assert sum(window["n_rows"] for window in report.window_summaries) == 800


def test_streaming_update_and_run_stream():
    ref = data(300, 0)
    runner = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=100,
        stride=100,
        sliding=True,
    )
    assert runner.update(data(60, 1)) == []
    reports = runner.update(data(60, 2))
    assert len(reports) == 1
    assert reports[0].n_rows == 100
    final = runner.update(pd.DataFrame(), final=True)
    assert len(final) == 1
    assert final[0].n_rows == 20

    stream_runner = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=100,
        stride=100,
        sliding=True,
    )
    streamed = list(stream_runner.run_stream([data(60, 1), data(60, 2), data(60, 3)]))
    assert len(streamed) == 2
    assert sum(report.n_rows for report in streamed) == 180


def test_short_sliding_input_respects_include_partial():
    ref = data(300, 0)
    report = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=100,
        sliding=True,
        include_partial=False,
    ).run(data(50, 1))
    assert report.n_windows == 0
    assert report.n_rows_analyzed == 0
    assert report.n_rows_skipped == 50
    assert report.results == []

    partial = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=100,
        sliding=True,
        include_partial=True,
    ).run(data(50, 1))
    assert partial.n_windows == 1
    assert partial.n_rows_analyzed == 50
    assert partial.start == 0
    assert partial.end == 50


def test_non_sliding_short_input_is_one_window():
    report = SlidingWindowRunner(
        reference=data(300, 0),
        monitors=["psi"],
        window_size=1000,
        sliding=False,
    ).run(data(50, 1))
    assert report.n_windows == 1
    assert report.n_rows_analyzed == 50
    assert report.n_rows_skipped == 0
    assert report.start == 0
    assert report.end == 50


def test_stride_gaps_are_skipped_and_reported():
    ref = data(300, 0)
    report = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=100,
        stride=150,
        sliding=True,
        include_partial=True,
    ).run(data(500, 1))
    assert [window["start"] for window in report.window_summaries] == [0, 150, 300, 450]
    assert [window["end"] for window in report.window_summaries] == [100, 250, 400, 500]
    assert report.n_rows_analyzed == 350
    assert report.n_rows_skipped == 150


def test_partial_window_does_not_trigger_volume_alert():
    report = SlidingWindowRunner(
        reference=data(300, 0),
        monitors=["volume"],
        window_size=100,
        stride=100,
        sliding=True,
        include_partial=True,
    ).run(data(1050, 1))
    assert report.n_windows == 11
    assert report.alerts == []


def test_run_and_stream_share_window_semantics():
    ref = data(300, 0)
    batch = data(470, 1)
    runner = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=100,
        stride=150,
        sliding=True,
        include_partial=True,
    )
    batch_report = runner.run(batch)

    stream_runner = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=100,
        stride=150,
        sliding=True,
        include_partial=True,
    )
    streamed = []
    for chunk in [batch.iloc[:70], batch.iloc[70:220], batch.iloc[220:410], batch.iloc[410:]]:
        streamed.extend(stream_runner.update(chunk))
    streamed.extend(stream_runner.update(pd.DataFrame(), final=True))

    assert [
        (window["start"], window["end"], window["n_rows"])
        for window in batch_report.window_summaries
    ] == [(report.start, report.end, report.n_rows) for report in streamed]
    assert batch_report.n_rows_analyzed == 320
    assert batch_report.n_rows_skipped == 150


def test_finalization_is_idempotent_and_locked():
    runner = SlidingWindowRunner(
        reference=data(300, 0),
        monitors=["psi"],
        window_size=100,
        stride=100,
        sliding=True,
        include_partial=True,
    )
    assert len(runner.update(data(150, 1), final=True)) == 2
    assert runner.update(pd.DataFrame(), final=True) == []
    with pytest.raises(RuntimeError):
        runner.update(data(10, 1))


def test_empty_updates_are_allowed_before_finalization():
    runner = SlidingWindowRunner(
        reference=data(300, 0),
        monitors=["psi"],
        window_size=100,
        stride=100,
        sliding=True,
    )
    assert runner.update(pd.DataFrame()) == []
    assert runner.update(data(100, 1))[0].n_rows == 100
