"""Cross-component integration test."""

from helpers import data

from driftalytics import SlidingWindowRunner


def test_all_monitors_run():
    ref = data(400, 0)
    batch = data(200, 1, shift=0.5).assign(
        c=["a", "b"] * 100,
        y_true=[0, 1] * 100,
        y_pred=[0, 1] * 100,
        question=["q"] * 200,
        context=["c"] * 200,
        answer=["a"] * 200,
    )
    names = [
        "answer_relevance",
        "cardinality",
        "chi_square",
        "drift_magnitude",
        "faithfulness",
        "ks",
        "missing_rate",
        "perf_classification",
        "perf_regression",
        "psi",
        "schema_change",
        "volume",
        "wasserstein",
    ]
    report = SlidingWindowRunner(reference=ref, monitors=names, window_size=100).run(batch)
    assert len(report.results) == 13
    assert all(result.status in {"ok", "skipped"} for result in report.results)
