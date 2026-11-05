"""Correctness and regression tests."""

import json

import numpy as np
import pandas as pd
import pytest

from driftalytics import (
    CorruptionVector,
    DamagePredictor,
    Monitor,
    MonitorResult,
    SlidingWindowRunner,
    compute_corruption_vector,
)
from driftalytics.llm import LLMJudge
from driftalytics.monitors.llm_monitors import LLMMonitor
from driftalytics.monitors.performance import (
    PerfClassificationMonitor,
    PerfRegressionMonitor,
)
from driftalytics.monitors.tabular import (
    KSMonitor,
    MissingRateMonitor,
    PSIMonitor,
    SchemaChangeMonitor,
    VolumeMonitor,
)
from driftalytics.train_damage import train_and_save


def _data(n=500, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(loc=shift, size=n),
            "b": rng.normal(size=n),
        }
    )


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response, 10


class AlwaysAlert(Monitor):
    name = "always"

    def on_batch(self, reference, batch):
        return MonitorResult(name=self.name, score=0.5, alert=False)


def test_all_monitors_run():
    ref = _data(400, 0)
    batch = _data(200, 1, shift=0.5).assign(
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


def test_monitor_threshold_is_honored():
    ref = _data(500, 0)
    batch = _data(200, 1, shift=1.0)
    monitor = PSIMonitor(threshold=1000.0)
    report = SlidingWindowRunner(
        reference=ref,
        monitors=[monitor],
        alert_threshold=0.0,
    ).run(batch)
    assert report.results[0].alert is False
    assert report.alerts == []


def test_global_threshold_only_applies_without_monitor_threshold():
    ref = _data(300, 0)
    batch = _data(100, 1)
    report = SlidingWindowRunner(
        reference=ref,
        monitors=[AlwaysAlert()],
        alert_threshold=0.1,
    ).run(batch)
    assert report.results[0].alert is True
    assert report.results[0].threshold == 0.1


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
    ref = _data(2000, 0)
    small = _data(2000, 1, shift=0.01)
    large = _data(2000, 1, shift=2.0)
    monitor = KSMonitor(alpha=0.05, min_effect=0.5)
    assert monitor.on_batch(ref, small).alert is False
    assert monitor.on_batch(ref, large).alert is True


def test_schema_and_volume_monitors():
    ref = _data(100, 0)
    batch = _data(50, 1)
    changed = batch.drop(columns=["b"])
    assert VolumeMonitor(expected_size=100).on_batch(ref, batch).score == pytest.approx(0.5)
    assert SchemaChangeMonitor().on_batch(ref, changed).score > 0


def test_performance_classification_and_regression():
    classification = pd.DataFrame({"y_true": [0, 1, 0, 1], "y_pred": [0, 1, 1, 1]})
    monitor = PerfClassificationMonitor(metric="accuracy", reference_score=1.0, threshold=0.1)
    result = monitor.on_batch(None, classification)
    assert result.score == pytest.approx(0.25)
    assert result.alert is True

    improved = PerfClassificationMonitor(metric="accuracy", reference_score=0.5, threshold=0.1)
    assert improved.on_batch(None, classification).alert is False

    regression = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "y_pred": [2.0, 3.0, 4.0]})
    reg = PerfRegressionMonitor(metric="rmse", reference_score=0.5, threshold=0.1)
    assert reg.on_batch(None, regression).score == pytest.approx(0.5)

    better = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "y_pred": [1.0, 2.0, 3.0]})
    assert reg.on_batch(None, better).alert is False


def test_performance_missing_columns_is_an_error():
    result = PerfClassificationMonitor(reference_score=1.0).on_batch(
        None, pd.DataFrame({"y_true": [0, 1]})
    )
    assert result.status == "error"
    assert "y_pred" in result.error


def test_llm_judge_budget_sampling_cache_and_clamping():
    budget_judge = LLMJudge(provider=FakeProvider("{}"), budget_tokens=0)
    result = budget_judge.evaluate(["q"], ["c"], ["a"])[0]
    assert result["_status"] == "budget"
    assert result["faithfulness"] is None

    sampled = LLMJudge(provider=FakeProvider("{}"), sample_rate=0.0)
    assert sampled.evaluate(["q"], ["c"], ["a"])[0]["_status"] == "sampled"

    provider = FakeProvider('{"faithfulness": 1.5, "answer_relevance": -1}')
    judge = LLMJudge(provider=provider, budget_tokens=10_000, cache=True)
    first = judge.evaluate(["q"], ["c"], ["a"])[0]
    second = judge.evaluate(["q"], ["c"], ["a"])[0]
    assert first["faithfulness"] == 1.0
    assert first["answer_relevance"] == 0.0
    assert first["_status"] == "ok"
    assert second["_status"] == "cached"
    assert provider.calls == 1

    invalid = LLMJudge(provider=FakeProvider("garbage"), budget_tokens=10_000)
    assert invalid.evaluate(["q"], ["c"], ["a"])[0]["_status"] == "error"

    failing = LLMJudge(provider=FakeProvider(RuntimeError("boom")), budget_tokens=10_000)
    assert failing.evaluate(["q"], ["c"], ["a"])[0]["_status"] == "error"


def test_llm_monitor_skips_without_results():
    judge = LLMJudge(provider=FakeProvider("{}"), budget_tokens=0)
    monitor = LLMMonitor(judge=judge)
    batch = pd.DataFrame({"question": ["q"], "context": ["c"], "answer": ["a"]})
    result = monitor.on_batch(None, batch)
    assert result.status == "skipped"
    assert result.alert is False

    missing = monitor.on_batch(None, pd.DataFrame({"question": ["q"]}))
    assert missing.status == "error"


def test_llm_monitor_scores_valid_results():
    judge = LLMJudge(
        provider=FakeProvider('{"faithfulness": 0.9, "answer_relevance": 0.8}'),
        budget_tokens=10_000,
    )
    monitor = LLMMonitor(metric="faithfulness", judge=judge, threshold=0.5)
    batch = pd.DataFrame({"question": ["q"], "context": ["c"], "answer": ["a"]})
    result = monitor.on_batch(None, batch)
    assert result.status == "ok"
    assert result.score == pytest.approx(0.1)
    assert result.alert is False


def test_damage_predictor_fit_save_load(tmp_path):
    features = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    target = np.array([0.0, 0.3, 0.25, 0.5])
    model = DamagePredictor().fit(features, target)
    prediction = model.predict(CorruptionVector(1.0, 0.0, 0.0, 0.0, 0.0))
    assert 0.0 <= prediction <= 1.0

    path = tmp_path / "damage.json"
    model.save(str(path))
    loaded = DamagePredictor.load(str(path))
    assert loaded.predict(CorruptionVector(1.0, 0.0, 0.0, 0.0, 0.0)) == pytest.approx(prediction)
    with pytest.raises(ValueError):
        loaded.predict([1.0])


def test_train_damage_writes_json(tmp_path):
    path = tmp_path / "model.json"
    train_and_save(str(path), n=300, seed=2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "driftalytics-damage-v1"
    assert len(payload["coef"]) == 5


def test_sliding_windows_include_partial_tail():
    ref = _data(300, 0)
    batch = _data(1050, 1, shift=0.3)
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
    ref = _data(300, 0)
    batch = _data(500, 1, shift=0.3)
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
    ref = _data(300, 0)
    runner = SlidingWindowRunner(
        reference=ref,
        monitors=["psi"],
        window_size=100,
        stride=100,
        sliding=True,
    )
    assert runner.update(_data(60, 1)) == []
    reports = runner.update(_data(60, 2))
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
    streamed = list(stream_runner.run_stream([_data(60, 1), _data(60, 2), _data(60, 3)]))
    assert len(streamed) == 2
    assert sum(report.n_rows for report in streamed) == 180
