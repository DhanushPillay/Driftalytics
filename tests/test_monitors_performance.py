"""Performance monitor tests."""

import pandas as pd
import pytest

from driftalytics.monitors.performance import (
    PerfClassificationMonitor,
    PerfRegressionMonitor,
)


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
