"""LLM monitor tests."""

import pandas as pd
import pytest
from helpers import FakeProvider

from driftalytics.llm import LLMJudge
from driftalytics.monitors.llm_monitors import LLMMonitor


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
