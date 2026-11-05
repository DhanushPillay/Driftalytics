"""LLM monitors (faithfulness, answer relevance)."""

from __future__ import annotations

import numpy as np

from ..core import Monitor, MonitorResult
from ..llm import LLMJudge

REQUIRED_COLUMNS = ("question", "context", "answer")


class LLMMonitor(Monitor):
    def __init__(
        self,
        metric: str = "faithfulness",
        judge: LLMJudge | None = None,
        threshold: float = 0.5,
    ):
        self.metric = metric
        self.judge = judge
        self.threshold = threshold
        self.name = metric

    def on_batch(self, reference, batch) -> MonitorResult:
        assert self.threshold is not None
        if self.judge is None:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=self.threshold,
                status="skipped",
                details={"reason": "no judge configured"},
            )
        missing = [column for column in REQUIRED_COLUMNS if column not in batch.columns]
        if missing:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=self.threshold,
                status="error",
                error=f"missing required column(s): {', '.join(missing)}",
            )
        if batch.empty:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=self.threshold,
                status="skipped",
                details={"reason": "empty batch"},
            )

        scores = self.judge.evaluate(
            batch["question"].tolist(),
            batch["context"].tolist(),
            batch["answer"].tolist(),
        )
        valid = [item[self.metric] for item in scores if item.get(self.metric) is not None]
        statuses: dict[str, int] = {}
        for item in scores:
            status = str(item.get("_status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1

        if not valid:
            return MonitorResult(
                name=self.name,
                score=0.0,
                alert=False,
                threshold=self.threshold,
                status="skipped",
                details={
                    "reason": "no judge results",
                    "evaluated": 0,
                    "total": len(scores),
                    "statuses": statuses,
                },
            )

        mean = float(np.mean(valid))
        score = float(1.0 - mean)
        return MonitorResult(
            name=self.name,
            score=score,
            alert=score > self.threshold,
            threshold=self.threshold,
            details={
                f"mean_{self.metric}": mean,
                "evaluated": len(valid),
                "total": len(scores),
                "statuses": statuses,
            },
        )


class FaithfulnessMonitor(LLMMonitor):
    def __init__(self, **kwargs):
        super().__init__(metric="faithfulness", **kwargs)


class AnswerRelevanceMonitor(LLMMonitor):
    def __init__(self, **kwargs):
        super().__init__(metric="answer_relevance", **kwargs)
