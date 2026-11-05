# LLM evaluation

LLM judging is cost-bounded by design: identical requests are cached, traffic can be
sampled, and every run has a token budget. There is no batching layer in v0.2.

## Setup

No provider SDK is required. The judge uses `urllib` against the provider HTTP APIs.

```bash
export OPENAI_API_KEY=...       # OpenAI
export ANTHROPIC_API_KEY=...    # Anthropic
# Ollama needs no key; run it locally on the default port
```

## Basic usage

```python
from driftalytics.llm import LLMJudge

judge = LLMJudge(
    provider="openai",          # "openai", "anthropic", or "ollama"
    model="gpt-4o-mini",
    metrics=["faithfulness", "answer_relevance"],
    budget_tokens=50_000,       # hard cap for this judge instance
    sample_rate=0.1,            # judge about 10% of requests
    cache=True,                 # reuse identical (question, context, answer) triples
)
results = judge.evaluate(questions, contexts, answers)
```

Each result is a dict with the requested metric keys plus `_status` and `_error`.

## Local judging

```python
judge = LLMJudge(
    provider="ollama",
    model="llama3.2",
    metrics=["faithfulness"],
    host="http://localhost:11434",
)
```

No API key is required. The model must already be available to the local Ollama server.

## Cost controls

- `budget_tokens`: stops issuing requests once the estimated spend would exceed the cap.
  Items skipped for budget have `_status="budget"`.
- `sample_rate`: a value below 1.0 randomly skips items. Skipped items have
  `_status="sampled"`. This is uniform sampling, not stratified by segment.
- `cache`: keyed by model and the exact question, context, and answer text. Hits have
  `_status="cached"` and cost no tokens.
- `timeout` and `retries`: retry 429 and 5xx responses with exponential backoff.
- `seed`: controls the sampling RNG for reproducible runs.

Token counts come from provider usage fields when available. Otherwise the judge uses a
rough `characters / 4` estimate, so the budget is protective rather than exact.

## Status handling

`LLMJudge.evaluate` never raises for provider or parsing failures. It returns per-item
statuses so one bad response does not stop a batch. The `faithfulness` and
`answer_relevance` monitors count only usable scores. If none are usable, the monitor
returns `status="skipped"` rather than an alert based on missing data.

Prompt injection is limited by treating the question, context, and answer as data inside
XML-like tags and asking for JSON only. This reduces risk but does not eliminate it; do
not send untrusted content to a judge with tool access.

## Reporting

LLM monitor results appear in the same `Report` as tabular and performance results, so
quality signals sit next to drift signals in the HTML and JSON output.

The repository does not include a human-labeled LLM evaluation set, so no
judge-versus-human agreement numbers are published.
