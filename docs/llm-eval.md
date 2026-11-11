# LLM evaluation

LLM judging is cost-bounded by design: identical requests are cached, traffic can be
sampled, and every run has a token budget. There is no batching layer in v0.3.

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
    cache=True,                 # reuse identical requests
    max_output_tokens=256,      # cap the judge's reply
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
  Items skipped for budget have `_status="budget"`. The judge reserves the estimated prompt
  cost before a request, reconciles actual usage on success, and charges the reservation on
  provider or parse failures. Invalid responses therefore cannot bypass the cap.
- `sample_rate`: a value below 1.0 randomly skips items. Skipped items have
  `_status="sampled"`. This is uniform sampling, not stratified by segment.
- `cache`: keyed by provider, model, metric list, prompt version, and the exact question,
  context, and answer. Hits have `_status="cached"` and cost no tokens.
- `max_output_tokens`: forwarded to each provider as an output cap. Non-positive values are
  rejected.
- `timeout` and `retries`: retry 429 and 5xx responses with jittered exponential backoff and
  honor `Retry-After` when present.
- `seed`: controls the sampling RNG for reproducible runs.

Token counts come from provider usage fields when available. Otherwise the judge uses a
rough `characters / 4` estimate, so the budget is protective rather than exact.

## Status handling

`LLMJudge.evaluate` never raises for provider or parsing failures. It returns per-item
statuses so one bad response does not stop a batch. The `faithfulness` and
`answer_relevance` monitors count only usable scores. If none are usable, the monitor
returns `status="skipped"` rather than an alert based on missing data.

Prompt injection is reduced by sending a separate system instruction, serializing the
question, context, and answer as a JSON data block, and asking for JSON only. The payload is
still untrusted text that reaches the provider, so this reduces risk but does not eliminate
it. Do not send untrusted content to a judge with tool access.

The cache key includes the prompt version (`PROMPT_VERSION`). Changing the prompt invalidates
old cache entries instead of silently reusing scores from a different rubric.

## Reporting

LLM monitor results appear in the same `Report` as tabular and performance results, so
quality signals sit next to drift signals in the HTML and JSON output.

The repository does not include a human-labeled LLM evaluation set, so no
judge-versus-human agreement numbers are published.
