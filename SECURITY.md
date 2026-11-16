# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.5.x | Yes |
| Earlier | No |

## Reporting a vulnerability

Do not open a public issue for a security problem. Report it privately through
[GitHub Security Advisories](https://github.com/DhanushPillay/Driftalytics/security/advisories/new).

Include the affected version, a minimal reproduction, and the impact you expect. We will
acknowledge the report as soon as possible and coordinate a fix and disclosure date.

## Scope

Driftalytics runs locally and does not operate a hosted service. Relevant reports include:

- Code execution or path traversal through dataset, model, or report inputs.
- HTML or JSON output that escapes its intended context.
- Prompt injection that causes an LLM judge to fabricate a score.
- Leakage of API keys or judged content through logs, caches, or error messages.

LLM providers are third-party services. Report provider-side issues to the provider.
