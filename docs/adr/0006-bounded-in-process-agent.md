# 0006. A bounded, opt-in agent loop inside the API function

- Status: accepted, 2026-10-24

## Context

Multi-part questions can need more than one search. An agent that plans and searches can find each part, at the cost of latency and tokens.

## Decision

Implement a fixed sequence (plan, search the question and each sub-question, optional web search, answer, one keyword retry) inside the API function, with step, token and wall-clock budgets checked before every step. Tools are registered with schemas and permissions; their scope comes from the server. Agentic mode is off unless the tenant enables it and the request asks for it.

## Alternatives

- **Bedrock AgentCore**: suited to long-running, resumable or multi-tool agents. This agent finishes in seconds with one read-only tool; a managed runtime adds a service and cost without changing the result.
- **Agentic by default**: rejected on measurement. On the evaluation set, agentic mode matched standard mode's accuracy (fact accuracy 1.0 for both) at twice the median latency (7.5 s against 3.8 s) and 28% more input tokens.
- **Open-ended tool-calling loop**: unbounded cost and harder to test.

## Consequences

- Agentic behaviour is deterministic in structure and testable offline.
- The approval gate for write tools exists but has no write tool behind it.
- Re-evaluate on larger corpora, where one search is less likely to cover every part of a question.
