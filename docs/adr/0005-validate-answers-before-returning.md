# 0005. Validate every answer before returning it (no token streaming)

- Status: accepted, 2026-10-23

## Context

A RAG answer is only useful if each claim can be traced to a document the caller may read. Models sometimes cite sources they were not given, or answer without citing at all.

## Decision

The model returns one JSON object with `[S#]` markers. The server parses it, removes markers that name no supplied source, downgrades an uncited answer to `insufficient_context`, optionally runs the Guardrails grounding check, and only then returns the response. Responses are not streamed.

## Alternatives

- **Stream tokens and validate afterwards**: the user would read text the server may then retract.
- **Native Converse citations**: tied to document content blocks and to models that support them (not confirmed for Nova); the marker scheme works with every provider, including the offline test double.

## Consequences

- Time to first token equals time to full answer (measured p50 3.8 s with Gemini Flash).
- API Gateway REST supports response streaming through Lambda, so streaming could be added for a mode that accepts unvalidated text; it is not implemented.
- `InvalidCitationsRemoved` and `AnswerFormatRetry` make model drift visible.
