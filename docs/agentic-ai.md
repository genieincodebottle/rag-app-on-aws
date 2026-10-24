# Agentic RAG

The platform has two answering modes. Standard mode makes one retrieval and one model call. Agentic mode adds a planning call and extra retrievals, inside fixed budgets. This guide covers when agentic mode helps, how it is bounded, and how it was evaluated.

## When it is worth it

Measured on the evaluation set with Gemini Flash (45 questions):

| Mode | Fact accuracy | Status accuracy | p50 latency | p95 latency | Input tokens (45 questions) |
|---|---|---|---|---|---|
| standard (hybrid) | 1.000 | 1.000 | 3.8 s | 9.2 s | 31,464 |
| agentic (hybrid) | 1.000 | 1.000 | 7.5 s | 12.0 s | 40,212 |

Agentic mode was as accurate, twice as slow at the median, and used 28% more input tokens. On this corpus one hybrid search already finds the evidence for multi-part questions, because the corpus is small and each part has distinctive terms. The evidence does not justify agentic mode as the default, so it is off unless a tenant enables it (`allow_agentic`) and the request asks for it (`"mode": "agentic"`).

Agentic mode earns its cost when one search cannot retrieve every part of a question: large corpora where each part competes for the top results with many near matches, or questions that need a second search based on the first answer. Measure it on your own data before enabling it for a tenant.

## The loop

`agent/runner.py` runs a fixed sequence, not an open-ended loop:

```
plan (1 model call, JSON)
  -> search the full question
  -> search each sub-question (at most 3)
  -> web search, only if allowed and nothing was found
  -> answer (the same generator and citation checks as standard mode)
  -> if insufficient: one keyword-only search, then answer again
```

1. **Plan.** The planner returns `{"strategy": "direct" | "decompose", "sub_questions": [...]}`. The `Plan` model validates it: at most three sub-questions, each at least two words and at most 300 characters. If the reply is not valid JSON or fails validation, the agent continues with `direct`, which is standard mode plus the planning call.
2. **Search.** Each search is a call to the `search_documents` tool, which runs the normal retriever under the caller's access scope. Results are merged by chunk id.
3. **Answer.** Sources are ranked by fused score and packed into the same context budget as standard mode.
4. **Retry.** If the answer is `insufficient_context` and budget remains, a keyword-only search runs once. Exact-term matching finds identifiers and error codes that embeddings blur.

## Budgets

Every step is checked against three budgets before it starts (`Budget.allows`):

| Budget | Setting | Default |
|---|---|---|
| Steps (model calls plus tool calls) | `RAG_AGENT_MAX_STEPS` | 4 |
| Tokens | `RAG_AGENT_MAX_TOKENS` | 24,000 |
| Wall clock | `RAG_AGENT_DEADLINE_S` | 25 s (the API Gateway limit is 29 s) |

A planning or search step that would exceed a budget is recorded as `skipped` in the trace and not run. The answer step always runs afterwards, using the evidence gathered so far, so a budget never leaves the caller without a response. `AgentBudgetExhausted` is emitted as a metric. The default of four steps covers the plan and three searches (the full question plus two sub-questions); a plan with three sub-questions skips its last search. `test_agent_stops_at_its_step_budget` checks this with a budget of two.

The per-tenant daily token budget also counts every agent call.

## Tools and permissions

`agent/tools.py` defines the tool contract:

- A tool is registered with a name, an argument model, a permission (`read` or `write`) and a handler.
- Arguments are validated with pydantic; `extra="forbid"` rejects any field the schema does not name. A model cannot add `"tenant_id": "globex"` to a search.
- The handler receives a `ToolContext` built by the server: the caller's `AccessScope`, filters and request id. Scope never comes from model output.
- `write` tools run only if an approver callback returns true for that exact call. No write tool is registered, so the agent cannot change data. The gate is there so that adding one cannot skip human approval; `test_write_tool_needs_approval` covers it.

| Tool | Permission | Registered when |
|---|---|---|
| `search_documents` | read | always |
| `web_search` | read | `RAG_WEB_SEARCH_URL` is set and the tenant has `allow_web_search` |

## Web search

The optional web-search tool calls the MCP server in `mcp_servers/` (SerpApi). Controls:

- The endpoint is configuration, HTTPS only (or `http://localhost` locally). The previous implementation took the URL from the request body, which let any caller make the function send requests to an address of their choice.
- Bearer key on every call; the server rejects other callers.
- No redirects (a redirect to an internal address is a classic server-side request forgery), 10 s timeout, 256 KB response cap, `https://` links only.
- A tool failure (the MCP result has `isError: true`) is raised, not read as "no results".
- Results enter the prompt flagged as `external_web` and carry the note that tells the model to treat them as data.
- Web results are used only when document search found nothing, so they never override the tenant's documents.

Run the server locally:

```bash
cd mcp_servers && pip install -r requirements.txt
SERPAPI_API_KEY=... MCP_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))") \
  python web_search_mcp_server.py --host 127.0.0.1 --port 8765
```

Then set `RAG_WEB_SEARCH_URL=http://localhost:8765/mcp` and `RAG_WEB_SEARCH_API_KEY` to the same key, and enable it for a tenant with `ragplatform tenant acme "Acme Corp" --agentic --web`. On AWS the server must be reachable over HTTPS from Lambda; hosting it is outside this stack.

## Traces

Every agentic response includes `agent_trace`. An example (timings illustrative):

```json
[
  {"step": 1, "kind": "plan", "name": "planner", "ms": 812.4, "strategy": "decompose", "sub_questions": 2},
  {"step": 2, "kind": "tool", "name": "search_documents", "ms": 95.1, "results": 6, "query_chars": 78},
  {"step": 3, "kind": "tool", "name": "search_documents", "ms": 61.0, "results": 6, "query_chars": 41},
  {"step": 4, "kind": "skipped", "name": "search_documents", "ms": 0.0, "reason": "budget"},
  {"step": 5, "kind": "answer", "name": "generator", "ms": 2410.7, "status": "answered", "citations": 2}
]
```

The trace records lengths and counts, not query text, so it is safe to log. With X-Ray enabled, `AgentRunner.run`, each retrieval and each generation also appear as subsegments.

## Failure handling

| Failure | Behaviour |
|---|---|
| Planner returns invalid JSON or an invalid plan | Continue as `direct`; trace records `invalid plan, using direct` |
| A tool call has invalid arguments or an unknown name | Recorded in the trace with the error; the agent continues |
| Web search is down or throttled | Recorded; the agent answers from documents |
| Model unavailable during the answer step | Fallback model if configured; otherwise 503 with `Retry-After` |
| Budget exhausted | Remaining steps skipped; answer from evidence so far |

## Evaluating agent changes

```bash
python -m eval.run_eval --provider gemini --mode agentic --label my-agent-change
python -m eval.run_eval --provider gemini --mode standard --label baseline
```

Compare `fact_accuracy`, `latency_ms_p50`, `input_tokens_total` and the `multi_part` row of `by_category`. A change to the agent is worth shipping only if it improves the questions it targets without making the rest slower or more expensive.
