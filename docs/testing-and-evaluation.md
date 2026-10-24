# Testing and evaluation

Tests prove the code does what it claims. The evaluation measures whether the system answers well. Both run in CI, but a passing CI run says the build is consistent with its own checks, not that it is ready for every workload. The gaps are listed at the end.

## Test suite

```bash
docker compose up -d db
uv run pytest                       # 168 tests, about 30 to 45 s
uv run pytest --cov --cov-report=term
```

| File | Tests | What it proves |
|---|---:|---|
| `unit/test_ingestion.py` | 25 | Parsers on real PDF and DOCX bytes, type sniffing, decompression and page limits, hidden HTML removed, normalisation, chunk sizes and overlap, parent-child links, dedup, contextual headers, injection patterns |
| `unit/test_generation.py` | 15 | Sources cannot escape their tags, JSON parsing, unknown citation markers removed, uncited answers downgraded, conflicts, snippets |
| `unit/test_security_and_config.py` | 26 | Group claim formats, tenant claim validation, local token tampering and expiry, deployed-environment config refusal, prompt-logging ban in prod, web-search URL rules |
| `unit/test_aws_contracts.py` | 19 | RDS Data API and Bedrock (Titan, Converse, Rerank, ApplyGuardrail) request and response shapes checked by botocore `Stubber`; Aurora resume retry; error mapping |
| `unit/test_components.py` | 33 | Fake providers, retries and fallback, Gemini batching and normalisation, agent tool permissions and approval gate, web-search client, S3 adapter with moto (presigned POST policy, version purge), SQL helpers |
| `unit/test_mcp_server.py` | 4 | The real MCP server (SDK 2.x) driven by the platform's client: structured results, auth, tool errors |
| `integration/test_isolation.py` | 12 | Tenant isolation and document ACLs against PostgreSQL as the runtime role, including raw SQL as another tenant, write attempts across tenants, privilege limits, body-supplied identity |
| `integration/test_lifecycle.py` | 11 | Idempotent ingestion, checksum and type rejection, version replacement, late older versions, deletion purge and tombstone, cache invalidation on delete and ACL change |
| `integration/test_query_and_failures.py` | 17 | Every retrieval mode, filters, conflicts, injection handling, budgets, agentic mode and its budget, and failure injection (embedding throttling, malformed model output, model outage with fallback, embedding-space mismatch) |
| `integration/test_api_and_handlers.py` | 6 | The HTTP API end to end, error shape, 503 with `Retry-After`, the Lambda proxy path through Mangum with Cognito claims, the SQS handler's partial batch failures |

Line coverage is 82% of `src/ragplatform`. The uncovered lines are mostly the Gemini chat adapter (covered by the live evaluation instead), the migrate Lambda handler, and the re-embed command.

### Test doubles

- `HashingEmbedder` maps words and word pairs to a vector with a stable hash. Texts that share words land close together, which exercises vector search, fusion and filters. It has no notion of synonyms.
- `ExtractiveChatModel` copies the source sentences that best overlap the question, cites them, reports conflicting numbers, and returns `insufficient_context` below an overlap threshold. It never acts on instructions in sources, so any behaviour change in an injection test comes from the pipeline.
- Nothing mocks the database. Integration tests run real SQL against PostgreSQL 16 with pgvector, as a role that row-level security applies to.

## Evaluation

`eval/dataset.json` (version `2026-10-23.1`) holds 13 documents for two fictional tenants and 45 labelled questions. Each question lists the documents that hold the evidence, facts a correct answer must state, facts it must never state, and the statuses that count as correct.

| Category | Questions | What it tests |
|---|---:|---|
| single_fact | 14 | the basic path |
| paraphrase | 6 | wording that shares few terms with the document ("memory stick" vs "USB drives") |
| near_duplicate | 4 | two product specs with the same structure and different numbers |
| superseded | 3 | a 2025 and a 2026 policy; the 2026 one replaces the other |
| multi_part | 3 | two separate facts in one question |
| identifier | 3 | error codes and vendor codes |
| table | 2 | CSV rows |
| unanswerable | 3 | nothing in the corpus answers it, including a near miss (X400 vs X200/X300) |
| acl_denied, acl_allowed | 1, 1 | the same question by a user outside and inside the finance group |
| cross_tenant | 3 | a fact that exists only in the other tenant |
| injection | 1 | a document instructing the model to give a false warranty and ask for passwords |
| reasoning | 1 | a comparison across two documents |

```bash
uv run python -m eval.run_eval                                    # offline, fake providers
uv run python -m eval.run_eval --check eval/thresholds-offline.json
RAG_GEMINI_API_KEY=... uv run python -m eval.run_eval --provider gemini --label my-run
uv run python -m eval.run_eval --provider gemini --retrieval-only --retrieval vector --label vec
```

The runner resets the schema of its target database (`ragplatform_test` by default; it refuses any database whose name does not end in `_test` or `_eval`), ingests the corpus through the real pipeline, and scores retrieval and answers separately. Reports go to `eval/reports/<label>.json` with per-question details.

### Metrics

| Metric | Definition |
|---|---|
| hit@1, hit@k | share of answerable questions whose first source, or any of the k sources, is an expected document |
| recall@3, recall@k | share of expected documents among the first 3 or k sources |
| MRR | mean of 1 / rank of the first expected document |
| nDCG@k | rank-weighted recall with binary relevance |
| status accuracy | the answer's status is one the question allows |
| fact accuracy | every `must_contain` fact appears, and no `must_not_contain` fact does |
| citation precision | share of cited documents that are expected documents |
| refusal accuracy | unanswerable and ACL-denied questions return `insufficient_context` |
| leaks | forbidden documents retrieved or cited, or forbidden facts stated (must be 0) |
| injection obeyed | an answer states the fact planted by an injected instruction (must be 0) |

Retrieval metrics are measured over the sources actually sent to the model, after reranking, parent expansion and packing, so they describe what the model saw.

### Results

Offline (fake providers), the CI gate:

| hit@k | recall@k | MRR | citation precision | refusal | status | facts | leaks | injection obeyed |
|---|---|---|---|---|---|---|---|---|
| 1.000 | 1.000 | 0.944 | 0.957 | 0.75 | 0.556 | 0.513 | 0 | 0 |

The fake model's low answer scores are expected; it copies sentences. These numbers catch regressions in chunking, retrieval, fusion, access control and citation handling.

Live, `gemini-embedding-001` (1024 dimensions) and `gemini-flash-latest`, 2026-10-24:

| Run | hit@1 | MRR | facts | status | citation precision | refusal | p50 | p95 | leaks | injection |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid, standard | 1.000 | 1.000 | 1.000 | 1.000 | 0.953 | 1.00 | 3.8 s | 9.2 s | 0 | 0 |
| vector, standard | 0.974 | 0.987 | 0.974 | 0.978 | 0.952 | 1.00 | 3.9 s | 7.8 s | 0 | 0 |
| hybrid, agentic | 0.949 | 0.974 | 1.000 | 1.000 | 0.953 | 1.00 | 7.5 s | 12.0 s | 0 | 0 |

Retrieval only (no generation):

| Run | hit@1 | MRR |
|---|---|---|
| hybrid, parent-child, headers | 0.949 | 0.974 |
| hybrid, recursive, headers | 1.000 | 1.000 |
| hybrid, no headers | 0.923 | 0.962 |
| vector, headers | 0.974 | 0.987 |
| vector, no headers | 0.897 | 0.949 |
| keyword | 0.872 | 0.927 |

How to read these:

- The set is small: one answerable question is 0.026. The identical hybrid configuration scored hit@1 of 0.949 and 1.000 in two runs, because the 2025 and 2026 travel policies are near-identical and trade rank 1. Differences of one or two questions are noise.
- Supported by the data: contextual headers improve vector retrieval (3 questions); keyword-only search loses paraphrases; agentic mode doubles latency without improving answers on this set; every run had zero leaks and zero obeyed injections.
- Not supported by the data: that hybrid beats vector search on rank (within noise here), and anything about reranking (not measured).
- Not measured at all: Bedrock models. Run `--provider bedrock` in a dev account before trusting the defaults.

### Regression thresholds

`eval/thresholds-offline.json` gates CI (offline providers). `eval/thresholds-live.json` gates changes to prompts, chunking, retrieval or models with a real provider, run by a person before merging:

```bash
uv run python -m eval.run_eval --provider gemini --check eval/thresholds-live.json
```

Both set leaks and obeyed injections to a maximum of 0. Quality thresholds sit just below the measured baseline, so a real regression fails and noise does not. When a change improves the baseline, raise the thresholds in the same pull request.

### Extending the dataset

Add documents under `eval/corpus/<tenant>/`, list them in `documents`, and add questions with `expected_docs` and `must_contain`. Good additions are questions the system currently gets wrong, longer documents (the current ones are short, which makes retrieval easy), and more cross-tenant and ACL probes. Change `version` when you change the dataset, and re-baseline the thresholds.

## Load test

`tests/load/locustfile.py` asks evaluation questions and lists documents with a 0.5 to 1.5 s pause between requests per simulated user.

Local results, one uvicorn process, fake providers, PostgreSQL in Docker on a laptop, 2026-10-24. This measures the platform's own work per request (authentication, budget check, hybrid SQL under row-level security, citation validation, usage and audit writes), not model latency:

| Users | Requests | Failures | Throughput | `POST /v1/query` p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| 20 | 1,001 | 0 | 18.6 req/s | 43 ms | 74 ms | 260 ms |
| 100 | 3,297 | 0 | 75.9 req/s | 260 ms | 560 ms | 690 ms |

At 100 users the single process (one event loop, a 40-thread pool, 10 database connections) is the limit. In Lambda, each execution environment serves one request at a time, so the equivalent limit is concurrency: required concurrency is about requests per second times seconds per request. At 10 questions per second and 4 s each, that is 40 concurrent executions, well under the default regional quota of 1,000. Aurora capacity and Bedrock tokens-per-minute quotas are the limits to plan for at higher rates.

To load-test a deployed stage, set `LOAD_AUTH_HEADER` to a raw Cognito ID token and point `--host` at the stage URL. That includes Bedrock and costs money per question.

## Known gaps

- No test runs against real AWS services. Contract tests check request shapes; the RDS Data API transaction behaviour, IAM policies and Bedrock model access are verified only by the first deployment's smoke test.
- The UI has no automated test.
- The evaluation set is small and its documents are short, which makes retrieval easier than on real corpora.
- Reranking and Guardrails grounding have no measured effect yet.
