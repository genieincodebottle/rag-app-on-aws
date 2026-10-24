# Implementation plan

This plan turns the audited codebase ([current-state assessment](../audit/current-state-assessment.md)) into a multi-tenant RAG platform on AWS. It records the target architecture, the gaps it closes, the milestones, and how each milestone is accepted.

## Target architecture

```
                       +------------------------------+
  Streamlit / curl --> | API Gateway (REST, Regional) |  Cognito authorizer, WAF, stage throttling
                       +--------------+---------------+
                                      |
                       +--------------v---------------+
                       | api Lambda (FastAPI+Mangum)  |  auth context from verified claims only
                       |  query, documents, admin     |
                       +--+-----------+-----------+---+
                          |           |           |
            RDS Data API  |   Bedrock |           | presigned POST
                          |           |           v
 +------------------------v--+   +----v------+  +-------------+   S3 event   +-----------+
 | Aurora PostgreSQL         |   | Titan V2  |  | S3 documents| -----------> | SQS queue | -> DLQ
 | Serverless v2 (0-N ACU)   |   | Converse  |  | (KMS, vers.)|              +-----+-----+
 | pgvector HNSW + GIN FTS   |   | Rerank    |  +-------------+                    |
 | row-level security        |   | Guardrails|                       +------------v-----------+
 +-------------^-------------+   +----^------+                       | ingest Lambda          |
               |                      |                              | parse, chunk, dedup,   |
               +----------------------+------------------------------+ embed, swap version    |
                                                                     +------------------------+
```

Three Lambda functions, each with its own IAM role.

| Function | Trigger | Job |
|---|---|---|
| `api` | API Gateway | Query answering, document registration, upload URLs, listing, deletion, health |
| `ingest` | SQS (fed by S3 events) | Parse, normalise, chunk, deduplicate, embed, write a new document version atomically |
| `migrate` | Invoked by the deploy pipeline | Apply SQL migrations, create the application database role and store its credentials |

The Lambdas run outside a VPC and reach Aurora through the RDS Data API, Bedrock, S3 and SQS through IAM-authenticated AWS endpoints. Aurora sits in private subnets with no inbound rules. There is no NAT gateway. The [service selection guide](../aws-service-selection.md) and the ADRs record the alternatives and why they lost.

### Request flow for a question

1. API Gateway verifies the Cognito ID token. The Lambda builds an `AuthContext` (user, tenant, groups) from the verified claims only. The request body cannot change it.
2. The usage service checks the tenant's daily query and token budget.
3. The answer cache is checked. A cache key includes the tenant, a hash of the caller's groups, the normalised question, the mode and the model, and an entry is valid only for the tenant's current corpus version.
4. The question is embedded. Vector search and full-text search run in one SQL statement inside a transaction that sets the tenant and groups for row-level security. Results are fused with reciprocal rank fusion.
5. Optional reranking (Bedrock Rerank). Child chunks are expanded to their parent sections, and the context is packed to a token budget.
6. The model answers in a fixed JSON shape with `[S1]`-style citation markers. The server checks every marker against the supplied sources and refuses answers with no valid citation.
7. The response returns the answer, citations (document, version, page, section), status (`answered`, `insufficient_context`, `conflicting_sources`), token usage and estimated cost.

### Ingestion flow

1. `POST /documents` registers the document (title, type, size, SHA-256, access groups) and returns a presigned S3 POST limited to the declared size.
2. The client uploads straight to S3, so file size is not bounded by API Gateway or Lambda payload limits.
3. S3 sends the event to SQS. The ingest Lambda claims the version with a conditional update, so a duplicate event does nothing.
4. The file is validated (size, magic bytes against the declared type, page and decompression limits), parsed, normalised and chunked.
5. Chunks identical to ones already embedded in the same embedding space reuse the stored vector.
6. The new version's chunks are written and made active in one transaction, the previous version's chunks are deleted, and the tenant's corpus version is bumped, which invalidates cached answers.
7. A failure marks the version `failed` with an error code. After five receives the message moves to the DLQ, and an alarm fires.

## Gap analysis

| Requirement | State at 81fa2a6 | Plan |
|---|---|---|
| Tenant isolation, document ACLs | None, user id from request body | Claims-only auth context, RLS on tenant and groups, tenant parameter in every query, cross-tenant tests |
| Multiple formats, validation | PDF, TXT, CSV with no validation | PDF, DOCX, HTML, Markdown, TXT, CSV with type sniffing and size, page and decompression limits |
| Chunking strategies | Fixed 1,000 characters | Recursive, Markdown-heading-aware, parent-child, plus optional contextual headers, configurable per deployment |
| Dedup, idempotency | None | Document SHA-256, chunk hashes, embedding reuse, conditional version claim |
| Versioning, deletion | None | Versions with atomic swap, tombstoned deletes, S3 version purge, cache invalidation |
| Hybrid retrieval, filters, rerank | Vector top 5 only | Vector, keyword, hybrid RRF, metadata filters, optional rerank, parent expansion |
| Citations, grounding | None | Server-validated citation markers, insufficient-context and conflict statuses, optional Guardrails grounding check |
| Agentic RAG | Web search triggered by a threshold | Bounded planner (decompose, retrieve, verify) with step, token and time budgets, read-only tools, approval gate for any write tool |
| Failure handling | Zero vectors, 200 on failure, no DLQ | Typed errors, retries with backoff and jitter, model fallback, DLQ, honest HTTP codes |
| Rate limits, budgets | None | API Gateway throttling, WAF rate rule, per-tenant daily budgets |
| Observability | Two error alarms | Structured JSON logs with correlation ids, EMF metrics (latency, tokens, cost, cache, grounding), X-Ray, dashboard, alarms |
| Evaluation | Self-grading with invented 0.5 scores | Versioned dataset, retrieval metrics (recall, MRR, nDCG), citation and refusal metrics, regression thresholds in CI |
| Tests | Mocks of every boundary | Unit, integration against real PostgreSQL with pgvector and moto, end-to-end, security, failure injection, load |
| IaC, CI/CD | Copied roots, static keys, `-lock=false` | One root with per-environment var files, S3 state locking, GitHub OIDC, plan then approved apply, migrations, smoke tests, rollback by alias |

## Milestones

Each milestone ends with its tests passing locally and nothing claimed that was not run.

| # | Milestone | Acceptance criteria |
|---|---|---|
| M0 | Audit and plan | Assessment and this plan in `docs/` |
| M1 | Core library and local stack | `docker compose up` gives PostgreSQL with pgvector; migrations apply; ingest-to-answer works with the `fake` provider; unit and integration tests pass |
| M2 | Security | Claims-only auth; RLS proven by tests that query as another tenant and as a user outside a document's groups; injection and SSRF tests pass; local auth refuses to start inside Lambda |
| M3 | Retrieval quality and generation | Hybrid, filters, parent expansion, citation validation, statuses; evaluation runner produces retrieval and answer metrics on the versioned dataset |
| M4 | Providers | Bedrock adapters contract-tested with botocore stubs; Gemini adapter verified live locally; embedding-space fingerprint enforced |
| M5 | Agentic mode | Bounded planner with budgets and traces; tests for step limits, tool permission, approval gate, injection in tool output |
| M6 | AWS runtime | Lambda handlers, Data API executor, S3 and SQS adapters; handler tests with moto; partial batch failure tests |
| M7 | Infrastructure | Terraform modules and one root pass `terraform validate` and checkov; per-environment var files; bootstrap stack for state and OIDC |
| M8 | CI/CD | CI runs lint, types, tests, evaluation thresholds, dependency audit, IaC scan, SBOM; deploy workflow with plan, approval, apply, migrate, smoke test, rollback |
| M9 | Observability and cost | Metrics and dashboards defined in code; cost model script with sourced prices; load test results recorded |
| M10 | UI | Thin Streamlit client against the API |
| M11 | Documentation | The 16 guides and ADRs, each checked against the code |
| M12 | Learning module | Blog-format project in the LLM track of the app |

## Repository structure

```
rag-app-on-aws/
  src/ragplatform/
    config.py            settings, validated per environment
    domain/              typed models and errors
    security/            auth context, local dev tokens, injection screening, redaction
    ingestion/           parsers, normalisation, chunking, pipeline
    providers/           embedder, chat model, reranker ports with bedrock, gemini, fake adapters
    store/               SQL executors (psycopg, Data API), migrations, repository
    retrieval/           hybrid search, fusion, rerank, context packing
    generation/          prompts, answer parsing, citation validation
    agent/               bounded planner, tools, budgets, traces
    services/            query, documents, usage budgets, answer cache
    observability/       logging, metrics, tracing, cost
    api/                 FastAPI app, routes, schemas
    handlers/            Lambda entry points (api, ingest, migrate)
    adapters/            blob storage and queue (S3, SQS, local)
  ui/                    Streamlit client
  mcp_servers/           optional web-search MCP server
  eval/                  dataset, runner, thresholds, reports
  tests/                 unit, integration, e2e, security, failure, contract, load
  infra/terraform/       bootstrap, modules, platform root, envs
  .github/workflows/     ci, deploy, destroy
  docs/                  guides, ADRs, audit, plan
```

## Test strategy

| Layer | Scope | Runs where |
|---|---|---|
| Unit | Chunkers, parsers, fusion, citation validation, budgets, config validation, prompt building | Everywhere, no network |
| Integration | Repository SQL, RLS, migrations against PostgreSQL 16 with pgvector; S3 and SQS adapters against moto | Local Docker, CI service container |
| End-to-end | Upload, ingest, query, cite, version, delete through the FastAPI app with the `fake` provider | Local, CI |
| Contract | Bedrock request and response shapes through botocore `Stubber` | Everywhere |
| Security | Cross-tenant reads, ACL bypass attempts, body-supplied identity, prompt injection in documents and tool output, SSRF, oversized and mislabelled files | Local, CI |
| Failure injection | Embedding timeouts, throttling, model failure and fallback, database resume errors, partial SQS batches | Local, CI |
| Evaluation | Retrieval and answer metrics on the versioned dataset with regression thresholds | Offline in CI with deterministic providers; live locally with a real provider |
| Load | Locust against the local API | Local, results recorded in the performance guide |

Live AWS verification (Bedrock calls, Data API transactions, a full `terraform apply`) needs an AWS account with spend approved by its owner. Until that happens, the documentation says so wherever a behaviour depends on it.
