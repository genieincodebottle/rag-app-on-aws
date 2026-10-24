# AWS service selection

Each choice below names the alternatives considered and why the selected service won for this workload: a multi-tenant document question-answering service with bursty, often idle traffic, strict tenant isolation, and documents that change.

Facts about AWS services were checked against AWS documentation on 2026-10-23 and 2026-10-24. Service behaviour changes; re-check the linked pages before relying on a limit.

## Retrieval store: Aurora PostgreSQL Serverless v2 with pgvector

| Option | Strengths | Why not chosen |
|---|---|---|
| **Aurora PostgreSQL Serverless v2 + pgvector** (chosen) | Vector (HNSW) and keyword (GIN full-text) search in one SQL statement; row-level security for tenant and document isolation; transactional version swaps and deletes; scales to 0 ACU when idle (PostgreSQL 16.3+); RDS Data API means no VPC for Lambda | Not built for billions of vectors; first request after a pause waits about 15 s |
| OpenSearch Serverless | Mature hybrid search, scales very large; collection groups can now scale to 0 OCU when idle | Billed per OCU-hour while active, in steps of whole OCUs, which is more than Aurora's half-ACU steps at this workload's size; tenant isolation through filters or separate collections; no transactions across documents and chunks, so a version swap is not atomic |
| Amazon S3 Vectors (GA December 2025) | Very low storage cost, up to 2 billion vectors per index | No keyword or hybrid search found in the documentation; metadata filters only, so tenant isolation is a query-time convention; up to 100 results per query |
| Bedrock Knowledge Bases | Managed parsing, chunking and sync | Ingestion is a sync job per data source, harder to tie to per-document versions and deletes; isolation by metadata filter; chunking and retrieval internals are the subject this repository teaches, so hiding them defeats its purpose |
| RDS PostgreSQL (provisioned) | Same SQL as Aurora | No scale to zero and no Data API on smaller instance classes, so Lambdas would need a VPC, NAT or endpoints, and pooling |

pgvector on Aurora PostgreSQL 16.13 is 0.8.1, the same version the local Docker image runs. 0.8 added iterative index scans (`hnsw.iterative_scan`), which keep HNSW scanning when filters remove candidates; the retriever turns it on per transaction.

## Compute: Lambda, three functions, outside a VPC

| Option | Why |
|---|---|
| **Lambda** (chosen) | Pay per request, scales from zero, one deployment artifact. Ingestion is naturally event-driven. A 29 s API Gateway integration limit is enough: measured p95 answer latency was 9.2 s with Gemini Flash |
| ECS Fargate | Better for long-lived connections or streaming. Always-on tasks cost money while idle, and nothing here needs a long-lived process |
| Step Functions for ingestion | Worth it for multi-step jobs that run longer than 15 minutes (OCR, very large PDFs). Ingestion here is one function call under the 5-minute timeout; the size and page limits keep it there. Adding Step Functions is the documented path if those limits are raised |

The functions run on arm64 (Graviton), which Lambda prices lower than x86, with Python 3.13 (supported to June 2029).

## Data access: RDS Data API instead of VPC-attached Lambdas

See [ADR 0003](adr/0003-data-api-instead-of-vpc-lambdas.md). In short, VPC-attached Lambdas reaching Bedrock, S3, SQS and Secrets Manager privately need either a NAT gateway or interface endpoints for Bedrock, SQS and Secrets Manager in each availability zone (S3 can use a free gateway endpoint). Both cost more per month than the rest of an idle dev stack. The Data API is an IAM-authenticated HTTPS endpoint, so the functions need no network path to the database. Aurora sits in subnets with no route out and a security group with no inbound rules.

Data API limits that shaped the code: 1 MiB response, 45 s default statement timeout, transactions roll back if unused for 3 minutes, no array parameters, and session state does not survive between calls because connections are pooled and reused, which is why row-level-security settings are set per transaction with `set_config(..., true)`.

## Models: Amazon Bedrock

| Purpose | Default | Alternatives |
|---|---|---|
| Embeddings | `amazon.titan-embed-text-v2:0`, 1024 dimensions (also 512, 256) | Cohere Embed v4 (`cohere.embed-v4:0`). Changing model or width needs `ragplatform reembed` |
| Generation | `us.amazon.nova-2-lite-v1:0` (geo inference profile; not available in-region in us-east-1) | `us.anthropic.claude-haiku-5-5` (prod fallback), `us.amazon.nova-micro-v1:0` for lowest cost |
| Reranking | off | `cohere.rerank-v3-5:0`. Amazon Rerank 1.0 is not offered in us-east-1 |
| Safety | off in dev | Bedrock Guardrails: prompt-attack filter on the question, secret and card-number blocking, contextual grounding check |

Bedrock keeps traffic and data inside the AWS account, uses IAM instead of API keys, and offers Guardrails. Gemini is supported only for local development (free tier, real model quality), and configuration validation refuses it in deployed environments.

Not measured: retrieval and answer quality with Titan and Nova on the evaluation set. The numbers in [testing-and-evaluation.md](testing-and-evaluation.md) are from Gemini models. Run the evaluation with `--provider bedrock` in a dev account before relying on the defaults.

## Agent runtime: in-process, not Bedrock AgentCore

AgentCore (GA October 2025) provides session runtimes up to 8 hours, managed memory, identity and a tool gateway. That is the right tool for long-running or resumable agents, many tools, or per-session isolation. The agent here makes at most four bounded steps inside one API request, uses one read-only tool, and must finish within the API's 29 s. An in-process loop is cheaper, testable offline, and adds no service to operate. See [ADR 0006](adr/0006-bounded-in-process-agent.md).

## Identity: Cognito user pools

Users are created by administrators only (no self-sign-up), assigned a tenant through `custom:tenant_id`, which the app client cannot write, and put in groups that drive document access. MFA is optional in dev and required in prod. API Gateway's Cognito authorizer validates tokens before the function runs. An external identity provider (SAML or OIDC federation into the user pool) can replace passwords without changing the application, because the application reads only the claims.

## Supporting services

| Service | Role |
|---|---|
| KMS | One customer-managed key per environment for S3, SQS, Aurora, secrets, Lambda environment variables and log groups |
| Secrets Manager | Aurora master secret (managed and rotated by RDS) and the runtime-role secret (written by the migrate function) |
| SQS | Ingestion buffer and DLQ, with event-source concurrency capped as backpressure |
| CloudWatch, X-Ray | Structured logs, EMF metrics, traces, dashboard, alarms |
| WAF | Managed common and known-bad-input rule groups, per-IP rate limit (test and prod) |
| AWS Budgets | Monthly cost alerts on the `Project` cost-allocation tag |

Deliberately not used: NAT gateways, VPC interface endpoints, RDS Proxy (blocks Aurora auto-pause, and the Data API makes it unnecessary), ElastiCache (the answer cache is a PostgreSQL table keyed by tenant and invalidated by corpus version), and API Gateway response caching (answers are per user).
