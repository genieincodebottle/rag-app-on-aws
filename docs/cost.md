# Cost

Monthly cost by scenario, what drives it, and the controls that cap it. The numbers come from `scripts/cost_model.py`, which holds every unit price with its source and every assumption in one file. Run it after changing an assumption:

```bash
uv run python scripts/cost_model.py          # tables
uv run python scripts/cost_model.py --json   # prices, quantities, assumptions and results
```

Unit prices are us-east-1 on-demand, read from the AWS Price List API on 2026-10-24. They are estimates for planning; the AWS bill is the record. After a month of real traffic, compare with Cost Explorer (filter on the `Project` tag) and correct the assumptions in the script.

## Scenarios

| Scenario | Description |
|---|---|
| idle-dev | dev stack, no traffic, Aurora paused all month |
| pilot | dev settings, 50 users, 2,000 questions and 200 documents a month |
| department | prod settings, 500 users, 30,000 questions and 2,000 documents a month |
| organisation | prod settings with reranking, 5,000 users, 300,000 questions a month |

| Line item | idle-dev | pilot | department | organisation |
|---|---:|---:|---:|---:|
| API Gateway | 0.00 | 0.01 | 0.13 | 1.26 |
| Aurora compute | 0.00 | 5.28 | 70.20 | 175.20 |
| Aurora storage and I/O | 0.02 | 0.12 | 1.32 | 8.80 |
| Bedrock Guardrails | 0.00 | 0.00 | 7.50 | 75.00 |
| Bedrock embeddings | 0.00 | 0.04 | 0.39 | 2.02 |
| Bedrock generation | 0.00 | 2.40 | 33.94 | 319.44 |
| Bedrock rerank | 0.00 | 0.00 | 0.00 | 480.00 |
| CloudWatch | 1.70 | 13.70 | 13.75 | 14.23 |
| Cognito | 0.00 | 0.00 | 0.00 | 0.00 |
| KMS and Secrets Manager | 1.80 | 1.81 | 2.01 | 3.79 |
| Lambda | 0.00 | 0.19 | 2.41 | 20.06 |
| RDS Data API | 0.00 | 0.01 | 0.20 | 1.93 |
| S3 and SQS | 0.00 | 0.02 | 0.24 | 1.45 |
| WAF | 0.00 | 0.00 | 8.02 | 8.22 |
| X-Ray | 0.00 | 0.00 | 0.00 | 1.00 |
| **Total per month (USD)** | **3.52** | **23.58** | **140.11** | **1,112.40** |
| Cost per question (USD) | n/a | 0.0118 | 0.0047 | 0.0037 |

## Assumptions

| Quantity | Value | Source |
|---|---|---|
| Input tokens per question | 700 | measured: 31,464 over 45 evaluation questions (Gemini Flash) |
| Output tokens per question | 400 | measured: 17,810 over 45 (Gemini counts reasoning tokens as output, so this is high for Nova) |
| Data API calls per question | 18 | code: three transactions, each a begin, a scope statement, one or more work statements and a commit |
| API function time per question | 4 s at 1 GB | measured p50 answer latency 3.8 s; Lambda bills while it waits for the model |
| Chunks per document | 40 | about 30 pages of prose at the default chunk sizes |
| Cache hit rate | 10% to 20% | assumption; the cache is per user and resets on every corpus change |
| Aurora ACU-hours | see the script | pilot: 4 active hours on 22 working days at 0.5 ACU; department: 0.5 ACU always plus 1 ACU for 10 hours on 22 days; organisation: 2 ACU average |
| Generation model | Nova 2 Lite through the `us.` profile, regional rate ($0.33 in, $2.75 out per million tokens) | Price List API |
| Guardrails | question only, about one text unit per question | code |
| Free tiers | ignored, except Cognito's 10,000 MAU and part of X-Ray | conservative |

## What drives cost

- **Idle**: the KMS key ($1.00), two secrets ($0.80) and 17 alarm metrics ($1.70). Aurora paused at 0 ACU bills storage only. There is no NAT gateway, which would otherwise be the largest idle cost.
- **Aurora compute** dominates small and medium deployments once the cluster cannot pause. Prod keeps a 0.5 ACU floor, which alone is about $44 a month, so the first request of the day does not wait 15 seconds.
- **Generation** grows linearly with questions. Output tokens cost about eight times input tokens with Nova 2 Lite, so the six-sentence answer limit in the prompt is also a cost control.
- **Reranking** is billed per query at $2.00 per 1,000. At 300,000 questions a month it costs more than generation. Enable it only after the evaluation shows it improves answers on your data.
- **CloudWatch custom metrics** are $0.30 per metric per month, billed for the hours they are published. The code emits about 40 metric and dimension combinations, so any active stack pays about $12 a month for them regardless of traffic.
- **Embedding** is cheap at Titan V2 prices; re-embedding the whole corpus after a model change costs less than a day of generation.

## Controls that cap spend

| Control | Where | Effect |
|---|---|---|
| Per-tenant daily query and token budgets | `tenants.daily_query_limit`, `daily_token_limit`; checked before every question | Returns 429 `budget_exceeded` |
| Stage throttling | `api_throttle_rate`, `api_throttle_burst` per environment | Caps request rate for the whole API |
| WAF per-IP rate limit | 1,000 requests per 5 minutes per IP (test, prod) | Blocks a single runaway client |
| Agent budgets | 4 steps, 24,000 tokens, 25 s per request | Bounds the most expensive mode |
| Agentic mode off by default | `allow_agentic` per tenant | Avoids the measured 28% token increase where it does not help |
| No model call without evidence | query service | Questions with no accessible sources cost no Bedrock tokens |
| Answer cache | per user and corpus version | Repeated questions skip embedding and generation |
| Embedding reuse | content hash | Re-uploads and new versions embed only changed chunks |
| Ingest concurrency cap | `ingest_max_concurrency` | Protects Bedrock quotas; a large upload waits in the queue instead of spiking |
| AWS Budgets | `monthly_budget_usd` per environment | Emails at 80% of actual and 100% of forecast spend (needs the `Project` tag activated as a cost allocation tag) |
| Cost per request in every response and the `EstimatedCostMicroUSD` metric | `observability/cost.py` | Makes spend visible per question and on the dashboard |

## Optimisations, with their trade-offs

| Change | Saves | Costs |
|---|---|---|
| Nova Micro instead of Nova 2 Lite | about 90% of generation | weaker answers; run the live evaluation first |
| Global inference profile (`global.amazon.nova-2-lite-v1:0`) | about 9% of generation | requests may be served from any region; check data residency rules |
| Remove the `code` and `reason` dimensions from `ApiError` and `IngestFailed` | a few dollars of custom metrics | less detail on the dashboard; logs still carry the codes |
| Prod floor of 0 ACU instead of 0.5 | about $44 a month | about 15 s on the first request after an idle period |
| Fewer Data API calls per question (combine statements) | small | slightly more complex repository code |
| Reranking only for tenants that need it | up to the whole rerank line | none, if the evaluation shows no gain for the others |
