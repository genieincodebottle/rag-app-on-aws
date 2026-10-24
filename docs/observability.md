# Observability

What the platform records, where to find it, and which signals mean something is wrong. Alarms and the dashboard are defined in `infra/terraform/modules/observability`; each alarm names its entry in [operations-runbook.md](operations-runbook.md).

## Logs

Every function logs JSON through Powertools Logger to its own encrypted log group (`/aws/lambda/ragplatform-<env>-<fn>`). Each line carries `service`, `level`, `timestamp`, `request_id` (the API Gateway request id, also returned to the client as `X-Request-Id`) and, once known, `tenant_id` and `document_id`.

| Event | Fields |
|---|---|
| `query answered` | status, citations, sources, input and output tokens, latency, question fingerprint |
| `query served from cache` | status, question fingerprint |
| `ingest complete` | document, version, chunks, reused embeddings, flagged chunks, duration |
| `ingest rejected` | document, version, reason code |
| `request failed` | error code, HTTP status, internal detail (truncated) |
| `primary model unavailable, using fallback` | primary, fallback, reason |

**What is never logged by default**: question text, answer text, document text, tokens or secrets. Questions appear as `sha256_16`, a 16-character hash prefix, so the same question can be correlated across requests without storing it. `RAG_LOG_PROMPTS=true` logs text for local debugging and is refused in prod. API Gateway access logs record method, path, status, latency, user `sub` and tenant, never bodies (`data_trace_enabled = false`), and WAF logs redact the `Authorization` header.

Useful CloudWatch Logs Insights queries:

```
# Slowest questions in the last hour
fields @timestamp, request_id, tenant_id, latency_ms, status, input_tokens
| filter message = "query answered" | sort latency_ms desc | limit 20

# Everything that happened in one request
fields @timestamp, level, message | filter request_id = "<id from X-Request-Id>" | sort @timestamp

# Ingestion failures by reason
filter message = "ingest rejected" | stats count() by reason

# Answers by status per tenant
filter message = "query answered" | stats count() by tenant_id, status
```

## Metrics

Namespace `RagPlatform`, dimension `service=ragplatform` (plus the listed extra dimension). They are emitted as CloudWatch Embedded Metric Format inside the log stream, flushed once per invocation, so they cost no API calls.

| Metric | Unit | Meaning |
|---|---|---|
| `QueryLatency` | ms | end-to-end time inside the function |
| `RetrievalLatency`, `ModelLatency` | ms | search transaction; one model call |
| `AnswerStatus` (`status`) | count | answered, insufficient_context, conflicting_sources |
| `InputTokens`, `OutputTokens` | count | per question |
| `EstimatedCostMicroUSD` | none | estimated cost per question in millionths of a dollar |
| `CacheHit`, `CacheMiss` | count | answer cache |
| `CitationsPerAnswer`, `InvalidCitationsRemoved` | count | citation health; removals mean the model cited sources it was not given |
| `AnswerFormatRetry` | count | malformed model JSON (retried once) |
| `GroundingCheckFailed` | count | answers downgraded by the Guardrails grounding check |
| `FlaggedSourcesInContext`, `ChunksFlaggedForInjection` | count | injection-like text reaching the model, found at ingest |
| `RetrievalCandidates` | count | rows the search returned before expansion |
| `AgentSteps`, `AgentBudgetExhausted` | count | agentic mode |
| `IngestDuration` | ms | one version |
| `ChunksWritten`, `EmbeddingsReused` | count | ingestion volume and reuse |
| `IngestFailed` (`reason`), `IngestRetried`, `IngestSkipped` | count | permanent failures, transient failures, duplicate deliveries |
| `DocumentsRegistered`, `DocumentsDeleted`, `BlobPurgeFailed` | count | lifecycle |
| `ApiError` (`code`) | count | every error response by code |

AWS metrics also used: API Gateway `Count`, `4XXError`, `5XXError`, `Latency`; Lambda `Errors`, `Throttles`, `Duration`, `ConcurrentExecutions`; SQS `ApproximateNumberOfMessagesVisible`, `ApproximateAgeOfOldestMessage`; RDS `ServerlessDatabaseCapacity`, `ACUUtilization`.

## Traces

X-Ray tracing is on for API Gateway and every function. Powertools Tracer adds subsegments for `QueryService.answer`, `Retriever.retrieve`, `Generator.answer`, `AgentRunner.run` and `IngestionPipeline.process`. Responses are not captured into traces (`capture_response=False`), so answers and documents do not end up in X-Ray. AWS SDK calls (Bedrock, Data API, S3) appear as their own subsegments, which makes it clear whether a slow request waited on the model, the database or neither.

## Dashboard

`ragplatform-<env>` has seven panels: API requests and errors; latency (API p50 and p95, retrieval p95, model p95); answer status per hour; tokens and estimated cost; cache and safety signals; ingestion and queue depth; Aurora capacity.

## Alarms

| Alarm | Condition | Usually means |
|---|---|---|
| `api-5xx` | 5 or more 5xx in 5 minutes | Bedrock throttling or outage, database resuming slowly, a bug |
| `api-latency-p95` | p95 over 15 s for 15 minutes | model slow, Aurora at capacity, agentic traffic |
| `internal-errors` | any unhandled exception | a bug; look up the request id |
| `<fn>-errors`, `<fn>-throttles` | function errors or throttles | a bug, or concurrency or quota limits |
| `ingest-dlq-not-empty` | any message in the DLQ | ingestion failed five times for a transient reason |
| `ingest-backlog` | oldest message over 15 minutes | ingest concurrency too low for the upload rate, or a stuck dependency |
| `blob-purge-failed` | a deleted document's file not removed | S3 permission or throttling; the deletion is incomplete |
| `insufficient-context-ratio` | over 50% of answers insufficient in an hour (with over 20 answers) | an embedding or index problem, a failed re-ingest, or users asking about content that is not there |
| `aurora-at-max-capacity` | ACU utilisation over 90% for 15 minutes | raise `db_max_acu` or find the expensive query |

Alarms notify the SNS topic `ragplatform-<env>-alerts`; set `alert_emails` in the environment's variables to subscribe. Each email address must confirm the subscription.

## Answer quality in production

Logs and metrics cannot tell whether an answer was right. What they can show, in order of usefulness:

1. `insufficient_context` share by tenant: a sudden rise follows broken ingestion or a model change.
2. `InvalidCitationsRemoved` and `AnswerFormatRetry`: the model drifting from the contract, typically after a model version change.
3. `GroundingCheckFailed` when Guardrails grounding is on.
4. `audit_events`: which documents get cited, which never do.

For the rest, run the evaluation against the deployed stack after every model, prompt or chunking change, and add real questions with known answers to the dataset.
