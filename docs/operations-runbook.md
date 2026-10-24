# Operations runbook

One entry per alarm and per routine operation. Each entry says how to confirm the problem, what to do, and how to know it is fixed. Commands assume `ENV=dev|test|prod` and credentials for that account.

Set once per shell:

```bash
ENV=dev; P=ragplatform-$ENV
CLUSTER=$(aws rds describe-db-clusters --db-cluster-identifier $P --query 'DBClusters[0].DBClusterArn' --output text)
```

## API 5xx errors (`api-5xx`, `internal-errors`)

1. Find the codes: Logs Insights on `/aws/lambda/$P-api`: `filter message = "request failed" and status >= 500 | stats count() by code`.
2. By code:
   - `upstream_unavailable` from Bedrock: throttling or an outage. Check the Bedrock service health dashboard and your quota (Service Quotas, Amazon Bedrock, tokens per minute for the model). Short term: configure `fallback_chat_model_id`. Long term: request a quota increase.
   - `database_resuming`: Aurora was paused and took longer than 25 s to resume (it can take 30 s or more after a day idle). Expected in dev and test. In prod it means the floor was set to 0.
   - `upstream_error`: a request Bedrock rejected (model access not granted, invalid model id, validation). Check `detail` in the log line.
   - `internal_error`: a bug. Take the `request_id`, read the whole request's log lines, reproduce locally, fix with a test.
3. Fixed when the alarm returns to OK and `ApiError` stops increasing.

## Slow answers (`api-latency-p95`)

1. X-Ray trace map for the API: which subsegment is slow?
2. Bedrock slow: compare `ModelLatency` p95 with the usual level. Consider a faster model or a lower `RAG_MAX_OUTPUT_TOKENS`.
3. Database slow: `RetrievalLatency` p95 high. Check `ACUUtilization`; check slow statements in the Aurora PostgreSQL log (statements over 1 s are logged). An HNSW scan over a tenant with many filtered-out rows can be slow; lower `RAG_HNSW_EF_SEARCH` trades recall for speed.
4. Agentic traffic: `AgentSteps` rising means more agentic questions, which take about twice as long.

## Lambda errors and throttles (`<fn>-errors`, `<fn>-throttles`)

- **Errors in api**: see API 5xx.
- **Errors in ingest**: transient failures raise on purpose so SQS retries; check `IngestRetried`. Persistent errors end in the DLQ.
- **Throttles**: account concurrency is exhausted (default 1,000 per region shared by all functions) or a reserved limit is reached. Check `ConcurrentExecutions` across the account. Ingest concurrency is capped by the event source (`ingest_max_concurrency`), which does not cause throttles; messages wait in the queue.

## Messages in the ingest DLQ (`ingest-dlq-not-empty`)

A message lands here after five failed receives, which only happens for transient failures (permanent ones are recorded on the version and acknowledged).

1. Look at a message without removing it:
   ```bash
   DLQ=$(aws sqs get-queue-url --queue-name $P-ingest-dlq --query QueueUrl --output text)
   aws sqs receive-message --queue-url $DLQ --max-number-of-messages 1 --visibility-timeout 30
   ```
2. The body is the S3 event. Find the version's state: in the ingest log group, `filter document_id = "<id>"`, and in the database `SELECT status, attempts, error_code FROM document_versions WHERE document_id = '<id>'` (run through the Data API as the owner, or use the query editor).
3. Fix the cause (quota, model access, a bug), then move the messages back:
   ```bash
   aws sqs start-message-move-task --source-arn $(aws sqs get-queue-attributes --queue-url $DLQ --attribute-names QueueArn --query Attributes.QueueArn --output text)
   ```
   Redriven messages re-claim versions left in `uploaded`, and versions stuck in `processing` become claimable once their lease (the ingest timeout) expires.
4. Fixed when the DLQ is empty and the versions are `ready`.

## Ingestion backlog (`ingest-backlog`)

Uploads arrive faster than `ingest_max_concurrency` processes them. If the backlog is a one-off bulk upload, let it drain. If it is steady, raise `ingest_max_concurrency` in the environment's tfvars, after checking the Bedrock embedding quota (requests per minute) can absorb it.

## Deletion purge failed (`blob-purge-failed`)

The document is already gone from the index and from answers; its file is still in S3.

1. Find the key: `filter message = "blob purge failed"` in the API log group gives `key`.
2. Delete every version:
   ```bash
   aws s3api list-object-versions --bucket $P-documents-<account> --prefix <key> \
     --query '{Objects: [Versions[].{Key:Key,VersionId:VersionId}, DeleteMarkers[].{Key:Key,VersionId:VersionId}][]}' > del.json
   aws s3api delete-objects --bucket $P-documents-<account> --delete file://del.json
   ```
3. The lifecycle rule also removes noncurrent versions after 30 days.

## Answer quality drop (`insufficient-context-ratio`)

1. Per tenant: `filter message = "query answered" | stats count() by tenant_id, status`. One tenant or all?
2. All tenants: recent change to the embedding model, chunking or prompt? Check `embedding_space` against the deployed `RAG_EMBEDDING_MODEL_ID`; a mismatch makes every query fail with `embedding_space_mismatch` rather than return insufficient answers, so a quiet drop points at chunking or the prompt. Run the evaluation against the stack.
3. One tenant: failed ingests (`document_versions.status = 'failed'`), documents deleted, or ACL changes that hid documents from most users.

## Database at capacity (`aurora-at-max-capacity`)

Raise `db_max_acu` in the environment's tfvars and deploy, or find the expensive statements in the PostgreSQL log and Performance Insights (prod).

## Routine operations

### Add a tenant

```bash
aws lambda invoke --function-name $P-migrate:live --cli-binary-format raw-in-base64-out \
  --payload '{"tenants": [{"tenant_id": "globex", "name": "Globex Inc", "daily_query_limit": 5000}]}' out.json
```

### Rotate the runtime database password

```bash
aws lambda invoke --function-name $P-migrate:live --cli-binary-format raw-in-base64-out \
  --payload '{"rotate_app_password": true}' out.json
```

The role's password changes first, then the secret. Functions read the secret on each Data API call, so requests in the next few seconds may fail once and succeed on retry.

### Change the embedding model

This re-embeds every chunk; search quality is degraded until it finishes. Do it in a maintenance window.

1. Set `embedding_model_id` and `embedding_dimensions` in the environment's tfvars and deploy. Queries now fail with `embedding_space_mismatch`.
2. Run `ragplatform reembed` with owner credentials against the cluster (from a machine with Data API access, setting `RAG_DB_BACKEND=data_api`, `RAG_DB_CLUSTER_ARN`, `RAG_DB_ADMIN_SECRET_ARN`). It changes the column width if needed, re-embeds in batches and records the new embedding space.

### Restore from backup

Aurora keeps automated backups for `db_backup_retention_days` (7 in dev, 35 in prod) with point-in-time restore to any second in that window.

1. Restore to a new cluster:
   ```bash
   aws rds restore-db-cluster-to-point-in-time --source-db-cluster-identifier $P \
     --db-cluster-identifier $P-restore --restore-to-time 2026-10-24T09:00:00Z \
     --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=4 \
     --db-subnet-group-name $P --vpc-security-group-ids <aurora sg> --enable-http-endpoint
   aws rds create-db-instance --db-cluster-identifier $P-restore --db-instance-identifier $P-restore-writer \
     --engine aurora-postgresql --db-instance-class db.serverless
   ```
2. Check the restored data, then either copy the needed rows back, or point the functions at the restored cluster (`RAG_DB_CLUSTER_ARN`) and import it into Terraform state in place of the old one.
3. S3 objects: versioning keeps overwritten and deleted objects for 30 days; restore with `aws s3api copy-object` from a version id.

Recovery objectives with this setup: data loss up to about 5 minutes (Aurora's continuous backup), recovery in about an hour for a point-in-time restore, within one region. Cross-region recovery (snapshot copy or Aurora Global Database, S3 replication) is not configured.

### Deploy, roll back, tear down

See [deployment.md](deployment.md).
