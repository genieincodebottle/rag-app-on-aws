# Troubleshooting

Problems met while building and testing this repository, and the failure modes the code is designed to report. Each entry gives the symptom, the cause and the fix.

## Local setup

**`password authentication failed for user "rag_owner"` even though the password is right.**
Another PostgreSQL is listening on the same port (on Windows, a local PostgreSQL service often holds 5432 or 5433), so the connection never reaches the container. Check with `netstat -ano | findstr :55432` (Windows) or `lsof -i :55432`. The compose file maps 55432 for this reason; if that is taken too, change the port in `docker-compose.yml`, `.env` and `RAG_TEST_ADMIN_DATABASE_URL`.

**The init script did not create `rag_owner`.**
`docker/initdb/01-roles.sql` runs only when the data volume is empty. If the volume existed before, run `docker compose down -v` (this deletes local data) and `docker compose up -d db`.

**`ragplatform migrate` exits with code 2 and "the index was built with a different embedding model".**
You changed `RAG_PROVIDER` or `RAG_EMBEDDING_MODEL_ID` after the first migrate. Run `uv run ragplatform reembed`, or start from an empty database.

**Queries fail with `embedding_space_mismatch`.**
Same cause, at query time. Vectors from two models are not comparable, so the platform refuses instead of returning poor results.

**Integration tests are skipped.**
The test database is not reachable. Start it with `docker compose up -d db`. CI sets `RAG_REQUIRE_DB=1` to turn this into a failure.

**`ragplatform` commands pick up settings you did not expect.**
Settings come from environment variables first and `.env` second. A `RAG_` variable exported in your shell overrides `.env`. `env | grep RAG_` (or `Get-ChildItem Env:RAG_*`) shows them.

**Building the Lambda bundle fails with "lxml has no usable wheels".**
uv older than 0.9 cannot match the dual manylinux tags newer wheels use. Upgrade uv, or run once with a current one: `UV="uvx --from uv>=0.9 uv" uv run python scripts/build_lambda.py`.

## Ingestion

| `error_code` on the version | Cause | Fix |
|---|---|---|
| `type_mismatch` | the bytes do not match the declared type (a `.pdf` that is really HTML, a zip that is not DOCX) | upload with the right type or file |
| `no_text` | scanned PDF or empty file | OCR it first; text extraction only |
| `encrypted` | password-protected PDF | remove the password |
| `too_many_pages`, `decompression_limit`, `too_large` | over a configured limit | split the file or raise the limit deliberately |
| `checksum_mismatch`, `size_mismatch` | the uploaded bytes differ from what was registered | register again with the right SHA-256 and size |
| `parse_error` | the parser library raised on a malformed file | open and re-save the file; check the exception type in the ingest log |
| `missing_object` | the upload never completed, or the object was deleted | upload again |

A version stuck in `processing` for longer than the ingest timeout becomes claimable again when its lease expires; a redelivered or redriven message then processes it.

## Answers

**Every answer is `insufficient_context`.**
With `--debug` (or `"debug": true`), check `retrieved`. Empty: nothing the caller can read matched. Confirm the user's groups overlap the documents' `acl_groups` and the tenant is right. Not empty but still insufficient: the model did not cite (`downgraded_reason: no_valid_citations`) or the sources do not contain the answer.

**The answer is right but `conflicting_sources`.**
Two documents disagree, typically an old and a new version uploaded as separate documents. Upload the new one as a version of the old (`POST /v1/documents/{id}/versions`) so only one is searchable.

**`upstream_error` with "malformed output twice".**
The model did not return the required JSON, twice. Usually a model change. Check `AnswerFormatRetry`; with Gemini, a very low `RAG_MAX_OUTPUT_TOKENS` can cut the JSON off, because reasoning tokens count against the limit.

**First request after a quiet period takes 15 s or returns 503 `database_resuming`.**
Aurora was paused at 0 ACU. Expected in dev and test; retry after `Retry-After`. In prod the floor is 0.5 ACU.

## AWS

**`AccessDeniedException` from Bedrock (502 `upstream_error`).**
Model access is not granted for that model in that region, or the function role lacks the model's ARN. For inference profiles (`us.` or `global.` ids), the role needs the profile ARN and the foundation-model ARN in every region the profile routes to; the Terraform grants `arn:aws:bedrock:*::foundation-model/<id>`.

**`ValidationException` "on-demand throughput isn't supported" or "model identifier is invalid".**
The model needs an inference profile id in this region. Nova 2 Lite and Claude Haiku 5.5 are not available in-region in us-east-1; use `us.amazon.nova-2-lite-v1:0` and `us.anthropic.claude-haiku-5-5`.

**Browser upload to the presigned POST fails with 403.**
The policy pins the exact size and content type: the file must be exactly `size_bytes` and the `Content-Type` field must match. For browser uploads, the page's origin must be in `upload_origins` (S3 CORS).

**Upload succeeds but nothing is ingested.**
Check the ingest queue depth and the ingest log group. If the queue is empty, the S3 notification is missing (it filters on the `raw/` prefix) or the queue policy rejected S3. If messages are there but not consumed, check the event source mapping is enabled.

**401 from the API with a valid-looking token.**
API Gateway's Cognito authorizer wants the raw ID token, not `Bearer <token>`, and not the access token. The ID token carries `custom:tenant_id`.

**403 `forbidden`: "user is not assigned to a tenant".**
The Cognito user has no `custom:tenant_id`. Set it with `admin-update-user-attributes`; the user must sign in again for a new token.

**Terraform plan job fails with "deletes or replaces a data-bearing resource".**
The plan would replace Aurora, the documents bucket, the user pool or the KMS key, often because a setting that forces replacement changed (for example a Cognito schema attribute). Review the plan; apply manually if the replacement is really intended and data is backed up.

**Deploy smoke test fails and the workflow rolls back aliases.**
Read the smoke-test step output (status code and body), fix forward, and redeploy. Terraform state still records the newer function versions until the next successful deploy.
