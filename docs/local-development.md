# Local development

Everything except the AWS adapters runs on a laptop: PostgreSQL in Docker, files in `.data/blobs`, ingestion in-process, and model providers that are either offline (`fake`) or Gemini. Start with [setup.md](setup.md).

## What runs where

| Concern | Local | AWS |
|---|---|---|
| Database | PostgreSQL 16 + pgvector 0.8.1 in Docker, through psycopg | Aurora PostgreSQL 16.13, through the RDS Data API |
| Database roles | `rag_owner` (non-superuser owner) runs migrations; `rag_app` serves requests | Aurora master user runs migrations; `rag_app` serves requests |
| Files | `.data/blobs/raw/...` | S3 `raw/...` |
| Upload | `PUT /v1/documents/{id}/versions/{v}/content` on the API | presigned POST straight to S3 |
| Ingestion | in-process after the upload | S3 event, SQS, ingest Lambda |
| Auth | HS256 tokens from `ragplatform token` | Cognito ID tokens checked by API Gateway |
| Models | `fake` or `gemini` | `bedrock` |

The owner role is deliberately not a superuser. Aurora's master user is not one either, and a superuser bypasses row-level security, so tests run as one would see every tenant's rows and could not exercise the policies that bind on Aurora.

## Common commands

```bash
uv run ragplatform serve                                   # API with reload-free uvicorn
uv run ragplatform ingest path/*.md --tenant acme --user alice --groups staff --acl staff
uv run ragplatform ask "question" --tenant acme --user bob --groups staff --debug
uv run ragplatform ask "question" --tenant acme --user bob --groups staff --mode agentic
uv run ragplatform tenant globex "Globex Inc"
uv run ragplatform token --user root --tenant acme --groups tenant-admin
uv run ragplatform reembed                                 # after changing the embedding model
uv run streamlit run ui/app.py
```

`--debug` adds `retrieved` to the response: each source's vector rank, keyword rank, fused score and whether it was flagged for injection-like text. Use it to see why a question retrieved what it did.

## Tests

```bash
uv run pytest                               # everything (about 30 s)
uv run pytest tests/unit                    # no database needed
uv run pytest tests/integration -k isolation
uv run pytest --cov --cov-report=term       # coverage
```

Integration tests use the `ragplatform_test` database, migrate it once per session, and truncate tables between tests. They connect as `rag_app_test`, a separate role, so they never change the password your dev database uses. If PostgreSQL is not running they are skipped with a message; CI sets `RAG_REQUIRE_DB=1`, which turns a missing database into a failure.

## Lint, format, types

```bash
uv run ruff check .
uv run ruff format .
uv run mypy                 # strict mode over src/ragplatform
```

## Working without the full AWS stack

- **Bedrock adapters** are covered by contract tests (`tests/unit/test_aws_contracts.py`) that use botocore's `Stubber`. It validates every request against the real service API model, so parameter names, nesting and required fields are checked without an account. The tests found one real bug this way: `CommitTransaction` does not accept a `database` parameter.
- **S3 and the ingest handler** are tested with moto and real SQS event payloads.
- **The Lambda proxy path** is tested by passing an API Gateway REST event (with Cognito claims) through Mangum into the FastAPI app.
- **The RDS Data API** cannot be emulated faithfully. The SQL is the same, and the executor's request shapes are contract-tested, but transaction behaviour against a real Aurora cluster has not been verified by this test suite. Run the deploy smoke test in a dev account to close that gap.

## Debugging

- Logs are JSON from Powertools Logger, the same format as in Lambda. Every line in a request carries `request_id`.
- Questions are logged as a SHA-256 prefix, not text. Set `RAG_LOG_PROMPTS=true` locally to log text.
- Inspect the database directly: `docker compose exec db psql -U rag_owner -d ragplatform`. As the owner you see every tenant. To see what the API sees, run as the runtime role and set the scope first:
  ```sql
  SET ROLE rag_app;
  SELECT set_config('app.tenant_id', 'acme', false), set_config('app.user_id', 'bob', false),
         set_config('app.groups', '["staff"]', false);
  SELECT title, status FROM documents;
  ```
- An ingestion failure is recorded on the version: `SELECT version, status, error_code, error_detail FROM document_versions;`

## Adding a parser, chunker or provider

- **A file type**: add the MIME type to `SUPPORTED_TYPES` and `EXTENSION_TYPES` in `ingestion/parsers.py`, a parser that returns `Section` objects with heading paths and pages, and a case in `check_declared_type` if it has magic bytes. Add a test with a real file in `tests/unit/test_ingestion.py`.
- **A provider**: implement the `Embedder`, `ChatModel` or `Reranker` protocol in `providers/`, add it to `build_providers`, and add contract tests. Embedders must call `check_vector` so a bad vector is rejected instead of stored.
- **A schema change**: add `src/ragplatform/store/migrations/0002_<name>.sql`. Never edit an applied migration; the migrator refuses to run when an applied file's checksum changes.
