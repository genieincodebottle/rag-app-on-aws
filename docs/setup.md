# Setup

This gets the platform running on your machine in about ten minutes, with no AWS account and no API key. [deployment.md](deployment.md) covers AWS.

## Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.12 or 3.13 | `python --version` |
| uv | 0.9 or later | `uv --version` |
| Docker | with Compose v2 | `docker compose version` |
| Git | any | `git --version` |

Install uv with `pip install uv` or from https://docs.astral.sh/uv/. Port 55432 must be free (the local PostgreSQL listens there, chosen to avoid clashing with a PostgreSQL service already on 5432 or 5433).

## Install and start

```bash
git clone https://github.com/genieincodebottle/rag-app-on-aws.git
cd rag-app-on-aws
uv sync --all-extras          # creates .venv from uv.lock
cp .env.example .env           # Windows PowerShell: Copy-Item .env.example .env
docker compose up -d db        # PostgreSQL 16 with pgvector 0.8.1
uv run ragplatform migrate     # schema, runtime role, embedding space
uv run ragplatform tenant acme "Acme Corp" --agentic
```

Expected output of `migrate`:

```
{
  "applied": ["0001_schema.sql"],
  "embedding_space": {"provider": "fake", "model_id": "hashing-v1", "dimensions": 1024}
}
```

Running `migrate` again prints `"applied": []`. It is safe to repeat.

## First document and first question

```bash
uv run ragplatform ingest eval/corpus/northwind/expense-policy.md --tenant acme --user alice --groups staff --acl staff
uv run ragplatform ask "Who approves an expense claim of 2,000 GBP?" --tenant acme --user bob --groups staff
```

`ingest` prints `expense-policy.md: ready (8 chunks, 0 embeddings reused) <document id>`. `ask` prints JSON with `"status": "answered"` and a citation pointing at `Expense Policy > Approval limits`.

The answer itself shows what the offline `fake` model is. It copies the source sentence that shares the most words with the question, so it answers "Your line manager approves claims up to 500 GBP", the first sentence of the right section, not the one about 2,000 GBP. Retrieval and citation found the right place; the wording needs a real model. With Gemini (below) the same question is answered with "director approval", as in the live evaluation.

Ask the same question as a user outside the `staff` group:

```bash
uv run ragplatform ask "Who approves an expense claim of 2,000 GBP?" --tenant acme --user eve --groups contractors
```

The status is `insufficient_context` and `usage.input_tokens` is 0: row-level security returned no rows, so no model was called.

## Run the API and the UI

```bash
uv run ragplatform serve                     # http://127.0.0.1:8000, docs at /docs
uv run streamlit run ui/app.py               # http://localhost:8501
```

In the UI, sign in with user `alice`, tenant `acme`, groups `staff`. The UI mints a local token with `RAG_LOCAL_AUTH_SECRET` from `.env`. To call the API yourself:

```bash
TOKEN=$(uv run ragplatform token --user alice --tenant acme --groups staff)
curl -s http://127.0.0.1:8000/v1/query -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"question": "Who approves a 2,000 GBP expense claim?"}'
```

## Use real models (optional, free tier)

The `fake` providers are deterministic and need no key, but their answers are copied sentences. For real answers, use Gemini locally:

1. Get a key at https://aistudio.google.com/apikey.
2. In `.env` set:
   ```
   RAG_PROVIDER=gemini
   RAG_GEMINI_API_KEY=<your key>
   RAG_EMBEDDING_MODEL_ID=gemini-embedding-001
   RAG_CHAT_MODEL_ID=gemini-flash-latest
   ```
3. The database was built with the fake embedder, and vectors from different models cannot be compared. Either start fresh (`docker compose down -v`, then `up -d db` and `migrate`), or run `uv run ragplatform reembed`.

`migrate` warns and exits with code 2 if the configured embedding model differs from the one the index holds, and queries fail with `embedding_space_mismatch` until you re-embed.

## Configuration reference

Every setting is an environment variable with the `RAG_` prefix, read from the environment or `.env`. The full list with defaults is in `src/ragplatform/config.py`. The ones you are most likely to change:

| Variable | Default | Meaning |
|---|---|---|
| `RAG_PROVIDER` | `fake` | `fake`, `gemini` (local only) or `bedrock` (required when deployed) |
| `RAG_EMBEDDING_MODEL_ID`, `RAG_EMBEDDING_DIMENSIONS` | Titan V2, 1024 | Embedding model and width (256, 512, 768 or 1024) |
| `RAG_CHAT_MODEL_ID`, `RAG_FALLBACK_CHAT_MODEL_ID` | Nova 2 Lite, none | Generation model, and the model to use when the first is throttled or down |
| `RAG_RERANK_MODEL_ID` | none | Enables reranking, e.g. `cohere.rerank-v3-5:0` |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `hybrid`, `vector` or `keyword` |
| `RAG_CHUNK_STRATEGY` | `parent_child` | or `recursive` |
| `RAG_CHUNK_TOKENS`, `RAG_PARENT_CHUNK_TOKENS`, `RAG_CHUNK_OVERLAP_TOKENS` | 220, 900, 30 | Chunk sizes (tokens estimated as characters / 4) |
| `RAG_CONTEXTUAL_HEADERS` | `true` | Prefix embedded text with the document title and section path |
| `RAG_FINAL_K` | 6 | Sources sent to the model |
| `RAG_MAX_UPLOAD_BYTES` | 52428800 | Upload size limit (50 MB) |
| `RAG_LOG_PROMPTS` | `false` | Log question and answer text (refused in prod) |

Configuration is validated at start-up. Inside Lambda, or with `RAG_ENVIRONMENT` set to dev, test or prod, the process refuses to start unless authentication is Cognito, the provider is Bedrock, the database backend is the Data API, and ingestion uses S3 and SQS. A misconfigured deployment fails at cold start instead of serving traffic with development settings.

## Verify the installation

```bash
uv run pytest                 # 168 tests; integration tests need the database
uv run python -m eval.run_eval --check eval/thresholds-offline.json
```

Both should finish without failures. If integration tests are reported as skipped, the database is not reachable; see [troubleshooting.md](troubleshooting.md).
