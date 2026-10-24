# API reference

Base URL: the deployed stage URL (`terraform output api_url`, for example `https://abc123.execute-api.us-east-1.amazonaws.com/dev`) or `http://127.0.0.1:8000` locally. Locally, the generated OpenAPI document is at `/openapi.json` and an interactive page at `/docs`. Both are disabled in Lambda.

## Authentication

| Environment | Header |
|---|---|
| AWS | `Authorization: <Cognito ID token>` (the raw token; API Gateway's Cognito authorizer validates it) |
| Local | `Authorization: Bearer <token from ragplatform token>` |

Identity comes only from the token: `sub` is the user, `custom:tenant_id` the tenant, `cognito:groups` the groups. Members of `tenant-admin` can read every document in their tenant and change any document's access. Fields in a request body that look like identity (`user_id`, `tenant_id`, `groups`) are ignored.

## Errors

Every error has the same shape and an HTTP status that matches it:

```json
{"error": {"code": "budget_exceeded", "message": "The daily usage budget for this tenant has been reached.", "request_id": "c1f0..."}}
```

| Status | Code | Meaning |
|---|---|---|
| 401 | `unauthenticated` | missing or invalid token |
| 403 | `forbidden` | no tenant assigned, not the owner, sharing with a group you are not in, agentic mode not enabled |
| 404 | `not_found` | does not exist, or you cannot see it (the two are indistinguishable on purpose) |
| 409 | `conflict` | identical file already a live version; concurrent registration |
| 413 | `payload_too_large` | over `RAG_MAX_UPLOAD_BYTES` |
| 415 | `unsupported_media_type` | not PDF, DOCX, HTML, Markdown, text or CSV |
| 422 | `validation_failed` | invalid fields (the message lists them); question too long |
| 422 | `guardrail_blocked` | Bedrock Guardrails blocked the question |
| 429 | `budget_exceeded` | tenant's daily query or token budget reached |
| 429 | (API Gateway) | stage throttling; the body is API Gateway's own |
| 502 | `upstream_error` | a dependency rejected the request, or the model returned malformed output twice |
| 503 | `upstream_unavailable`, `database_resuming` | retry after the `Retry-After` header (5 s, or 15 s while Aurora resumes) |
| 500 | `internal_error` | a bug; quote the `request_id` |

Every response carries `X-Request-Id`, `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.

## `GET /v1/health`

No authentication, touches nothing. `{"status": "ok", "version": "2.0.0"}`.

## `GET /v1/health/ready`

No authentication. Checks the database and the embedding space. 200 `{"status": "ready"}` or 503. Every call wakes a paused Aurora cluster, so do not poll it on a schedule against dev or test.

## `POST /v1/query`

```json
{
  "question": "Who approves an expense claim over 5,000 GBP?",
  "mode": "standard",
  "filters": {"document_ids": null, "tags": ["policy"], "content_types": null},
  "retrieval_mode": null,
  "use_cache": true,
  "debug": false
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `question` | string | required | 1 to `RAG_MAX_QUESTION_CHARS` (2,000) characters after whitespace is collapsed |
| `mode` | `standard` or `agentic` | `standard` | agentic needs `allow_agentic` on the tenant |
| `filters` | object | none | restrict to document ids, tags or content types |
| `retrieval_mode` | `hybrid`, `vector`, `keyword` | server setting | for evaluation and debugging |
| `use_cache` | bool | true | |
| `debug` | bool | false | adds `retrieved` (ranks and scores); disables the cache |

Response:

```json
{
  "request_id": "c1f0...",
  "status": "answered",
  "answer": "Claims above 5,000 GBP need approval from the Chief Financial Officer [S1].",
  "citations": [{
    "marker": "S1", "chunk_id": "...", "document_id": "...", "version": 1,
    "title": "Expense Policy", "section_path": "Expense Policy > Approval limits",
    "page_start": null, "page_end": null,
    "snippet": "Your line manager approves claims up to 500 GBP. Claims from 500 GBP to 5,000 GBP need director approval..."
  }],
  "conflicts": [],
  "model_id": "us.amazon.nova-2-lite-v1:0",
  "usage": {"input_tokens": 742, "output_tokens": 61},
  "cost_usd": 0.0004136,
  "cached": false,
  "latency_ms": 2214.6,
  "timings_ms": {"embed": 81.2, "search": 64.5, "generate": 2040.3},
  "downgraded_reason": null,
  "retrieved": null,
  "agent_trace": null
}
```

| `status` | Meaning |
|---|---|
| `answered` | every factual sentence cites a supplied source |
| `insufficient_context` | the accessible documents do not answer it; `answer` is a fixed message and `citations` is empty |
| `conflicting_sources` | sources disagree; `answer` states each version with its citation and `conflicts` describes the disagreement |

`downgraded_reason` explains a server-side change: `no_retrieval_results` (nothing accessible matched; no model call), `no_valid_citations` (the model answered without valid citations), `grounding_check_failed` (Guardrails). `cost_usd` is null when the model has no known price.

## `POST /v1/documents`

Registers a document and returns how to upload it.

```json
{
  "source_name": "expense-policy.pdf",
  "size_bytes": 482113,
  "sha256": "9f2c...64 hex characters",
  "content_type": "application/pdf",
  "title": null,
  "acl_groups": ["finance", "staff"],
  "tags": ["policy"],
  "metadata": {"owner_team": "finance"}
}
```

`content_type` is inferred from the extension when omitted. Without a `title`, the document's own title (PDF metadata, DOCX title, Markdown H1, HTML `<title>`) replaces the file name after ingestion. `acl_groups` must be groups you belong to (tenant admins can use any). An empty list means only the owner and tenant admins can read it.

Response `201`:

```json
{
  "document": {"document_id": "...", "status": "pending_upload", "...": "..."},
  "version": {"version": 1, "status": "pending_upload", "storage_key": "raw/acme/<id>/v1", "...": "..."},
  "upload": {"method": "POST", "url": "https://<bucket>.s3.amazonaws.com/", "fields": {"key": "...", "policy": "...", "...": "..."}, "expires_in_s": 900, "max_bytes": 482113},
  "duplicate": false
}
```

Upload with the returned instruction:

- **AWS** (`method: POST`): multipart form POST to `url` with every entry of `fields`, then the file as the last field named `file`. The policy accepts exactly `size_bytes` bytes of the declared content type.
  ```bash
  curl -F key=... -F policy=... -F x-amz-signature=... [all fields] -F file=@expense-policy.pdf "$URL"
  ```
- **Local** (`method: PUT`): `PUT` the bytes to `url` (relative to the API) with the same `Authorization` header.

If you already uploaded the same bytes, the response is `200` with `duplicate: true`, the existing document, and no upload instruction. Poll `GET /v1/documents/{id}` until `status` is `ready` or `failed`.

## `POST /v1/documents/{id}/versions`

`{"size_bytes": 512000, "sha256": "..."}`. Owner or tenant admin. Returns the same shape with the new version and its upload instruction. When the new version is ready, it replaces the previous one in search in a single transaction.

## `GET /v1/documents`

Query parameters: `limit` (1 to 100, default 25), `cursor` (from the previous page's `next_cursor`), `status`. Returns `{"items": [Document...], "next_cursor": "..." | null}`, newest first, only documents you can read.

## `GET /v1/documents/{id}`

`{"document": {...}, "versions": [{"version": 1, "status": "superseded", ...}, {"version": 2, "status": "ready", ...}]}`. A failed version carries `error_code` (`type_mismatch`, `corrupt`, `encrypted`, `no_text`, `too_many_pages`, `decompression_limit`, `checksum_mismatch`, `size_mismatch`, `parse_error`, `missing_object`, `unsupported_type`, `too_large`) and `error_detail`.

## `PATCH /v1/documents/{id}`

`{"title": "...", "acl_groups": [...], "tags": [...]}`, any subset. Owner or tenant admin. Takes effect on the next query; cached answers for the tenant are invalidated.

## `DELETE /v1/documents/{id}`

Owner or tenant admin. Removes the document from search immediately, deletes every stored version of the file, and keeps a tombstone (id only) for the audit trail. `{"document_id": "...", "deleted": true, "objects_purged": 2, "purge_failures": 0}`.

## `GET /v1/usage`

`{"tenant_id": "acme", "today": {"queries": 42, "input_tokens": 30110, "output_tokens": 9800, "cost_usd": 0.037}, "limits": {"daily_query_limit": 2000, "daily_token_limit": 5000000}}`

## Limits

| Limit | Value | Setting |
|---|---|---|
| Question length | 2,000 characters | `RAG_MAX_QUESTION_CHARS` |
| File size | 50 MB | `RAG_MAX_UPLOAD_BYTES` |
| PDF pages | 1,500 | `RAG_MAX_PDF_PAGES` |
| DOCX uncompressed size | 200 MB | `RAG_MAX_DOCX_UNCOMPRESSED_BYTES` |
| ACL groups and tags per document | 20 each | |
| Request time | 29 s | API Gateway integration timeout |
| Upload URL lifetime | 15 minutes | `RAG_UPLOAD_URL_TTL_S` |
| Request rate | per environment | `api_throttle_rate`, `api_throttle_burst`; WAF 1,000 per 5 minutes per IP |
