# Current-state assessment of rag-app-on-aws

Audit of branch `develop` at commit `81fa2a6`, run on 2026-10-23 before the platform rewrite. Every finding below was checked against the code at that commit. File and line references point at that commit, so `git show 81fa2a6:<path>` shows the exact lines.

## What the system was

A Terraform-provisioned serverless RAG app.

| Layer | Implementation at 81fa2a6 |
|---|---|
| Client | Streamlit app (`rag_ui/app.py`, 1,200 lines) |
| API | API Gateway REST API, Cognito user-pool authorizer on `/query` and `/upload`, `/auth` open |
| Compute | Five Python 3.11 Lambdas sharing one IAM role: `auth_handler`, `upload_handler`, `document_processor`, `query_processor`, `db_init` |
| Models | Google Gemini over the public internet through a NAT gateway (`gemini-embedding-001` at 768 dimensions, `gemini-flash-latest`) |
| Retrieval store | RDS PostgreSQL `db.t3.micro` with pgvector, one `ivfflat` index |
| Metadata | DynamoDB table, written as a fallback when PostgreSQL fails |
| Web search | Optional MCP server (`mcp_servers/`), called by URL |
| Delivery | GitHub Actions with static IAM access keys, Terraform with three copied environment roots, 2,000 lines of import and cleanup shell scripts |

The happy path worked like this. The UI base64-encoded a file into a JSON body and posted it to `/upload`. The upload Lambda wrote the file to S3 and a row to PostgreSQL. The S3 event invoked the processor, which chunked the file with LangChain, embedded each chunk with one Gemini call, and inserted the rows. `/query` embedded the question, ran a top-5 cosine search, optionally called a web-search MCP server, generated an answer, and then graded it with three more Gemini calls.

## Findings

Severity follows impact on a multi-user deployment. **Critical** means data exposure or a system that cannot work. **High** means silent corruption, a security control missing, or a failure with no recovery path.

### Critical

| ID | Finding | Evidence |
|---|---|---|
| C1 | **Any signed-in user can read any other user's documents.** The query Lambda takes `user_id` from the request body and ignores the Cognito claims the authorizer already verified. Sending another user's id returns their chunks. Uploads have the same flaw, so a user can also write into another user's namespace. | `src/query_processor/query_processor.py:632`, `src/upload_handler/upload_handler.py:126` |
| C2 | **Server-side request forgery from inside the VPC.** The client chooses `mcp_server_url`, and the Lambda POSTs to it from a private subnet with NAT egress. The response is then pasted into the prompt, so the same field is also a prompt-injection channel. | `query_processor.py:637`, `:666`, `:444` |
| C3 | **The ingestion Lambda cannot import on a fresh install.** `document_processor.py` imports `langchain.text_splitter` and `langchain.schema`. Both modules were removed in LangChain 1.x, and `requirements.txt` pins `langchain>=0.3.24`, so a fresh build resolves 1.x (1.4.7 on the audit date) and fails with `ModuleNotFoundError` at cold start. No document can be ingested. | `src/document_processor/document_processor.py:18-19`, `src/document_processor/requirements.txt` |
| C4 | **The test suite cannot detect C3 or any SQL error.** `conftest.py` replaces `boto3`, `botocore`, `psycopg2`, `google.genai` and `langchain` with `MagicMock` objects in `sys.modules`. The 48 tests pass against mocks of every boundary, and no test executes SQL against PostgreSQL. | `src/tests/unit/conftest.py:77-86` |

### High

| ID | Finding | Evidence |
|---|---|---|
| H1 | **Embedding failures store zero vectors.** On any embedding error both Lambdas return `[0.0] * 768` instead of raising. Cosine distance against a zero vector is undefined in pgvector, so those chunks can never be retrieved correctly, and nothing reports it. The query side does the same, so a failed query embedding still returns "results". | `document_processor.py:113-115`, `query_processor.py:286-288` |
| H2 | **Ingestion is not idempotent and leaves duplicates.** The upload Lambda inserts a `documents` row with status `uploaded`, and the processor inserts a second row for the same `document_id` with status `processed`. There is no unique constraint on `document_id`. S3 delivers events at least once, so a retried event inserts every chunk again. | `upload_handler.py:167`, `document_processor.py:326`, `src/db_init/db_init.py:170-186` |
| H3 | **Partial failures leave a document marked processed.** The document row is committed as `processed` before any chunk is embedded. A failure halfway through leaves a "processed" document with some of its chunks. | `document_processor.py:326-343` |
| H4 | **Failed ingestions disappear.** S3 invokes the processor asynchronously with no dead-letter queue or failure destination. After the built-in retries the event is dropped and the document stays in whatever state it reached. | `modules/compute/main.tf` (bucket notification, no `on_failure`) |
| H5 | **Only the first record of an S3 event is processed.** | `document_processor.py:450-451` |
| H6 | **No update or delete path.** A document cannot be replaced, versioned or deleted through the system, so stale content stays retrievable indefinitely and there is no way to honour a deletion request. | whole codebase |
| H7 | **Prompts, documents and personal data are written to CloudWatch.** Every handler logs the full API Gateway event, which contains the question, and the upload body, which contains the whole file in base64. The query Lambda also logs the raw web-search results. | `query_processor.py:616`, `:697`, `upload_handler.py:94` |
| H8 | **Documents and web results go into the prompt with no trust boundary.** Retrieved text is concatenated straight into the instruction prompt, so instructions inside a document are indistinguishable from the system's instructions. | `query_processor.py:446-460` |
| H9 | **Evaluation scores are invented on failure.** Every query runs three extra Gemini calls to grade itself (on by default). When grading fails, or the model returns no number, the score is reported as 0.5. A dashboard built on these numbers would show plausible quality whether or not grading ran. | `query_processor.py:47`, `:522-527`, `:607-611` |
| H10 | **Generation failures return HTTP 200.** A failed model call returns the string "Sorry, I couldn't generate a response" as a successful answer, so error-rate alarms never fire. | `query_processor.py:476-478` |
| H11 | **CI deploys with long-lived IAM access keys** stored as repository secrets, and `terraform plan` runs with `-lock=false`, so two concurrent runs can corrupt state. | `.github/workflows/deploy.yml:210-214`, `:299` |
| H12 | **The upload path caps file size at about 4.5 MB.** Files travel base64-encoded inside a JSON body, so they hit the 6 MB synchronous Lambda payload limit. | `upload_handler.py:123-145` |
| H13 | **The vector index is built empty.** `ivfflat` learns its cluster centres when the index is created. `db_init` creates it on an empty table with `lists = 100`, so the centres carry no information about the data and recall degrades as the table grows. The fallback creates a B-tree on a vector column, which cannot serve similarity search. | `db_init.py:222-238` |

### Medium

| ID | Finding | Evidence |
|---|---|---|
| M1 | One IAM role for all five Lambdas. It grants `logs:*` on every log group, `bedrock:InvokeModel` on `*` (unused), DynamoDB `Scan`, and Cognito admin actions to the document processor. | `modules/compute/main.tf` (`lambda_policy`) |
| M2 | CORS allows every origin on the API and on the documents bucket. | `modules/api/main.tf`, `modules/storage/main.tf` |
| M3 | Cognito uses the implicit OAuth flow, `USER_PASSWORD_AUTH`, and MFA off. An extra Lambda proxies user passwords to Cognito. | `modules/auth/main.tf`, `src/auth_handler/auth_handler.py` |
| M4 | Internal exception text is returned to clients (`Internal error: {str(e)}`). | `query_processor.py:747`, `upload_handler.py:229` |
| M5 | The Gemini API key and the database password pass through Terraform variables and `random_password`, so both sit in plain text in the Terraform state file. | `modules/compute/main.tf`, `modules/database/main.tf` |
| M6 | Embedding is one sequential API call per chunk with no batching, retry or backoff, so a large document hits rate limits. | `document_processor.py:347-352` |
| M7 | If the Gemini secret cannot be read at import, the processor logs the error and continues, then fails later with `NameError: client`. | `document_processor.py:80-84` |
| M8 | Upload metadata falls back to DynamoDB when PostgreSQL fails, so the two stores disagree about which documents exist. | `upload_handler.py:187-204` |
| M9 | The web-search trigger compares raw cosine similarity with a fixed 0.7. That threshold is specific to one embedding model and dimension and was never calibrated. | `query_processor.py:362-405` |
| M10 | Every environment root (`environments/{dev,staging,prod}/main.tf`) is a 249-line copy. A fix applied to one environment silently misses the others. | `environments/*/main.tf` |
| M11 | Resource drift is managed with `import_resources.sh` (602 lines) and `cleanup.sh` (1,412 lines) rather than through state, and `prevent_destroy` sits on resources that the cleanup workflow then deletes by script. | `scripts/` |
| M12 | A NAT gateway is required only so Lambdas can reach Gemini. It costs more per month than the rest of an idle dev stack, which the README's "~$3 to experiment" estimate does not reflect. | `modules/vpc/main.tf` |

### Low

| ID | Finding | Evidence |
|---|---|---|
| L1 | `db_init` builds SQL with f-strings from configuration values. | `db_init.py:98`, `:103` |
| L2 | Python 3.11 runtime; dependencies are unpinned lower bounds with no lock file. | `modules/compute/main.tf`, `requirements*.txt` |
| L3 | Dev bastion security group allows `0.0.0.0/0`. | `environments/dev/terraform.tfvars` |
| L4 | No bucket versioning, no TLS-only bucket policy, S3-managed keys rather than KMS. | `modules/storage/main.tf` |

## What the system lacked entirely

Measured against the target requirements in [the implementation plan](../plan/implementation-plan.md), these capabilities had no implementation at all.

- Tenant isolation and document-level access control.
- Hybrid (keyword plus vector) retrieval, metadata filters, reranking, parent-child retrieval.
- Citations, and any check that an answer is supported by the retrieved text.
- A handled "not enough information" outcome distinct from a normal answer.
- Document versioning, deletion, and propagation of either to the index.
- A dead-letter path for failed ingestion, and retry with backoff around model calls.
- Rate limits or per-tenant usage budgets.
- Structured logs, correlation IDs, latency, token and cost metrics, dashboards.
- A version-controlled evaluation dataset and retrieval-quality metrics.
- Any test that runs against a real PostgreSQL, S3 or SQS implementation.
- Infrastructure security scanning, dependency scanning, or a lock file.

## What is worth keeping

- **PostgreSQL with pgvector as the retrieval store.** It gives keyword search, vector search, transactions and row-level security in one engine. The rewrite keeps it and moves to Aurora Serverless v2 (see the service selection guide).
- **The serverless shape.** Lambda plus an event-driven ingestion path fits a bursty, low-baseline workload.
- **Terraform.** It stays, restructured into one module set with thin per-environment roots.
- **The MCP web-search server**, as an optional, server-configured tool with its output treated as untrusted.
- **A thin Streamlit client** for demos, rewritten to call the API only.

Everything else is replaced. The rewrite has no compatibility constraint, because C1, C3 and H2 mean no deployment of this commit could hold correct multi-user data.
