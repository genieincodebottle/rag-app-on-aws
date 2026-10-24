# Security

This guide is the threat model, the controls that answer it, how each control is tested, and what is still open. [architecture.md](architecture.md) shows where the trust boundaries sit.

## Assets

| Asset | Why it matters |
|---|---|
| Document content and chunks | Customer data; may include personal or confidential information |
| Which documents exist | Titles and existence alone can leak (a file called "layoffs-q4.pdf") |
| Tenant boundaries | A single cross-tenant read is a breach |
| Answers and questions | Questions reveal intent; answers carry document content |
| Credentials | Database secrets, the KMS key, deploy roles |
| Budget | Model calls cost money; a runaway client or agent can spend it |

## Threats and controls

STRIDE categories, highest impact first.

| # | Threat | Control | Tested by |
|---|---|---|---|
| T1 | A user reads another tenant's data | Identity from verified Cognito claims only; row-level security on every table keyed to `app.tenant_id`; explicit tenant predicate in search queries; runtime role is not the table owner | `test_isolation.py`: raw SQL as another tenant with ACL bypass returns 0 rows; cross-tenant ids return 404; eval `cross_tenant` questions (0 leaks) |
| T2 | A user reads a document their groups do not allow | Restrictive RLS policy `app_can_read(owner, groups)` on documents and chunks; groups come from the token | ACL tests, eval `acl_denied` (0 leaks) |
| T3 | A user changes their own tenant or groups | `custom:tenant_id` is not in the app client's `write_attributes`; groups are set by administrators | Terraform `modules/auth`; manual check in a deployed pool |
| T4 | Identity supplied in the request body or a header | Never read. In Cognito mode a bearer header is ignored and only authorizer claims count | `test_body_supplied_identity_is_ignored_by_the_api`, `test_lambda_proxy_path_trusts_only_authorizer_claims` |
| T5 | Prompt injection in a document | Sources in delimited blocks with `<` escaped; system prompt treats sources as data; ingest-time screening flags instruction-like text (shown to the model as a note); no write tools in standard mode; citations validated; optional Guardrails | `test_source_text_cannot_close_its_own_block`, injection integration test, eval `injection` question (0 obeyed) |
| T6 | Hidden instructions (zero-width characters, `display:none` HTML) | Normalisation removes invisible and bidirectional control characters; the HTML parser drops hidden elements, scripts and styles | `test_normalize_*`, `test_html_drops_scripts_and_hidden_text*` |
| T7 | Server-side request forgery through web search | Endpoint fixed in configuration, HTTPS only (localhost only in local mode), no redirects, 256 KB response cap | `test_web_search_url_must_be_https`, `test_web_search_does_not_follow_redirects` |
| T8 | Data exfiltration through the answer (a markdown image whose URL carries data, fetched when the browser renders it) | Injection screening flags image links with query strings at ingest; the API returns plain text, never HTML; the UI removes markdown images and shows links as plain text (`safe_markdown` in `ui/app.py`). Any other client must do the same | `exfil_link` pattern test |
| T9 | Model invents facts | Server-side citation validation; uncited answers downgraded; optional Guardrails contextual grounding; `insufficient_context` returned instead of a guess | `test_uncited_answer_is_downgraded`, eval `unanswerable` (refusal accuracy 1.0 live) |
| T10 | Malicious or malformed upload | Declared type checked against magic bytes; PDF page and DOCX decompression limits; exact size and SHA-256 checked after upload; parser exceptions become permanent failures | parser tests, `test_checksum_mismatch_fails_the_version_permanently` |
| T11 | Existence leak through deduplication | Deduplication is per owner, not per tenant; `GET` of a hidden document returns 404, same as a missing one | lifecycle and isolation tests |
| T12 | Runaway cost | Per-tenant daily query and token budgets; API Gateway throttling; WAF rate limit; agent step, token and time budgets; ingest concurrency cap; AWS Budgets alerts | `test_daily_query_budget`, `test_agent_stops_at_its_step_budget` |
| T13 | Sensitive data in logs | Questions logged as a SHA-256 prefix; no request or response bodies in API Gateway logs (`data_trace_enabled = false`); WAF logs redact `Authorization`; prompt logging refused in prod | `test_prompt_logging_is_refused_in_prod` |
| T14 | Deletion request not honoured | Chunks deleted and document tombstoned in one transaction; every S3 version deleted; cache invalidated; failures alarmed; noncurrent versions expire after 30 days as a backstop | `test_delete_removes_chunks_bytes_and_cached_answers`, `test_s3_delete_removes_every_version` |
| T15 | Development settings in production | Config validation refuses local auth, non-Bedrock providers, psycopg and local storage when deployed | `test_lambda_refuses_local_development_settings` |
| T16 | Tampered build or dependency | `uv.lock` pins every dependency; `pip-audit` in CI; SHA-pinned GitHub Actions; build provenance attestation; bundle SHA-256 pinned in Terraform; Dependabot | CI workflow |
| T17 | Stolen CI credentials | No long-lived keys: GitHub OIDC only; roles trust one repository and one GitHub environment; prod deploys need reviewers; deploy role denied IAM user and access-key creation | bootstrap Terraform |
| T18 | Destructive infrastructure change | Plan job refuses deletes or replacements of Aurora, the documents bucket, Cognito and KMS; deletion protection in prod; destroy workflow cannot target prod | deploy and destroy workflows |

## Identity and authorization

- **Authentication**: Cognito user pool, administrator-created users only, 14-character passwords, MFA required in prod. API Gateway validates the ID token's signature, expiry and audience.
- **Authorization layers**, outermost first:
  1. API Gateway rejects unauthenticated requests (except `/v1/health`).
  2. The service checks tenant provisioning, budgets and feature flags (agentic mode).
  3. Owner-or-admin checks for changing or deleting a document; users can share only with groups they belong to.
  4. Row-level security in PostgreSQL filters every read and write.

  Layer 4 is the one that holds even if a bug skips layer 3.
- **Row-level security** is applied per transaction with `set_config(..., true)`, because the RDS Data API resets session state between pooled uses. If the tenant setting is missing, `current_setting` returns NULL and no row matches. Tenant admins get `acl_bypass`, which skips document ACLs inside their tenant only; nothing can skip the tenant policy.

## Secrets and encryption

| Secret | Where | Rotation |
|---|---|---|
| Aurora master password | Secrets Manager, created and rotated by RDS (`manage_master_user_password`) | Automatic (RDS default schedule) |
| Runtime role password | Secrets Manager, written by the migrate function; never in Terraform state | `{"rotate_app_password": true}` to the migrate function |
| Cognito, Bedrock, S3, SQS | IAM roles, no keys | n/a |
| Web-search key (optional) | Function environment, encrypted with the KMS key | Manual |

Encryption at rest uses one customer-managed KMS key per environment (S3 with bucket keys, SQS, Aurora, secrets, Lambda environment variables, log groups). In transit: TLS everywhere; the documents and state buckets deny non-TLS requests; Aurora's parameter group sets `rds.force_ssl = 1`.

## Network

Aurora sits in subnets with no route to the internet and a security group with no inbound rules. The Lambda functions run outside any VPC and reach AWS services over TLS with IAM. There is no NAT gateway, so there is no general egress path to restrict. The one outbound call to a non-AWS endpoint, optional web search, goes only to the configured URL.

## Audit

- `audit_events` (append-only for the runtime role): document registration, version, access change, deletion, ingestion, and every query (question hash, status, cited document ids).
- CloudTrail records AWS API calls. Enable S3 data events on the documents bucket for object-level access logs (not configured by this Terraform, because most accounts manage trails centrally).
- API Gateway access logs record method, path, status, latency, user sub and tenant, never bodies.

## Prompt injection: what the controls do and do not do

The screening patterns catch common, low-effort attacks and nothing clever. They flag; they do not block, because blocking on a phrase lets an attacker suppress legitimate documents by adding that phrase. The structure limits the damage:

- In standard mode the model has no tools, so an injected instruction can only change the text of one answer.
- That answer must cite supplied sources, and the citations are checked.
- Agent tools are read-only, take their tenant from the server, and reject extra arguments.
- A write tool would need an approver callback that returns true for that exact call; none is registered.

An injected instruction can still bias an answer within those limits, for example by making a cited source's misleading sentence look authoritative. Treat retrieved content as untrusted, and use Guardrails' grounding check when a wrong answer is costly.

## Incident response

1. **Contain.** Suspected cross-tenant exposure: set API Gateway stage throttling to 0 (`aws apigateway update-stage ... throttling/rateLimit=0`) or detach the WAF allow rule. Suspected compromised user: `aws cognito-idp admin-user-global-sign-out` and disable the user.
2. **Preserve.** Export the relevant log groups and `audit_events` rows before retention removes them. Do not delete documents involved until they are copied for investigation.
3. **Investigate.** `audit_events` answers which user queried which documents and when; API Gateway access logs map request ids to users; CloudTrail covers infrastructure changes.
4. **Recover and review.** Fix, add a test that reproduces the failure, and record the decision as an ADR if a control changed.

## Known gaps

| Gap | Risk | Planned control |
|---|---|---|
| Runtime-role secret has no scheduled rotation | A leaked password stays valid until rotated by hand | Secrets Manager rotation function using the migrate code path |
| Lambda code signing is not enforced | A bundle uploaded outside the pipeline would run | AWS Signer profile and a code signing configuration in Enforce mode |
| S3 data events are not enabled by this Terraform | No object-level access trail unless the account's trail has it | Optional CloudTrail resource |
| Row-level security through the Data API is contract-tested, not live-tested | The `set_config` pattern is reasoned from documentation | First deployment smoke test with a two-tenant check |
| No automated malware scan of uploads | Malicious files are stored (they are parsed as text and never executed) | GuardDuty Malware Protection for S3 |
