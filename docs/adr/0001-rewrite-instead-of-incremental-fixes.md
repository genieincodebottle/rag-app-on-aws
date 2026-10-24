# 0001. Rewrite instead of fixing the existing implementation

- Status: accepted, 2026-10-23
- Context: [current-state assessment](../audit/current-state-assessment.md)

## Context

The audited commit (`81fa2a6`) had four critical findings: any signed-in user could read any other user's documents (identity taken from the request body), a client-chosen URL let callers make the Lambda send requests anywhere, the ingestion function could not import on a fresh install, and the tests replaced every external library with mocks, so none of this was detectable. Many high findings (zero vectors on failure, duplicate rows, no DLQ, no delete path) sat in the same few functions.

## Decision

Replace the application code, infrastructure and pipeline rather than patch them. Keep what the assessment found sound: PostgreSQL with pgvector, a serverless shape, Terraform, a thin Streamlit client and the MCP web-search server.

## Alternatives

- **Patch the critical findings only.** Fixes C1 to C4 but leaves no tenant model, no versioning, no deletion, untestable module-level clients, and copied Terraform roots. Every later requirement would have been built on top of that.
- **Incremental refactor behind the same API.** The old API's contract itself was the problem (identity and MCP URL in the body), so compatibility had no value worth the effort.

## Consequences

- The API is new (`/v1/...`), and old clients do not work with it. No deployment of the old commit could hold correct multi-user data, so there is nothing to migrate.
- The old YouTube walkthrough describes the previous version. Tag the old commit before merging so it stays reachable.
