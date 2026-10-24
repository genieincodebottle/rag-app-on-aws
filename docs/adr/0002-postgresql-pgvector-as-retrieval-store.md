# 0002. One PostgreSQL database for vectors, keyword search, metadata and access control

- Status: accepted, 2026-10-23

## Context

Retrieval needs vector search, keyword search and metadata filters. The platform also needs tenant isolation, per-document access control, atomic version replacement, deletion that reaches the index, an answer cache and an audit log.

## Decision

Use Aurora PostgreSQL Serverless v2 with pgvector (HNSW, cosine) and a generated `tsvector` column with a GIN index, and run vector search, keyword search and reciprocal rank fusion in one SQL statement. Keep documents, versions, chunks, cache, usage and audit in the same database.

## Alternatives

OpenSearch Serverless, Amazon S3 Vectors and Bedrock Knowledge Bases are compared in [aws-service-selection.md](../aws-service-selection.md). The deciding properties: only PostgreSQL gives transactions across documents and chunks (a version swap readers see all at once) and row-level security (isolation enforced below the application). S3 Vectors has no keyword search; OpenSearch would split the data model across two stores with no transaction between them.

## Consequences

- A document update, deletion or access change is one transaction, and the answer cache is invalidated in the same place.
- One isolation mechanism covers every table.
- Scale ceiling: pgvector HNSW on one writer suits millions of chunks, not billions. Beyond that, a dedicated vector service with its own isolation design is needed.
- Keyword ranking is `ts_rank_cd`, not BM25.
