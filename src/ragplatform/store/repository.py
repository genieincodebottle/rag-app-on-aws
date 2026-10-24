"""Every SQL statement the runtime issues.

All methods take an open transaction. The caller decides the transaction
boundary and the ``AccessScope``, so a multi-step change (activating a document
version, deleting a document) commits or rolls back as one unit.

Tenant isolation is enforced twice: row-level security on every table, and an
explicit ``tenant_id = :tenant`` predicate in the queries that matter most. A
bug in one layer does not expose another tenant's data on its own.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from ragplatform.domain.models import (
    ChunkDraft,
    ChunkKind,
    Document,
    DocumentVersion,
    RetrievedChunk,
    SearchFilters,
)
from ragplatform.store.db import Row, Transaction

_INSERT_BATCH = 25  # keeps a Data API request well under its payload limits
_TERM = re.compile(r"[a-z0-9]{2,}")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        loaded = json.loads(value) if value.startswith("[") else value.strip("{}").split(",")
        return [v for v in loaded if v != ""]
    return list(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        loaded: dict[str, Any] = json.loads(value)
        return loaded
    return dict(value)


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def parse_vector(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")]


def keyword_query(question: str, max_terms: int = 32) -> str | None:
    """Build an OR query for ``to_tsquery`` from a natural-language question.

    ``websearch_to_tsquery`` ANDs every word, so a question phrased differently
    from the document matches nothing. OR-ing the terms and ranking by
    ``ts_rank_cd`` rewards chunks that match more of them. Only ``[a-z0-9]``
    tokens reach ``to_tsquery``, so no query syntax can be injected.
    """
    terms = list(dict.fromkeys(_TERM.findall(question.lower())))[:max_terms]
    return " | ".join(terms) if terms else None


def _document(row: Row) -> Document:
    return Document(
        document_id=row["document_id"],
        tenant_id=row["tenant_id"],
        owner_id=row["owner_id"],
        title=row["title"],
        source_name=row["source_name"],
        content_type=row["content_type"],
        acl_groups=as_list(row["acl_groups"]),
        tags=as_list(row["tags"]),
        status=row["status"],
        current_version=row["current_version"],
        latest_version=row["latest_version"],
        metadata=_as_dict(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _version(row: Row) -> DocumentVersion:
    return DocumentVersion.model_validate(dict(row))


_DOC_COLUMNS = (
    "document_id, tenant_id, owner_id, title, source_name, content_type,"
    " to_jsonb(acl_groups) AS acl_groups, to_jsonb(tags) AS tags, status,"
    " current_version, latest_version, metadata, created_at, updated_at"
)
_VERSION_COLUMNS = (
    "document_id, version, sha256, size_bytes, storage_key, status, chunk_count,"
    " attempts, error_code, error_detail, created_at, processed_at"
)


class Repository:
    # ------------------------------------------------------------------ tenants

    def get_tenant(self, tx: Transaction, tenant_id: str) -> Row | None:
        rows = tx.execute(
            "SELECT tenant_id, name, corpus_version, daily_query_limit, daily_token_limit,"
            " allow_agentic, allow_web_search FROM tenants WHERE tenant_id = :t",
            {"t": tenant_id},
        )
        return rows[0] if rows else None

    def bump_corpus_version(self, tx: Transaction, tenant_id: str) -> None:
        tx.execute(
            "UPDATE tenants SET corpus_version = corpus_version + 1 WHERE tenant_id = :t",
            {"t": tenant_id},
        )

    def get_embedding_space(self, tx: Transaction) -> Row | None:
        rows = tx.execute("SELECT provider, model_id, dimensions FROM embedding_space")
        return rows[0] if rows else None

    # ------------------------------------------------------------------ documents

    def create_document(
        self,
        tx: Transaction,
        *,
        tenant_id: str,
        owner_id: str,
        title: str,
        source_name: str,
        content_type: str,
        acl_groups: Sequence[str],
        tags: Sequence[str],
        metadata: dict[str, Any],
        sha256: str,
        size_bytes: int,
        storage_key_for: Any,
    ) -> tuple[Document, DocumentVersion]:
        document_id = uuid.uuid4()
        doc_rows = tx.execute(
            "INSERT INTO documents (document_id, tenant_id, owner_id, title, source_name,"
            " content_type, acl_groups, tags, status, latest_version, metadata)"
            " VALUES (CAST(:id AS uuid), :tenant, :owner, :title, :source, :ctype,"
            " text_array(CAST(:acl AS jsonb)), text_array(CAST(:tags AS jsonb)),"
            " 'pending_upload', 1, CAST(:meta AS jsonb))"
            f" RETURNING {_DOC_COLUMNS}",
            {
                "id": document_id,
                "tenant": tenant_id,
                "owner": owner_id,
                "title": title,
                "source": source_name,
                "ctype": content_type,
                "acl": list(acl_groups),
                "tags": list(tags),
                "meta": metadata,
            },
        )
        version = self._insert_version(
            tx,
            tenant_id,
            owner_id,
            document_id,
            1,
            sha256,
            size_bytes,
            storage_key_for(document_id, 1),
        )
        return _document(doc_rows[0]), version

    def add_version(
        self,
        tx: Transaction,
        *,
        tenant_id: str,
        owner_id: str,
        document_id: UUID,
        sha256: str,
        size_bytes: int,
        storage_key_for: Any,
    ) -> DocumentVersion:
        rows = tx.execute(
            "UPDATE documents SET latest_version = latest_version + 1, updated_at = now()"
            " WHERE document_id = CAST(:id AS uuid) AND tenant_id = :tenant"
            " AND status <> 'deleted' RETURNING latest_version",
            {"id": document_id, "tenant": tenant_id},
        )
        if not rows:
            raise LookupError(str(document_id))
        version = int(rows[0]["latest_version"])
        return self._insert_version(
            tx,
            tenant_id,
            owner_id,
            document_id,
            version,
            sha256,
            size_bytes,
            storage_key_for(document_id, version),
        )

    def _insert_version(
        self,
        tx: Transaction,
        tenant_id: str,
        owner_id: str,
        document_id: UUID,
        version: int,
        sha256: str,
        size_bytes: int,
        storage_key: str,
    ) -> DocumentVersion:
        rows = tx.execute(
            "INSERT INTO document_versions (document_id, version, tenant_id, owner_id, sha256,"
            " size_bytes, storage_key, status) VALUES (CAST(:id AS uuid), :v, :tenant, :owner,"
            f" :sha, :size, :key, 'pending_upload') RETURNING {_VERSION_COLUMNS}",
            {
                "id": document_id,
                "v": version,
                "tenant": tenant_id,
                "owner": owner_id,
                "sha": sha256,
                "size": size_bytes,
                "key": storage_key,
            },
        )
        return _version(rows[0])

    def release_claim(self, tx: Transaction, document_id: UUID, version: int) -> None:
        """Hand a version back after a transient failure so the retry can claim it."""
        tx.execute(
            "UPDATE document_versions SET status = 'uploaded', claimed_at = NULL"
            " WHERE document_id = CAST(:id AS uuid) AND version = :v AND status = 'processing'",
            {"id": document_id, "v": version},
        )

    def find_document_by_sha(
        self, tx: Transaction, tenant_id: str, owner_id: str, sha256: str
    ) -> UUID | None:
        rows = tx.execute(
            "SELECT document_id FROM document_versions WHERE tenant_id = :tenant"
            " AND owner_id = :owner AND sha256 = :sha"
            " AND status NOT IN ('failed', 'superseded', 'deleted') LIMIT 1",
            {"tenant": tenant_id, "owner": owner_id, "sha": sha256},
        )
        return UUID(str(rows[0]["document_id"])) if rows else None

    def get_document(self, tx: Transaction, document_id: UUID) -> Document | None:
        rows = tx.execute(
            f"SELECT {_DOC_COLUMNS} FROM documents WHERE document_id = CAST(:id AS uuid)"
            " AND status <> 'deleted'",
            {"id": document_id},
        )
        return _document(rows[0]) if rows else None

    def get_version(
        self, tx: Transaction, document_id: UUID, version: int
    ) -> DocumentVersion | None:
        rows = tx.execute(
            f"SELECT {_VERSION_COLUMNS} FROM document_versions"
            " WHERE document_id = CAST(:id AS uuid) AND version = :v",
            {"id": document_id, "v": version},
        )
        return _version(rows[0]) if rows else None

    def list_versions(self, tx: Transaction, document_id: UUID) -> list[DocumentVersion]:
        rows = tx.execute(
            f"SELECT {_VERSION_COLUMNS} FROM document_versions"
            " WHERE document_id = CAST(:id AS uuid) ORDER BY version",
            {"id": document_id},
        )
        return [_version(r) for r in rows]

    def list_documents(
        self,
        tx: Transaction,
        *,
        limit: int,
        before: tuple[datetime, UUID] | None,
        status: str | None,
    ) -> list[Document]:
        clauses = ["status <> 'deleted'"]
        params: dict[str, Any] = {"limit": limit}
        if status:
            clauses.append("status = :status")
            params["status"] = status
        if before:
            clauses.append(
                "(updated_at, document_id) < (CAST(:b_at AS timestamptz), CAST(:b_id AS uuid))"
            )
            params["b_at"], params["b_id"] = before
        rows = tx.execute(
            f"SELECT {_DOC_COLUMNS} FROM documents WHERE {' AND '.join(clauses)}"
            " ORDER BY updated_at DESC, document_id DESC LIMIT :limit",
            params,
        )
        return [_document(r) for r in rows]

    def update_access(
        self,
        tx: Transaction,
        document_id: UUID,
        *,
        title: str | None,
        acl_groups: Sequence[str] | None,
        tags: Sequence[str] | None,
    ) -> Document | None:
        sets: list[str] = ["updated_at = now()"]
        params: dict[str, Any] = {"id": document_id}
        chunk_sets: list[str] = []
        if title is not None:
            sets.append("title = :title")
            params["title"] = title
        if acl_groups is not None:
            sets.append("acl_groups = text_array(CAST(:acl AS jsonb))")
            chunk_sets.append("acl_groups = text_array(CAST(:acl AS jsonb))")
            params["acl"] = list(acl_groups)
        if tags is not None:
            sets.append("tags = text_array(CAST(:tags AS jsonb))")
            chunk_sets.append("tags = text_array(CAST(:tags AS jsonb))")
            params["tags"] = list(tags)
        rows = tx.execute(
            f"UPDATE documents SET {', '.join(sets)}"
            " WHERE document_id = CAST(:id AS uuid) AND status <> 'deleted'"
            f" RETURNING {_DOC_COLUMNS}",
            params,
        )
        if rows and chunk_sets:
            tx.execute(
                f"UPDATE chunks SET {', '.join(chunk_sets)} WHERE document_id = CAST(:id AS uuid)",
                {k: v for k, v in params.items() if k != "title"},
            )
        return _document(rows[0]) if rows else None

    def mark_uploaded(self, tx: Transaction, document_id: UUID, version: int) -> None:
        tx.execute(
            "UPDATE document_versions SET status = 'uploaded'"
            " WHERE document_id = CAST(:id AS uuid) AND version = :v AND status = 'pending_upload'",
            {"id": document_id, "v": version},
        )

    def delete_document(self, tx: Transaction, document_id: UUID) -> list[str]:
        """Remove every chunk and tombstone the document. Returns storage keys to purge."""
        keys = [
            r["storage_key"]
            for r in tx.execute(
                "UPDATE document_versions SET status = 'deleted'"
                " WHERE document_id = CAST(:id AS uuid) RETURNING storage_key",
                {"id": document_id},
            )
        ]
        tx.execute("DELETE FROM chunks WHERE document_id = CAST(:id AS uuid)", {"id": document_id})
        # The tombstone keeps the id for the audit trail. Title, source name and
        # metadata may hold personal data, so they are scrubbed.
        tx.execute(
            "UPDATE documents SET status = 'deleted', deleted_at = now(), updated_at = now(),"
            " title = 'deleted document', source_name = 'deleted', metadata = '{}'::jsonb,"
            " current_version = NULL WHERE document_id = CAST(:id AS uuid)",
            {"id": document_id},
        )
        return keys

    # ------------------------------------------------------------------ ingestion

    def claim_version(
        self, tx: Transaction, document_id: UUID, version: int, lease_s: int
    ) -> DocumentVersion | None:
        """Take exclusive ownership of a version for processing.

        A duplicate S3 or SQS delivery finds the version already processing (and
        its lease not expired) or already ready, gets no row back, and stops.
        """
        rows = tx.execute(
            "UPDATE document_versions SET status = 'processing', attempts = attempts + 1,"
            " claimed_at = now(), error_code = NULL, error_detail = NULL"
            " WHERE document_id = CAST(:id AS uuid) AND version = :v AND ("
            "   status IN ('pending_upload', 'uploaded')"
            "   OR (status = 'processing' AND claimed_at < now() - make_interval(secs => :lease))"
            f" ) RETURNING {_VERSION_COLUMNS}",
            {"id": document_id, "v": version, "lease": lease_s},
        )
        if not rows:
            return None
        tx.execute(
            "UPDATE documents SET status = 'processing', updated_at = now()"
            " WHERE document_id = CAST(:id AS uuid) AND current_version IS NULL",
            {"id": document_id},
        )
        return _version(rows[0])

    def fail_version(
        self, tx: Transaction, document_id: UUID, version: int, code: str, detail: str
    ) -> None:
        tx.execute(
            "UPDATE document_versions SET status = 'failed', error_code = :code,"
            " error_detail = :detail, processed_at = now()"
            " WHERE document_id = CAST(:id AS uuid) AND version = :v",
            {"id": document_id, "v": version, "code": code, "detail": detail[:1000]},
        )
        # A failed new version leaves an existing ready version in service.
        tx.execute(
            "UPDATE documents SET status = CASE WHEN current_version IS NULL THEN 'failed'"
            " ELSE 'ready' END, updated_at = now() WHERE document_id = CAST(:id AS uuid)",
            {"id": document_id},
        )

    def embeddings_for_hashes(
        self, tx: Transaction, tenant_id: str, hashes: Iterable[str]
    ) -> dict[str, list[float]]:
        wanted = sorted(set(hashes))
        if not wanted:
            return {}
        rows = tx.execute(
            "SELECT DISTINCT ON (content_hash) content_hash, embedding::text AS embedding"
            " FROM chunks WHERE tenant_id = :tenant AND embedding IS NOT NULL"
            " AND content_hash = ANY(text_array(CAST(:hashes AS jsonb)))",
            {"tenant": tenant_id, "hashes": wanted},
        )
        return {r["content_hash"]: parse_vector(r["embedding"]) for r in rows}

    def activate_version(
        self,
        tx: Transaction,
        *,
        document: Document,
        version: int,
        chunks: Sequence[ChunkDraft],
        embeddings: dict[int, list[float]],
        title: str | None = None,
    ) -> int:
        """Write a version's chunks and make it the live version, in one transaction.

        Readers see either the old version or the new one, never a mix, because
        the old chunks are deleted and the new ones inserted as active in the
        same transaction.
        """
        current = tx.execute(
            "SELECT current_version FROM documents WHERE document_id = CAST(:id AS uuid)"
            " FOR UPDATE",
            {"id": document.document_id},
        )
        current_version = current[0]["current_version"] if current else None
        if current_version is not None and current_version > version:
            # An older upload finished after a newer one. Never roll back.
            tx.execute(
                "UPDATE document_versions SET status = 'superseded', processed_at = now()"
                " WHERE document_id = CAST(:id AS uuid) AND version = :v",
                {"id": document.document_id, "v": version},
            )
            return 0

        ids = {c.chunk_index: uuid.uuid4() for c in chunks}
        records = []
        for c in chunks:
            vector = embeddings.get(c.chunk_index)
            if (c.kind == ChunkKind.PARENT) != (vector is None):
                raise ValueError(f"chunk {c.chunk_index}: parents have no vector, others need one")
            records.append(
                {
                    "chunk_id": str(ids[c.chunk_index]),
                    "parent_chunk_id": str(ids[c.parent_index])
                    if c.parent_index is not None
                    else None,
                    "chunk_index": c.chunk_index,
                    "kind": c.kind.value,
                    "section_path": c.section_path,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "text": c.text,
                    "content_hash": c.content_hash,
                    "token_count": c.token_count,
                    "embedding": vector_literal(vector) if vector is not None else None,
                    "risk_flags": list(c.risk_flags),
                }
            )
        # Parents first, so the parent_chunk_id foreign key resolves.
        records.sort(key=lambda r: (r["kind"] != ChunkKind.PARENT.value, r["chunk_index"]))

        tx.execute(
            "DELETE FROM chunks WHERE document_id = CAST(:id AS uuid) AND version <> :v",
            {"id": document.document_id, "v": version},
        )
        tx.execute(
            "DELETE FROM chunks WHERE document_id = CAST(:id AS uuid) AND version = :v",
            {"id": document.document_id, "v": version},
        )
        for start in range(0, len(records), _INSERT_BATCH):
            tx.execute(
                "INSERT INTO chunks (chunk_id, tenant_id, document_id, version, chunk_index, kind,"
                " parent_chunk_id, section_path, page_start, page_end, text, content_hash,"
                " token_count, embedding, risk_flags, owner_id, acl_groups, tags, content_type,"
                " is_active)"
                " SELECT r.chunk_id, :tenant, CAST(:doc AS uuid), :v, r.chunk_index, r.kind,"
                " r.parent_chunk_id, r.section_path, r.page_start, r.page_end, r.text,"
                " r.content_hash, r.token_count, CAST(r.embedding AS vector),"
                " text_array(r.risk_flags), :owner, text_array(CAST(:acl AS jsonb)),"
                " text_array(CAST(:tags AS jsonb)), :ctype, true"
                " FROM jsonb_to_recordset(CAST(:rows AS jsonb)) AS r(chunk_id uuid,"
                " parent_chunk_id uuid, chunk_index int, kind text, section_path text,"
                " page_start int, page_end int, text text, content_hash text, token_count int,"
                " embedding text, risk_flags jsonb)",
                {
                    "tenant": document.tenant_id,
                    "doc": document.document_id,
                    "v": version,
                    "owner": document.owner_id,
                    "acl": document.acl_groups,
                    "tags": document.tags,
                    "ctype": document.content_type,
                    "rows": records[start : start + _INSERT_BATCH],
                },
            )
        tx.execute(
            "UPDATE document_versions SET status = 'superseded'"
            " WHERE document_id = CAST(:id AS uuid) AND version <> :v AND status = 'ready'",
            {"id": document.document_id, "v": version},
        )
        tx.execute(
            "UPDATE document_versions SET status = 'ready', chunk_count = :n, processed_at = now()"
            " WHERE document_id = CAST(:id AS uuid) AND version = :v",
            {"id": document.document_id, "v": version, "n": len(records)},
        )
        tx.execute(
            "UPDATE documents SET status = 'ready', current_version = :v, updated_at = now(),"
            " title = coalesce(:title, title) WHERE document_id = CAST(:id AS uuid)",
            {"id": document.document_id, "v": version, "title": title},
        )
        self.bump_corpus_version(tx, document.tenant_id)
        return len(records)

    # ------------------------------------------------------------------ retrieval

    def search(
        self,
        tx: Transaction,
        *,
        tenant_id: str,
        query_vector: Sequence[float] | None,
        question: str | None,
        filters: SearchFilters | None,
        vector_candidates: int,
        keyword_candidates: int,
        rrf_k: int,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Vector search, keyword search, or both fused with reciprocal rank fusion.

        RRF scores each chunk ``sum(1 / (rrf_k + rank))`` over the lists it
        appears in. It needs no score normalisation, which matters because
        cosine distance and ``ts_rank_cd`` are on unrelated scales.
        """
        params: dict[str, Any] = {"tenant": tenant_id, "rrf_k": rrf_k, "limit": limit}
        where = ["c.tenant_id = :tenant", "c.is_active", "c.kind <> 'parent'"]
        if filters:
            if filters.document_ids:
                where.append(
                    "c.document_id = ANY(CAST(text_array(CAST(:f_docs AS jsonb)) AS uuid[]))"
                )
                params["f_docs"] = [str(d) for d in filters.document_ids]
            if filters.tags:
                where.append("c.tags && text_array(CAST(:f_tags AS jsonb))")
                params["f_tags"] = filters.tags
            if filters.content_types:
                where.append("c.content_type = ANY(text_array(CAST(:f_ct AS jsonb)))")
                params["f_ct"] = filters.content_types
        predicate = " AND ".join(where)

        branches: list[str] = []
        ctes: list[str] = []
        if query_vector is not None:
            params["qvec"] = vector_literal(query_vector)
            params["k_vec"] = vector_candidates
            ctes.append(
                "vec AS (SELECT c.chunk_id, c.embedding <=> CAST(:qvec AS vector) AS dist"
                f" FROM chunks c WHERE {predicate}"
                " ORDER BY c.embedding <=> CAST(:qvec AS vector) LIMIT :k_vec)"
            )
            branches.append(
                "SELECT chunk_id, row_number() OVER (ORDER BY dist) AS rnk, 'v' AS src FROM vec"
            )
        tsquery = keyword_query(question) if question else None
        if tsquery is not None:
            params["tsq"] = tsquery
            params["k_kw"] = keyword_candidates
            ctes.append(
                "kw AS (SELECT c.chunk_id, ts_rank_cd(c.tsv, q.query) AS rank"
                " FROM chunks c, (SELECT to_tsquery('english', :tsq) AS query) q"
                f" WHERE {predicate} AND c.tsv @@ q.query ORDER BY rank DESC LIMIT :k_kw)"
            )
            branches.append(
                "SELECT chunk_id, row_number() OVER (ORDER BY rank DESC) AS rnk, 'k' AS src FROM kw"
            )
        if not branches:
            return []
        ctes.append(
            "fused AS (SELECT chunk_id, SUM(1.0 / (:rrf_k + rnk)) AS score,"
            " MIN(CASE WHEN src = 'v' THEN rnk END) AS vector_rank,"
            " MIN(CASE WHEN src = 'k' THEN rnk END) AS keyword_rank"
            f" FROM ({' UNION ALL '.join(branches)}) u GROUP BY chunk_id)"
        )
        sql = (
            f"WITH {', '.join(ctes)} "
            "SELECT c.chunk_id, c.document_id, c.version, c.section_path, c.page_start, c.page_end,"
            " c.text, c.parent_chunk_id, to_jsonb(c.risk_flags) AS risk_flags, d.title,"
            " d.updated_at, f.score, f.vector_rank, f.keyword_rank"
            " FROM fused f JOIN chunks c ON c.chunk_id = f.chunk_id"
            " JOIN documents d ON d.document_id = c.document_id"
            " ORDER BY f.score DESC, c.chunk_id LIMIT :limit"
        )
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                version=r["version"],
                title=r["title"],
                section_path=r["section_path"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                text=r["text"],
                score=float(r["score"]),
                vector_rank=r["vector_rank"],
                keyword_rank=r["keyword_rank"],
                parent_chunk_id=r["parent_chunk_id"],
                risk_flags=as_list(r["risk_flags"]),
                updated_at=r["updated_at"],
            )
            for r in tx.execute(sql, params)
        ]

    def parent_texts(self, tx: Transaction, parent_ids: Iterable[UUID]) -> dict[UUID, Row]:
        wanted = sorted({str(p) for p in parent_ids})
        if not wanted:
            return {}
        rows = tx.execute(
            "SELECT chunk_id, text, section_path, page_start, page_end,"
            " to_jsonb(risk_flags) AS risk_flags FROM chunks"
            " WHERE chunk_id = ANY(CAST(text_array(CAST(:ids AS jsonb)) AS uuid[]))",
            {"ids": wanted},
        )
        return {UUID(str(r["chunk_id"])): r for r in rows}

    # ------------------------------------------------------------------ cache, usage, audit

    def cache_get(
        self, tx: Transaction, tenant_id: str, key: str, corpus_version: int, ttl_s: int
    ) -> dict[str, Any] | None:
        rows = tx.execute(
            "SELECT response FROM answer_cache WHERE tenant_id = :t AND cache_key = :k"
            " AND corpus_version = :cv AND created_at > now() - make_interval(secs => :ttl)",
            {"t": tenant_id, "k": key, "cv": corpus_version, "ttl": ttl_s},
        )
        return _as_dict(rows[0]["response"]) if rows else None

    def cache_put(
        self,
        tx: Transaction,
        tenant_id: str,
        key: str,
        corpus_version: int,
        response: dict[str, Any],
    ) -> None:
        tx.execute(
            "INSERT INTO answer_cache (tenant_id, cache_key, corpus_version, response)"
            " VALUES (:t, :k, :cv, CAST(:r AS jsonb)) ON CONFLICT (tenant_id, cache_key)"
            " DO UPDATE SET corpus_version = EXCLUDED.corpus_version,"
            " response = EXCLUDED.response, created_at = now()",
            {"t": tenant_id, "k": key, "cv": corpus_version, "r": response},
        )

    def cache_purge_stale(self, tx: Transaction, tenant_id: str, corpus_version: int) -> int:
        rows = tx.execute(
            "DELETE FROM answer_cache WHERE tenant_id = :t AND corpus_version < :cv"
            " RETURNING cache_key",
            {"t": tenant_id, "cv": corpus_version},
        )
        return len(rows)

    def usage_today(self, tx: Transaction, tenant_id: str) -> Row:
        rows = tx.execute(
            "SELECT queries, input_tokens, output_tokens, cost_usd FROM usage_daily"
            " WHERE tenant_id = :t AND day = current_date",
            {"t": tenant_id},
        )
        return (
            rows[0]
            if rows
            else {"queries": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}
        )

    def usage_add(
        self,
        tx: Transaction,
        tenant_id: str,
        *,
        queries: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        tx.execute(
            "INSERT INTO usage_daily"
            " (tenant_id, day, queries, input_tokens, output_tokens, cost_usd)"
            " VALUES (:t, current_date, :q, :i, :o, :c)"
            " ON CONFLICT (tenant_id, day) DO UPDATE SET"
            " queries = usage_daily.queries + EXCLUDED.queries,"
            " input_tokens = usage_daily.input_tokens + EXCLUDED.input_tokens,"
            " output_tokens = usage_daily.output_tokens + EXCLUDED.output_tokens,"
            " cost_usd = usage_daily.cost_usd + EXCLUDED.cost_usd",
            {"t": tenant_id, "q": queries, "i": input_tokens, "o": output_tokens, "c": cost_usd},
        )

    def audit(
        self,
        tx: Transaction,
        *,
        tenant_id: str,
        actor_id: str,
        action: str,
        resource_type: str,
        resource_id: str | None,
        request_id: str | None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        tx.execute(
            "INSERT INTO audit_events (tenant_id, actor_id, action, resource_type, resource_id,"
            " request_id, detail) VALUES (:t, :a, :act, :rt, :rid, :req, CAST(:d AS jsonb))",
            {
                "t": tenant_id,
                "a": actor_id,
                "act": action,
                "rt": resource_type,
                "rid": resource_id,
                "req": request_id,
                "d": detail or {},
            },
        )
