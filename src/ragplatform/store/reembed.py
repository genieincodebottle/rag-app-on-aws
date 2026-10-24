"""Re-embedding every chunk after an embedding model or dimension change.

This is a maintenance operation. While it runs, search results mix two
embedding spaces (same dimension) or vector search is unavailable (dimension
change), so run it in a maintenance window. It runs as the owner role, which
row-level security does not filter.

Steps for a dimension change:
1. drop the HNSW index and the parent/vector check constraint;
2. change the column type, clearing every vector;
3. embed every searchable chunk in batches, committing each batch;
4. restore the constraint and the index, and record the new embedding space.

The embedding space is recorded only at the end. An interrupted run therefore
starts over from the beginning when rerun, which is slower but never leaves a
mix of vectors from two models behind.
"""

from __future__ import annotations

from typing import Any

from ragplatform.ingestion.chunking import embed_text_for
from ragplatform.observability.telemetry import logger
from ragplatform.providers.base import Embedder
from ragplatform.store.db import Database
from ragplatform.store.repository import vector_literal


def reembed_all(
    db: Database, embedder: Embedder, *, contextual_headers: bool, batch_size: int = 64
) -> dict[str, Any]:
    space = embedder.space
    with db.transaction() as tx:
        stored = tx.execute("SELECT provider, model_id, dimensions FROM embedding_space")
        current_dims = int(stored[0]["dimensions"]) if stored else space.dimensions
        if current_dims != space.dimensions:
            logger.info(
                "changing vector dimension", extra={"from": current_dims, "to": space.dimensions}
            )
            tx.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")
            tx.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_parent_has_no_vector")
            tx.execute(
                f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({int(space.dimensions)})"
                " USING NULL"
            )
        elif stored and stored[0]["model_id"] != space.model_id:
            # Same width, different model: clear vectors so the loop re-embeds them.
            tx.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_parent_has_no_vector")
            tx.execute("UPDATE chunks SET embedding = NULL WHERE kind <> 'parent'")
        else:
            # Same model and width: only chunks left without a vector are embedded.
            tx.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_parent_has_no_vector")

    total = 0
    while True:
        with db.transaction() as tx:
            rows = tx.execute(
                "SELECT c.chunk_id, c.text, c.section_path, d.title FROM chunks c"
                " JOIN documents d ON d.document_id = c.document_id"
                " WHERE c.kind <> 'parent' AND c.embedding IS NULL"
                " ORDER BY c.chunk_id LIMIT :n",
                {"n": batch_size},
            )
            if not rows:
                break
            texts = [
                embed_text_for(r["title"], r["section_path"], r["text"], contextual_headers)
                for r in rows
            ]
            vectors = embedder.embed_documents(texts)
            for row, vector in zip(rows, vectors, strict=True):
                tx.execute(
                    "UPDATE chunks SET embedding = CAST(:v AS vector)"
                    " WHERE chunk_id = CAST(:id AS uuid)",
                    {"v": vector_literal(vector), "id": row["chunk_id"]},
                )
        total += len(rows)
        logger.info("re-embedded batch", extra={"total": total})

    with db.transaction() as tx:
        tx.execute(
            "ALTER TABLE chunks ADD CONSTRAINT chunks_parent_has_no_vector"
            " CHECK ((kind = 'parent') = (embedding IS NULL))"
        )
        tx.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks"
            " USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
        tx.execute(
            "INSERT INTO embedding_space (id, provider, model_id, dimensions)"
            " VALUES (1, :p, :m, :d)"
            " ON CONFLICT (id) DO UPDATE SET provider = EXCLUDED.provider,"
            " model_id = EXCLUDED.model_id, dimensions = EXCLUDED.dimensions, updated_at = now()",
            {"p": space.provider, "m": space.model_id, "d": space.dimensions},
        )
        # Every cached answer was produced from the old vectors.
        tx.execute("DELETE FROM answer_cache")
    return {
        "reembedded_chunks": total,
        "embedding_space": {
            "provider": space.provider,
            "model_id": space.model_id,
            "dimensions": space.dimensions,
        },
    }
