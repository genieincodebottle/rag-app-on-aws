"""Retrieval: embed, search (vector, keyword or hybrid), rerank, expand, pack.

The stages, in order:

1. Embed the question (skipped in keyword mode).
2. One SQL statement runs the vector and keyword searches and fuses them with
   reciprocal rank fusion, under row-level security for the caller.
3. Optional reranking re-scores the top candidates with a cross-encoder
   (Bedrock Rerank). It reads the matched child text, which is short and precise.
4. Child chunks are replaced by their parent section, and children of the same
   parent collapse into one source.
5. Sources are packed into the context token budget in rank order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ragplatform.config import RetrievalMode, Settings
from ragplatform.domain.errors import EmbeddingSpaceMismatch
from ragplatform.domain.models import AccessScope, RetrievedChunk, SearchFilters
from ragplatform.ingestion.chunking import estimate_tokens
from ragplatform.observability.telemetry import Telemetry, tracer
from ragplatform.providers.base import Embedder, Reranker
from ragplatform.store.db import Database
from ragplatform.store.repository import Repository, as_list


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    snippets: dict[str, str]  # chunk_id -> matched child text, for citation snippets
    candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    reranked: bool = False


class Retriever:
    def __init__(
        self,
        *,
        db: Database,
        repo: Repository,
        embedder: Embedder,
        reranker: Reranker | None,
        settings: Settings,
        telemetry: Telemetry,
    ) -> None:
        self.db = db
        self.repo = repo
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        self.telemetry = telemetry
        self._space_checked = False

    def check_embedding_space(self) -> None:
        if self._space_checked:
            return
        with self.db.transaction() as tx:
            stored = self.repo.get_embedding_space(tx)
        configured = self.embedder.space
        if stored is None or (stored["model_id"], int(stored["dimensions"])) != (
            configured.model_id,
            configured.dimensions,
        ):
            raise EmbeddingSpaceMismatch(
                f"index built with {stored}, configured {configured}. "
                "Run `ragplatform reembed` after changing the embedding model."
            )
        self._space_checked = True

    @tracer.capture_method(capture_response=False)
    def retrieve(
        self,
        scope: AccessScope,
        question: str,
        *,
        filters: SearchFilters | None = None,
        mode: RetrievalMode | None = None,
        final_k: int | None = None,
        use_reranker: bool = True,
    ) -> RetrievalResult:
        mode = mode or self.settings.retrieval_mode
        final_k = final_k or self.settings.final_k
        timings: dict[str, float] = {}

        query_vector = None
        if mode != RetrievalMode.KEYWORD:
            self.check_embedding_space()
            started = time.perf_counter()
            query_vector = self.embedder.embed_query(question)
            timings["embed"] = (time.perf_counter() - started) * 1000

        rerank = self.reranker is not None and use_reranker
        # Fetch more than needed: reranking needs candidates to reorder, and
        # parent collapse can merge several hits into one source.
        limit = self.settings.rerank_candidates if rerank else final_k * 3
        started = time.perf_counter()
        with self.db.transaction(scope, search_tuning=self.settings.hnsw_ef_search) as tx:
            hits = self.repo.search(
                tx,
                tenant_id=scope.tenant_id,
                query_vector=query_vector,
                question=question if mode != RetrievalMode.VECTOR else None,
                filters=filters,
                vector_candidates=self.settings.vector_candidates,
                keyword_candidates=self.settings.keyword_candidates,
                rrf_k=self.settings.rrf_k,
                limit=limit,
            )
            parents = self.repo.parent_texts(
                tx, [h.parent_chunk_id for h in hits if h.parent_chunk_id is not None]
            )
        timings["search"] = (time.perf_counter() - started) * 1000
        candidates = len(hits)

        if rerank and hits and self.reranker is not None:
            started = time.perf_counter()
            order = self.reranker.rerank(question, [h.text for h in hits], top_n=len(hits))
            hits = [
                hits[index].model_copy(update={"rerank_score": score}) for index, score in order
            ]
            timings["rerank"] = (time.perf_counter() - started) * 1000

        snippets: dict[str, str] = {}
        expanded: list[RetrievedChunk] = []
        seen_parents: set[str] = set()
        for hit in hits:
            snippets[str(hit.chunk_id)] = hit.text
            if hit.parent_chunk_id is not None and hit.parent_chunk_id in parents:
                key = str(hit.parent_chunk_id)
                if key in seen_parents:
                    continue
                seen_parents.add(key)
                parent = parents[hit.parent_chunk_id]
                flags = sorted(set(hit.risk_flags) | set(as_list(parent.get("risk_flags"))))
                hit = hit.model_copy(
                    update={
                        "text": parent["text"],
                        "page_start": parent["page_start"],
                        "page_end": parent["page_end"],
                        "risk_flags": flags,
                    }
                )
            expanded.append(hit)
            if len(expanded) == final_k:
                break

        self.telemetry.add("RetrievalCandidates", candidates)
        self.telemetry.add("RetrievalLatency", timings["search"], "ms")
        return RetrievalResult(
            chunks=expanded,
            snippets=snippets,
            candidates=candidates,
            timings_ms=timings,
            reranked=rerank,
        )


def pack_context(chunks: list[RetrievedChunk], budget_tokens: int) -> list[RetrievedChunk]:
    """Keep chunks in rank order until the token budget is spent.

    A chunk that does not fit is skipped rather than truncated: a truncated
    source can drop the sentence that qualifies the one the model quotes.
    """
    packed: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk.text) + 30  # tag and attribute overhead
        if used + cost > budget_tokens:
            continue
        packed.append(chunk)
        used += cost
    return packed
