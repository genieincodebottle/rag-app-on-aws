"""Answering a question: budget, cache, retrieve, generate, record.

The answer cache key covers everything that can change an answer for this
caller: tenant, user, groups, admin bypass, question, filters, mode and model.
An entry is valid only for the tenant's current corpus version, which every
ingestion and deletion increments, so a cached answer never cites a deleted or
replaced document.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from ragplatform.agent.runner import AgentRunner
from ragplatform.config import RetrievalMode, Settings
from ragplatform.domain.errors import BudgetExceeded, Forbidden, ValidationFailed
from ragplatform.domain.models import (
    AccessScope,
    Answer,
    AnswerStatus,
    AuthContext,
    Citation,
    RetrievedChunk,
    SearchFilters,
    TokenUsage,
)
from ragplatform.generation.answers import INSUFFICIENT_MESSAGE
from ragplatform.generation.generator import Generator
from ragplatform.observability.cost import chat_cost, embedding_cost, rerank_cost
from ragplatform.observability.telemetry import (
    Telemetry,
    logger,
    question_fingerprint,
    safe_text,
    tracer,
)
from ragplatform.retrieval.retriever import Retriever, pack_context
from ragplatform.store.db import Database
from ragplatform.store.repository import Repository


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    filters: SearchFilters | None = None
    mode: Literal["standard", "agentic"] = "standard"
    retrieval_mode: RetrievalMode | None = None
    use_cache: bool = True
    debug: bool = False

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("question is empty")
        return value


class RetrievedSummary(BaseModel):
    marker: str
    chunk_id: UUID
    document_id: UUID
    title: str
    score: float
    vector_rank: int | None
    keyword_rank: int | None
    rerank_score: float | None
    flagged: bool


class QueryResponse(BaseModel):
    request_id: str
    status: AnswerStatus
    answer: str
    citations: list[Citation]
    conflicts: list[str]
    model_id: str
    usage: TokenUsage
    cost_usd: float | None
    cached: bool
    latency_ms: float
    timings_ms: dict[str, float] = Field(default_factory=dict)
    downgraded_reason: str | None = None
    retrieved: list[RetrievedSummary] | None = None
    agent_trace: list[dict[str, Any]] | None = None


class QueryService:
    def __init__(
        self,
        *,
        db: Database,
        repo: Repository,
        retriever: Retriever,
        generator: Generator,
        agent: AgentRunner,
        settings: Settings,
        telemetry: Telemetry,
    ) -> None:
        self.db = db
        self.repo = repo
        self.retriever = retriever
        self.generator = generator
        self.agent = agent
        self.settings = settings
        self.telemetry = telemetry

    def _cache_key(self, ctx: AuthContext, request: QueryRequest) -> str:
        material = {
            "u": ctx.user_id,
            "g": sorted(ctx.groups),
            "a": ctx.is_tenant_admin,
            "q": request.question.lower(),
            "f": request.filters.model_dump(mode="json") if request.filters else None,
            "m": request.mode,
            "r": (request.retrieval_mode or self.settings.retrieval_mode).value,
            "model": self.generator.chat.model_id,
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    @tracer.capture_method(capture_response=False)
    def answer(self, ctx: AuthContext, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        if len(request.question) > self.settings.max_question_chars:
            raise ValidationFailed(
                f"question is longer than {self.settings.max_question_chars} characters",
                public_message=f"The question must be at most {self.settings.max_question_chars}"
                " characters.",
            )
        scope = AccessScope.for_user(ctx)
        log = {
            "request_id": ctx.request_id,
            "tenant_id": ctx.tenant_id,
            "question": safe_text(request.question, allowed=self.settings.log_prompts),
            "mode": request.mode,
        }

        cache_key = self._cache_key(ctx, request)
        with self.db.transaction(scope) as tx:
            tenant = self.repo.get_tenant(tx, ctx.tenant_id)
            if tenant is None:
                raise Forbidden(f"tenant {ctx.tenant_id} is not provisioned")
            usage = self.repo.usage_today(tx, ctx.tenant_id)
            if int(usage["queries"]) >= int(tenant["daily_query_limit"]):
                raise BudgetExceeded("daily query limit reached")
            if int(usage["input_tokens"]) + int(usage["output_tokens"]) >= int(
                tenant["daily_token_limit"]
            ):
                raise BudgetExceeded("daily token limit reached")
            if request.mode == "agentic" and not tenant["allow_agentic"]:
                raise Forbidden(
                    "agentic mode is not enabled for this tenant",
                    public_message="Agentic mode is not enabled for your organisation.",
                )
            corpus_version = int(tenant["corpus_version"])
            cached = (
                self.repo.cache_get(
                    tx, ctx.tenant_id, cache_key, corpus_version, self.settings.answer_cache_ttl_s
                )
                if request.use_cache and not request.debug
                else None
            )
            if cached is not None:
                self.repo.usage_add(
                    tx, ctx.tenant_id, queries=1, input_tokens=0, output_tokens=0, cost_usd=0.0
                )

        if cached is not None:
            self.telemetry.add("CacheHit", 1)
            response = QueryResponse.model_validate(
                {
                    **cached,
                    "request_id": ctx.request_id,
                    "cached": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "timings_ms": {},
                    "cost_usd": 0.0,
                }
            )
            logger.info("query served from cache", extra={**log, "status": response.status})
            return response
        self.telemetry.add("CacheMiss", 1)

        timings: dict[str, float] = {}
        trace: list[dict[str, Any]] | None = None
        rerank_queries = 0
        if request.mode == "agentic":
            run = self.agent.run(
                scope=scope,
                question=request.question,
                filters=request.filters,
                allow_web=bool(tenant["allow_web_search"]),
                request_id=ctx.request_id,
            )
            answer, sources, trace = run.answer, run.sources, run.trace
        else:
            retrieval = self.retriever.retrieve(
                scope, request.question, filters=request.filters, mode=request.retrieval_mode
            )
            timings.update(retrieval.timings_ms)
            rerank_queries = 1 if retrieval.reranked else 0
            sources = pack_context(retrieval.chunks, self.settings.context_token_budget)
            if not sources:
                # Nothing the caller can see matched. No model call is made, so
                # an empty or fully restricted corpus costs nothing to query.
                answer = Answer(
                    status=AnswerStatus.INSUFFICIENT_CONTEXT,
                    answer=INSUFFICIENT_MESSAGE,
                    model_id=self.generator.chat.model_id,
                    downgraded_reason="no_retrieval_results",
                )
            else:
                gen_started = time.perf_counter()
                answer = self.generator.answer(request.question, sources, retrieval.snippets)
                timings["generate"] = (time.perf_counter() - gen_started) * 1000

        cost = self._cost(answer, request.question, rerank_queries)
        response = QueryResponse(
            request_id=ctx.request_id,
            status=answer.status,
            answer=answer.answer,
            citations=answer.citations,
            conflicts=answer.conflicts,
            model_id=answer.model_id,
            usage=answer.usage,
            cost_usd=cost,
            cached=False,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            downgraded_reason=answer.downgraded_reason,
            retrieved=_summaries(sources) if request.debug else None,
            agent_trace=trace,
        )

        with self.db.transaction(scope) as tx:
            self.repo.usage_add(
                tx,
                ctx.tenant_id,
                queries=1,
                input_tokens=answer.usage.input_tokens,
                output_tokens=answer.usage.output_tokens,
                cost_usd=cost or 0.0,
            )
            if request.use_cache and not request.debug:
                self.repo.cache_put(
                    tx,
                    ctx.tenant_id,
                    cache_key,
                    corpus_version,
                    response.model_dump(
                        mode="json",
                        exclude={"request_id", "cached", "latency_ms", "timings_ms", "cost_usd"},
                    ),
                )
            self.repo.audit(
                tx,
                tenant_id=ctx.tenant_id,
                actor_id=ctx.user_id,
                action="query",
                resource_type="query",
                resource_id=None,
                request_id=ctx.request_id,
                detail={
                    "question_sha256_16": question_fingerprint(request.question),
                    "status": answer.status.value,
                    "mode": request.mode,
                    "cited_documents": sorted({str(c.document_id) for c in answer.citations}),
                },
            )

        self._metrics(response, sources)
        logger.info(
            "query answered",
            extra={
                **log,
                "status": answer.status.value,
                "citations": len(answer.citations),
                "sources": len(sources),
                "input_tokens": answer.usage.input_tokens,
                "output_tokens": answer.usage.output_tokens,
                "latency_ms": response.latency_ms,
            },
        )
        return response

    def _cost(self, answer: Answer, question: str, rerank_queries: int) -> float | None:
        parts = [
            chat_cost(answer.model_id, answer.usage),
            embedding_cost(self.retriever.embedder.space.model_id, len(question) // 4 + 1),
        ]
        if rerank_queries and self.retriever.reranker is not None:
            parts.append(rerank_cost(self.retriever.reranker.model_id, rerank_queries))
        known = [p for p in parts if p is not None]
        return round(sum(known), 8) if known and parts[0] is not None else None

    def _metrics(self, response: QueryResponse, sources: list[RetrievedChunk]) -> None:
        t = self.telemetry
        t.add("QueryLatency", response.latency_ms, "ms")
        t.add("AnswerStatus", 1, status=response.status.value)
        t.add("InputTokens", response.usage.input_tokens)
        t.add("OutputTokens", response.usage.output_tokens)
        t.add("CitationsPerAnswer", len(response.citations))
        if response.cost_usd is not None:
            t.add("EstimatedCostMicroUSD", round(response.cost_usd * 1e6), "none")
        flagged = sum(1 for s in sources if s.risk_flags)
        if flagged:
            t.add("FlaggedSourcesInContext", flagged)


def _summaries(sources: list[RetrievedChunk]) -> list[RetrievedSummary]:
    return [
        RetrievedSummary(
            marker=f"S{i}",
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            title=c.title,
            score=round(c.score, 6),
            vector_rank=c.vector_rank,
            keyword_rank=c.keyword_rank,
            rerank_score=c.rerank_score,
            flagged=bool(c.risk_flags),
        )
        for i, c in enumerate(sources, start=1)
    ]
