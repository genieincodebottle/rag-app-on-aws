"""Bounded agentic retrieval.

Standard mode makes one retrieval and one model call. Agentic mode adds a
planning call and up to a few more retrievals, for questions that ask about
several separate things ("What is the hotel limit in London, and who approves
exceptions?") where one search tends to find only one half.

The loop is fixed, not open-ended:

    plan -> search each sub-question -> (web search, if allowed and needed)
         -> answer -> (one keyword-only retry if the answer was insufficient)

Every step is checked against three budgets (steps, tokens, wall-clock time)
before it starts. A step that would exceed a budget is not run, and the agent
answers with the evidence it already has. If planning fails or returns
something invalid, the agent falls back to the standard single search.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from ragplatform.agent.tools import (
    SearchArgs,
    Tool,
    ToolArgumentsInvalid,
    ToolContext,
    ToolDenied,
    ToolOutput,
    ToolRegistry,
    WebSearchArgs,
)
from ragplatform.agent.web_search import WebSearchClient
from ragplatform.config import RetrievalMode, Settings
from ragplatform.domain.errors import UpstreamUnavailable
from ragplatform.domain.models import (
    AccessScope,
    Answer,
    AnswerStatus,
    RetrievedChunk,
    SearchFilters,
    TokenUsage,
)
from ragplatform.generation.answers import INSUFFICIENT_MESSAGE
from ragplatform.generation.generator import Generator
from ragplatform.generation.prompts import PLAN_SYSTEM, plan_prompt
from ragplatform.observability.telemetry import Telemetry, tracer
from ragplatform.providers.base import ChatModel, ChatRequest
from ragplatform.retrieval.retriever import Retriever, pack_context


class Plan(BaseModel):
    strategy: str = Field(pattern="^(direct|decompose)$")
    sub_questions: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("sub_questions")
    @classmethod
    def _bounded(cls, value: list[str]) -> list[str]:
        cleaned = [" ".join(q.split())[:300] for q in value if len(q.split()) >= 2]
        return cleaned[:3]


@dataclass
class Budget:
    max_steps: int
    max_tokens: int
    deadline: float
    steps: int = 0
    tokens: int = 0

    def allows(self, estimated_tokens: int = 0) -> bool:
        return (
            self.steps < self.max_steps
            and self.tokens + estimated_tokens <= self.max_tokens
            and time.monotonic() < self.deadline
        )

    def spend(self, tokens: int = 0) -> None:
        self.steps += 1
        self.tokens += tokens


@dataclass
class AgentRun:
    answer: Answer
    sources: list[RetrievedChunk]
    trace: list[dict[str, Any]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    stopped_by_budget: bool = False


class AgentRunner:
    def __init__(
        self,
        *,
        chat: ChatModel,
        retriever: Retriever,
        generator: Generator,
        settings: Settings,
        telemetry: Telemetry,
        web_search: WebSearchClient | None = None,
    ) -> None:
        self.chat = chat
        self.retriever = retriever
        self.generator = generator
        self.settings = settings
        self.telemetry = telemetry
        self.web_search = web_search

    def _registry(self, allow_web: bool) -> ToolRegistry:
        registry = ToolRegistry()

        def search(ctx: ToolContext, args: SearchArgs) -> ToolOutput:
            result = self.retriever.retrieve(ctx.scope, args.query, filters=ctx.filters)
            return ToolOutput(chunks=result.chunks, snippets=result.snippets)

        registry.register(
            Tool("search_documents", "Search the tenant's documents", SearchArgs, "read", search)
        )
        if allow_web and self.web_search is not None:
            client = self.web_search

            def web(ctx: ToolContext, args: WebSearchArgs) -> ToolOutput:
                chunks = client.search(args.query)
                return ToolOutput(
                    chunks=chunks,
                    snippets={str(c.chunk_id): c.text for c in chunks},
                    note="external",
                )

            registry.register(
                Tool("web_search", "Search the public web", WebSearchArgs, "read", web)
            )
        return registry

    @tracer.capture_method(capture_response=False)
    def run(
        self,
        *,
        scope: AccessScope,
        question: str,
        filters: SearchFilters | None,
        allow_web: bool,
        request_id: str,
    ) -> AgentRun:
        budget = Budget(
            max_steps=self.settings.agent_max_steps,
            max_tokens=self.settings.agent_max_tokens,
            deadline=time.monotonic() + self.settings.agent_deadline_s,
        )
        registry = self._registry(allow_web)
        ctx = ToolContext(scope=scope, filters=filters, request_id=request_id)
        trace: list[dict[str, Any]] = []
        usage = TokenUsage()
        evidence: dict[str, RetrievedChunk] = {}
        snippets: dict[str, str] = {}

        def record(kind: str, name: str, started: float, **extra: Any) -> None:
            trace.append(
                {
                    "step": len(trace) + 1,
                    "kind": kind,
                    "name": name,
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                    **extra,
                }
            )

        def call_tool(name: str, args: dict[str, Any]) -> None:
            if not budget.allows():
                record("skipped", name, time.perf_counter(), reason="budget")
                return
            started = time.perf_counter()
            budget.spend()
            try:
                output = registry.call(name, args, ctx)
            except (ToolDenied, ToolArgumentsInvalid) as exc:
                record("tool", name, started, error=str(exc))
                return
            except UpstreamUnavailable as exc:
                # Web search is optional evidence; its outage must not fail the answer.
                record("tool", name, started, error=f"unavailable: {exc}"[:200])
                return
            for chunk in output.chunks:
                evidence.setdefault(str(chunk.chunk_id), chunk)
            snippets.update(output.snippets)
            record(
                "tool",
                name,
                started,
                results=len(output.chunks),
                query_chars=len(str(args.get("query", ""))),
            )

        # Step: plan.
        plan = Plan(strategy="direct")
        started = time.perf_counter()
        if budget.allows(1000):
            budget.spend()
            result = self.chat.generate(
                ChatRequest(
                    system=PLAN_SYSTEM,
                    prompt=plan_prompt(question),
                    max_tokens=300,
                    temperature=0.0,
                    purpose="plan",
                    question=question,
                )
            )
            usage = usage + result.usage
            budget.tokens += result.usage.input_tokens + result.usage.output_tokens
            try:
                plan = Plan.model_validate(_json_object(result.text))
            except (ValueError, ValidationError):
                record("plan", "planner", started, error="invalid plan, using direct")
            else:
                record(
                    "plan",
                    "planner",
                    started,
                    strategy=plan.strategy,
                    sub_questions=len(plan.sub_questions),
                )

        queries = plan.sub_questions if plan.strategy == "decompose" and plan.sub_questions else []
        for query in [question, *queries]:
            call_tool("search_documents", {"query": query})

        if allow_web and "web_search" in registry.names() and not evidence:
            call_tool("web_search", {"query": question})

        ranked = sorted(evidence.values(), key=lambda c: -c.score)
        sources = pack_context(ranked, self.settings.context_token_budget)
        answer = self._answer(question, sources, snippets, budget, trace)
        usage = usage + answer.usage

        if answer.status == AnswerStatus.INSUFFICIENT_CONTEXT and budget.allows(4000):
            # One deterministic retry: exact-term matching catches identifiers
            # and codes that the embedding model blurs.
            started = time.perf_counter()
            budget.spend()
            retry = self.retriever.retrieve(
                scope, question, filters=filters, mode=RetrievalMode.KEYWORD
            )
            record("tool", "search_documents_keyword", started, results=len(retry.chunks))
            new = [c for c in retry.chunks if str(c.chunk_id) not in evidence]
            if new:
                snippets.update(retry.snippets)
                sources = pack_context(sources + new, self.settings.context_token_budget)
                second = self._answer(question, sources, snippets, budget, trace)
                usage = usage + second.usage
                answer = second

        stopped = not budget.allows()
        self.telemetry.add("AgentSteps", budget.steps)
        if stopped:
            self.telemetry.add("AgentBudgetExhausted", 1)
        return AgentRun(
            answer=answer.model_copy(update={"usage": usage}),
            sources=sources,
            trace=trace,
            usage=usage,
            stopped_by_budget=stopped,
        )

    def _answer(
        self,
        question: str,
        sources: list[RetrievedChunk],
        snippets: dict[str, str],
        budget: Budget,
        trace: list[dict[str, Any]],
    ) -> Answer:
        started = time.perf_counter()
        if not sources:
            trace.append(
                {
                    "step": len(trace) + 1,
                    "kind": "answer",
                    "name": "generator",
                    "ms": 0.0,
                    "skipped": "no evidence",
                }
            )
            return Answer(
                status=AnswerStatus.INSUFFICIENT_CONTEXT,
                answer=INSUFFICIENT_MESSAGE,
                model_id=self.generator.chat.model_id,
            )
        budget.spend()
        answer = self.generator.answer(question, sources, snippets)
        budget.tokens += answer.usage.input_tokens + answer.usage.output_tokens
        trace.append(
            {
                "step": len(trace) + 1,
                "kind": "answer",
                "name": "generator",
                "ms": round((time.perf_counter() - started) * 1000, 1),
                "status": answer.status.value,
                "citations": len(answer.citations),
            }
        )
        return answer


def _json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("not an object")
    return value
