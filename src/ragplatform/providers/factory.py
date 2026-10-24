"""Build the configured providers once per process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ragplatform.config import Settings
from ragplatform.domain.errors import UpstreamUnavailable
from ragplatform.observability.telemetry import logger
from ragplatform.providers.base import ChatModel, ChatRequest, ChatResult, Embedder, Reranker


class GroundingCheck(Protocol):
    def is_grounded(self, *, sources: str, question: str, answer: str) -> bool: ...


class FallbackChatModel:
    """Try the primary model, then the fallback, on throttling or outage only.

    A validation error (bad request, content policy) is not retried on another
    model: it would fail the same way and double the cost.
    """

    def __init__(self, primary: ChatModel, fallback: ChatModel) -> None:
        self.primary = primary
        self.fallback = fallback
        self.model_id = primary.model_id

    def generate(self, request: ChatRequest) -> ChatResult:
        try:
            return self.primary.generate(request)
        except UpstreamUnavailable as exc:
            logger.warning(
                "primary model unavailable, using fallback",
                extra={
                    "primary": self.primary.model_id,
                    "fallback": self.fallback.model_id,
                    "reason": str(exc)[:200],
                },
            )
            return self.fallback.generate(request)


@dataclass
class Providers:
    embedder: Embedder
    chat: ChatModel
    reranker: Reranker | None
    grounding: GroundingCheck | None


def build_providers(settings: Settings, **clients: Any) -> Providers:
    if settings.provider == "fake":
        from ragplatform.providers.fake import ExtractiveChatModel, HashingEmbedder, OverlapReranker

        return Providers(
            embedder=HashingEmbedder(settings.embedding_dimensions),
            chat=ExtractiveChatModel(),
            reranker=OverlapReranker() if settings.rerank_model_id else None,
            grounding=None,
        )

    if settings.provider == "gemini":
        from ragplatform.providers.gemini import GeminiChatModel, GeminiEmbedder

        if settings.gemini_api_key is None:
            raise ValueError("RAG_PROVIDER=gemini needs RAG_GEMINI_API_KEY")
        key = settings.gemini_api_key.get_secret_value()
        gemini_chat: ChatModel = GeminiChatModel(
            api_key=key, model_id=settings.chat_model_id, attempts=settings.model_max_attempts
        )
        if settings.fallback_chat_model_id:
            gemini_chat = FallbackChatModel(
                gemini_chat,
                GeminiChatModel(
                    api_key=key,
                    model_id=settings.fallback_chat_model_id,
                    attempts=settings.model_max_attempts,
                ),
            )
        return Providers(
            embedder=GeminiEmbedder(
                api_key=key,
                model_id=settings.embedding_model_id,
                dimensions=settings.embedding_dimensions,
                attempts=settings.model_max_attempts,
            ),
            chat=gemini_chat,
            reranker=None,
            grounding=None,
        )

    from ragplatform.providers.bedrock import (
        BedrockReranker,
        ConverseChatModel,
        GuardrailGroundingCheck,
        TitanEmbedder,
    )

    common: dict[str, Any] = {
        "region": settings.aws_region,
        "timeout_s": settings.model_timeout_s,
        "attempts": settings.model_max_attempts,
    }
    chat: ChatModel = ConverseChatModel(
        model_id=settings.chat_model_id,
        guardrail_id=settings.guardrail_id,
        guardrail_version=settings.guardrail_version,
        client=clients.get("runtime"),
        **common,
    )
    if settings.fallback_chat_model_id:
        chat = FallbackChatModel(
            chat,
            ConverseChatModel(
                model_id=settings.fallback_chat_model_id,
                guardrail_id=settings.guardrail_id,
                guardrail_version=settings.guardrail_version,
                client=clients.get("runtime"),
                **common,
            ),
        )
    return Providers(
        embedder=TitanEmbedder(
            model_id=settings.embedding_model_id,
            dimensions=settings.embedding_dimensions,
            concurrency=settings.embed_concurrency,
            client=clients.get("runtime"),
            **common,
        ),
        chat=chat,
        reranker=BedrockReranker(
            model_id=settings.rerank_model_id, client=clients.get("agent_runtime"), **common
        )
        if settings.rerank_model_id
        else None,
        grounding=GuardrailGroundingCheck(
            guardrail_id=settings.guardrail_id,
            guardrail_version=settings.guardrail_version,
            client=clients.get("runtime"),
            **common,
        )
        if settings.guardrail_id and settings.guardrail_grounding_check
        else None,
    )
