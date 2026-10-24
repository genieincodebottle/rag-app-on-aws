"""Provider ports: embedder, chat model, reranker.

Three implementations exist. ``bedrock`` is the only one allowed in a deployed
environment (enforced in ``config.py``). ``gemini`` gives a free-tier way to run
the whole system on a laptop with real models. ``fake`` is deterministic and
offline, for tests and for the evaluation gate in CI.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ragplatform.domain.errors import UpstreamError, UpstreamUnavailable
from ragplatform.domain.models import TokenUsage


@dataclass(frozen=True)
class EmbeddingSpace:
    provider: str
    model_id: str
    dimensions: int


@dataclass(frozen=True)
class SourceBlock:
    """A retrieved source as the model sees it. Test doubles read these directly."""

    marker: str
    title: str
    section: str
    text: str
    flagged: bool = False


@dataclass(frozen=True)
class ChatRequest:
    system: str
    prompt: str
    max_tokens: int
    temperature: float
    purpose: str  # "answer" | "plan" | "verify"; for logs and test doubles
    question: str = ""
    sources: Sequence[SourceBlock] = field(default_factory=tuple)


@dataclass(frozen=True)
class ChatResult:
    text: str
    usage: TokenUsage
    model_id: str
    stop_reason: str = "end_turn"
    latency_ms: float = 0.0


class Embedder(Protocol):
    space: EmbeddingSpace

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class ChatModel(Protocol):
    model_id: str

    def generate(self, request: ChatRequest) -> ChatResult: ...


class Reranker(Protocol):
    model_id: str

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        """Return ``(index into documents, relevance score)`` best first."""
        ...


def check_vector(vector: Sequence[float], dimensions: int) -> list[float]:
    """Reject vectors that would corrupt the index instead of storing them.

    A zero vector has no direction, so cosine distance to it is undefined and
    the chunk can never be ranked correctly. The previous implementation stored
    zero vectors whenever the embedding call failed.
    """
    values = [float(x) for x in vector]
    if len(values) != dimensions:
        raise UpstreamError(f"embedding has {len(values)} dimensions, expected {dimensions}")
    if not all(math.isfinite(x) for x in values):
        raise UpstreamError("embedding contains NaN or infinity")
    norm = math.sqrt(sum(x * x for x in values))
    if norm < 1e-6:
        raise UpstreamError("embedding is a zero vector")
    return values


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def with_retries[T](
    call: Callable[[], T],
    *,
    attempts: int,
    is_transient: Callable[[Exception], bool],
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry transient failures with exponential backoff and full jitter."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                if is_transient(exc):
                    raise UpstreamUnavailable(str(exc)) from exc
                raise
            # Jitter only, not a security use of randomness.
            delay = random.uniform(0, min(max_delay_s, base_delay_s * 2 ** (attempt - 1)))  # noqa: S311
            sleep(delay)
    raise AssertionError("unreachable")
