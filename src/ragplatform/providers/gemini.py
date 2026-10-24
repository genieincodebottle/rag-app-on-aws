"""Google Gemini adapters for running the system locally with real models.

Free-tier quota is enough to ingest the evaluation corpus and run the
evaluation set. Not allowed in deployed environments (see ``config.py``):
production traffic stays on Bedrock inside the AWS account.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from ragplatform.domain.errors import UpstreamError
from ragplatform.domain.models import TokenUsage
from ragplatform.providers.base import (
    ChatRequest,
    ChatResult,
    EmbeddingSpace,
    check_vector,
    l2_normalize,
    with_retries,
)

_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return isinstance(code, int) and code in _TRANSIENT_STATUS


class GeminiEmbedder:
    _BATCH = 50

    def __init__(
        self, *, api_key: str, model_id: str, dimensions: int, attempts: int, client: Any = None
    ) -> None:
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self._attempts = attempts
        self.space = EmbeddingSpace("gemini", model_id, dimensions)

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH):
            batch = list(texts[start : start + self._BATCH])
            response = with_retries(
                lambda b=batch: self._client.models.embed_content(  # type: ignore[misc]
                    model=self.space.model_id,
                    contents=b,
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=self.space.dimensions
                    ),
                ),
                attempts=self._attempts,
                is_transient=_is_transient,
                base_delay_s=2.0,
                max_delay_s=30.0,
            )
            # Below the model's full 3072 dimensions the output is not unit
            # length. Normalising keeps cosine and inner product consistent.
            vectors.extend(
                check_vector(l2_normalize(e.values), self.space.dimensions)
                for e in response.embeddings
            )
        if len(vectors) != len(texts):
            raise UpstreamError(f"gemini returned {len(vectors)} embeddings for {len(texts)} texts")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiChatModel:
    def __init__(self, *, api_key: str, model_id: str, attempts: int, client: Any = None) -> None:
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self._attempts = attempts
        self.model_id = model_id

    def generate(self, request: ChatRequest) -> ChatResult:
        from google.genai import types

        started = time.perf_counter()
        response = with_retries(
            lambda: self._client.models.generate_content(
                model=self.model_id,
                contents=request.prompt,
                config=types.GenerateContentConfig(
                    system_instruction=request.system,
                    temperature=request.temperature,
                    # Thinking tokens count against this limit on current Gemini
                    # models, so it is set well above the answer size.
                    max_output_tokens=max(request.max_tokens, 4096),
                    response_mime_type="application/json",
                ),
            ),
            attempts=self._attempts,
            is_transient=_is_transient,
            base_delay_s=2.0,
            max_delay_s=30.0,
        )
        meta = response.usage_metadata
        usage = TokenUsage(
            input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0)
            + int(getattr(meta, "thoughts_token_count", 0) or 0),
        )
        finish = ""
        if response.candidates:
            finish = str(getattr(response.candidates[0], "finish_reason", "") or "")
        return ChatResult(
            text=response.text or "",
            usage=usage,
            model_id=self.model_id,
            stop_reason=finish,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
