"""Amazon Bedrock adapters: Titan Text Embeddings V2, Converse, Rerank, Guardrails.

Model ids and regional availability (checked 2026-10-23 against the Bedrock
user guide; re-check before changing defaults):

* ``amazon.titan-embed-text-v2:0``: 1024, 512 or 256 dimensions, up to 8,192
  tokens or 50,000 characters per input, one text per InvokeModel call.
* ``us.amazon.nova-2-lite-v1:0``: Nova 2 Lite through the US geo inference
  profile. Not available in-region in us-east-1, so the profile id is required.
* ``us.anthropic.claude-haiku-5-5``: Claude Haiku 5.5, geo profile only in us-east-1.
* ``cohere.rerank-v3-5:0``: Cohere Rerank 3.5, available in us-east-1. Amazon
  Rerank 1.0 is not offered in us-east-1.

These adapters are covered by contract tests (botocore Stubber) that pin the
request and response shapes. They have not been run against a live account by
this repository's test suite.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ragplatform.domain.errors import GuardrailBlocked, UpstreamError, UpstreamUnavailable
from ragplatform.domain.models import TokenUsage
from ragplatform.providers.base import ChatRequest, ChatResult, EmbeddingSpace, check_vector

_TRANSIENT_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "InternalServerException",
    "ModelTimeoutException",
    "TooManyRequestsException",
}


def _client(service: str, region: str, timeout_s: float, attempts: int, client: Any = None) -> Any:
    if client is not None:
        return client
    import boto3
    from botocore.config import Config

    # "adaptive" retries add client-side rate limiting on top of exponential
    # backoff, which is what you want against per-model RPM quotas.
    return boto3.client(
        service,
        region_name=region,
        config=Config(
            connect_timeout=5,
            read_timeout=timeout_s,
            retries={"mode": "adaptive", "max_attempts": attempts},
        ),
    )


def _translate(exc: Exception, what: str) -> Exception:
    from botocore.exceptions import (
        ClientError,
        ConnectTimeoutError,
        EndpointConnectionError,
        ReadTimeoutError,
    )

    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        message = exc.response.get("Error", {}).get("Message", "")
        if code in _TRANSIENT_CODES:
            return UpstreamUnavailable(f"{what}: {code}: {message}")
        return UpstreamError(f"{what}: {code}: {message}")
    if isinstance(exc, ReadTimeoutError | ConnectTimeoutError | EndpointConnectionError):
        return UpstreamUnavailable(f"{what}: {exc}")
    return exc


class TitanEmbedder:
    def __init__(
        self,
        *,
        model_id: str,
        dimensions: int,
        region: str,
        timeout_s: float,
        attempts: int,
        concurrency: int,
        client: Any = None,
    ) -> None:
        if dimensions not in (256, 512, 1024):
            raise ValueError("Titan Text Embeddings V2 supports 256, 512 or 1024 dimensions")
        self.space = EmbeddingSpace("bedrock", model_id, dimensions)
        self._client = _client("bedrock-runtime", region, timeout_s, attempts, client)
        self._concurrency = concurrency
        self.input_tokens = 0

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps(
            {"inputText": text, "dimensions": self.space.dimensions, "normalize": True}
        )
        try:
            response = self._client.invoke_model(
                modelId=self.space.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
        except Exception as exc:
            raise _translate(exc, "titan embed") from exc
        payload = json.loads(response["body"].read())
        self.input_tokens += int(payload.get("inputTextTokenCount", 0))
        return check_vector(payload["embedding"], self.space.dimensions)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) <= 1 or self._concurrency <= 1:
            return [self._embed_one(t) for t in texts]
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            return list(pool.map(self._embed_one, texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class ConverseChatModel:
    def __init__(
        self,
        *,
        model_id: str,
        region: str,
        timeout_s: float,
        attempts: int,
        guardrail_id: str | None = None,
        guardrail_version: str = "DRAFT",
        client: Any = None,
    ) -> None:
        self.model_id = model_id
        self._client = _client("bedrock-runtime", region, timeout_s, attempts, client)
        self._guardrail = (
            {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version,
                "trace": "disabled",
            }
            if guardrail_id
            else None
        )

    def generate(self, request: ChatRequest) -> ChatResult:
        content: list[dict[str, Any]] = []
        if self._guardrail and request.question:
            # With a guardContent block present, Guardrails evaluates only that
            # block: the user's question. Retrieved sources are not screened by
            # the prompt-attack filter, which would flag legitimate documents.
            content.append({"guardContent": {"text": {"text": request.question}}})
        content.append({"text": request.prompt})
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": request.system}],
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {
                "maxTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if self._guardrail:
            kwargs["guardrailConfig"] = self._guardrail
        started = time.perf_counter()
        try:
            response = self._client.converse(**kwargs)
        except Exception as exc:
            raise _translate(exc, f"converse {self.model_id}") from exc
        stop_reason = response.get("stopReason", "")
        if stop_reason == "guardrail_intervened":
            raise GuardrailBlocked("bedrock guardrail intervened")
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in blocks)
        usage = response.get("usage", {})
        return ChatResult(
            text=text,
            usage=TokenUsage(
                input_tokens=int(usage.get("inputTokens", 0)),
                output_tokens=int(usage.get("outputTokens", 0)),
            ),
            model_id=self.model_id,
            stop_reason=stop_reason,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class BedrockReranker:
    def __init__(
        self, *, model_id: str, region: str, timeout_s: float, attempts: int, client: Any = None
    ) -> None:
        self.model_id = model_id
        self._model_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
        self._client = _client("bedrock-agent-runtime", region, timeout_s, attempts, client)

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        try:
            response = self._client.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": query}}],
                sources=[
                    {
                        "type": "INLINE",
                        "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": d}},
                    }
                    for d in documents
                ],
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "numberOfResults": min(top_n, len(documents)),
                        "modelConfiguration": {"modelArn": self._model_arn},
                    },
                },
            )
        except Exception as exc:
            raise _translate(exc, "rerank") from exc
        return [(int(r["index"]), float(r["relevanceScore"])) for r in response["results"]]


class GuardrailGroundingCheck:
    """Contextual grounding check through ApplyGuardrail, run on the final answer.

    Limits from the Bedrock docs: 100,000 characters of source, 1,000 of query,
    5,000 of answer. Inputs are truncated to fit; a truncated source makes the
    check stricter, never looser.
    """

    def __init__(
        self,
        *,
        guardrail_id: str,
        guardrail_version: str,
        region: str,
        timeout_s: float,
        attempts: int,
        client: Any = None,
    ) -> None:
        self._id = guardrail_id
        self._version = guardrail_version
        self._client = _client("bedrock-runtime", region, timeout_s, attempts, client)

    def is_grounded(self, *, sources: str, question: str, answer: str) -> bool:
        try:
            response = self._client.apply_guardrail(
                guardrailIdentifier=self._id,
                guardrailVersion=self._version,
                source="OUTPUT",
                content=[
                    {"text": {"text": sources[:100_000], "qualifiers": ["grounding_source"]}},
                    {"text": {"text": question[:1_000], "qualifiers": ["query"]}},
                    {"text": {"text": answer[:5_000], "qualifiers": ["guard_content"]}},
                ],
            )
        except Exception as exc:
            raise _translate(exc, "apply_guardrail") from exc
        return bool(response.get("action") != "GUARDRAIL_INTERVENED")
