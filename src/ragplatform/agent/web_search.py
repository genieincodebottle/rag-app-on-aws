"""Client for the optional web-search MCP server (``mcp_servers/``).

The endpoint comes only from ``RAG_WEB_SEARCH_URL``. The previous
implementation took the URL from the request body, which let any caller make
the Lambda send requests to an address of their choosing.

Results are external, unvetted text. They enter the prompt as flagged sources
and are never cached across tenants.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from ragplatform.domain.errors import UpstreamError, UpstreamUnavailable
from ragplatform.domain.models import RetrievedChunk

_NAMESPACE = uuid.UUID("6f3f2a76-1d65-4b8e-9d6c-0d3c9e2a51f7")
_MAX_RESPONSE_BYTES = 256 * 1024


class WebSearchClient:
    def __init__(
        self,
        url: str,
        *,
        api_key: str | None,
        timeout_s: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            timeout=timeout_s, headers=headers, transport=transport, follow_redirects=False
        )

    def search(self, query: str, max_results: int = 5) -> list[RetrievedChunk]:
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "web_search",
                "arguments": {"query": query, "num_results": max_results},
            },
        }
        try:
            response = self._client.post(self.url, json=request)
        except httpx.TimeoutException as exc:
            raise UpstreamUnavailable("web search timed out") from exc
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable(f"web search unreachable: {type(exc).__name__}") from exc
        if response.status_code >= 500 or response.status_code == 429:
            raise UpstreamUnavailable(f"web search returned {response.status_code}")
        if not 200 <= response.status_code < 300:
            # Includes redirects: following one could send the request to an
            # internal address, so a redirect is treated as an error.
            raise UpstreamError(f"web search returned {response.status_code}")
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise UpstreamError("web search response too large")
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError("web search returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise UpstreamError("web search returned an unexpected payload")
        return self._to_chunks(payload, max_results)

    @staticmethod
    def _to_chunks(payload: dict[str, Any], max_results: int) -> list[RetrievedChunk]:
        if "error" in payload:
            raise UpstreamError(f"web search error: {payload['error']}")
        result = payload.get("result", {})
        # A tool failure comes back as a normal result with isError set, not as
        # a JSON-RPC error. Reading it as "no results" would hide the failure.
        if isinstance(result, dict) and result.get("isError"):
            text = " ".join(
                str(b.get("text", "")) for b in result.get("content", []) if isinstance(b, dict)
            )
            raise UpstreamError(f"web search tool error: {text[:200]}")
        items: list[dict[str, Any]] = []
        if isinstance(result, dict):
            structured = result.get("structuredContent", {})
            if isinstance(structured, dict) and isinstance(structured.get("results"), list):
                items = structured["results"]
            else:
                for block in result.get("content", []):
                    if block.get("type") == "text":
                        try:
                            parsed = json.loads(block["text"])
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                            items = parsed["results"]
        chunks = []
        for item in items[:max_results]:
            link = str(item.get("link") or item.get("url") or "")
            snippet = " ".join(str(item.get("snippet") or "").split())[:1200]
            if not link.startswith("https://") or not snippet:
                continue
            doc_id = uuid.uuid5(_NAMESPACE, link)
            chunks.append(
                RetrievedChunk(
                    chunk_id=doc_id,
                    document_id=doc_id,
                    version=0,
                    title=f"Web: {str(item.get('title') or link)[:200]}",
                    section_path=link,
                    page_start=None,
                    page_end=None,
                    text=snippet,
                    score=0.0,
                    risk_flags=["external_web"],
                )
            )
        return chunks
