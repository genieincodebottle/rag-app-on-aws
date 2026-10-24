"""Web search MCP server (SerpApi), used by the platform's optional web_search tool.

Streamable HTTP transport, stateless, JSON responses. One tool, ``web_search``,
returns structured results ({"results": [{"title", "link", "snippet"}]}) so
the client never has to parse formatted text.

    python web_search_mcp_server.py --host 127.0.0.1 --port 8765

Settings (environment or .env):
    SERPAPI_API_KEY   SerpApi key (https://serpapi.com)
    MCP_API_KEY       shared secret callers must send as "Authorization: Bearer <key>".
                      Required unless --allow-unauthenticated is passed for a
                      local experiment bound to 127.0.0.1.

The platform calls this server only at the URL in RAG_WEB_SEARCH_URL, which is
server configuration. Results are treated as untrusted text by the caller.
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Only this server's own .env, never one from the working directory.
load_dotenv(Path(__file__).with_name(".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("web-search-mcp")

SERPAPI_URL = "https://serpapi.com/search.json"
MAX_RESULTS = 10
MAX_SNIPPET_CHARS = 600


class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str


class SearchResults(BaseModel):
    query: str
    results: list[SearchResult] = Field(default_factory=list)


MCP_PATH = "/mcp"

# MCP Python SDK 2.x (the class was called FastMCP in 1.x).
mcp = MCPServer(
    name="web-search",
    instructions="Web search through SerpApi. Returns titles, links and snippets.",
)

# Replaced in tests with an httpx.MockTransport.
transport: httpx.BaseTransport | None = None


def _clean(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def search_serpapi(query: str, num_results: int, api_key: str) -> SearchResults:
    params = {
        "q": query,
        "num": num_results,
        "api_key": api_key,
        "hl": "en",
        "safe": "active",
        "engine": "google",
    }
    with httpx.Client(timeout=10.0, transport=transport) as client:
        response = client.get(SERPAPI_URL, params=params)
    if response.status_code != 200:
        # The key is in the query string; never log the URL or the response body.
        raise RuntimeError(f"SerpApi returned HTTP {response.status_code}")
    payload = response.json()
    results = []
    for item in payload.get("organic_results", [])[:num_results]:
        link = str(item.get("link") or "")
        if not link.startswith("https://"):
            continue
        results.append(
            SearchResult(
                title=_clean(item.get("title"), 200),
                link=link,
                snippet=_clean(item.get("snippet"), MAX_SNIPPET_CHARS),
            )
        )
    return SearchResults(query=query, results=results)


@mcp.tool(name="web_search", description="Search the public web. Returns up to 10 results.")
def web_search(query: str, num_results: int = 5) -> SearchResults:
    query = _clean(query, 300)
    if len(query) < 3:
        raise ValueError("query must be at least 3 characters")
    num_results = max(1, min(int(num_results), MAX_RESULTS))
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not configured on the server")
    logger.info("web_search chars=%d results=%d", len(query), num_results)
    return search_serpapi(query, num_results, api_key)


class BearerAuth(BaseHTTPMiddleware):
    def __init__(self, app: Any, key: str | None) -> None:
        super().__init__(app)
        self.key = key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path == "/health":
            return JSONResponse({"status": "ok"})
        if self.key is not None:
            supplied = request.headers.get("authorization", "")
            expected = f"Bearer {self.key}"
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def build_app(api_key: str | None, host: str = "127.0.0.1") -> Any:
    app = mcp.streamable_http_app(
        streamable_http_path=MCP_PATH,
        json_response=True,  # one JSON body per call, no event stream
        stateless_http=True,  # no session state, so any instance can serve any call
        max_request_body_size=64 * 1024,
        host=host,
    )
    app.add_middleware(BearerAuth, key=api_key)
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Web search MCP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="local experiments only; refused unless bound to 127.0.0.1",
    )
    args = parser.parse_args()

    key = os.environ.get("MCP_API_KEY")
    if key is None:
        if not args.allow_unauthenticated or args.host != "127.0.0.1":
            raise SystemExit(
                "Set MCP_API_KEY, or pass --allow-unauthenticated with --host 127.0.0.1"
            )
        logger.warning("running WITHOUT authentication on 127.0.0.1")
    print(f"MCP endpoint: http://{args.host}:{args.port}{MCP_PATH}")
    uvicorn.run(build_app(key, args.host), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
