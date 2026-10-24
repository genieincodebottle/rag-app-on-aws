"""The web-search MCP server, driven end to end by the platform's own client.

The server runs for real on a local port; only SerpApi is replaced. This
checks that the platform's stateless ``tools/call`` request is accepted by the
MCP SDK version in uv.lock and that the structured results come back intact.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from ragplatform.agent.web_search import WebSearchClient
from ragplatform.domain.errors import UpstreamError

SERVER_PATH = Path(__file__).resolve().parents[2] / "mcp_servers" / "web_search_mcp_server.py"


@pytest.fixture(scope="module")
def server_module() -> Any:
    spec = importlib.util.spec_from_file_location("web_search_mcp_server", SERVER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["web_search_mcp_server"] = module
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def running_server(server_module: Any) -> Any:
    import uvicorn

    def serpapi(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "serp-test-key"
        return httpx.Response(
            200,
            json={
                "organic_results": [
                    {
                        "title": "AWS Lambda quotas",
                        "link": "https://docs.aws.amazon.com/lambda/quotas",
                        "snippet": "Function timeout: 900 seconds.",
                    },
                    {"title": "Insecure mirror", "link": "http://mirror.example/x", "snippet": "x"},
                ]
            },
        )

    server_module.transport = httpx.MockTransport(serpapi)
    import os

    os.environ["SERPAPI_API_KEY"] = "serp-test-key"
    port = _free_port()
    config = uvicorn.Config(
        server_module.build_app("mcp-test-key"), host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}{server_module.MCP_PATH}"
    server.should_exit = True
    thread.join(timeout=5)


def test_platform_client_gets_structured_results(running_server: str) -> None:
    client = WebSearchClient(running_server, api_key="mcp-test-key", timeout_s=5)
    # The config validator only allows http:// for localhost; this test calls
    # the client directly, which mirrors a local development setup.
    chunks = client.search("lambda timeout limit", max_results=5)
    assert [c.title for c in chunks] == ["Web: AWS Lambda quotas"]
    assert chunks[0].text == "Function timeout: 900 seconds."
    assert chunks[0].risk_flags == ["external_web"]


def test_wrong_key_is_rejected(running_server: str) -> None:
    client = WebSearchClient(running_server, api_key="wrong", timeout_s=5)
    with pytest.raises(UpstreamError, match="401"):
        client.search("lambda timeout limit")


def test_health_needs_no_key(running_server: str) -> None:
    base = running_server.rsplit("/", 1)[0]
    assert httpx.get(f"{base}/health", timeout=5).json() == {"status": "ok"}


def test_short_query_is_an_error_result(running_server: str) -> None:
    client = WebSearchClient(running_server, api_key="mcp-test-key", timeout_s=5)
    with pytest.raises(UpstreamError):
        client.search("ab")
