# Web search MCP server

An optional Model Context Protocol server that gives the platform's agentic mode a `web_search` tool, backed by SerpApi. It uses the streamable HTTP transport in stateless JSON mode, so any instance can serve any call and responses are single JSON bodies.

The tool returns structured results:

```json
{"query": "lambda timeout limit", "results": [{"title": "...", "link": "https://...", "snippet": "..."}]}
```

## Run

```bash
cd mcp_servers
uv venv && uv pip install -r requirements.txt     # or: pip install -r requirements.txt
cp .env.example .env                             # set SERPAPI_API_KEY and MCP_API_KEY
python web_search_mcp_server.py --host 127.0.0.1 --port 8765
```

Requires the MCP Python SDK 2.x (the server class is `MCPServer`; in 1.x it was `FastMCP`).

| Setting | Purpose |
|---|---|
| `SERPAPI_API_KEY` | SerpApi key, from https://serpapi.com/dashboard |
| `MCP_API_KEY` | shared secret; callers send `Authorization: Bearer <key>`. Required unless you pass `--allow-unauthenticated` with `--host 127.0.0.1` for a local experiment |

`GET /health` answers without a key.

## Connect the platform

In the platform's `.env`:

```
RAG_WEB_SEARCH_URL=http://localhost:8765/mcp
RAG_WEB_SEARCH_API_KEY=<the same MCP_API_KEY>
```

and enable web search for a tenant: `uv run ragplatform tenant acme "Acme Corp" --agentic --web`. The agent calls the tool only in agentic mode and only when document search found nothing. Results enter the prompt flagged as external, untrusted text.

Deployed environments accept only an `https://` URL, so host the server behind TLS (a container service, or a tunnel for a short test). The URL is server configuration; it is never taken from a request.

## Safety

- SerpApi is called with `safe=active`, a 10 s timeout, and the key never logged (it is in the query string).
- Only `https://` result links are returned; snippets are trimmed to 600 characters and at most 10 results.
- Request bodies over 64 KB are rejected.
- Tool failures are returned as MCP tool errors (`isError: true`), which the platform's client raises rather than treating as "no results".

`tests/unit/test_mcp_server.py` starts this server on a local port with SerpApi mocked and drives it with the platform's client.
