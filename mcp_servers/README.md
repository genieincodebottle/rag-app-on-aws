### 🔎 SerpApi Web Search (Model Context Protocol) MCP Server

This project implements a streamable HTTP-based MCP server for performing Google web searches using SerpApi. Built using FastMCP, it exposes a web_search tool and a health_check endpoint for integration with LLM agents or custom apps.

### 🚀 Features

- 📡 Real-time Google search via SerpApi

- 🧠 Answer box, knowledge graph & related questions parsing

- ⚡ Fast stateless HTTP streamable transport with JSON responses MCP Server based on FastMCP

- 🩺 Built-in health check endpoint

### 🚀 Installation

```bash
git clone https://github.com/genieincodebottle/rag-app-on-aws.git
cd rag-app-on-aws/mcp_server
pip install uv # If uv doesn't exist in your system
uv venv
.venv\Scripts\activate   # Linux: source .venv/bin/activate
uv pip install -r requirements.txt
```

### 🛠️ Configuration

Create a `.env` file:

```env
SERPAPI_API_KEY=your_serpai_api_key
```
SerpAPI API Key (Free Quota) -> https://serpapi.com/dashboard

### 💡 Usage

Run the following command to start the MCP server on localhost at port 8000 (you can change the port if needed)

```bash
python web_search_mcp_server.py --port 8000
```

Run the following command in Windows PowerShell or Git Bash to expose your local server over the internet. This is necessary because AWS Lambda cannot access localhost. Update the port if you're not using 8000. This will generate a URL required for using the MCP Server-based Web Search in the RAG UI.

```bash
ssh -R 80:localhost:8000 serveo.net
```
