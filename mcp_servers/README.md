## 🧠🔗 Remote MCP Server based on Streaming Http Transport protocol for Real Time Web Search

Connect your LLM powered app with real-time Google search results using this streamable Http Transport based Model Context Protocol (MCP) Server.

Built with ⚡FastMCP and 🔍SerpApi - perfect for RAG pipelines, agent tools or experimental AI workflows.

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
python web_search_mcp_server.py --host localhost --port 8000
```

To expose your local server to the internet (required because AWS Lambda cannot access localhost), choose one of the following methods. Be sure to update the port if you're not using 8000.

✅ Option 1 (Recommended): Use Cloudflare Tunnel (Free without login)
Run the following commands in Windows PowerShell to start a secure tunnel and get a public URL. This URL allows external access to your local MCP server in the RAG UI for testing purposes.

  ```bash
  iwr -useb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile cloudflared.exe
  
  cloudflared tunnel --url http://localhost:8000
  ```

✅ Option 2: Use Serveo (Quick SSH Tunnel)
Run this command in Windows PowerShell or Git Bash to open an SSH tunnel and expose your local server for testing purpose:

 ```bash
 ssh -R 80:localhost:8000 serveo.net
 ```
