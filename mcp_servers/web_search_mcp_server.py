"""
SerpApi based Web Search MCP Server with HTTP Streamable Transport
Configured for Stateless Http Requests
"""
import os
import logging
import asyncio
import argparse
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP
import uvicorn
from dotenv import load_dotenv
from serpapi.google_search import GoogleSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize FastMCP server with Streamable HTTP transport
mcp = FastMCP(
    name="SerpApi Search Server",
    instructions="Web search server using SerpApi",
    # If needs to return JSON responses
    json_response=True,
    # Stateless when each request is independent
    stateless_http=True,
    # Production settings
    warn_on_duplicate_tools=True,
    warn_on_duplicate_resources=True,
    warn_on_duplicate_prompts=True,
    debug=True  # Disable debug mode for production
)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    logger.warning("SERPAPI_API_KEY environment variable not set. Server will not function properly.")

class SerpApiClient:
    """Client for interacting with SerpApi"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def search(self, query: str, num_results: int = 10, location: str = None) -> Dict[str, Any]:
        """Perform a search using SerpApi"""
        params = {
            "q": query,
            "api_key": self.api_key,
            "hl": "en",
            "gl": "us",
            "google_domain": "google.com",
            "device": "desktop",
            "safe": "active",
            "num": min(num_results, 100),
            "output": "json"
        }
        
        if location:
            params["location"] = location
        
        try:
            loop = asyncio.get_event_loop()
            search = GoogleSearch(params)
            results = await loop.run_in_executor(None, search.get_dict)
            return results
        except Exception as e:
            logger.error(f"SerpApi search error: {e}")
            raise

# Initialize SerpApi client
serpapi_client = SerpApiClient(SERPAPI_API_KEY) if SERPAPI_API_KEY else None

@mcp.tool(
    name="web_search",
    description="Search the web using SerpApi with support for localized results."
)
async def web_search(query: str, num_results: int = 10, location: str = None) -> str:
    """
    Search the web using SerpApi
    
    Args:
        query: The search query (required)
        num_results: Number of results to return (default: 10, max: 100)
        location: Optional location for localized results (e.g., "New York, NY", "London, UK")
    
    Returns:
        Formatted search results optimized for Lambda client consumption
    """
    if not serpapi_client:
        return "ERROR: SERPAPI_API_KEY not configured. Please set the environment variable."
    
    if not query or not query.strip():
        return "ERROR: Search query cannot be empty."
    
    # Validate num_results
    if not isinstance(num_results, int) or num_results < 1 or num_results > 100:
        num_results = 10  # Default fallback
        logger.warning(f"Invalid num_results provided, using default: {num_results}")
    
    try:
        logger.info(f"Lambda client search request: '{query}' ({num_results} results, location: {location})")
        results = await serpapi_client.search(query, num_results, location)
        
        # Format results for optimal Lambda consumption (structured but readable)
        formatted_results = []
        
        # Header with search info
        formatted_results.append(f"SEARCH RESULTS FOR: {query}")
        if location:
            formatted_results.append(f"LOCATION: {location}")
        
        search_metadata = results.get("search_information", {})
        if search_metadata.get("total_results"):
            formatted_results.append(f"TOTAL RESULTS: {search_metadata['total_results']:,}")
        
        formatted_results.append("-" * 60)
        
        # Organic search results
        organic_results = results.get("organic_results", [])
        if organic_results:
            formatted_results.append("\nORGANIC RESULTS:")
            for i, result in enumerate(organic_results[:num_results], 1):
                title = result.get("title", "No title").strip()
                link = result.get("link", "No link").strip()
                snippet = result.get("snippet", "No description").strip()
                
                formatted_results.append(f"\n{i}. {title}")
                formatted_results.append(f"   URL: {link}")
                formatted_results.append(f"   Summary: {snippet}")
                
                # Add displayed link if different from actual link
                if result.get("displayed_link"):
                    formatted_results.append(f"   Display URL: {result['displayed_link']}")
        
        # Answer box (featured snippet)
        if "answer_box" in results:
            answer_box = results["answer_box"]
            formatted_results.append(f"\n\nFEATURED ANSWER:")
            
            if answer_box.get("title"):
                formatted_results.append(f"Title: {answer_box['title']}")
            
            if answer_box.get("answer"):
                formatted_results.append(f"Answer: {answer_box['answer']}")
            elif answer_box.get("snippet"):
                formatted_results.append(f"Snippet: {answer_box['snippet']}")
            
            if answer_box.get("link"):
                formatted_results.append(f"Source: {answer_box['link']}")
        
        # Knowledge graph
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            formatted_results.append(f"\n\nKNOWLEDGE PANEL:")
            
            if kg.get("title"):
                formatted_results.append(f"Entity: {kg['title']}")
            if kg.get("type"):
                formatted_results.append(f"Type: {kg['type']}")
            if kg.get("description"):
                formatted_results.append(f"Description: {kg['description']}")
            if kg.get("source", {}).get("name"):
                formatted_results.append(f"Source: {kg['source']['name']}")
        
        # Related questions (People Also Ask)
        related_questions = results.get("related_questions", [])
        if related_questions:
            formatted_results.append(f"\n\nRELATED QUESTIONS:")
            for i, question in enumerate(related_questions[:3], 1):  # Limit to 3 for Lambda
                if question.get("question"):
                    formatted_results.append(f"{i}. {question['question']}")
                    if question.get("snippet"):
                        formatted_results.append(f"   Answer: {question['snippet']}")
        
        # News results if available
        if "news_results" in results:
            news = results["news_results"][:3]  # Limit news results
            if news:
                formatted_results.append(f"\n\nNEWS:")
                for i, article in enumerate(news, 1):
                    formatted_results.append(f"{i}. {article.get('title', 'No title')}")
                    if article.get("date"):
                        formatted_results.append(f"   Date: {article['date']}")
                    if article.get("source"):
                        formatted_results.append(f"   Source: {article['source']}")
        
        # Add search timing for Lambda monitoring
        if search_metadata.get("time_taken_displayed"):
            formatted_results.append(f"\n\nSearch completed in: {search_metadata['time_taken_displayed']}")
        
        result_text = "\n".join(formatted_results)
        logger.info(f"Search completed for Lambda client. Returned {len(organic_results)} organic results.")
        return result_text
        
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        logger.error(f"Lambda client search failed: {error_msg}")
        return f"ERROR: {error_msg}"

@mcp.tool(
    name="health_check",
    description="Health check endpoint for monitoring the MCP server"
)
async def health_check() -> str:
    """
    Health check for client monitoring
    
    Returns:
        Server status information
    """
    import time
    
    status_items = [
        "STATUS: HEALTHY",
        f"TIMESTAMP: {int(time.time())}",
        f"SERVER: SerpApi MCP Server",
        f"TRANSPORT: Streamable HTTP (Stateless)",
        f"CONFIGURATION:",
        f"  - JSON Response: {mcp.settings.json_response}",
        f"  - Stateless: {mcp.settings.stateless_http}",
        f"API_KEY: {'CONFIGURED' if SERPAPI_API_KEY else 'MISSING'}",
        f"TOOLS: web_search, health_check",
    ]
    
    return "\n".join(status_items)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCP SerpApi server optimized for AWS Lambda clients")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to (0.0.0.0 for production)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    
    # Update configuration
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.settings.log_level = args.log_level
    
    print("SerpApi MCP Server based on Streamable HTTP Transport")
    print(f"MCP Endpoint: http://{args.host}:{args.port}{mcp.settings.streamable_http_path}")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print(f"Server Info: http://{args.host}:{args.port}/info")
    print("Transport: streamable_http")
    print("Mode: stateless (no session persistence needed)")
    
    try:
        uvicorn.run(
            mcp.streamable_http_app(),
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            access_log=True,
            # Production optimizations
            workers=1,  # Single worker for MCP
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        logger.error(f"Server startup failed: {e}")