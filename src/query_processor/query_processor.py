"""
Lambda function with LangGraph-based Agentic RAG
Combines traditional RAG, web search using MCP Server, and intelligent decision-making
"""
import os
import json
import boto3
import logging
import psycopg2
import asyncio
import httpx
import time
import traceback
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from decimal import Decimal
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated

# Logger setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')
secretsmanager = boto3.client('secretsmanager')

# Environment variables
DOCUMENTS_BUCKET = os.environ.get('DOCUMENTS_BUCKET')
METADATA_TABLE = os.environ.get('METADATA_TABLE')
STAGE = os.environ.get('STAGE')
DB_SECRET_ARN = os.environ.get('DB_SECRET_ARN')
GEMINI_SECRET_ARN = os.environ.get('GEMINI_SECRET_ARN')
GEMINI_EMBEDDING_MODEL = os.environ.get('GEMINI_EMBEDDING_MODEL')
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.7'))
MAX_OUTPUT_TOKENS = int(os.environ.get('MAX_OUTPUT_TOKENS', '1000'))
TOP_K = int(os.environ.get('TOP_K', '40'))
TOP_P = float(os.environ.get('TOP_P', '0.95'))
ENABLE_EVALUATION = os.environ.get('ENABLE_EVALUATION', 'true').lower() == 'true'
GEMINI_MODEL = "gemini-2.0-flash"

# MCP Configuration
MCP_TIMEOUT = int(os.environ.get('MCP_TIMEOUT', '60'))
RAG_CONFIDENCE_THRESHOLD = float(os.environ.get('RAG_CONFIDENCE_THRESHOLD', '0.7'))
MIN_CONTEXT_LENGTH = int(os.environ.get('MIN_CONTEXT_LENGTH', '100'))

# Get Gemini API key from Secrets Manager
def get_gemini_api_key():
    try:
        response = secretsmanager.get_secret_value(SecretId=GEMINI_SECRET_ARN)
        return json.loads(response['SecretString'])['GEMINI_API_KEY']
    except Exception as e:
        logger.error(f"Error fetching Gemini API key: {str(e)}")
        raise

# Initialize Gemini clients
try:
    GEMINI_API_KEY = get_gemini_api_key()
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        top_k=TOP_K,
        top_p=TOP_P
    )
except Exception as e:
    logger.error(f"Error configuring Gemini API client: {str(e)}")
    raise

# Custom JSON encoder for Decimal types
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

# State definition for LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    user_id: str
    traditional_rag_results: List[Dict[str, Any]]
    web_search_results: Optional[str]
    rag_assessment: Dict[str, Any]
    search_strategy: str  # "traditional_only", "web_only", "hybrid", "iterative"
    iteration_count: int
    final_response: str
    confidence_score: float
    context_sources: List[str]
    reasoning_steps: List[str]
    evaluation_results: Dict[str, float]

# Stateless MCP Client for making JSON-RPC requests
class StatelessMCPClient:
    def __init__(self, mcp_url: str, timeout: float = 30.0, headers: Optional[Dict[str, str]] = None):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(headers or {})
        }
        self._request_id_counter = 0
    
    def _generate_request_id(self) -> str:
        self._request_id_counter += 1
        return f"req_{self._request_id_counter}_{int(time.time() * 1000)}"
    
    async def _make_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        request_id = self._generate_request_id()
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as http_client:
                response = await http_client.post(
                    self.mcp_url,
                    json=jsonrpc_request,
                    headers=self.headers
                )
                response.raise_for_status()
                response_data = response.json()
                return {"success": True, "data": response_data, "error": None}
        except Exception as e:
            logger.error(f"MCP request error: {e}")
            return {"success": False, "data": None, "error": str(e)}
    
    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._make_request(
            method="tools/call",
            params={"name": name, "arguments": arguments or {}}
        )
    
    async def search_with_mcp(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        try:
            result = await self.call_tool("web_search", {
                "query": query,
                "num_results": max_results
            })
            
            if not result["success"]:
                return {"success": False, "error": result["error"], "data": None}
            
            search_data = self._extract_tool_result(result["data"])
            return {"success": True, "data": search_data, "source": "mcp_web_search"}
        except Exception as e:
            logger.error(f"MCP search failed: {str(e)}")
            return {"success": False, "error": str(e), "data": None}
    
    def _extract_tool_result(self, response_data) -> Any:
        try:
            if isinstance(response_data, dict):
                if "result" in response_data:
                    result = response_data["result"]
                    if isinstance(result, str):
                        return result
                    elif isinstance(result, dict):
                        return str(result.get('content', result.get('text', str(result))))
                    else:
                        return str(result)
                elif "error" in response_data:
                    error = response_data["error"]
                    if isinstance(error, dict):
                        return f"MCP Error: {error.get('message', str(error))}"
                    else:
                        return f"MCP Error: {str(error)}"
        except Exception as e:
            return f"Error extracting result: {str(e)}"

# Function to embed query text using Gemini API
def embed_query(text: str) -> List[float]:
    try:
        result = genai_client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        return list(result.embeddings[0].values)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return [0.0] * 768

# Function to get Postgres credentials from Secrets Manager
def get_postgres_credentials():
    try:
        response = secretsmanager.get_secret_value(SecretId=DB_SECRET_ARN)
        return json.loads(response['SecretString'])
    except Exception as e:
        logger.error(f"Error fetching DB credentials: {str(e)}")
        raise

# Function to establish a Postgres connection using credentials
def get_postgres_connection(creds):
    return psycopg2.connect(
        host=creds['host'],
        port=creds['port'],
        user=creds['username'],
        password=creds['password'],
        dbname=creds['dbname']
    )

# Function to perform similarity search in Postgres
def similarity_search(query_embedding: List[float], user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    credentials = get_postgres_credentials()
    conn = get_postgres_connection(credentials)

    try:
        cursor = conn.cursor()
        vector_str = '[' + ','.join([str(x) for x in query_embedding]) + ']'

        cursor.execute(f"""
            SELECT 
                c.chunk_id, c.document_id, c.user_id, c.content, c.metadata,
                d.file_name, 1 - (c.embedding <=> '{vector_str}'::vector) AS similarity_score
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE c.user_id = %s
            ORDER BY c.embedding <=> '{vector_str}'::vector
            LIMIT %s
        """, (user_id, limit))

        rows = cursor.fetchall()
        results = []
        for row in rows:
            chunk_id, document_id, user_id, content, metadata, file_name, similarity_score = row
            results.append({
                'chunk_id': chunk_id,
                'document_id': document_id,
                'user_id': user_id,
                'content': content,
                'metadata': metadata,
                'file_name': file_name,
                'similarity_score': float(similarity_score)
            })
        return results
    except Exception as e:
        logger.error(f"Similarity search failed: {str(e)}")
        raise e
    finally:
        cursor.close()
        conn.close()

# Function to analyze query characteristics and determine search strategy
async def analyze_query_node(state: AgentState) -> AgentState:
    """Analyze the query to determine the search strategy"""
    query = state["query"]
    reasoning_steps = state.get("reasoning_steps", [])
    
    # Use LLM to analyze query characteristics
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query analyzer for a RAG system. Analyze the query and determine:
        1. Is this asking for recent/current information? (temporal keywords: "recent", "latest", "current", "today", etc.)
        2. Is this asking for specific document content? (keywords: "my document", "file", "report I uploaded")
        3. Is this a general knowledge question?
        4. What level of confidence would traditional document search likely have?
        
        Respond with JSON format:
        {{
            "query_type": "temporal|document_specific|general_knowledge|mixed",
            "needs_current_info": boolean,
            "document_focus": boolean,
            "recommended_strategy": "traditional_only|web_only|hybrid|iterative",
            "reasoning": "explanation of the decision"
        }}"""),
        ("human", f"Query: {query}")
    ])
    
    try:
        response = await llm.ainvoke(analysis_prompt.format_messages())
        analysis = json.loads(response.content)
    except Exception as e:
        logger.warning(f"Query analysis failed, using default: {e}")
        analysis = {
            "query_type": "mixed",
            "needs_current_info": False,
            "document_focus": True,
            "recommended_strategy": "hybrid",
            "reasoning": "Default analysis due to parsing error"
        }
    
    reasoning_steps.append(f"Query Analysis: {analysis['reasoning']}")
    
    return {
        **state,
        "search_strategy": analysis["recommended_strategy"],
        "reasoning_steps": reasoning_steps
    }

# Function to perform traditional RAG search
async def traditional_rag_node(state: AgentState) -> AgentState:
    """Perform traditional RAG search"""
    query = state["query"]
    user_id = state["user_id"]
    reasoning_steps = state.get("reasoning_steps", [])
    
    try:
        # Get embeddings and search
        query_embedding = embed_query(query)
        relevant_chunks = similarity_search(query_embedding, user_id)
        
        # Assess quality
        rag_assessment = assess_rag_quality(relevant_chunks, query)
        
        reasoning_steps.append(f"Traditional RAG: Found {len(relevant_chunks)} chunks, "
                             f"max similarity: {rag_assessment.get('max_similarity', 0):.3f}")
        
        return {
            **state,
            "traditional_rag_results": relevant_chunks,
            "rag_assessment": rag_assessment,
            "reasoning_steps": reasoning_steps
        }
    except Exception as e:
        logger.error(f"Traditional RAG failed: {e}")
        return {
            **state,
            "traditional_rag_results": [],
            "rag_assessment": {"needs_web_search": True, "reason": f"RAG failed: {e}"},
            "reasoning_steps": reasoning_steps + [f"Traditional RAG failed: {e}"]
        }

# Function to perform web search using MCP
async def web_search_node(state: AgentState, mcp_client: StatelessMCPClient) -> AgentState:
    """Perform web search using MCP"""
    query = state["query"]
    reasoning_steps = state.get("reasoning_steps", [])
    
    try:
        search_result = await mcp_client.search_with_mcp(query)
        
        if search_result["success"]:
            web_data = search_result["data"]
            reasoning_steps.append(f"Web Search: Successfully retrieved current information")
        else:
            web_data = None
            reasoning_steps.append(f"Web Search: Failed - {search_result.get('error', 'Unknown error')}")
        
        return {
            **state,
            "web_search_results": web_data,
            "reasoning_steps": reasoning_steps
        }
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            **state,
            "web_search_results": None,
            "reasoning_steps": reasoning_steps + [f"Web search failed: {e}"]
        }

# Function to decide next action based on current results
async def decision_node(state: AgentState) -> AgentState:
    """Decide next action based on current results"""
    strategy = state["search_strategy"]
    rag_results = state.get("traditional_rag_results", [])
    web_results = state.get("web_search_results")
    rag_assessment = state.get("rag_assessment", {})
    iteration_count = state.get("iteration_count", 0)
    reasoning_steps = state.get("reasoning_steps", [])
    
    # Decision logic
    next_action = "generate_response"
    
    if strategy == "iterative" and iteration_count < 2:
        # Check if we need more information
        if not rag_results and not web_results:
            next_action = "traditional_rag" if not rag_results else "web_search"
        elif rag_assessment.get("needs_web_search", False) and not web_results:
            next_action = "web_search"
        elif web_results and not rag_results:
            next_action = "traditional_rag"
    
    reasoning_steps.append(f"Decision: Next action is {next_action}")
    
    return {
        **state,
        "reasoning_steps": reasoning_steps,
        "iteration_count": iteration_count + 1
    }

# Function to generate final response based on available context
async def generate_response_node(state: AgentState) -> AgentState:
    """Generate final response using available context"""
    query = state["query"]
    rag_results = state.get("traditional_rag_results", [])
    web_results = state.get("web_search_results")
    reasoning_steps = state.get("reasoning_steps", [])
    
    # Prepare context
    traditional_context = ""
    if rag_results:
        traditional_context = "\n\n".join([
            f"Document: {c['file_name']}\nContent: {c['content'][:500]}..." 
            for c in rag_results[:3]  # Limit context length
        ])
    
    web_context = ""
    if web_results:
        web_context = f"\n\nCurrent Web Information:\n{str(web_results)[:1000]}..."
    
    # Generate response
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to both user documents and current web information.
        
        Provide a comprehensive answer using the available context. 
        - If using document information, cite the document name
        - If using web information, mention it's from recent web sources
        - If information is missing, clearly state what's needed
        - Be concise but thorough
        
        Context from User Documents:
        {traditional_context}
        
        Current Web Information:
        {web_context}
        
        Reasoning Process:
        {reasoning_process}"""),
        ("human", "{query}")
    ])
    
    try:
        response = await llm.ainvoke(response_prompt.format_messages(
            query=query,
            traditional_context=traditional_context,
            web_context=web_context,
            reasoning_process="\n".join(reasoning_steps)
        ))
        
        final_response = response.content
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(rag_results, web_results, state.get("rag_assessment", {}))
        
        # Determine context sources
        context_sources = []
        if rag_results:
            context_sources.extend([f"Document: {r['file_name']}" for r in rag_results])
        if web_results:
            context_sources.append("Web Search Results")
        
        reasoning_steps.append(f"Response generated with confidence: {confidence_score:.3f}")
        
        return {
            **state,
            "final_response": final_response,
            "confidence_score": confidence_score,
            "context_sources": context_sources,
            "reasoning_steps": reasoning_steps
        }
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return {
            **state,
            "final_response": "I apologize, but I encountered an error generating the response. Please try again.",
            "confidence_score": 0.0,
            "context_sources": [],
            "reasoning_steps": reasoning_steps + [f"Response generation failed: {e}"]
        }

# Function to calculate confidence score based on RAG and web results
def calculate_confidence_score(rag_results: List[Dict], web_results: Optional[str], rag_assessment: Dict) -> float:
    """Calculate confidence score based on available information"""
    score = 0.0
    
    if rag_results:
        max_similarity = max([r.get('similarity_score', 0) for r in rag_results], default=0)
        score += max_similarity * 0.6
    
    if web_results:
        score += 0.4  # Base score for having web results
    
    # Penalize if RAG assessment indicates problems
    if rag_assessment.get("needs_web_search", False) and not web_results:
        score *= 0.7
    
    return min(score, 1.0)

# Function to assess the quality of traditional RAG results
def assess_rag_quality(relevant_chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Assess the quality of traditional RAG results"""
    if not relevant_chunks:
        return {
            "needs_web_search": True,
            "reason": "No relevant documents found",
            "confidence": 0.0,
            "context_length": 0
        }
    
    max_similarity = max([chunk.get('similarity_score', 0.0) for chunk in relevant_chunks])
    avg_similarity = sum([chunk.get('similarity_score', 0.0) for chunk in relevant_chunks]) / len(relevant_chunks)
    total_context_length = sum([len(chunk.get('content', '')) for chunk in relevant_chunks])
    
    needs_web_search = (
        max_similarity < RAG_CONFIDENCE_THRESHOLD or
        avg_similarity < (RAG_CONFIDENCE_THRESHOLD * 0.8) or
        total_context_length < MIN_CONTEXT_LENGTH
    )
    
    reason = ""
    if max_similarity < RAG_CONFIDENCE_THRESHOLD:
        reason = f"Low similarity score: {max_similarity:.3f} < {RAG_CONFIDENCE_THRESHOLD}"
    elif avg_similarity < (RAG_CONFIDENCE_THRESHOLD * 0.8):
        reason = f"Low average similarity: {avg_similarity:.3f}"
    elif total_context_length < MIN_CONTEXT_LENGTH:
        reason = f"Insufficient context length: {total_context_length} < {MIN_CONTEXT_LENGTH}"
    else:
        reason = "Traditional RAG results are sufficient"
    
    return {
        "needs_web_search": needs_web_search,
        "reason": reason,
        "confidence": max_similarity,
        "context_length": total_context_length,
        "max_similarity": max_similarity,
        "avg_similarity": avg_similarity
    }

# Gemini RAG Evaluator Class
class GeminiRagEvaluator:
    def __init__(self, model_name, google_api_key):
        self.model_name = model_name
        self.google_api_key = google_api_key
        self.client = genai.Client(api_key=self.google_api_key)
    
    def evaluate_response(self, query: str, answer: str, contexts: List[str], 
                         ground_truth: Optional[str] = None) -> Dict[str, float]:
        results = {}
        results["answer_relevancy"] = self._evaluate_answer_relevancy(query, answer)
        results["faithfulness"] = self._evaluate_faithfulness(query, answer, contexts)
        
        if ground_truth:
            results["context_precision"] = self._evaluate_context_precision(answer, ground_truth)
        
        return results
    
    def _evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        prompt = f"""On a scale of 0 to 1, rate how directly this answer addresses the query.
        Only respond with a number between 0 and 1.
        
        Query: {query}
        Answer: {answer}
        
        Rating:"""
        
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=[prompt])
            rating_text = response.text.strip()
            import re
            matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
            if matches:
                return min(max(float(matches[0]), 0.0), 1.0)
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating answer relevancy: {e}")
            return 0.5
    
    def _evaluate_faithfulness(self, query: str, answer: str, contexts: List[str]) -> float:
        context_text = "\n\n---\n\n".join(contexts)
        prompt = f"""On a scale of 0 to 1, evaluate how factually accurate this answer is based on the context.
        Only respond with a number between 0 and 1.
        
        Query: {query}
        Context: {context_text}
        Answer: {answer}
        
        Rating:"""
        
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=[prompt])
            rating_text = response.text.strip()
            import re
            matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
            if matches:
                return min(max(float(matches[0]), 0.0), 1.0)
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {e}")
            return 0.5
    
    def _evaluate_context_precision(self, answer: str, ground_truth: str) -> float:
        prompt = f"""On a scale of 0 to 1, rate how well the answer matches the ground truth.
        Only respond with a number between 0 and 1.
        
        Answer: {answer}
        Ground Truth: {ground_truth}
        
        Rating:"""
        
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=[prompt])
            rating_text = response.text.strip()
            import re
            matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
            if matches:
                return min(max(float(matches[0]), 0.0), 1.0)
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating context precision: {e}")
            return 0.5

# Main Agentic RAG Class
class AgenticRAG:
    def __init__(self, mcp_server_url: Optional[str] = None):
        self.mcp_client = StatelessMCPClient(mcp_server_url, MCP_TIMEOUT) if mcp_server_url else None
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Use the new syntax for LangGraph 0.2.x
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_wrapper)
        workflow.add_node("traditional_rag", self._traditional_rag_wrapper)
        workflow.add_node("web_search", self._web_search_wrapper)
        workflow.add_node("decision", self._decision_wrapper)
        workflow.add_node("generate_response", self._generate_response_wrapper)
        
        # Define edges
        workflow.set_entry_point("analyze_query")
        
        # From analyze_query, go to initial search based on strategy
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_after_analysis,
            {
                "traditional_rag": "traditional_rag",
                "web_search": "web_search",
                "both": "traditional_rag"  # Start with traditional, then decide
            }
        )
        
        # From traditional_rag, go to decision
        workflow.add_edge("traditional_rag", "decision")
        
        # From web_search, go to decision
        workflow.add_edge("web_search", "decision")
        
        # From decision, route to next action
        workflow.add_conditional_edges(
            "decision",
            self._route_after_decision,
            {
                "traditional_rag": "traditional_rag",
                "web_search": "web_search",
                "generate_response": "generate_response"
            }
        )
        
        # Generate response is the end
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _analyze_query_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for async analyze_query_node"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(analyze_query_node(state))
        finally:
            loop.close()
    
    def _traditional_rag_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for async traditional_rag_node"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(traditional_rag_node(state))
        finally:
            loop.close()
    
    def _web_search_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for async web_search_node"""
        if not self.mcp_client:
            return {
                **state,
                "web_search_results": None,
                "reasoning_steps": state.get("reasoning_steps", []) + ["Web search skipped - no MCP client"]
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(web_search_node(state, self.mcp_client))
        finally:
            loop.close()
    
    def _decision_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for async decision_node"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(decision_node(state))
        finally:
            loop.close()
    
    def _generate_response_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for async generate_response_node"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(generate_response_node(state))
        finally:
            loop.close()
    
    def _route_after_analysis(self, state: AgentState) -> str:
        """Route after query analysis"""
        strategy = state.get("search_strategy", "hybrid")
        if strategy == "traditional_only":
            return "traditional_rag"
        elif strategy == "web_only":
            return "web_search"
        else:  # hybrid or iterative
            return "both"
    
    def _route_after_decision(self, state: AgentState) -> str:
        """Route after decision node"""
        strategy = state["search_strategy"]
        rag_results = state.get("traditional_rag_results", [])
        web_results = state.get("web_search_results")
        rag_assessment = state.get("rag_assessment", {})
        iteration_count = state.get("iteration_count", 0)
        
        # If we have enough information or max iterations reached, generate response
        if iteration_count >= 3:
            return "generate_response"
        
        # If strategy is iterative, check what we need
        if strategy == "iterative":
            if not rag_results:
                return "traditional_rag"
            elif rag_assessment.get("needs_web_search", False) and not web_results and self.mcp_client:
                return "web_search"
        
        return "generate_response"
    
    def process_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """Process a query through the agentic RAG workflow"""
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            query=query,
            user_id=user_id,
            traditional_rag_results=[],
            web_search_results=None,
            rag_assessment={},
            search_strategy="hybrid",
            iteration_count=0,
            final_response="",
            confidence_score=0.0,
            context_sources=[],
            reasoning_steps=[],
            evaluation_results={}
        )
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Agentic RAG processing failed: {e}")
            return {
                **initial_state,
                "final_response": "I apologize, but I encountered an error processing your query. Please try again.",
                "confidence_score": 0.0,
                "reasoning_steps": [f"Processing failed: {str(e)}"]
            }

# Function to evaluate RAG response using GeminiRagEvaluator
def evaluate_rag_response(model_name: str, query: str, answer: str, contexts: List[Dict], ground_truth: Optional[str] = None) -> Dict[str, float]:
    """Evaluate RAG response using GeminiRagEvaluator"""
    try:
        if not ENABLE_EVALUATION:
            results = {"answer_relevancy": 0.0, "faithfulness": 0.0}
            if ground_truth:
                results["context_precision"] = 0.0
            return results
            
        evaluator = GeminiRagEvaluator(model_name, GEMINI_API_KEY)
        context_strings = []
        
        # Convert contexts to strings
        for context in contexts:
            if isinstance(context, dict):
                context_strings.append(context.get("content", str(context)))
            else:
                context_strings.append(str(context))
        
        return evaluator.evaluate_response(
            query=query,
            answer=answer,
            contexts=context_strings,
            ground_truth=ground_truth
        )
    except Exception as e:
        logger.error(f"RAG evaluation failed: {str(e)}")
        results = {"answer_relevancy": 0.5, "faithfulness": 0.5}
        if ground_truth:
            results["context_precision"] = 0.5
        return results

# Lambda handler function
def handler(event, context):
    """Enhanced Lambda handler with Agentic RAG"""
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Extract body from the request
        body = {}
        if 'body' in event:
            if isinstance(event.get('body'), str) and event.get('body'):
                try:
                    body = json.loads(event['body'])
                except json.JSONDecodeError:
                    body = {}
            elif isinstance(event.get('body'), dict):
                body = event.get('body')
        
        # Extract parameters
        query = body.get('query')
        user_id = body.get('user_id', 'system')
        ground_truth = body.get('ground_truth')
        enable_evaluation = body.get('enable_evaluation', ENABLE_EVALUATION)
        model_name = body.get('model_name', GEMINI_MODEL)
        mcp_server_url = body.get('mcp_server_url', None)
        force_strategy = body.get('force_strategy', None)  # "traditional_only", "web_only", "hybrid", "iterative"
        
        # Health check
        if event.get('action') == 'healthcheck' or body.get('action') == 'healthcheck':
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'Agentic RAG with LangGraph is healthy',
                    'stage': STAGE,
                    'mcp_server_url': mcp_server_url,
                    'capabilities': [
                        'traditional_rag',
                        'web_search_mcp',
                        'agentic_workflow',
                        'adaptive_strategy',
                        'iterative_search'
                    ],
                    'version': '2.0.0'
                })
            }

        # Validate query
        if not query:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'message': 'Query is required'})
            }

        # Initialize Agentic RAG
        agentic_rag = AgenticRAG(mcp_server_url)
        
        # Override strategy if forced
        if force_strategy:
            # We'll need to modify the initial state after creation
            pass
        
        # Process query through agentic workflow
        start_time = time.time()
        result_state = agentic_rag.process_query(query, user_id)
        processing_time = time.time() - start_time
        
        # Override strategy if forced (modify the result)
        if force_strategy:
            result_state["search_strategy"] = force_strategy
            result_state["reasoning_steps"].append(f"Strategy forced to: {force_strategy}")
        
        # Extract results
        final_response = result_state.get("final_response", "No response generated")
        traditional_rag_results = result_state.get("traditional_rag_results", [])
        web_search_results = result_state.get("web_search_results")
        rag_assessment = result_state.get("rag_assessment", {})
        confidence_score = result_state.get("confidence_score", 0.0)
        context_sources = result_state.get("context_sources", [])
        reasoning_steps = result_state.get("reasoning_steps", [])
        search_strategy = result_state.get("search_strategy", "unknown")
        iteration_count = result_state.get("iteration_count", 0)
        
        # Perform evaluation if enabled
        evaluation_results = {}
        if enable_evaluation and final_response:
            evaluation_contexts = []
            
            # Add traditional RAG contexts
            if traditional_rag_results:
                evaluation_contexts.extend(traditional_rag_results)
            
            # Add web search context
            if web_search_results:
                evaluation_contexts.append({"content": str(web_search_results)})
            
            evaluation_results = evaluate_rag_response(
                model_name,
                query=query,
                answer=final_response,
                contexts=evaluation_contexts,
                ground_truth=ground_truth
            )
        
        # Prepare comprehensive response
        response_data = {
            'query': query,
            'response': final_response,
            'agentic_workflow': {
                'strategy_used': search_strategy,
                'iterations': iteration_count,
                'reasoning_steps': reasoning_steps,
                'processing_time_seconds': round(processing_time, 3),
                'confidence_score': confidence_score,
                'context_sources': context_sources
            },
            'traditional_rag': {
                'results': traditional_rag_results,
                'count': len(traditional_rag_results),
                'assessment': rag_assessment
            },
            'web_search': {
                'used': web_search_results is not None,
                'data': web_search_results if web_search_results else None,
                'mcp_server_url': mcp_server_url
            },
            'evaluation': evaluation_results,
            'metadata': {
                'user_id': user_id,
                'model_name': model_name,
                'enable_evaluation': enable_evaluation,
                'force_strategy': force_strategy,
                'agentic_version': '2.0.0',
                'framework': 'langgraph'
            }
        }
        
        logger.info(f"Agentic RAG completed: strategy={search_strategy}, "
                   f"iterations={iteration_count}, confidence={confidence_score:.3f}")

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps(response_data, cls=DecimalEncoder)
        }

    except Exception as e:
        logger.error(f"Unhandled error in agentic RAG: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'message': f"Internal error: {str(e)}",
                'error_type': 'agentic_rag_error',
                'framework': 'langgraph'
            })
        }