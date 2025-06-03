"""
Lambda function with MCP Stateless HTTP Streaming integration
Works with MCP servers deployed locally or in the cloud.
This function performs Web Search if traiditional RAG is inefficient using stateless MCP http streamling calls.
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
from typing import List, Dict, Any, Optional
from decimal import Decimal
from google import genai
from google.genai import types

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
TEMPERATURE = float(os.environ.get('TEMPERATURE'))
MAX_OUTPUT_TOKENS = int(os.environ.get('MAX_OUTPUT_TOKENS'))
TOP_K = int(os.environ.get('TOP_K'))
TOP_P = float(os.environ.get('TOP_P'))
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

# Gemini client
try:
    GEMINI_API_KEY = get_gemini_api_key()
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Error configuring Gemini API client: {str(e)}")
    raise

# Custom JSON encoder for Decimal types to handle JSON serialization 
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

class StatelessMCPClient:
    """
    Stateless MCP client for Lambda integration.
    Each request is independent and doesn't maintain session state.
    """
    
    def __init__(
        self,
        mcp_url: str,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the stateless MCP client.
        
        Args:
            mcp_url: The MCP server URL (Lambda function URL or HTTP endpoint)
            timeout: HTTP timeout for requests
            headers: Optional additional headers
        """
        self.mcp_url = mcp_url
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream", # Accept both JSON and SSE responses
            **(headers or {})
        }
        self._request_id_counter = 0
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_id_counter += 1
        return f"req_{self._request_id_counter}_{int(time.time() * 1000)}"
    
    async def _make_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a stateless JSON-RPC request to the MCP server.
        
        Args:
            method: The MCP method to call
            params: Optional parameters for the method
            
        Returns:
            JSON-RPC response or error dict
        """
        request_id = self._generate_request_id()
        
        # Create JSON-RPC request
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
                # Parse response
                response_data = response.json()
                return {
                    "success": True,
                    "data": response_data,
                    "error": None
                }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return {
                "success": False,
                "data": None,
                "error": f"HTTP error: {e.response.status_code}"
            }
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }
    
    async def list_tools(self) -> Dict[str, Any]:
        """
        List available tools from the MCP server.
        
        Returns:
            Tools list response
        """
        return await self._make_request(method="tools/list")
    
    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call a specific tool on the MCP server.
        
        Args:
            name: Tool name to call
            arguments: Tool arguments
            
        Returns:
            Tool execution response
        """
        return await self._make_request(
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments or {}
            }
        )
    
    async def search_with_mcp(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Perform web search using MCP Server stateless client.
        """
        try:
            logger.info(f"Making stateless MCP request to: {self.mcp_url}")
            
            # Call the web_search tool
            result = await self.call_tool("web_search", {
                "query": query,
                "num_results": max_results
            })
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "data": None,
                    "source": "mcp_client"
                }
            
            # Extract search results from MCP response
            search_data = self._extract_tool_result(result["data"])
            
            logger.info("MCP search completed successfully")
            return {
                "success": True,
                "data": search_data,
                "source": "mcp_web_search"
            }
                    
        except Exception as e:
            logger.error(f"MCP search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "source": "mcp_client"
            }
    
    def _extract_tool_result(self, response_data) -> Any:
        """Extract tool result from MCP response"""
        try:
            # Handle different response formats
            if isinstance(response_data, dict):
                # Check for JSON-RPC response format
                if "result" in response_data:
                    result = response_data["result"]
                    # FastMCP with json_response=True returns direct tool results
                    if isinstance(result, str):
                        return result
                # Check for error in response
                elif "error" in response_data:
                    error = response_data["error"]
                    if isinstance(error, dict):
                        error_message = error.get('message', str(error))
                        error_code = error.get('code', 'unknown')
                        return f"MCP Error ({error_code}): {error_message}"
                    else:
                        return f"MCP Error: {str(error)}"
                # Handle direct content (for some FastMCP configurations)
                elif "content" in response_data:
                    return str(response_data["content"])
                # Fallback to string representation
                else:
                    return str(response_data)
            else:
                return str(response_data)
        except Exception as e:
            logger.error(f"Error extracting tool result: {e}")
            return f"Error extracting result: {str(e)}"

# Function to embed query using Gemini model
def embed_query(text: str) -> List[float]:
    try:
        result = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        return list(result.embeddings[0].values)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return [0.0] * 768

# Function to fetch Postgres credentials from AWS Secrets Manager
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
                c.chunk_id,
                c.document_id,
                c.user_id,
                c.content,
                c.metadata,
                d.file_name,
                1 - (c.embedding <=> '{vector_str}'::vector) AS similarity_score
            FROM 
                chunks c
            JOIN 
                documents d ON c.document_id = d.document_id
            WHERE 
                c.user_id = %s
            ORDER BY 
                c.embedding <=> '{vector_str}'::vector
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

# Function to assess quality of traditional RAG results
def assess_rag_quality(relevant_chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Assess the quality of traditional RAG results to determine if web search is needed
    """
    if not relevant_chunks:
        return {
            "needs_web_search": True,
            "reason": "No relevant documents found",
            "confidence": 0.0,
            "context_length": 0
        }
    
    # Check similarity scores
    max_similarity = max([chunk.get('similarity_score', 0.0) for chunk in relevant_chunks])
    avg_similarity = sum([chunk.get('similarity_score', 0.0) for chunk in relevant_chunks]) / len(relevant_chunks)
    
    # Check total context length
    total_context_length = sum([len(chunk.get('content', '')) for chunk in relevant_chunks])
    
    # Decision logic
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

# Function to perform web search using stateless MCP client
async def perform_mcp_web_search(query: str, mcp_client) -> Dict[str, Any]:
    """
    Perform agentic web search using stateless MCP client
    """
    if not mcp_client:
        return {
            "success": False,
            "error": "Agentic RAG not enabled or MCP client not available",
            "data": None
        }
    try:
        search_result = await mcp_client.search_with_mcp(query)
        return search_result
    except Exception as e:
        logger.error(f"Agentic search failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "data": None
        }

def generate_response(model_name: str, query: str, relevant_chunks: List[Dict[str, Any]], web_search_data: Optional[str] = None) -> str:
    """
    Generate response using both traditional RAG context and web search data
    """
    # Prepare traditional context
    traditional_context = ""
    if relevant_chunks:
        traditional_context = "\n\n".join([
            f"Document: {c['file_name']}\nContent: {c['content']}" 
            for c in relevant_chunks
        ])
    
    # Prepare web search context
    web_context = ""
    if web_search_data:
        web_context = f"\n\nAdditional Web Search Results:\n{web_search_data}"
    
    prompt = f"""
    Answer the following question using the provided context. Use information from both 
    your document library and recent web search results when available.
    
    If the answer is not sufficiently covered in the context, clearly state what 
    information is missing and suggest where the user might find more details.

    Context from Documents:
    {traditional_context}
    {web_context}

    Question: {query}

    Answer:
    """
    
    try:
        config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type='text/plain'
        )
        result = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        return result.text
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        return "Sorry, I couldn't generate a response. Please try again later."

# Evaluator class for RAG responses using Google's Gemini model
class GeminiRagEvaluator:
    """RAG Evaluator using Google's Gemini model"""
    
    def __init__(self, model_name, google_api_key=None):
        self.model_name = model_name
        self.google_api_key = google_api_key
        self.client = genai.Client(api_key=self.google_api_key)
    
    # Evaluates the RAG response based on query, answer, and contexts
    def evaluate_response(self, query: str, answer: str, contexts: List[str], 
                         ground_truth: Optional[str] = None) -> Dict[str, float]:
        results = {}
        results["answer_relevancy"] = self._evaluate_answer_relevancy(query, answer)
        results["faithfulness"] = self._evaluate_faithfulness(query, answer, contexts)
        
        if ground_truth:
            results["context_precision"] = self._evaluate_context_precision(answer, ground_truth)
        
        return results
    
    # Evaluates how relevant the answer is to the query
    def _evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        prompt = f"""On a scale of 0 to 1 (where 1 is best), rate how directly this answer addresses the query.
        Only respond with a number between 0 and 1, with up to 2 decimal places.
        
        Query: {query}
        Answer: {answer}
        
        Rating (0-1):"""
        
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=[prompt])
            rating_text = response.text.strip()
            
            if rating_text:
                try:
                    import re
                    matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
                    if matches:
                        rating = float(matches[0])
                        return min(max(rating, 0.0), 1.0)
                except:
                    return 0.5
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating answer relevancy with Gemini: {str(e)}")
            return 0.5

    # Evaluates how factually accurate and faithful the answer is based on the provided context            
    def _evaluate_faithfulness(self, query: str, answer: str, contexts: List[str]) -> float:
        context_text = "\n\n---\n\n".join(contexts)
        
        prompt = f"""On a scale of 0 to 1 (where 1 is best), evaluate how factually accurate and faithful this answer is based ONLY on the provided context.
        Does the answer contain claims not supported by the context? 
        Does it contradict the context?
        Only respond with a number between 0 and 1, with up to 2 decimal places.
        
        Query: {query}
        Context: {context_text}
        Answer: {answer}
        
        Faithfulness rating (0-1):"""
        
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=[prompt])
            rating_text = response.text.strip()
            
            if rating_text:
                try:
                    import re
                    matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
                    if matches:
                        rating = float(matches[0])
                        return min(max(rating, 0.0), 1.0)
                except:
                    return 0.5
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating faithfulness with Gemini: {str(e)}")
            return 0.5
        
    # Evaluates how well the answer matches the ground truth context
    def _evaluate_context_precision(self, answer: str, ground_truth: str) -> float:
        prompt = f"""On a scale of 0 to 1 (where 1 is best), rate how well the given answer matches the ground truth.
        Consider factual accuracy, completeness, and correctness.
        Only respond with a number between 0 and 1, with up to 2 decimal places.
        
        Answer: {answer}
        Ground Truth: {ground_truth}
        
        Rating (0-1):"""
        
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=[prompt])
            rating_text = response.text.strip()
            
            if rating_text:
                try:
                    import re
                    matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
                    if matches:
                        rating = float(matches[0])
                        return min(max(rating, 0.0), 1.0)
                except:
                    return 0.5
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating context precision with Gemini: {str(e)}")
            return 0.5

# Function to evaluate RAG response using GeminiRagEvaluator
def evaluate_rag_response(model_name: str, query: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict[str, float]:
    try:
        if not ENABLE_EVALUATION:
            results = {"answer_relevancy": 0.0, "faithfulness": 0.0}
            if ground_truth:
                results["context_precision"] = 0.0
            return results
            
        evaluator = GeminiRagEvaluator(model_name, GEMINI_API_KEY)
        return evaluator.evaluate_response(
            query=query,
            answer=answer,
            contexts=[c["content"] for c in contexts],
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
        web_search_with_mcp = body.get('web_search_with_mcp', False)
        mcp_server_url = body.get('mcp_server_url', None)
                
        # Health check
        if event.get('action') == 'healthcheck' or body.get('action') == 'healthcheck':
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'Enhanced query processor with stateless agentic RAG is healthy',
                    'stage': STAGE,
                    'mcp_server_url': mcp_server_url,
                    'client_type': 'stateless_http'
                })
            }

        # Validate query
        if not query:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'message': 'Query is required'})
            }

        # Initialize stateless MCP client
        mcp_client = None
        if mcp_server_url:
            mcp_client = StatelessMCPClient(mcp_server_url, MCP_TIMEOUT)

        # Step 1: Traditional RAG
        query_embedding = embed_query(query)
        relevant_chunks = similarity_search(query_embedding, user_id)
        
        # Step 2: Assess RAG quality
        rag_assessment = assess_rag_quality(relevant_chunks, query)
        
        # Step 3: MCP Based Web search if needed
        web_search_data = None
        mcp_web_search_used = False
        web_search_error = None
        
        # Check if we need to perform web search
        if rag_assessment["needs_web_search"] and (mcp_client or web_search_with_mcp):
            logger.info(f"Triggering stateless agentic search. Reason: {rag_assessment['reason']}")
            
            # Run async MCP based search
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                search_result = loop.run_until_complete(perform_mcp_web_search(query, mcp_client))
                if search_result["success"]:
                    web_search_data = search_result["data"]
                    mcp_web_search_used = True
                else:
                    web_search_error = search_result["error"]
            finally:
                loop.close()
        
        # Step 4: Generate response
        response = generate_response(model_name, query, relevant_chunks, web_search_data)
        logger.info(f"Generated response: {response}")
        # Step 5: Evaluate if enabled
        evaluation_results = {}
        if enable_evaluation:
            evaluation_contexts = relevant_chunks.copy()
            if web_search_data:
                evaluation_contexts.append({"content": str(web_search_data)})
            
            evaluation_results = evaluate_rag_response(
                model_name,
                query=query,
                answer=response,
                contexts=evaluation_contexts,
                ground_truth=ground_truth
            )

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'query': query,
                'response': response,
                'traditional_rag': {
                    'results': relevant_chunks,
                    'count': len(relevant_chunks),
                    'assessment': rag_assessment
                },
                'mcp_web_search': {
                    'used': mcp_web_search_used,
                    'data': web_search_data if mcp_web_search_used else None,
                    'error': web_search_error,
                    'client_type': 'stateless_http'
                },
                'evaluation': evaluation_results,
                'metadata': {
                    'force_web_search': web_search_with_mcp,
                    'mcp_server_url': mcp_server_url,
                    'mcp_client_type': 'stateless_http'
                }
            }, cls=DecimalEncoder)
        }

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'message': f"Internal error: {str(e)}"})
        }