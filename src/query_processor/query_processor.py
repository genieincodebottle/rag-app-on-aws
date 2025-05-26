"""
Lambda function to process queries and retrieve relevant documents using RAG.
Includes RAG evaluation functionality.
"""
import os
import json
import boto3
import logging
import psycopg2
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

# Convert Decimal
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

# Embed a query using Gemini embedding model
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

# Get RDS credentials from Secrets Manager
def get_postgres_credentials():
    try:
        response = secretsmanager.get_secret_value(SecretId=DB_SECRET_ARN)
        return json.loads(response['SecretString'])
    except Exception as e:
        logger.error(f"Error fetching DB credentials: {str(e)}")
        raise

# PostgreSQL connection
def get_postgres_connection(creds):
    return psycopg2.connect(
        host=creds['host'],
        port=creds['port'],
        user=creds['username'],
        password=creds['password'],
        dbname=creds['dbname']
    )


# Vector similarity search using pgvector
def similarity_search(query_embedding: List[float], user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    credentials = get_postgres_credentials()
    conn = get_postgres_connection(credentials)

    try:
        cursor = conn.cursor()

        # Manually convert the Python list to PostgreSQL vector string format
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


# Generate a response from Gemini using relevant context
def generate_response(model_name: str, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join([f"Document: {c['file_name']}\nContent: {c['content']}" for c in relevant_chunks])
    prompt = f"""
    Answer the following question based on the provided context.
    If the answer is not in the context, say "I don't have enough information."

    Context:
    {context}

    Question: {query}

    Answer:
    """
    try:
        config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type='application/json'
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

# RAG Evaluation functionality
class GeminiRagEvaluator:
    """RAG Evaluator using Google's Gemini model"""
    
    def __init__(self, model_name, google_api_key=None):
        """Initialize the evaluator with the Gemini model"""
        self.model_name = model_name
        self.google_api_key = google_api_key
        self.client = genai.Client(api_key=self.google_api_key)
        
    def evaluate_response(self, query: str, answer: str, contexts: List[str], 
                         ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate RAG response using Gemini model
        
        Args:
            query: The user's query
            answer: The generated answer
            contexts: Retrieved context passages
            ground_truth: Optional ground truth answer
            
        Returns:
            Dict with evaluation metrics
        """
        results = {}
        
        # Evaluate answer relevancy
        results["answer_relevancy"] = self._evaluate_answer_relevancy(query, answer)
        
        # Evaluate faithfulness to the context
        results["faithfulness"] = self._evaluate_faithfulness(query, answer, contexts)
        
        # If ground truth is provided, evaluate precision
        if ground_truth:
            results["context_precision"] = self._evaluate_context_precision(answer, ground_truth)
        
        return results
    
    def _evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the query"""
        prompt = f"""On a scale of 0 to 1 (where 1 is best), rate how directly this answer addresses the query.
        Only respond with a number between 0 and 1, with up to 2 decimal places.
        
        Query: {query}
        Answer: {answer}
        
        Rating (0-1):"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt]
            )
            rating_text = response.text.strip()
            
            # Extract the numerical rating
            if rating_text:
                try:
                    # Find the first float in the response
                    import re
                    matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
                    if matches:
                        rating = float(matches[0])
                        return min(max(rating, 0.0), 1.0)  # Ensure value is between 0 and 1
                except:
                    return 0.5
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating answer relevancy with Gemini: {str(e)}")
            return 0.5
                
    def _evaluate_faithfulness(self, query: str, answer: str, contexts: List[str]) -> float:
        """Evaluate how faithful the answer is to the provided contexts"""
        # Join contexts with separators for clarity
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt]
            )
            rating_text = response.text.strip()
            
            # Extract the numerical rating
            if rating_text:
                try:
                    # Find the first float in the response
                    import re
                    matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
                    if matches:
                        rating = float(matches[0])
                        return min(max(rating, 0.0), 1.0)  # Ensure value is between 0 and 1
                except:
                    return 0.5
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating faithfulness with Gemini: {str(e)}")
            return 0.5
    
    def _evaluate_context_precision(self, answer: str, ground_truth: str) -> float:
        """Evaluate how close the answer is to the ground truth"""
        prompt = f"""On a scale of 0 to 1 (where 1 is best), rate how well the given answer matches the ground truth.
        Consider factual accuracy, completeness, and correctness.
        Only respond with a number between 0 and 1, with up to 2 decimal places.
        
        Answer: {answer}
        Ground Truth: {ground_truth}
        
        Rating (0-1):"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt]
            )
            rating_text = response.text.strip()
            
            # Extract the numerical rating
            if rating_text:
                try:
                    # Find the first float in the response
                    import re
                    matches = re.findall(r"0\.\d+|\d+\.?\d*", rating_text)
                    if matches:
                        rating = float(matches[0])
                        return min(max(rating, 0.0), 1.0)  # Ensure value is between 0 and 1
                except:
                    return 0.5
            return 0.5
        except Exception as e:
            logger.error(f"Error evaluating context precision with Gemini: {str(e)}")
            return 0.5

# Function to evaluate the RAG response
def evaluate_rag_response(model_name: str, query: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate the RAG response quality using Gemini
    
    Args:
        query: The user's question
        answer: The generated answer
        contexts: List of context passages used for generation
        ground_truth: Optional ground truth answer
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        if not ENABLE_EVALUATION:
            # Return placeholder values when evaluation is disabled
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
        # Return default values on error
        results = {"answer_relevancy": 0.5, "faithfulness": 0.5}
        if ground_truth:
            results["context_precision"] = 0.5
        return results

# Lambda handler
def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    try:
         # Extract body from the request for API Gateway calls
        body = {}
        if 'body' in event:
            if isinstance(event.get('body'), str) and event.get('body'):
                try:
                    body = json.loads(event['body'])
                except json.JSONDecodeError:
                    body = {}
            elif isinstance(event.get('body'), dict):
                body = event.get('body')
                
        # Check if this is a health check request
        if event.get('action') == 'healthcheck' or body.get('action') == 'healthcheck':
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'Query processor is healthy',
                    'stage': STAGE
                })
            }

        query = body.get('query')
        user_id = body.get('user_id', 'system')
        ground_truth = body.get('ground_truth')
        enable_evaluation = body.get('enable_evaluation', ENABLE_EVALUATION)
        model_name = body.get('model_name', GEMINI_MODEL)

        if not query:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'message': 'Query is required'})
            }

        query_embedding = embed_query(query)
        relevant_chunks = similarity_search(query_embedding, user_id)
        response = generate_response(model_name, query, relevant_chunks)
        
        # Evaluate the response if enabled
        evaluation_results = {}
        if enable_evaluation:
            evaluation_results = evaluate_rag_response(
                model_name,
                query=query,
                answer=response,
                contexts=relevant_chunks,
                ground_truth=ground_truth
            )

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'query': query,
                'response': response,
                'results': relevant_chunks,
                'count': len(relevant_chunks),
                'evaluation': evaluation_results
            }, cls=DecimalEncoder)
        }

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'message': f"Internal error: {str(e)}"})
        }