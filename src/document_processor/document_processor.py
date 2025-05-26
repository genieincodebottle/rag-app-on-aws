"""
Lambda function to process documents uploaded to S3.
Extracts text from documents, chunks it, creates embeddings, and stores in PostgreSQL.
"""
import os
import json
import boto3
import logging
import tempfile
import psycopg2
import uuid
import urllib.parse
from datetime import datetime
from typing import List, Tuple

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from google import genai
from google.genai import types

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
secretsmanager = boto3.client('secretsmanager')

# Get environment variables
DOCUMENTS_BUCKET = os.environ.get('DOCUMENTS_BUCKET')
METADATA_TABLE = os.environ.get('METADATA_TABLE')
DB_SECRET_ARN = os.environ.get('DB_SECRET_ARN')
GEMINI_SECRET_ARN = os.environ.get('GEMINI_SECRET_ARN')
STAGE = os.environ.get('STAGE')

GEMINI_EMBEDDING_MODEL = os.environ.get('GEMINI_EMBEDDING_MODEL')
TEMPERATURE = float(os.environ.get('TEMPERATURE'))
MAX_OUTPUT_TOKENS = int(os.environ.get('MAX_OUTPUT_TOKENS'))
TOP_K = int(os.environ.get('TOP_K'))
TOP_P = float(os.environ.get('TOP_P'))
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD'))
GEMINI_EMBEDDING_MODEL = "text-embedding-004"


def get_gemini_api_key():
    """
    Get Gemini API key from Secrets Manager.
    """
    try:
        secret_response = secretsmanager.get_secret_value(
            SecretId=GEMINI_SECRET_ARN
        )
        secret = json.loads(secret_response['SecretString'])
        return secret['GEMINI_API_KEY']
    except Exception as e:
        logger.error(f"Error getting Gemini API key: {str(e)}")
        raise e

# Set up Gemini API key from Secrets Manager
try:
    GEMINI_API_KEY = get_gemini_api_key()
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Error configuring Gemini API: {str(e)}")


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of documents
    """
    embeddings = []
    for text in texts:
        embedding = embed_query(text)
        embeddings.append(embedding)
    return embeddings


def embed_query(text: str) -> List[float]:
    """
    Embed text using Gemini and return a flat list of floats for pgvector.
    """
    try:
        result = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        # Access the first embedding object and return its .values
        return list(result.embeddings[0].values)
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        return [0.0] * 768


def get_postgres_credentials():
    """
    Get PostgreSQL credentials from Secrets Manager.
    """
    try:
        secret_response = secretsmanager.get_secret_value(
            SecretId=DB_SECRET_ARN
        )
        secret = json.loads(secret_response['SecretString'])
        return secret
    except Exception as e:
        logger.error(f"Error getting PostgreSQL credentials: {str(e)}")
        raise e


def get_postgres_connection(credentials):
    """
    Get a connection to PostgreSQL.
    """
    conn = psycopg2.connect(
        host=credentials['host'],
        port=credentials['port'],
        user=credentials['username'],
        password=credentials['password'],
        dbname=credentials['dbname']
    )
    return conn


def get_document_loader(file_path, mime_type):
    """
    Get the appropriate document loader based on file type.
    
    Args:
        file_path (str): Path to the file
        mime_type (str): MIME type of the file
        
    Returns:
        LangChain document loader
    """
    if mime_type == 'application/pdf':
        return PyPDFLoader(file_path)
    elif mime_type == 'text/plain':
        return TextLoader(file_path)
    elif mime_type in ['text/csv', 'application/csv']:
        return CSVLoader(file_path)
    else:
        # Default to text loader for unknown types
        return TextLoader(file_path)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks.
    
    Args:
        documents (List[Document]): List of LangChain documents
        
    Returns:
        List[Document]: List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_s3_object_with_various_encoding(bucket: str, key: str) -> str:
    """
    Try to access an S3 object with different URL encoding methods.
    Returns the correct key that works.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key to try
        
    Returns:
        str: The correct S3 key that works
    
    Raises:
        Exception: If object cannot be found with any encoding approach
    """
    logger.info(f"Trying to access S3 object with various encodings: s3://{bucket}/{key}")
    
    # List of keys to try
    keys_to_try = []
    
    # Original key
    keys_to_try.append(key)
    
    # URL decoded key 
    decoded_key = urllib.parse.unquote_plus(key)
    if decoded_key != key:
        keys_to_try.append(decoded_key)
    
    # URL encoded key (convert spaces to +)
    encoded_key = urllib.parse.quote_plus(decoded_key)
    if encoded_key != key and encoded_key != decoded_key:
        keys_to_try.append(encoded_key)
    
    # URL encoded key (convert spaces to %20)
    encoded_key_alt = urllib.parse.quote(decoded_key, safe='/')
    if encoded_key_alt != key and encoded_key_alt not in keys_to_try:
        keys_to_try.append(encoded_key_alt)
    
    # Try all possible keys
    for attempt_key in keys_to_try:
        try:
            logger.info(f"Trying key: {attempt_key}")
            s3_client.head_object(Bucket=bucket, Key=attempt_key)
            logger.info(f"Successfully found S3 object with key: {attempt_key}")
            return attempt_key
        except Exception as e:
            logger.warning(f"Failed to access S3 object with key: {attempt_key}, error: {str(e)}")
    
    # If we get here, try listing objects with the same prefix
    try:
        prefix = '/'.join(key.split('/')[:-1]) + '/'
        logger.info(f"Listing objects with prefix: {prefix}")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' in response:
            for obj in response.get('Contents', []):
                actual_key = obj['Key']
                logger.info(f"Found object in bucket: {actual_key}")
                
                # Check if the filename part matches (ignoring encoding differences)
                expected_filename = key.split('/')[-1].replace('+', ' ').replace('%20', ' ')
                actual_filename = actual_key.split('/')[-1].replace('+', ' ').replace('%20', ' ')
                
                if expected_filename == actual_filename:
                    logger.info(f"Found matching object: {actual_key}")
                    # Test if we can access this object
                    try:
                        s3_client.head_object(Bucket=bucket, Key=actual_key)
                        logger.info(f"Successfully accessed matching object: {actual_key}")
                        return actual_key
                    except Exception as e:
                        logger.warning(f"Found matching object but can't access it: {actual_key}, error: {str(e)}")
                
            logger.warning(f"No matching object found after checking {len(response.get('Contents', []))} objects with prefix {prefix}")
        else:
            logger.warning(f"No objects found with prefix: {prefix}")
    except Exception as e:
        logger.error(f"Error listing objects in bucket: {str(e)}")
    
    # If we still can't find the object, raise exception with details
    raise Exception(f"Could not find S3 object in bucket '{bucket}' with key '{key}' or any variation. Tried keys: {keys_to_try}")


def process_document(bucket: str, key: str, document_id: str, user_id: str, mime_type: str) -> Tuple[int, List[str]]:
    """
    Process a document, chunk it, create embeddings, and store in PostgreSQL.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        document_id (str): Document ID
        user_id (str): User ID
        mime_type (str): MIME type of the document
        
    Returns:
        Tuple[int, List[str]]: Number of chunks created and list of chunk IDs
    """
    # Find the correct key encoding
    try:
        working_key = get_s3_object_with_various_encoding(bucket, key)
        logger.info(f"Using corrected S3 key: {working_key}")
        
        if working_key != key:
            logger.info(f"Original key '{key}' was modified to '{working_key}' to match actual S3 object")
            key = working_key
    except Exception as e:
        logger.error(f"Failed to find S3 object with any encoding variation: {str(e)}")
        raise
    
    # Download the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        logger.info(f"Downloading S3 object from s3://{bucket}/{key} to {temp_file.name}")
        s3_client.download_file(bucket, key, temp_file.name)
        file_path = temp_file.name
    
    try:
        # Load document using appropriate loader
        loader = get_document_loader(file_path, mime_type)
        documents = loader.load()
        
        logger.info(f"Loaded {len(documents)} document(s)")
        
        # Chunk documents
        chunks = chunk_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Get PostgreSQL credentials
        credentials = get_postgres_credentials()
        conn = get_postgres_connection(credentials)
        cursor = conn.cursor()
        
        # Get file name from key (handle encoding)
        file_name = key.split('/')[-1]
        
        # Store document in PostgreSQL
        cursor.execute("""
        INSERT INTO documents (document_id, user_id, file_name, mime_type, status, bucket, key, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """, (
            document_id,
            user_id,
            file_name,
            mime_type,
            'processed',
            bucket,
            key,
            datetime.now(),
            datetime.now()
        ))
        
        # Commit the transaction
        conn.commit()
        
        # Store chunks with embeddings in PostgreSQL
        chunk_ids = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            # Create embedding
            embedding = embed_query(chunk.page_content)
            
            # Prepare metadata
            metadata = {
                "source": key,
                "page": chunk.metadata.get("page", 0) if hasattr(chunk, "metadata") else 0
            }
            
            # Store in PostgreSQL
            cursor.execute("""
            INSERT INTO chunks (chunk_id, document_id, user_id, content, metadata, embedding, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                chunk_id,
                document_id,
                user_id,
                chunk.page_content,
                json.dumps(metadata),
                embedding,
                datetime.now(),
                datetime.now()
            ))
        
        # Commit the transaction
        conn.commit()
        
        return len(chunks), chunk_ids
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise e
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
            logger.info(f"Temporary file {file_path} cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file {file_path}: {str(e)}")


def handler(event, context):
    """
    Lambda function to process documents uploaded to S3.
    
    This function is triggered when a document is uploaded to the S3 bucket.
    It extracts text from the document, chunks it, creates embeddings, and stores in PostgreSQL.
    
    Args:
        event (dict): S3 event notification
        context (object): Lambda context
        
    Returns:
        dict: Response with status code and body
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # For direct invocation or API Gateway calls
        if isinstance(event, dict) and event.get('action') == 'healthcheck':
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'Document processor is healthy',
                    'stage': STAGE
                })
            }
            
        # Also check for healthcheck in body (API Gateway)
        body = {}
        if 'body' in event:
            if isinstance(event.get('body'), str) and event.get('body'):
                try:
                    body = json.loads(event['body'])
                except json.JSONDecodeError:
                    body = {}
            elif isinstance(event.get('body'), dict):
                body = event.get('body')
                
        if body.get('action') == 'healthcheck':
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'Document processor is healthy',
                    'stage': STAGE
                })
            }
            
        # Extract bucket and key from the S3 event
        if 'Records' in event:
            # This is an S3 event notification
            bucket = event['Records'][0]['s3']['bucket']['name']
            key = event['Records'][0]['s3']['object']['key']
            
            # URL decode the key to handle potential encoding issues
            try:
                decoded_key = urllib.parse.unquote_plus(key)
                if decoded_key != key:
                    logger.info(f"URL decoded key from '{key}' to '{decoded_key}'")
                    key = decoded_key
            except Exception as e:
                logger.warning(f"Error decoding key: {str(e)}")
            
            # Process the document
            logger.info(f"Processing document: {key} from bucket: {bucket}")
            
            # Extract document ID and user ID from key
            # Format: uploads/{user_id}/{document_id}/{file_name}
            parts = key.split('/')
            if len(parts) >= 4:
                user_id = parts[1]
                document_id = parts[2]
                file_name = parts[3]
            else:
                # Fallback if key format is different
                document_id = key.split('/')[-1].split('.')[0]
                user_id = 'system'
                file_name = key.split('/')[-1]
            
            # Determine MIME type from file extension
            file_extension = file_name.split('.')[-1].lower()
            if file_extension == 'pdf':
                mime_type = 'application/pdf'
            elif file_extension == 'txt':
                mime_type = 'text/plain'
            elif file_extension == 'csv':
                mime_type = 'text/csv'
            else:
                mime_type = 'application/octet-stream'
            
            # Process the document
            num_chunks, chunk_ids = process_document(bucket, key, document_id, user_id, mime_type)
            
            # Store metadata in DynamoDB
            metadata_table = dynamodb.Table(METADATA_TABLE)
            metadata_table.put_item(
                Item={
                    'id': f"doc#{document_id}",
                    'document_id': document_id,
                    'user_id': user_id,
                    'status': 'processed',
                    'bucket': bucket,
                    'key': key,
                    'num_chunks': num_chunks,
                    'chunk_ids': chunk_ids,
                    'created_at': int(datetime.now().timestamp() * 1000),
                    'updated_at': int(datetime.now().timestamp() * 1000)
                }
            )
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f"Successfully processed document: {document_id}",
                    'document_id': document_id,
                    'num_chunks': num_chunks
                })
            }
        else:
            # This is a direct invocation
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Document processor is healthy',
                    'stage': STAGE
                })
            }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f"Error processing document: {str(e)}"
            })
        }