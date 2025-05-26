"""
Lambda function to handle document uploads.
"""
import os
import json
import boto3
import logging
import uuid
import base64
import psycopg2
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
secretsmanager = boto3.client('secretsmanager')
lambda_client = boto3.client('lambda')

# Get environment variables
DOCUMENTS_BUCKET = os.environ.get('DOCUMENTS_BUCKET')
METADATA_TABLE = os.environ.get('METADATA_TABLE')
DB_SECRET_ARN = os.environ.get('DB_SECRET_ARN')
STAGE = os.environ.get('STAGE')

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


def get_mime_type(file_name):
    """
    Determine MIME type from file extension.
    
    Args:
        file_name (str): File name
        
    Returns:
        str: MIME type
    """
    file_extension = file_name.split('.')[-1].lower()
    mime_types = {
        'pdf': 'application/pdf',
        'txt': 'text/plain',
        'csv': 'text/csv',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'xls': 'application/vnd.ms-excel',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'json': 'application/json',
        'md': 'text/markdown'
    }
    return mime_types.get(file_extension, 'application/octet-stream')


def handler(event, context):
    """
    Lambda function to handle document uploads.
    
    Args:
        event (dict): API Gateway event containing upload details
        context (object): Lambda context
        
    Returns:
        dict: Response with status code and body
    """
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
                    'message': 'Upload handler is healthy',
                    'stage': STAGE
                })
            }
        
        # Extract file data and metadata
        file_content_base64 = body.get('file_content', '')
        file_name = body.get('file_name', '')
        mime_type = body.get('mime_type', None)
        user_id = body.get('user_id', 'system')
        
        if not file_content_base64 or not file_name:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'File content and name are required'
                })
            }
        
        # Determine MIME type if not provided
        if not mime_type:
            mime_type = get_mime_type(file_name)
            
        # Decode base64 content
        file_content = base64.b64decode(file_content_base64)
        
        # Generate a unique document ID
        document_id = str(uuid.uuid4())
        
        # Upload file to S3
        s3_key = f"uploads/{user_id}/{document_id}/{file_name}"
        s3_client.put_object(
            Bucket=DOCUMENTS_BUCKET,
            Key=s3_key,
            Body=file_content,
            ContentType=mime_type
        )
        
        # Store initial metadata in PostgreSQL
        try:
            # Get PostgreSQL credentials
            credentials = get_postgres_credentials()
            conn = get_postgres_connection(credentials)
            cursor = conn.cursor()
            
            # Insert document record
            cursor.execute("""
            INSERT INTO documents (document_id, user_id, file_name, mime_type, status, bucket, key, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                document_id,
                user_id,
                file_name,
                mime_type,
                'uploaded',
                DOCUMENTS_BUCKET,
                s3_key,
                datetime.now(),
                datetime.now()
            ))
            
            # Commit the transaction
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing metadata in PostgreSQL: {str(e)}")
            # Continue with Storing metadata in DynamoDB as fallback
            metadata_table = dynamodb.Table(METADATA_TABLE)
            metadata_table.put_item(
                Item={
                    'id': f"doc#{document_id}",
                    'document_id': document_id,
                    'user_id': user_id,
                    'file_name': file_name,
                    'mime_type': mime_type,
                    'status': 'uploaded',
                    'bucket': DOCUMENTS_BUCKET,
                    'key': s3_key,
                    'created_at': int(datetime.now().timestamp() * 1000),
                    'updated_at': int(datetime.now().timestamp() * 1000)
                }
            )
        
        # Return success response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'File uploaded successfully',
                'document_id': document_id,
                'file_name': file_name
            })
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': f"Error uploading file: {str(e)}"
            })
        }