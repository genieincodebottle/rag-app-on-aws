"""Test cases for the query_processor Lambda function."""
import json
import os
import unittest
from unittest.mock import MagicMock, patch
from decimal import Decimal

"""Set up test environment."""
# Set environment variables
os.environ["DOCUMENTS_BUCKET"] = "test-bucket"
os.environ["METADATA_TABLE"] = "test-table"
os.environ["STAGE"] = "test"
os.environ["DB_SECRET_ARN"] = "test-db-secret"
os.environ["GEMINI_SECRET_ARN"] = "test-gemini-secret"
os.environ["GEMINI_EMBEDDING_MODEL"] = "test-embedding-model"
os.environ["TEMPERATURE"] = "0.2"
os.environ["MAX_OUTPUT_TOKENS"] = "1024"
os.environ["TOP_K"] = "40"
os.environ["TOP_P"] = "0.8"
MODEL_NAME = "gemini-2.0-flash"

# Now import the module under test - mocks are already in place globally from conftest
from query_processor.query_processor import (
    handler, get_gemini_api_key, get_postgres_credentials, get_postgres_connection,
    embed_query, similarity_search, generate_response, DecimalEncoder
)

class TestQueryProcessor(unittest.TestCase):
    """Test cases for the query_processor Lambda function."""

    def setUp(self):
        # Mock boto3 clients
        self.s3_patcher = patch("query_processor.query_processor.s3_client")
        self.secrets_patcher = patch("query_processor.query_processor.secretsmanager")
        
        self.mock_s3 = self.s3_patcher.start()
        self.mock_secretsmanager = self.secrets_patcher.start()

    def tearDown(self):
        """Clean up test environment."""
        # Clean up environment variables
        for key in [
            "DOCUMENTS_BUCKET", "METADATA_TABLE", "STAGE", "DB_SECRET_ARN",
            "GEMINI_SECRET_ARN", "GEMINI_EMBEDDING_MODEL", 
            "TEMPERATURE", "MAX_OUTPUT_TOKENS", "TOP_K", "TOP_P"
        ]:
            if key in os.environ:
                del os.environ[key]
                
        # Stop patchers
        self.s3_patcher.stop()
        self.secrets_patcher.stop()

    @patch("query_processor.query_processor.secretsmanager")
    def test_get_gemini_api_key(self, mock_secretsmanager):
        """Test getting Gemini API key from Secrets Manager."""
        # Mock the Secrets Manager response
        mock_secret_string = json.dumps({"GEMINI_API_KEY": "mock-api-key"})
        mock_response = {"SecretString": mock_secret_string}
        mock_secretsmanager.get_secret_value.return_value = mock_response

        # Call the function
        api_key = get_gemini_api_key()

        # Verify results
        self.assertEqual(api_key, "mock-api-key")
        mock_secretsmanager.get_secret_value.assert_called_once_with(
            SecretId="test-gemini-secret"
        )
        
    @patch("query_processor.query_processor.secretsmanager")
    def test_get_postgres_credentials(self, mock_secretsmanager):
        """Test getting PostgreSQL credentials from Secrets Manager."""
        # Mock the Secrets Manager response
        mock_credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }
        mock_response = {"SecretString": json.dumps(mock_credentials)}
        mock_secretsmanager.get_secret_value.return_value = mock_response

        # Call the function
        credentials = get_postgres_credentials()

        # Verify results
        self.assertEqual(credentials, mock_credentials)
        mock_secretsmanager.get_secret_value.assert_called_once_with(
            SecretId="test-db-secret"
        )

    @patch("query_processor.query_processor.psycopg2")
    def test_get_postgres_connection(self, mock_psycopg2):
        """Test getting a PostgreSQL connection."""
        # Mock the psycopg2 connection
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn

        # Test credentials
        credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }

        # Call the function
        conn = get_postgres_connection(credentials)

        # Verify results
        self.assertEqual(conn, mock_conn)
        mock_psycopg2.connect.assert_called_once_with(
            host="test-host",
            port=5432,
            user="test-user",
            password="test-password",
            dbname="test-db"
        )

    @patch("query_processor.query_processor.client")
    def test_embed_query(self, mock_client):
        """Test embedding a query using Gemini."""
        # Mock the Gemini embedding response
        mock_embeddings = MagicMock()
        mock_embeddings.embeddings = [MagicMock()]
        mock_embeddings.embeddings[0].values = [0.1, 0.2, 0.3]
        mock_client.models.embed_content.return_value = mock_embeddings

        # Call the function
        result = embed_query("Test query")

        # Verify results
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_client.models.embed_content.assert_called_once()

    @patch("query_processor.query_processor.get_postgres_credentials")
    @patch("query_processor.query_processor.get_postgres_connection")
    def test_similarity_search(self, mock_get_conn, mock_get_creds):
        """Test similarity search using pgvector."""
        # Mock the PostgreSQL connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Mock credentials
        mock_get_creds.return_value = {"host": "test-host"}
        
        # Mock the query results
        mock_cursor.fetchall.return_value = [
            ("chunk-1", "doc-1", "user-1", "Content 1", {"page": 1}, "file1.pdf", 0.95),
            ("chunk-2", "doc-2", "user-1", "Content 2", {"page": 2}, "file2.pdf", 0.85)
        ]
        
        # Test query embedding
        query_embedding = [0.1, 0.2, 0.3]
        user_id = "user-1"
        
        # Call the function
        results = similarity_search(query_embedding, user_id, limit=2)
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "chunk-1")
        self.assertEqual(results[0]["document_id"], "doc-1")
        self.assertEqual(results[0]["content"], "Content 1")
        self.assertEqual(results[0]["file_name"], "file1.pdf")
        self.assertEqual(results[0]["similarity_score"], 0.95)
        
        # Verify SQL query execution
        mock_cursor.execute.assert_called_once()
        # Verify query contains the user_id parameter
        mock_cursor.execute.assert_called_with(unittest.mock.ANY, ("user-1", 2))

    @patch("query_processor.query_processor.client")
    def test_generate_response(self, mock_client):
        """Test generating a response using Gemini."""
        # Mock the Gemini response
        mock_result = MagicMock()
        mock_result.text = "This is the generated response."
        mock_client.models.generate_content.return_value = mock_result
        
        # Test query and relevant chunks
        query = "What is RAG?"
        relevant_chunks = [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1", 
                "user_id": "user-1",
                "content": "RAG stands for Retrieval-Augmented Generation",
                "metadata": {"page": 1},
                "file_name": "file1.pdf",
                "similarity_score": 0.95
            }
        ]
        
        # Call the function
        response = generate_response(MODEL_NAME, query, relevant_chunks)
        
        # Verify results
        self.assertEqual(response, "This is the generated response.")
        mock_client.models.generate_content.assert_called_once()
        
    def test_decimal_encoder(self):
        """Test the DecimalEncoder JSON encoder."""
        # Create an object with Decimal values
        obj = {
            "score1": Decimal("0.95"),
            "score2": Decimal("0.85"),
            "text": "test",
            "number": 42
        }
        
        # Encode the object to JSON
        json_str = json.dumps(obj, cls=DecimalEncoder)
        
        # Decode the JSON
        decoded_obj = json.loads(json_str)
        
        # Verify results
        self.assertEqual(decoded_obj["score1"], 0.95)
        self.assertEqual(decoded_obj["score2"], 0.85)
        self.assertEqual(decoded_obj["text"], "test")
        self.assertEqual(decoded_obj["number"], 42)

    def test_handler_healthcheck(self):
        """Test the Lambda handler for a health check."""
        # Create a health check event
        event = {"action": "healthcheck"}

        # Call the handler
        response = handler(event, {})

        # Verify results
        self.assertEqual(response["statusCode"], 200)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["message"], "Query processor is healthy")
        self.assertEqual(response_body["stage"], "test")

    def test_handler_missing_query(self):
        """Test the Lambda handler when the query is missing."""
        # Create an event with missing query
        event = {
            "body": json.dumps({
                "user_id": "user-1",
                "model_name": "gemini-2.0-flash"
            })
        }

        # Call the handler
        response = handler(event, {})

        # Verify results
        self.assertEqual(response["statusCode"], 400)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["message"], "Query is required")

    @patch("query_processor.query_processor.embed_query")
    @patch("query_processor.query_processor.similarity_search")
    @patch("query_processor.query_processor.generate_response")
    def test_handler_query_success(self, mock_generate, mock_search, mock_embed):
        """Test the Lambda handler for a successful query."""
        # Mock embedding
        mock_embed.return_value = [0.1, 0.2, 0.3]
        
        # Mock similarity search results
        mock_chunks = [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1", 
                "user_id": "user-1",
                "content": "RAG stands for Retrieval-Augmented Generation",
                "metadata": {"page": 1},
                "file_name": "file1.pdf",
                "similarity_score": 0.95
            }
        ]
        mock_search.return_value = mock_chunks
        
        # Mock response generation
        mock_generate.return_value = "RAG stands for Retrieval-Augmented Generation. It combines retrieval and generation techniques."
        
        # Create a query event
        event = {
            "body": json.dumps({
                "query": "What is RAG?",
                "user_id": "user-1",
                "model_name": "gemini-2.0-flash"
            })
        }

        # Call the handler
        response = handler(event, {})

        # Verify results
        self.assertEqual(response["statusCode"], 200)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["query"], "What is RAG?")
        self.assertEqual(response_body["response"], "RAG stands for Retrieval-Augmented Generation. It combines retrieval and generation techniques.")
        self.assertEqual(len(response_body["results"]), 1)
        self.assertEqual(response_body["count"], 1)
        
        # Verify function calls
        mock_embed.assert_called_once_with("What is RAG?")
        mock_search.assert_called_once_with([0.1, 0.2, 0.3], "user-1")
        mock_generate.assert_called_once_with("gemini-2.0-flash", "What is RAG?", mock_chunks)

    @patch("query_processor.query_processor.embed_query")
    def test_handler_error_handling(self, mock_embed):
        """Test the Lambda handler error handling."""
        # Mock embedding to raise an exception
        mock_embed.side_effect = Exception("Error embedding query")
        
        # Create a query event
        event = {
            "body": json.dumps({
                "query": "What is RAG?",
                "user_id": "user-1",
                "model_name": "gemini-2.0-flash"
            })
        }

        # Call the handler
        response = handler(event, {})

        # Verify results
        self.assertEqual(response["statusCode"], 500)
        response_body = json.loads(response["body"])
        self.assertTrue("Internal error" in response_body["message"])


if __name__ == "__main__":
    unittest.main()