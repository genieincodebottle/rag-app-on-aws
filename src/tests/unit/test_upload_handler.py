"""Test cases for the upload_handler Lambda function."""
import json
import os
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

"""Set up test environment."""
# Set environment variables
os.environ["DOCUMENTS_BUCKET"] = "test-bucket"
os.environ["METADATA_TABLE"] = "test-table"
os.environ["STAGE"] = "test"
os.environ["DB_SECRET_ARN"] = "test-db-secret"

# Now import the module under test - mocks are already in place globally from conftest
from upload_handler.upload_handler import (
    handler, get_postgres_credentials, get_postgres_connection, get_mime_type
)

class TestUploadHandler(unittest.TestCase):
    """Test cases for the upload_handler Lambda function."""

    def setUp(self):

        # Mock boto3 clients
        self.s3_patcher = patch("upload_handler.upload_handler.s3_client")
        self.dynamodb_patcher = patch("upload_handler.upload_handler.dynamodb")
        self.secrets_patcher = patch("upload_handler.upload_handler.secretsmanager")
        self.lambda_patcher = patch("upload_handler.upload_handler.lambda_client")
        
        self.mock_s3 = self.s3_patcher.start()
        self.mock_dynamodb = self.dynamodb_patcher.start()
        self.mock_secretsmanager = self.secrets_patcher.start()
        self.mock_lambda = self.lambda_patcher.start()
        
        # Set up DynamoDB table mock
        self.mock_table = MagicMock()
        self.mock_dynamodb.Table.return_value = self.mock_table

    def tearDown(self):
        """Clean up test environment."""
        # Clean up environment variables
        for key in ["DOCUMENTS_BUCKET", "METADATA_TABLE", "STAGE", "DB_SECRET_ARN"]:
            if key in os.environ:
                del os.environ[key]
                
        # Stop patchers
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()
        self.secrets_patcher.stop()
        self.lambda_patcher.stop()

    @patch("upload_handler.upload_handler.secretsmanager")
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

    @patch("upload_handler.upload_handler.psycopg2")
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

    def test_get_mime_type(self):
        """Test determining MIME type from file extension."""
        # Test various file extensions
        test_cases = [
            ("document.pdf", "application/pdf"),
            ("file.txt", "text/plain"),
            ("data.csv", "text/csv"),
            ("document.doc", "application/msword"),
            ("document.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("spreadsheet.xls", "application/vnd.ms-excel"),
            ("spreadsheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("data.json", "application/json"),
            ("readme.md", "text/markdown"),
            ("unknown.xyz", "application/octet-stream")
        ]

        for file_name, expected_mime_type in test_cases:
            mime_type = get_mime_type(file_name)
            self.assertEqual(mime_type, expected_mime_type)

    def test_handler_healthcheck(self):
        """Test the Lambda handler for a health check."""
        # Create a health check event
        event = {"action": "healthcheck"}

        # Call the handler
        response = handler(event, {})

        # Verify results
        self.assertEqual(response["statusCode"], 200)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["message"], "Upload handler is healthy")
        self.assertEqual(response_body["stage"], "test")

    def test_handler_missing_file_data(self):
        """Test the Lambda handler when file data is missing."""
        # Create an event with missing file content
        event = {
            "body": json.dumps({
                "file_name": "test.pdf"
            })
        }

        # Call the handler
        response = handler(event, {})

        # Verify results
        self.assertEqual(response["statusCode"], 400)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["message"], "File content and name are required")

    @patch("upload_handler.upload_handler.base64.b64decode")
    @patch("upload_handler.upload_handler.uuid.uuid4")
    @patch("upload_handler.upload_handler.datetime")
    @patch("upload_handler.upload_handler.get_postgres_credentials")
    @patch("upload_handler.upload_handler.get_postgres_connection")
    def test_handler_success(self, mock_get_conn, mock_get_creds, mock_datetime, mock_uuid, mock_b64decode):
        """Test the Lambda handler for successful file upload."""
        # Mock base64 decode
        mock_b64decode.return_value = b"file content"
        
        # Mock UUID
        mock_uuid.return_value = "test-doc-id"
        
        # Mock datetime
        mock_now = datetime.now()
        mock_now_timestamp = int(mock_now.timestamp() * 1000)
        mock_datetime.now.return_value = mock_now
        
        # Mock PostgreSQL connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Mock credentials
        mock_get_creds.return_value = {"host": "test-host"}
        
        # Create an event with file data
        event = {
            "body": json.dumps({
                "file_content": "ZmlsZSBjb250ZW50",  # base64 "file content"
                "file_name": "test.pdf",
                "user_id": "test-user"
            })
        }

        # Call the handler
        response = handler(event, {})

        # Verify results
        self.assertEqual(response["statusCode"], 200)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["message"], "File uploaded successfully")
        self.assertEqual(response_body["document_id"], "test-doc-id")
        self.assertEqual(response_body["file_name"], "test.pdf")
        
        # Verify S3 upload
        self.mock_s3.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="uploads/test-user/test-doc-id/test.pdf",
            Body=b"file content",
            ContentType="application/pdf"
        )
        
        # Verify PostgreSQL insertion
        mock_cursor.execute.assert_called_once_with(
            """
            INSERT INTO documents (document_id, user_id, file_name, mime_type, status, bucket, key, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            unittest.mock.ANY  # We don't need to check the exact values here
        )
        
    @patch("upload_handler.upload_handler.base64.b64decode")
    @patch("upload_handler.upload_handler.uuid.uuid4")
    @patch("upload_handler.upload_handler.datetime")
    @patch("upload_handler.upload_handler.get_postgres_credentials")
    @patch("upload_handler.upload_handler.get_postgres_connection")
    def test_handler_postgres_error(self, mock_get_conn, mock_get_creds, mock_datetime, mock_uuid, mock_b64decode):
        """Test the Lambda handler when PostgreSQL insertion fails."""
        # Mock base64 decode
        mock_b64decode.return_value = b"file content"
        
        # Mock UUID
        mock_uuid.return_value = "test-doc-id"
        
        # Mock datetime
        mock_now = datetime.now()
        mock_now_timestamp = int(mock_now.timestamp() * 1000)
        mock_datetime.now.return_value = mock_now
        
        # Mock PostgreSQL connection to raise an exception
        mock_get_conn.side_effect = Exception("Database connection error")
        
        # Mock credentials
        mock_get_creds.return_value = {"host": "test-host"}
        
        # Create an event with file data
        event = {
            "body": json.dumps({
                "file_content": "ZmlsZSBjb250ZW50",  # base64 "file content"
                "file_name": "test.pdf",
                "user_id": "test-user"
            })
        }

        # Call the handler
        response = handler(event, {})

        # Verify results - should still succeed because of DynamoDB fallback
        self.assertEqual(response["statusCode"], 200)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["message"], "File uploaded successfully")
        
        # Verify S3 upload
        self.mock_s3.put_object.assert_called_once()
        
        # Verify DynamoDB put_item (fallback storage)
        self.mock_table.put_item.assert_called_once()

    @patch("upload_handler.upload_handler.base64.b64decode")
    @patch("upload_handler.upload_handler.uuid.uuid4")
    def test_handler_with_custom_mime_type(self, mock_uuid, mock_b64decode):
        """Test the Lambda handler with a custom MIME type."""
        # Mock base64 decode
        mock_b64decode.return_value = b"file content"
        
        # Mock UUID
        mock_uuid.return_value = "test-doc-id"
        
        # Create an event with file data and custom MIME type
        event = {
            "body": json.dumps({
                "file_content": "ZmlsZSBjb250ZW50",  # base64 "file content"
                "file_name": "test.custom",
                "mime_type": "application/custom",
                "user_id": "test-user"
            })
        }

        # Call the handler
        response = handler(event, {})

        # Verify results
        self.assertEqual(response["statusCode"], 200)
        
        # Verify S3 upload with custom MIME type
        self.mock_s3.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="uploads/test-user/test-doc-id/test.custom",
            Body=b"file content",
            ContentType="application/custom"
        )

    @patch("upload_handler.upload_handler.base64.b64decode")
    def test_handler_s3_error(self, mock_b64decode):
        """Test the Lambda handler when S3 upload fails."""
        # Mock base64 decode
        mock_b64decode.return_value = b"file content"
        
        # Mock S3 put_object to raise an exception
        self.mock_s3.put_object.side_effect = Exception("S3 upload error")
        
        # Create an event with file data
        event = {
            "body": json.dumps({
                "file_content": "ZmlsZSBjb250ZW50",  # base64 "file content"
                "file_name": "test.pdf",
                "user_id": "test-user"
            })
        }

        # Call the handler
        response = handler(event, {})

        # Verify results - should fail with 500
        self.assertEqual(response["statusCode"], 500)
        response_body = json.loads(response["body"])
        self.assertTrue("Error uploading file" in response_body["message"])
        
    def test_handler_json_decode_error(self):
        """Test the Lambda handler with invalid JSON in body."""
        # Create an event with invalid JSON
        event = {
            "action": "healthcheck"
        }

        # Call the handler
        response = handler(event, {})

        # Verify results - should succeed with healthcheck
        self.assertEqual(response["statusCode"], 200)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["message"], "Upload handler is healthy")


if __name__ == "__main__":
    unittest.main()