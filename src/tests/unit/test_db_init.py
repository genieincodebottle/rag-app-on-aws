"""Test cases for the db_init Lambda function."""
import json
import os
import socket
import unittest
from unittest.mock import MagicMock, patch, call

"""Set up test environment."""
# Set environment variables
os.environ["DB_SECRET_ARN"] = "test-db-secret"
os.environ["STAGE"] = "test"
os.environ["MAX_RETRIES"] = "3"
os.environ["RETRY_DELAY"] = "1"  # Short delay for tests

# Now import the module under test - mocks are already in place globally from conftest
from db_init.db_init import (
    handler, get_postgres_credentials, check_dns_resolution,
    create_database_if_not_exists, initialize_database
)

class TestDbInit(unittest.TestCase):
    """Test cases for the db_init Lambda function."""

    def tearDown(self):
        """Clean up test environment."""
        # Clean up environment variables
        for key in ["DB_SECRET_ARN", "STAGE", "MAX_RETRIES", "RETRY_DELAY"]:
            if key in os.environ:
                del os.environ[key]

    @patch("db_init.db_init.secretsmanager")
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

    @patch("db_init.db_init.socket.gethostbyname")
    def test_check_dns_resolution_success(self, mock_gethostbyname):
        """Test successful DNS resolution."""
        # Mock the socket.gethostbyname function
        mock_gethostbyname.return_value = "192.168.1.1"

        # Call the function
        result = check_dns_resolution("test-host")

        # Verify results
        self.assertTrue(result)
        mock_gethostbyname.assert_called_once_with("test-host")

    @patch("db_init.db_init.socket.gethostbyname")
    def test_check_dns_resolution_failure(self, mock_gethostbyname):
        """Test failed DNS resolution."""
        # Mock the socket.gethostbyname function to raise an exception
        mock_gethostbyname.side_effect = socket.gaierror()

        # Call the function
        result = check_dns_resolution("test-host")

        # Verify results
        self.assertFalse(result)
        mock_gethostbyname.assert_called_once_with("test-host")
        
    @patch("db_init.db_init.psycopg2")
    @patch("db_init.db_init.check_dns_resolution")
    @patch("db_init.db_init.time.sleep")
    def test_create_database_if_not_exists_success(self, mock_sleep, mock_check_dns, mock_psycopg2):
        """Test creating a database successfully."""
        # Mock DNS resolution
        mock_check_dns.return_value = True
        
        # Mock the psycopg2 connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        # Mock cursor fetchone result (database does not exist)
        mock_cursor.fetchone.return_value = None
        
        # Test credentials
        credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }
        
        # Call the function
        result = create_database_if_not_exists(credentials, "test-db")
        
        # Verify results
        self.assertTrue(result)
        mock_check_dns.assert_called_once_with("test-host")
        mock_psycopg2.connect.assert_called_once_with(
            host="test-host",
            port=5432,
            user="test-user",
            password="test-password",
            dbname="postgres",
            connect_timeout=10
        )
        
        # Verify database creation
        mock_cursor.execute.assert_any_call("SELECT 1 FROM pg_database WHERE datname = 'test-db'")
        mock_cursor.execute.assert_any_call("CREATE DATABASE test-db")
        
    @patch("db_init.db_init.psycopg2")
    @patch("db_init.db_init.check_dns_resolution")
    @patch("db_init.db_init.time.sleep")
    def test_create_database_if_not_exists_already_exists(self, mock_sleep, mock_check_dns, mock_psycopg2):
        """Test when database already exists."""
        # Mock DNS resolution
        mock_check_dns.return_value = True
        
        # Mock the psycopg2 connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        # Mock cursor fetchone result (database exists)
        mock_cursor.fetchone.return_value = (1,)
        
        # Test credentials
        credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }
        
        # Call the function
        result = create_database_if_not_exists(credentials, "test-db")
        
        # Verify results
        self.assertTrue(result)
        mock_check_dns.assert_called_once_with("test-host")
        mock_psycopg2.connect.assert_called_once()
        
        # Verify database check but no creation
        mock_cursor.execute.assert_called_once_with("SELECT 1 FROM pg_database WHERE datname = 'test-db'")
        
    @patch("db_init.db_init.check_dns_resolution")
    @patch("db_init.db_init.time.sleep")
    def test_create_database_if_not_exists_dns_failure(self, mock_sleep, mock_check_dns):
        """Test handling DNS resolution failure with retries."""
        # Mock DNS resolution to fail
        mock_check_dns.return_value = False
        
        # Test credentials
        credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }
        
        # Call the function (max retries is 3 from setup)
        result = create_database_if_not_exists(credentials, "test-db")
        
        # Verify results
        self.assertFalse(result)
        self.assertEqual(mock_check_dns.call_count, 4)  # Initial + 3 retries
        self.assertEqual(mock_sleep.call_count, 3)  # Sleep between retries
        
    @patch("db_init.db_init.psycopg2")
    @patch("db_init.db_init.check_dns_resolution")
    @patch("db_init.db_init.time.sleep")
    def test_initialize_database_success(self, mock_sleep, mock_check_dns, mock_psycopg2):
        """Test successful database initialization."""
        # Mock DNS resolution
        mock_check_dns.return_value = True
        
        # Mock the psycopg2 connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
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
        result = initialize_database(credentials)
        
        # Verify results
        self.assertTrue(result)
        mock_check_dns.assert_called_once_with("test-host")
        mock_psycopg2.connect.assert_called_once_with(
            host="test-host",
            port=5432,
            user="test-user",
            password="test-password",
            dbname="test-db",
            connect_timeout=10
        )
        
        # Verify SQL executions
        self.assertTrue(mock_cursor.execute.call_count >= 7)  # Several SQL statements are executed
        # Check that pgvector extension is created
        mock_cursor.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS vector")
        
    @patch("db_init.db_init.check_dns_resolution")
    @patch("db_init.db_init.time.sleep")
    def test_initialize_database_dns_failure(self, mock_sleep, mock_check_dns):
        """Test handling DNS resolution failure with retries in initialize_database."""
        # Mock DNS resolution to fail
        mock_check_dns.return_value = False
        
        # Test credentials
        credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }
        
        # Call the function (max retries is 3 from setup)
        result = initialize_database(credentials)
        
        # Verify results
        self.assertFalse(result)
        self.assertEqual(mock_check_dns.call_count, 4)  # Initial + 3 retries
        self.assertEqual(mock_sleep.call_count, 3)  # Sleep between retries
        
    @patch("db_init.db_init.psycopg2")
    @patch("db_init.db_init.check_dns_resolution")
    def test_initialize_database_connection_error(self, mock_check_dns, mock_psycopg2):
        """Test handling database connection errors."""
        # Mock DNS resolution to succeed
        mock_check_dns.return_value = True
        
        # Mock the psycopg2 connection to raise an error
        mock_psycopg2.OperationalError = Exception
        mock_psycopg2.connect.side_effect = mock_psycopg2.OperationalError("Connection refused")
        
        # Test credentials
        credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }
        
        # Call the function (max retries is 3 from setup)
        with patch("db_init.db_init.time.sleep") as mock_sleep:
            result = initialize_database(credentials)
        
        # Verify results
        self.assertFalse(result)
        mock_check_dns.assert_called_with("test-host")                  # was called at least once
        self.assertEqual(mock_check_dns.call_count, 4)                  # was called exactly 4 times
        mock_check_dns.assert_has_calls([call("test-host")] * 4)        # was called 4 times with same arg
        self.assertEqual(mock_psycopg2.connect.call_count, 4)  # Initial + 3 retries
        self.assertEqual(mock_sleep.call_count, 3)  # Sleep between retries
        
    @patch("db_init.db_init.get_postgres_credentials")
    @patch("db_init.db_init.create_database_if_not_exists")
    @patch("db_init.db_init.initialize_database")
    def test_handler_success(self, mock_initialize, mock_create_db, mock_get_creds):
        """Test the Lambda handler for successful execution."""
        # Mock credential retrieval
        mock_credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }
        mock_get_creds.return_value = mock_credentials
        
        # Mock database creation and initialization
        mock_create_db.return_value = True
        mock_initialize.return_value = True
        
        # Call the handler
        response = handler({}, {})
        
        # Verify results
        self.assertEqual(response["statusCode"], 200)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["message"], "Database initialization completed successfully")
        
        # Verify function calls
        mock_get_creds.assert_called_once()
        mock_create_db.assert_called_once_with(mock_credentials, "test-db")
        mock_initialize.assert_called_once_with(mock_credentials)
        
    @patch("db_init.db_init.get_postgres_credentials")
    @patch("db_init.db_init.create_database_if_not_exists")
    def test_handler_create_db_failure(self, mock_create_db, mock_get_creds):
        """Test the Lambda handler when database creation fails."""
        # Mock credential retrieval
        mock_credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }
        mock_get_creds.return_value = mock_credentials
        
        # Mock database creation failure
        mock_create_db.return_value = False
        
        # Call the handler
        response = handler({}, {})
        
        # Verify results
        self.assertEqual(response["statusCode"], 500)
        response_body = json.loads(response["body"])
        self.assertEqual(
            response_body["message"],
            "Failed to create database. Please check that the RDS instance is available."
        )
        
    @patch("db_init.db_init.get_postgres_credentials")
    @patch("db_init.db_init.create_database_if_not_exists")
    @patch("db_init.db_init.initialize_database")
    def test_handler_initialize_db_failure(self, mock_initialize, mock_create_db, mock_get_creds):
        """Test the Lambda handler when database initialization fails."""
        # Mock credential retrieval
        mock_credentials = {
            "host": "test-host",
            "port": 5432,
            "username": "test-user",
            "password": "test-password",
            "dbname": "test-db"
        }
        mock_get_creds.return_value = mock_credentials
        
        # Mock database creation success but initialization failure
        mock_create_db.return_value = True
        mock_initialize.return_value = False
        
        # Call the handler
        response = handler({}, {})
        
        # Verify results
        self.assertEqual(response["statusCode"], 500)
        response_body = json.loads(response["body"])
        self.assertEqual(
            response_body["message"],
            "Failed to initialize database schema. Please check logs for details."
        )
        
    def test_handler_healthcheck(self):
        """Test the Lambda handler for a health check."""
        # Override environment variable to ensure it's correct
        os.environ["STAGE"] = "test"
        
        # Create a health check event
        event = {"action": "healthcheck"}

        # Call the handler
        response = handler(event, {})

        # Verify results
        self.assertEqual(response["statusCode"], 200)
        response_body = json.loads(response["body"])
        self.assertEqual(response_body["message"], "DB initialization function is healthy")
        self.assertEqual(response_body["stage"], "test")


if __name__ == "__main__":
    unittest.main()