"""
Database connectivity test script.
This script tests connectivity to the PostgreSQL database and validates DNS resolution.
"""
import json
import boto3
import logging
import argparse
import psycopg2
import socket
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_secret(secret_arn):
    """
    Get database credentials from AWS Secrets Manager.
    
    Args:
        secret_arn (str): ARN of the secret in Secrets Manager
    
    Returns:
        dict: Database credentials
    """
    try:
        secretsmanager = boto3.client('secretsmanager')
        secret_response = secretsmanager.get_secret_value(
            SecretId=secret_arn
        )
        secret = json.loads(secret_response['SecretString'])
        return secret
    except Exception as e:
        logger.error(f"Error getting database credentials: {str(e)}")
        raise e

def check_dns_resolution(host):
    """
    Check if a hostname can be resolved to an IP address.
    
    Args:
        host (str): Hostname to resolve
    
    Returns:
        tuple: (bool, str) - Success flag and IP address or error message
    """
    try:
        ip_address = socket.gethostbyname(host)
        return True, ip_address
    except socket.gaierror as e:
        return False, str(e)

def test_database_connection(credentials, timeout=5):
    """
    Test connection to the PostgreSQL database.
    
    Args:
        credentials (dict): Database credentials
        timeout (int): Connection timeout in seconds
    
    Returns:
        tuple: (bool, str) - Success flag and connection info or error message
    """
    try:
        conn = psycopg2.connect(
            host=credentials['host'],
            port=credentials.get('port', 5432),
            user=credentials['username'],
            password=credentials['password'],
            dbname=credentials.get('dbname', 'postgres'),
            connect_timeout=timeout
        )
        
        # Get connection info
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.execute("SELECT current_database();")
        current_db = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        return True, f"Connected to {current_db} on {credentials['host']}. PostgreSQL version: {version}"
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Test connectivity to PostgreSQL database')
    parser.add_argument('--secret-arn', required=True, help='ARN of the database credentials secret')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum number of retry attempts')
    parser.add_argument('--retry-delay', type=int, default=5, help='Delay between retries in seconds')
    args = parser.parse_args()
    
    try:
        # Get database credentials
        logger.info(f"Getting database credentials from {args.secret_arn}")
        credentials = get_db_secret(args.secret_arn)
        
        # Check DNS resolution
        logger.info(f"Checking DNS resolution for {credentials['host']}")
        dns_success, dns_result = check_dns_resolution(credentials['host'])
        
        if dns_success:
            logger.info(f"DNS resolution successful: {credentials['host']} resolves to {dns_result}")
        else:
            logger.error(f"DNS resolution failed: {dns_result}")
            
            # Retry DNS resolution
            retry_count = 0
            while not dns_success and retry_count < args.max_retries:
                retry_count += 1
                logger.info(f"Retrying DNS resolution ({retry_count}/{args.max_retries})...")
                time.sleep(args.retry_delay)
                dns_success, dns_result = check_dns_resolution(credentials['host'])
                
                if dns_success:
                    logger.info(f"DNS resolution successful after {retry_count} retries: {credentials['host']} resolves to {dns_result}")
                elif retry_count == args.max_retries:
                    logger.error(f"DNS resolution failed after {args.max_retries} retries: {dns_result}")
        
        # Test database connection
        logger.info(f"Testing database connection to {credentials['host']}")
        conn_success, conn_result = test_database_connection(credentials)
        
        if conn_success:
            logger.info(f"Database connection successful: {conn_result}")
        else:
            logger.error(f"Database connection failed: {conn_result}")
            
            # Retry database connection
            retry_count = 0
            while not conn_success and retry_count < args.max_retries:
                retry_count += 1
                logger.info(f"Retrying database connection ({retry_count}/{args.max_retries})...")
                time.sleep(args.retry_delay)
                conn_success, conn_result = test_database_connection(credentials)
                
                if conn_success:
                    logger.info(f"Database connection successful after {retry_count} retries: {conn_result}")
                elif retry_count == args.max_retries:
                    logger.error(f"Database connection failed after {args.max_retries} retries: {conn_result}")
        
        # Summarize results
        logger.info("\n----- Connectivity Test Summary -----")
        logger.info(f"Database Host: {credentials['host']}")
        logger.info(f"DNS Resolution: {'Success' if dns_success else 'Failed'}")
        if dns_success:
            logger.info(f"IP Address: {dns_result}")
        else:
            logger.info(f"DNS Error: {dns_result}")
        
        logger.info(f"Database Connection: {'Success' if conn_success else 'Failed'}")
        if conn_success:
            logger.info(f"Connection Info: {conn_result}")
        else:
            logger.info(f"Connection Error: {conn_result}")
        
        # Exit with appropriate status code
        if dns_success and conn_success:
            logger.info("All tests passed! Database is reachable and connectable.")
            return 0
        else:
            logger.error("One or more tests failed. Please check the logs for details.")
            return 1
        
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())