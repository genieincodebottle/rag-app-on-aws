import streamlit as st
import requests
import json
import base64
import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

# Configuration from environment variables or defaults
DEFAULT_API_BASE = os.getenv("API_ENDPOINT")
API_ENDPOINTS = {
    "base_url": DEFAULT_API_BASE,
    "upload": os.getenv("UPLOAD_ENDPOINT", "/upload"),
    "query": os.getenv("QUERY_ENDPOINT", "/query"),
    "auth": os.getenv("AUTH_ENDPOINT", "/auth")
}
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "test-user")
DEFAULT_API_KEY = os.getenv("API_KEY", "")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "")
ENABLE_EVALUATION = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
# Set page config
st.set_page_config(
    page_title="RAG Application",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = DEFAULT_USER_ID
    
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []

if 'api_key' not in st.session_state:
    st.session_state.api_key = DEFAULT_API_KEY

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'access_token' not in st.session_state:
    st.session_state.access_token = None

if 'id_token' not in st.session_state:
    st.session_state.id_token = None

if 'refresh_token' not in st.session_state:
    st.session_state.refresh_token = None

if 'token_expiry' not in st.session_state:
    st.session_state.token_expiry = None

if 'user_email' not in st.session_state:
    st.session_state.user_email = None

# Function to get headers with correct authentication token format
def get_headers():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Access-Control-Allow-Origin": "*"  # For CORS support
    }
    
    # Add authentication token if available
    if st.session_state.id_token:
        # Use the id_token with Bearer prefix
        headers["Authorization"] = f"Bearer {st.session_state.id_token}"
    elif st.session_state.access_token:
        # Fallback to access_token with Bearer prefix
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    elif st.session_state.api_key:
        headers["x-api-key"] = st.session_state.api_key
    
    # Log headers for debugging (remove sensitive info in production)
    logger.info(f"Request headers (partial): {dict((k, v[:20] + '...' if k == 'Authorization' and v else v) for k, v in headers.items())}")
    
    return headers

# Function to validate email format
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

# Function to validate password strength
def is_strong_password(password):
    # At least 8 characters with 1 uppercase, 1 lowercase, 1 number, and 1 special character
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter."
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number."
    
    if not any(c in "!@#$%^&*()_-+=<>?/|" for c in password):
        return False, "Password must contain at least one special character."
    
    return True, "Password is strong."

# Function to register a new user
def register_user(email, password, name=""):
    payload = {
        "operation": "register",
        "email": email,
        "password": password,
        "name": name
    }
    
    auth_url = f"{API_ENDPOINTS['base_url']}{API_ENDPOINTS['auth']}"
    
    try:
        response = requests.post(
            auth_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Log response details for debugging
        logger.info(f"Register response status: {response.status_code}")
        try:
            logger.info(f"Register response body: {response.json()}")
        except:
            logger.info(f"Register response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            return True, result.get("message", "Registration successful.")
        else:
            return False, response.json().get("message", f"Error: {response.status_code}")
    
    except Exception as e:
        logger.error(f"Register error: {str(e)}")
        return False, f"Error: {str(e)}"

# Function to verify a user's email
def verify_user(email, confirmation_code):
    payload = {
        "operation": "verify",
        "email": email,
        "confirmation_code": confirmation_code
    }
    
    auth_url = f"{API_ENDPOINTS['base_url']}{API_ENDPOINTS['auth']}"
    
    try:
        response = requests.post(
            auth_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Log response details for debugging
        logger.info(f"Verify response status: {response.status_code}")
        try:
            logger.info(f"Verify response body: {response.json()}")
        except:
            logger.info(f"Verify response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            return True, result.get("message", "Verification successful.")
        else:
            return False, response.json().get("message", f"Error: {response.status_code}")
    
    except Exception as e:
        logger.error(f"Verify error: {str(e)}")
        return False, f"Error: {str(e)}"

# Function to login a user
def login_user(email, password):
    payload = {
        "operation": "login",
        "email": email,
        "password": password
    }
    
    auth_url = f"{API_ENDPOINTS['base_url']}{API_ENDPOINTS['auth']}"
    
    try:
        response = requests.post(
            auth_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Log response details for debugging
        #logger.info(f"Login response status: {response.status_code}")
        #try:
        #    logger.info(f"Login response body: {response.json()}")
        #except:
        #    logger.info(f"Login response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract tokens and expiry
            access_token = result.get("access_token")
            id_token = result.get("id_token")
            refresh_token = result.get("refresh_token")
            expires_in = result.get("expires_in", 3600)
            
            # Calculate expiry time
            expiry_time = datetime.now() + timedelta(seconds=expires_in)
            
            # Extract user ID from token
            try:
                # Decode the JWT token to get the subject (user ID)
                token_parts = id_token.split('.')
                if len(token_parts) == 3:
                    payload = json.loads(base64.b64decode(token_parts[1] + '==').decode('utf-8'))
                    user_id = payload.get('sub')
                    user_email = payload.get('email')
                else:
                    user_id = "unknown"
                    user_email = email
            except:
                user_id = "unknown"
                user_email = email
            
            return True, {
                "message": result.get("message", "Login successful."),
                "access_token": access_token,
                "id_token": id_token,
                "refresh_token": refresh_token,
                "token_expiry": expiry_time,
                "user_id": user_id,
                "user_email": user_email
            }
        else:
            return False, response.json().get("message", f"Error: {response.status_code}")
    
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return False, f"Error: {str(e)}"

# Function to refresh tokens
def refresh_token_func(refresh_token_value):
    payload = {
        "operation": "refresh_token",
        "refresh_token": refresh_token_value
    }
    
    auth_url = f"{API_ENDPOINTS['base_url']}{API_ENDPOINTS['auth']}"
    
    try:
        response = requests.post(
            auth_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Log response details for debugging
        logger.info(f"Refresh token response status: {response.status_code}")
        try:
            logger.info(f"Refresh token response body: {response.json()}")
        except:
            logger.info(f"Refresh token response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract tokens and expiry
            access_token = result.get("access_token")
            id_token = result.get("id_token")
            expires_in = result.get("expires_in", 3600)
            
            # Calculate expiry time
            expiry_time = datetime.now() + timedelta(seconds=expires_in)
            
            return True, {
                "message": result.get("message", "Tokens refreshed successfully."),
                "access_token": access_token,
                "id_token": id_token,
                "token_expiry": expiry_time
            }
        else:
            return False, response.json().get("message", f"Error: {response.status_code}")
    
    except Exception as e:
        logger.error(f"Refresh token error: {str(e)}")
        return False, f"Error: {str(e)}"

# Function to initiate forgot password
def forgot_password(email):
    payload = {
        "operation": "forgot_password",
        "email": email
    }
    
    auth_url = f"{API_ENDPOINTS['base_url']}{API_ENDPOINTS['auth']}"
    
    try:
        response = requests.post(
            auth_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Log response details for debugging
        logger.info(f"Forgot password response status: {response.status_code}")
        try:
            logger.info(f"Forgot password response body: {response.json()}")
        except:
            logger.info(f"Forgot password response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            return True, result.get("message", "Password reset initiated.")
        else:
            return False, response.json().get("message", f"Error: {response.status_code}")
    
    except Exception as e:
        logger.error(f"Forgot password error: {str(e)}")
        return False, f"Error: {str(e)}"

# Function to confirm forgot password
def confirm_forgot_password(email, confirmation_code, new_password):
    payload = {
        "operation": "confirm_forgot_password",
        "email": email,
        "confirmation_code": confirmation_code,
        "new_password": new_password
    }
    
    auth_url = f"{API_ENDPOINTS['base_url']}{API_ENDPOINTS['auth']}"
    
    try:
        response = requests.post(
            auth_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Log response details for debugging
        logger.info(f"Confirm forgot password response status: {response.status_code}")
        try:
            logger.info(f"Confirm forgot password response body: {response.json()}")
        except:
            logger.info(f"Confirm forgot password response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            return True, result.get("message", "Password reset confirmed.")
        else:
            return False, response.json().get("message", f"Error: {response.status_code}")
    
    except Exception as e:
        logger.error(f"Confirm forgot password error: {str(e)}")
        return False, f"Error: {str(e)}"

# Check if token needs to be refreshed
def check_token_refresh():
    if st.session_state.authenticated and st.session_state.token_expiry:
        # If token expires in less than 5 minutes, refresh it
        if st.session_state.token_expiry < datetime.now() + timedelta(minutes=5):
            if st.session_state.refresh_token:
                success, result = refresh_token_func(st.session_state.refresh_token)
                if success:
                    st.session_state.access_token = result["access_token"]
                    st.session_state.id_token = result["id_token"]
                    st.session_state.token_expiry = result["token_expiry"]
                    logger.info("Token refreshed successfully")
                    return True
                else:
                    # If refresh fails, log out the user
                    logger.warning("Token refresh failed, logging out user")
                    logout_user()
                    return False
            else:
                # If no refresh token, log out the user
                logger.warning("No refresh token available, logging out user")
                logout_user()
                return False
    return True

# Function to log out the user
def logout_user():
    st.session_state.authenticated = False
    st.session_state.access_token = None
    st.session_state.id_token = None
    st.session_state.refresh_token = None
    st.session_state.token_expiry = None
    st.session_state.user_email = None
    st.session_state.user_id = DEFAULT_USER_ID

# Function to test authentication token
def test_auth_token():
    """Test if the current authentication token is valid by making a lightweight API call"""
    if not st.session_state.authenticated:
        return False
    
    # Use the /auth endpoint with 'healthcheck' action
    payload = {"action": "healthcheck"}
    auth_url = f"{API_ENDPOINTS['base_url']}{API_ENDPOINTS['auth']}"
    
    try:
        response = requests.post(
            auth_url,
            json=payload,
            headers=get_headers()
        )
        
        # Check if response is successful (even 401 response means the auth endpoint is working)
        if response.status_code in [200, 401]:
            if response.status_code == 200:
                logger.info("Auth token test successful")
                return True
            else:
                logger.warning("Auth token test failed, token is invalid")
                return False
        else:
            logger.error(f"Auth endpoint error: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Auth token test error: {str(e)}")
        return False

# Application title and description
st.title("RAG App on AWS")
# Function to render the login page
def render_login_page():
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Forgot Password"])
    with tab1:
        email = st.text_input("Email:", key="login_email")
        password = st.text_input("Password:", type="password", key="login_password")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Login", use_container_width=True):
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    with st.spinner("Logging in..."):
                        success, result = login_user(email, password)
                        
                        if success:
                            st.success(result.get("message", "Login successful!"))
                            
                            # Set session state
                            st.session_state.authenticated = True
                            st.session_state.access_token = result["access_token"]
                            st.session_state.id_token = result["id_token"]
                            st.session_state.refresh_token = result["refresh_token"]
                            st.session_state.token_expiry = result["token_expiry"]
                            st.session_state.user_id = result["user_id"]
                            st.session_state.user_email = result["user_email"]
                            
                            # Rerun to show the main application
                            st.rerun()
                        else:
                            error_msg = result
                            if "UserNotConfirmed" in error_msg:
                                st.error("Email not verified. Please check your email for verification code.")
                                st.session_state.verify_email = email
                                st.session_state.current_tab = "Verify"
                            else:
                                st.error(error_msg)
        
    with tab2:
        reg_email = st.text_input("Email:", key="reg_email")
        reg_name = st.text_input("Name (optional):", key="reg_name")
        reg_password = st.text_input("Password:", type="password", key="reg_password")
        reg_confirm_password = st.text_input("Confirm Password:", type="password", key="reg_confirm_password")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Register", use_container_width=True):
                # Validate inputs
                if not reg_email:
                    st.error("Email is required.")
                elif not is_valid_email(reg_email):
                    st.error("Please enter a valid email address.")
                elif not reg_password:
                    st.error("Password is required.")
                elif reg_password != reg_confirm_password:
                    st.error("Passwords do not match.")
                else:
                    # Validate password strength
                    is_valid, message = is_strong_password(reg_password)
                    if not is_valid:
                        st.error(message)
                    else:
                        with st.spinner("Registering..."):
                            success, result = register_user(reg_email, reg_password, reg_name)
                            
                            if success:
                                st.success(result)
                                # Store email for verification
                                st.session_state.verify_email = reg_email
                                # Switch to verification tab
                                st.session_state.current_tab = "Verify"
                            else:
                                st.error(result)
    
    # Verification tab shown after registration
    if st.session_state.get("current_tab") == "Verify" or "verify_email" in st.session_state:
        with st.expander("Verify Your Email", expanded=True):
            st.write(f"Please enter the verification code sent to {st.session_state.get('verify_email', '')}")
            
            verification_code = st.text_input("Verification Code:", key="verification_code")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("Verify Email", use_container_width=True):
                    if not verification_code:
                        st.error("Please enter the verification code.")
                    else:
                        with st.spinner("Verifying..."):
                            success, result = verify_user(
                                st.session_state.get("verify_email", ""), 
                                verification_code
                            )
                            
                            if success:
                                st.success(result)
                                # Clear verification state
                                if "verify_email" in st.session_state:
                                    del st.session_state.verify_email
                                if "current_tab" in st.session_state:
                                    del st.session_state.current_tab
                            else:
                                st.error(result)
    
    with tab3:
        forgot_email = st.text_input("Email:", key="forgot_email")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Reset Password", use_container_width=True):
                if not forgot_email:
                    st.error("Please enter your email.")
                elif not is_valid_email(forgot_email):
                    st.error("Please enter a valid email address.")
                else:
                    with st.spinner("Processing request..."):
                        success, result = forgot_password(forgot_email)
                        
                        if success:
                            st.success(result)
                            # Store email for reset confirmation
                            st.session_state.reset_email = forgot_email
                            # Show reset confirmation form
                            st.session_state.show_reset_confirm = True
                        else:
                            st.error(result)
    
    # Reset confirmation form shown after forgot password
    if st.session_state.get("show_reset_confirm", False):
        with st.expander("Confirm Password Reset", expanded=True):
            st.write(f"Please enter the confirmation code sent to {st.session_state.get('reset_email', '')}")
            
            reset_code = st.text_input("Confirmation Code:", key="reset_code")
            reset_password = st.text_input("New Password:", type="password", key="reset_password")
            reset_confirm_password = st.text_input("Confirm New Password:", type="password", key="reset_confirm_password")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("Confirm Reset", use_container_width=True):
                    # Validate inputs
                    if not reset_code:
                        st.error("Please enter the confirmation code.")
                    elif not reset_password:
                        st.error("Please enter a new password.")
                    elif reset_password != reset_confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        # Validate password strength
                        is_valid, message = is_strong_password(reset_password)
                        if not is_valid:
                            st.error(message)
                        else:
                            with st.spinner("Resetting password..."):
                                success, result = confirm_forgot_password(
                                    st.session_state.get("reset_email", ""),
                                    reset_code,
                                    reset_password
                                )
                                
                                if success:
                                    st.success(result)
                                    # Clear reset state
                                    if "reset_email" in st.session_state:
                                        del st.session_state.reset_email
                                    if "show_reset_confirm" in st.session_state:
                                        del st.session_state.show_reset_confirm
                                else:
                                    st.error(result)

# Add a user profile in the sidebar
def render_user_sidebar():
    if st.session_state.authenticated:
        st.sidebar.markdown("---")
        st.sidebar.subheader("User Profile")
        st.sidebar.write(f"Email: {st.session_state.user_email}")
        
        # Calculate token expiry
        if st.session_state.token_expiry:
            now = datetime.now()
            expiry = st.session_state.token_expiry
            
            if expiry > now:
                minutes_left = int((expiry - now).total_seconds() / 60)
                if minutes_left > 60:
                    hours = minutes_left // 60
                    mins = minutes_left % 60
                    expiry_text = f"Token expires in {hours}h {mins}m"
                else:
                    expiry_text = f"Token expires in {minutes_left}m"
                
                if minutes_left < 5:
                    st.sidebar.warning(expiry_text)
                else:
                    st.sidebar.info(expiry_text)
            else:
                st.sidebar.error("Token expired. Please log in again.")
                
            # Check and refresh tokens automatically if needed
            check_token_refresh()
        
        # Logout button
        if st.sidebar.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()

# Function to upload document
def upload_document(file, user_id):
    # 🔐 Validate session
    if not check_token_refresh():
        st.error("Session expired. Please log in again.")
        logout_user()
        st.rerun()
        return False, "Authentication failed."

    # 📦 Prepare file payload
    payload = {
        "file_name": file.name,
        "mime_type": file.type or "application/octet-stream",
        "user_id": user_id,
        "file_content": base64.b64encode(file.getvalue()).decode()
    }

    # 🌐 Prepare API request
    upload_url = f"{API_ENDPOINTS['base_url']}{API_ENDPOINTS['upload']}"
    headers = get_headers()

    try:
        response = requests.post(upload_url, json=payload, headers=headers)
        logger.info(f"Upload response: {response.status_code}")
        return handle_response(response, file.name, user_id)

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return show_error("Exception during upload", str(e))


def handle_response(response, file_name, user_id):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if response.status_code == 200:
        result = response.json()
        document_id = result.get("document_id")
        st.session_state.uploaded_docs.append({
            "document_id": document_id,
            "file_name": file_name,
            "upload_time": timestamp,
            "status": "Uploaded",
            "user_id": user_id
        })

        show_success_tabs(file_name, document_id, timestamp, user_id, result)
        return True, result

    elif response.status_code == 401:
        st.error("Authentication failed. Please log in again.")
        logout_user()
        st.rerun()
        return False, "Authentication failed."

    else:
        return show_error(f"Upload failed (Error {response.status_code})", response)


def show_success_tabs(file_name, document_id, timestamp, user_id, result):
    tab1, tab2, tab3 = st.tabs(["Upload Summary", "API Response", "Recent Uploads"])
    with tab1:
        st.write(f"**File Name:** {file_name}")
        st.write(f"**Document ID:** {document_id}")
        st.write(f"**Upload Time:** {timestamp}")
        st.write(f"**User ID:** {user_id}")
        st.write("**Status:** Success")

    with tab2:
        st.json(result)

    with tab3:
        show_upload_history()


def show_error(title, details):
    tab1, tab2 = st.tabs(["Error Details", "Recent Uploads"])
    with tab1:
        st.subheader("Error Information")
        if isinstance(details, str):
            st.write(details)
        else:
            try:
                st.json(details.json())
            except:
                st.code(details.text)

    with tab2:
        show_upload_history()

    st.error(title)
    return False, title


def show_upload_history():
    if not st.session_state.get("uploaded_docs"):
        st.info("No upload history available.")
        return

    recent = sorted(st.session_state.uploaded_docs, key=lambda x: x.get('upload_time', ''), reverse=True)[:5]
    df = pd.DataFrame(recent)
    st.dataframe(df, use_container_width=True)

    if st.button("Clear Upload History", key="clear_history_upload_func"):
        st.session_state.uploaded_docs = []
        st.rerun()
    
# Function to query documents
def query_documents(selected_model, query_text, user_id, ground_truth=None, enable_evaluation=ENABLE_EVALUATION, web_search_with_mcp=False, mcp_server_url=None):
    # Ensure authentication is valid
    if not check_token_refresh():
        st.error("Your session has expired. Please log in again.")
        logout_user()
        st.rerun()
        return False, "Authentication failed. Please log in again."
    
    # Prepare the request payload
    payload = {
        "query": query_text, 
        "user_id": user_id,
        "enable_evaluation": enable_evaluation,
        "model_name": selected_model,
        "web_search_with_mcp": web_search_with_mcp,
        "mcp_server_url": mcp_server_url
    }
    
    # Add ground truth if provided
    if ground_truth:
        payload["ground_truth"] = ground_truth
    
    # Send request to API Gateway
    try:
        query_url = f"{API_ENDPOINTS['base_url']}{API_ENDPOINTS['query']}"
        
        # Log request details
        logger.info(f"Sending query request to: {query_url}")
        logger.info(f"Query payload: {payload}")
        
        # Show what's being sent
        st.write("Sending request to:", query_url)
        st.json(payload)
        
        response = requests.post(
            query_url,
            json=payload,
            headers=get_headers()
        )
        print(f"RESPONSE {response}")
        # Log response details
        logger.info(f"Query response status: {response.status_code}")
        logger.info(f"Query response headers: {dict(response.headers)}")
        try:
            logger.info(f"Query response body: {response.json()}")
        except:
            logger.info(f"Query response text: {response.text}")
        
        # Display raw response for debugging
        st.write(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Add to query history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.query_history.append({
                "query": query_text,
                "timestamp": timestamp,
                "num_results": len(result.get("results", [])),
                "has_evaluation": "evaluation" in result and bool(result["evaluation"])
            })
            
            return True, result
        elif response.status_code == 401:
            # Authentication failed
            error_message = "Authentication token expired or invalid."
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_message = error_data["message"]
            except:
                pass
            
            st.warning(f"{error_message} Please log in again.")
            logout_user()
            st.rerun()
            return False, "Authentication failed. Please log in again."
        else:
            # Handle other errors without logging out
            error_message = f"Error: {response.status_code}"
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_message += f" - {error_data['message']}"
            except:
                error_message += f" - {response.text}"
            
            return False, error_message
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return False, f"Error: {str(e)}"

# Function to create evaluation chart
def create_evaluation_chart(eval_results):
    """Create a visualization for RAG evaluation metrics"""
    # Define friendly metric names
    metric_names = {
        "answer_relevancy": "Answer Relevancy",
        "faithfulness": "Faithfulness",
        "context_precision": "Context Precision"
    }
    
    # Colors for each metric
    colors = {
        "answer_relevancy": "#2563eb",  # Blue
        "faithfulness": "#16a34a",      # Green
        "context_precision": "#d97706"  # Amber
    }
    
    # Prepare data for chart
    x_values = [metric_names.get(k, k) for k in eval_results.keys()]
    y_values = list(eval_results.values())
    bar_colors = [colors.get(k, "#6b7280") for k in eval_results.keys()]
    
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=x_values,
            y=y_values,
            text=[f"{v:.2f}" for v in y_values],
            textposition="auto",
            marker_color=bar_colors
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="RAG Evaluation Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score (0-1)",
        yaxis_range=[0, 1],
        template="plotly_white"
    )
    
    return fig

# Define sidebar
def render_sidebar():
    st.sidebar.title("📚 App Navigation")
    selected_model =""
    # Determine current page
    if st.session_state.get("authenticated", False):
        with st.sidebar.expander("⚙️ API Settings", expanded=False):
            new_url = st.text_input("Base API URL", value=API_ENDPOINTS["base_url"])
            if st.button("Save Settings"):
                API_ENDPOINTS["base_url"] = new_url
                st.success("✅ Settings saved for this session.")
        selected_model = st.selectbox(
            "Select Model",
            options=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-flash-preview-04-17"],
            index=0,
            help="Select the model to use"
        )
        st.sidebar.markdown("---")
        page = st.sidebar.radio("Select an action:", ["Upload Documents", "Query Documents", "View Documents"])
        render_user_sidebar()  # Show user info
    else:
        page = "Login"

    st.sidebar.markdown("---")
    st.sidebar.caption("🔖 Version: RAG App on AWS v0.1")

    return page, selected_model

# Main application logic
def main():
    # Render sidebar and get selected page
    page, selected_model = render_sidebar()
    
    # If not authenticated, show login page and return
    if not st.session_state.authenticated and page != "Login":
        render_login_page()
        return
    elif page == "Login":
        render_login_page()
        return

    # Upload Documents Page
    if page == "Upload Documents":
        st.header("Upload Documents")
        # File uploader with multiple file types
        uploaded_file = st.file_uploader(
            "Choose a file to upload", 
            type=["pdf", "txt", "docx", "doc", "csv", "xlsx", "json", "md"],
            help="Select a document to upload. Supported formats include PDF, text, Word documents, spreadsheets, and more."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Custom user ID for this upload
            upload_user_id = st.text_input(
                "User ID for this upload:",
                value=st.session_state.user_id,
                help="Documents will be associated with this user ID"
            )
        
        with col2:
            st.write("Upload Status:")
            upload_status = st.empty()
            upload_status.info("No file selected yet")
        
        if uploaded_file is not None:
            # Update status
            upload_status.info("File selected, ready to upload")
            
            st.write("File Details:")
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "MIME type": uploaded_file.type or "application/octet-stream"
            }
            st.json(file_details)
            
            # Upload button with user confirmation
            st.write("Ready to upload?")
            if st.button("Upload Document"):
                upload_status.warning("Uploading in progress...")
                
                with st.spinner("Uploading document..."):
                    success, result = upload_document(uploaded_file, upload_user_id)
                    
                    if success:
                        upload_status.success("Upload complete!")
                        st.success(f"Document uploaded successfully! Document ID: {result.get('document_id')}")
                        
                    else:
                        upload_status.error("Upload failed!")
                        st.error(result)

    # Query Documents Page
    elif page == "Query Documents":
        st.header("Query Documents")

        # Add RAG Evaluation configuration
        eval_expander = st.expander("RAG Evaluation Settings", expanded=False)
        with eval_expander:
            enable_evaluation = st.toggle("Enable RAG Evaluation", value=ENABLE_EVALUATION, 
                                        help="Evaluate the quality of RAG responses using metrics like faithfulness and relevancy")
            
            use_ground_truth = st.checkbox("Provide Ground Truth", value=False,
                                        help="Add a ground truth answer to compare with the generated response")
            
            st.info("RAG evaluation uses Gemini to assess the quality of responses based on retrieved context.")
        
        agentic_expander = st.expander("Agentic RAG Settings", expanded=False)
        with agentic_expander:
            web_search_with_mcp = st.checkbox("Web Search with MCP", value=False,
                                        help="Always use web search regardless of traditional RAG quality")
            
            st.info("When enabled, the system will always perform web search using MCP Server in addition to traditional RAG.")
            if web_search_with_mcp:
                mcp_server_url = st.text_input( "Web Search MCP Server URL", 
                                                value="",
                                                help="URL of the MCP server for web search")

        # Two column layout for query input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Query input with placeholder
            query = st.text_area(
                "Enter your question:",
                placeholder="e.g., What are the key points in the latest financial report?",
                height=100
            )
            
            # Ground truth input if enabled
            ground_truth = None
            if use_ground_truth:
                ground_truth = st.text_area(
                    "Ground Truth Answer (optional):",
                    placeholder="Enter the correct answer for evaluation purposes.",
                    height=100
                )
        
        with col2:
            # User ID selection
            query_user_id = st.text_input(
                "User ID for query:",
                value=st.session_state.user_id,
                help="Retrieves documents associated with this user ID"
            )
            
            # Submit button
            submit_button = st.button("Submit Query", use_container_width=True)
            
            # Clear results button
            if 'last_query_result' in st.session_state:
                clear_button = st.button("Clear Results", use_container_width=True)
                if clear_button:
                    del st.session_state.last_query_result
                    st.rerun()
        
        # Execute query when submit button is clicked
        if submit_button:
            if not query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Processing query..."):
                    success, result = query_documents(
                        selected_model,
                        query, 
                        query_user_id, 
                        ground_truth=ground_truth,  
                        enable_evaluation=enable_evaluation,
                        web_search_with_mcp=web_search_with_mcp,
                        mcp_server_url  = mcp_server_url if web_search_with_mcp else None
                    )
                    
                    # Store result in session state
                    if success:
                        st.session_state.last_query_result = result
        
        # Display query results if available
        if 'last_query_result' in st.session_state:
            result = st.session_state.last_query_result
    
            # Create tabs for different views of the results
            if "evaluation" in result and result["evaluation"]:
                tab1, tab2, tab3 = st.tabs(["AI Response", "Document Details", "Evaluation"])
            else:
                tab1, tab2 = st.tabs(["AI Response", "Document Details"])
        
            with tab1:
                # Display the AI-generated response
                if "response" in result:
                    response_data = json.loads(result["response"])
                    st.markdown(response_data.get("answer", "No answer found."))
                else:
                    st.info("No AI-generated response available.")
            
            with tab2:
                # Display document results
                if "results" in result and result["results"]:
                    st.markdown("### Retrieved Documents")
                    st.write(f"Found {len(result['results'])} relevant documents")
                    
                    # Create expandable sections for each document
                    for i, doc in enumerate(result["results"]):
                        score = doc.get('similarity_score', 0)
                        score_display = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                        doc_name = doc.get('file_name', doc.get('document_id', f'Document {i+1}'))
                        
                        with st.expander(f"{doc_name} - Relevance Score: {score_display}"):
                            # Two columns for metadata and content
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("#### Document Metadata")
                                # Filter out large fields like vectors
                                metadata = {k: v for k, v in doc.items() 
                                           if k not in ['embedding_vector'] and not isinstance(v, list) or len(v) < 100}
                                st.json(metadata)
                            
                            with col2:
                                st.markdown("#### Document Content")
                                if "content" in doc:
                                    st.write(doc["content"])
                                else:
                                    st.info("No content available")
                else:
                    st.info("No relevant documents found. Try a different query or upload more documents.")
            
            if "evaluation" in result and result["evaluation"]:
                with tab3:
                    st.markdown("### RAG Response Evaluation")
                    
                    eval_results = result["evaluation"]
                    
                    # Display metrics
                    metrics_cols = st.columns(len(eval_results))
                    for i, (metric, value) in enumerate(eval_results.items()):
                        with metrics_cols[i]:
                            # Format metric name for display
                            display_name = " ".join(word.capitalize() for word in metric.split("_"))
                            st.metric(display_name, f"{value:.2f}")
                    
                    # Display evaluation chart
                    chart = create_evaluation_chart(eval_results)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Explain metrics
                    with st.expander("Understanding Evaluation Metrics"):
                        st.markdown("""
                        ### RAG Evaluation Metrics Explained
                        
                        - **Answer Relevancy (0-1)**: Measures how directly the answer addresses the question.
                        
                        - **Faithfulness (0-1)**: Measures how factually accurate the answer is based only on the provided context.
                        
                        - **Context Precision (0-1)**: When ground truth is provided, measures how well the answer aligns with the known correct answer.
                        
                        A higher score indicates better performance. Scores above 0.7 are generally considered good.
                        """)

        # Display query history
        with st.expander("Query History", expanded=False):
            if st.session_state.query_history:
                df = pd.DataFrame(st.session_state.query_history)
                st.dataframe(df)
                
                if st.button("Clear Query History"):
                    st.session_state.query_history = []
                    st.rerun()
            else:
                st.info("No query history available.")

    # View Documents Page
    elif page == "View Documents":
        st.header("Document Management")
        # Settings for document retrieval
        col1, col2 = st.columns(2)
        
        with col1:
            view_user_id = st.text_input(
                "User ID to view documents:",
                value=st.session_state.user_id,
                help="Filter documents by this user ID"
            )
        
        with col2:
            # This would be implemented with a real API call in production
            if st.button("Refresh Document List"):
                with st.spinner("Fetching documents..."):
                    # Here you would typically query the API for documents
                    # For now, we'll just use what's in session state
                    filtered_docs = [
                        doc for doc in st.session_state.uploaded_docs 
                        if doc.get('user_id') == view_user_id
                    ]
                    st.session_state.filtered_docs = filtered_docs
                    time.sleep(1)
                    st.success(f"Found {len(filtered_docs)} documents.")
        
        # Display documents
        if 'filtered_docs' in st.session_state and st.session_state.filtered_docs:
            st.subheader(f"Documents for User: {view_user_id}")
            
            # Create a nicer display using a DataFrame
            df = pd.DataFrame(st.session_state.filtered_docs)
            
            # Add styling
            st.dataframe(
                df,
                column_config={
                    "document_id": st.column_config.TextColumn("Document ID"),
                    "file_name": st.column_config.TextColumn("File Name"),
                    "upload_time": st.column_config.DatetimeColumn("Upload Time"),
                    "status": st.column_config.TextColumn("Status"),
                },
                use_container_width=True
            )
       
        elif 'filtered_docs' in st.session_state:
            st.info(f"No documents found for user: {view_user_id}")
        else:
            st.info("Click 'Refresh Document List' to fetch documents.")

# Run the app
if __name__ == "__main__":
    main()