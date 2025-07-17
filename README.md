[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/genieincodebottle-rag-app-on-aws-badge.png)](https://mseep.ai/app/genieincodebottle-rag-app-on-aws)

<div align="center">
    <a target="_blank" href="https://www.youtube.com/@genieincodebottle"><img src="https://img.shields.io/badge/YouTube-11.4K-blue"></a>&nbsp;
    <a target="_blank" href="https://www.linkedin.com/in/rajesh-srivastava"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://www.instagram.com/genieincodebottle/"><img src="https://img.shields.io/badge/51K-C13584?style=flat-square&labelColor=C13584&logo=instagram&logoColor=white&link=https://www.instagram.com/eduardopiresbr/"></a>
    <a target="_blank" href="https://x.com/zero2nn"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=%20%40zero2nn"></a>
    <h3><a target="_blank" href="https://github.com/genieincodebottle/generative-ai/blob/main/GenAI_Roadmap.md">👉 GenAI Roadmap - 2025</a></h3>
</div>

<div align="center">
    <img src="./images/rag-app.png" alt="RAG App on"/>
</div>

### End-to-End RAG App with Evaluation on AWS, Integrating Web Search via Remote MCP Server

Terraform-based **Infrastructure as Code (IaC)** to deploy a complete AWS backend for a Retrieval-Augmented Generation (RAG) application. It integrates with Google’s free-tier Gemini Pro and Embedding models for AI powered document querying and includes a Streamlit UI with token-based authentication for interacting with the app.

👉 Related Remote MCP Server: [Web Search using SerpAPI](https://github.com/genieincodebottle/rag-app-on-aws/tree/main/mcp_servers)  Remote MCP Server based on Streaming Http Transport protocol for Real Time Web Search. It's located within the `mcp_servers/` directory of this repository.

👉 Related UI: [RAG UI (Streamlit Frontend)](https://github.com/genieincodebottle/rag-app-on-aws/tree/main/rag_ui)  A Streamlit-based frontend application designed to interact with the backend infrastructure deployed by this project. It's located within the `rag_ui/` directory of this repository.

💰 Estimated cost: ~$3 (~₹250) to experiment without the AWS Free Tier, primarily for RDS and NAT Gateway if active.

🎥 YouTube Video: Walkthrough on setting up the application, building, deploying, and running it end-to-end 👇

[![Watch the video](https://img.youtube.com/vi/x2P4Ee6PYNg/0.jpg)](https://www.youtube.com/watch?v=x2P4Ee6PYNg)

---

### 🔍 Overview

This repository contains the complete Terraform codebase for provisioning and managing the AWS infrastructure that powers a RAG application. It allows users to upload documents, which are then processed, embedded, and stored for efficient semantic search and AI-driven querying.

📌 Key features include:
- **IaC with Terraform**: For consistent and repeatable deployments across environments.
- **Serverless Compute**: AWS Lambda for backend logic (document processing, querying, uploads, authentication, DB initialization).
- **Vector Storage**: PostgreSQL RDS with the `pgvector` extension for storing and searching text embeddings.
- **AI Integration**: Leverages Google's Gemini Pro (for generation) and Gemini Embedding models (for text embeddings).
- **Authentication**: Secure user management with AWS Cognito.
- **CI/CD Workflows**: GitHub Actions for automated deployment, testing, and cleanup.
- **Multi-Environment Support**: Designed for `dev`, `staging`, and `production` environments.
- **Comprehensive Testing**: Includes unit and integration tests for backend Lambda functions.
- **Streamlit UI**: Includes a login page, document upload, query interface, and RAG evaluation dashboard.
---

### 🏗️ High Level Architecture

![architecture](./images/architecture.png)

---
### 🌐 Network Flow Walkthrough (Referencing the Architecture)

#### 🗂️ Document Processing Flow with Network Components:

1. User uploads document → API Gateway → `upload_handler` Lambda
2. `upload_handler` Lambda → S3 Gateway Endpoint → S3 Bucket
3. S3 Event → `document_processor` Lambda (in private subnet)
4. `document_processor` Lambda → NAT Gateway → Internet Gateway → Gemini API (for embeddings)
5. `document_processor` Lambda → RDS Security Group → PostgreSQL Database (stores chunks/vectors)

#### 💬 Query Processing Flow with Network Components:

1. User submits query → API Gateway → `query_processor` Lambda (in private subnet)
2. `query_processor` Lambda → RDS Security Group → PostgreSQL Database (vector search)
3. `query_processor` Lambda → NAT Gateway → Internet Gateway → Gemini API (for answer generation)
4. `query_processor` Lambda → API Gateway → User (returns answer)

This network architecture ensures that sensitive operations and data are processed in a secure environment, while still allowing the necessary external communications through controlled channels.

---

### 🔁 GitHub Action Pipeline

![pipeline](./images/pipeline.png)

---

### 🔁 AWS Infra Provisioning Flow Diagram

🗺️ [Infra Provisioning Lifecycle Flow](https://github.com/genieincodebottle/rag-app-on-aws/blob/main/images/infra_provisioning_sequence.png) (Illustrates the Terraform provisioning sequence)

---

### 🗂️ Repository Structure

```
.
├── .github/workflows/       # CI/CD via GitHub Actions
│   ├── deploy.yml           # Infrastructure deployment workflow
│   └── manual_cleanup.yml   # Resource cleanup workflow
├── environments/            # Environment-specific configs (dev, staging, prod)
│   └── dev/                 # Example 'dev' environment
│       ├── main.tf          # Root Terraform file for the environment
│       ├── providers.tf     # Terraform provider configurations
│       └── variables.tf     # Environment-specific variable definitions
├── modules/                 # Reusable Terraform modules
│   ├── api/                 # API Gateway configuration
│   ├── auth/                # Cognito authentication
│   ├── compute/             # Lambda functions & IAM roles
│   ├── database/            # PostgreSQL RDS with pgvector & Secrets Manager
│   ├── monitoring/          # CloudWatch Logs, Alarms & SNS Topic
│   ├── storage/             # S3 Buckets & DynamoDB Table
│   └── vpc/                 # VPC, Subnets, NAT, Security Groups, Endpoints
├── rag_ui/                  # Streamlit UI application
│   ├── app.py               # Main Streamlit application code
│   └── README.md            # README specific to the UI
├── scripts/                 # Utility shell scripts
│   ├── cleanup.sh           # Comprehensive resource cleanup script
│   ├── import_resources.sh  # Script to import existing AWS resources into Terraform state
│   └── network-diagnostics.sh # Script for troubleshooting network connectivity (e.g., Lambda to RDS)
├── src/                     # Lambda backend source code (Python)
│   ├── auth_handler/        # Lambda for Cognito authentication operations
│   ├── db_init/             # Lambda for database schema and pgvector initialization
│   ├── document_processor/  # Lambda for processing uploaded documents
│   ├── query_processor/     # Lambda for handling user queries and RAG
│   ├── tests/               # Unit and integration tests
│   │   ├── integration/     # Integration tests for deployed services
│   │   │   └── run_integration_tests.py
│   │   ├── unit/            # Unit tests for Lambda functions
│   │   │   ├── conftest.py  # Pytest common fixtures and mocks
│   │   │   ├── test_*.py    # Individual unit test files
│   │   └── __init__.py
│   ├── upload_handler/      # Lambda for handling file uploads via API
│   └── utils/               # Shared utility code (e.g., db_connectivity_test.py)
├── sonar-project.properties # SonarQube configuration file
└── tox.ini                  # tox configuration for running tests and linters
```

---
### 🧱 Infrastructure Components

The infrastructure is modularized using Terraform modules:

#### 1. **Networking (VPC - `modules/vpc`)**
-   Custom VPC with public, private, and database subnets across multiple Availability Zones.
-   Internet Gateway for public subnet access.
-   NAT Gateways (configurable for single or multiple AZs) for private subnet outbound access.
-   Route Tables for managing traffic flow.
-   Security Groups to control access to Lambdas, RDS, and Bastion hosts.
-   VPC Endpoints for S3 and DynamoDB, allowing private access from within the VPC.
-   Optional VPC Flow Logs for network traffic monitoring (enabled for `prod`).

#### 2. **Compute (Lambda Functions - `modules/compute`, `src/`)**
-   All functions are Python 3.11 based.
-   **Authentication Handler (`auth_handler`)**: Manages user authentication lifecycle with Cognito (registration, login, email verification, password reset, token refresh).
-   **Document Processor (`document_processor`)**:
    -   Triggered by S3 uploads to the `uploads/` prefix in the documents bucket.
    -   Downloads the uploaded file (PDF, TXT, CSV, etc.).
    -   Loads and chunks the document content.
    -   Generates text embeddings for chunks using the Gemini Embedding model.
    -   Stores document metadata and text chunks (with embeddings) in the PostgreSQL RDS database.
-   **Query Processor (`query_processor`)**:
    -   Handles user queries from the API.
    -   Generates an embedding for the user's query using the Gemini Embedding model.
    -   Performs a vector similarity search in PostgreSQL (using `pgvector`) against stored document chunks.
    -   Retrieves relevant chunks and prepares a context.
    -   Generates a final answer using the Gemini Pro model with the retrieved context.
    -   Optionally performs RAG evaluation (faithfulness, relevancy, context precision).
-   **Upload Handler (`upload_handler`)**:
    -   API endpoint for initiating file uploads.
    -   Receives file content (base64 encoded), name, and user ID.
    -   Uploads the raw file to a specific S3 path (`uploads/{user_id}/{document_id}/{file_name}`).
    -   Stores initial document metadata in PostgreSQL and DynamoDB.
-   **DB Initialization (`db_init`)**:
    -   A Lambda function invoked during CI/CD deployment.
    -   Connects to the PostgreSQL RDS instance.
    -   Creates necessary database tables (`documents`, `chunks`) if they don't exist.
    -   Enables the `pgvector` extension required for vector operations.
-   **IAM Roles & Policies**: Granular permissions for Lambda functions to access S3, DynamoDB, RDS (via Secrets Manager), Secrets Manager, and CloudWatch Logs.

#### 3. **Storage (`modules/storage`, `modules/database`, `environments/dev/main.tf`)**
-   **S3 Buckets**:
    -   `{project_name}-{stage}-documents`: Stores uploaded documents. S3 event notifications trigger the `document_processor` Lambda. Configured with CORS and lifecycle rules.
    -   `{project_name}-{stage}-lambda-code`: Stores Lambda function deployment packages (ZIP files).
    -   `{project_name}-terraform-state`: Central S3 bucket for storing Terraform state files (versioning enabled).
-   **DynamoDB**:
    -   `{project_name}-{stage}-metadata`: Stores metadata related to documents (e.g., status, S3 key, user ID). Used by `upload_handler` and `document_processor`. Features Global Secondary Indexes (GSIs) on `user_id` and `document_id`, and Point-in-Time Recovery (PITR).
    -   `{project_name}-{stage}-terraform-state-lock`: DynamoDB table for Terraform state locking, ensuring safe concurrent operations.
-   **PostgreSQL RDS with `pgvector` (`modules/database`)**:
    -   Managed PostgreSQL database instance.
    -   Utilizes the `pgvector` extension for efficient storage and similarity search of text embeddings.
    -   Stores structured document information in a `documents` table and text chunks with their corresponding vector embeddings in a `chunks` table.
    -   Database credentials are securely managed by AWS Secrets Manager.
 
#### 4. **API & Authentication (`modules/api`, `modules/auth`)**
-   **API Gateway (REST API)**:
    -   Provides public HTTP(S) endpoints for backend Lambda functions.
    -   Routes include `/upload`, `/query`, and `/auth`.
    -   Configured with CORS for frontend integration.
    -   Amazon API Gateway has a default timeout of 30 seconds. However, GenAI use cases may require longer processing times. To        support this, you can request an increased timeout via the AWS support form. After logging into your AWS account, use the following URL to access the form. In our case, we’ve configured the timeout to 150,000 milliseconds (2.5 minutes). Select United States (N. Virginia) as the region since it's set as the default in terraform.tfvars. If you're using a different region, choose the appropriate one accordingly. Keep all other settings unchanged.

        https://us-east-1.console.aws.amazon.com/servicequotas/home/template/add 
    
        ![aws-quota](./images/aws-quota.png)

-   **Cognito User Pools**:
    -   Manages user identities, including registration, sign-in, email verification, and password reset functionalities.
    -   Defines password policies and user attributes.
    -   Issues JWT (JSON Web Tokens) upon successful authentication.
    -   Includes an App Client configured for the frontend application.
-   **JWT-based API Authorization**:
    -   API Gateway utilizes a Cognito JWT authorizer to protect the `/upload` and `/query` endpoints, ensuring only authenticated users can access them.
    -   The `/auth` endpoint is public to allow user registration and login.
-   **Secrets Management (`modules/compute`, `modules/database`)**:
    -   **AWS Secrets Manager**: Used to securely store and manage sensitive information:
        -   `{project_name}-{stage}-gemini-api-key`: Stores the Google Gemini API Key used by `document_processor` and `query_processor`.
        -   `{project_name}-{stage}-db-credentials`: Stores the master credentials for the PostgreSQL RDS instance, automatically rotated or managed by Terraform.

#### 5. **Monitoring & Alerts (`modules/monitoring`)**
-   **CloudWatch Logs**: Centralized logging for API Gateway requests and all Lambda function executions. Log groups are configured with retention policies.
-   **CloudWatch Alarms**: Monitors key metrics for Lambda functions (e.g., `Errors` for `document_processor`, `query_processor`).
-   **SNS Topic (`{project_name}-{stage}-alerts`)**:
    -   Acts as a notification channel.
    -   CloudWatch Alarms publish messages to this topic when an alarm state is reached.
    -   Can be configured with subscriptions (e.g., email) to notify administrators of issues.

---

## ⚙️ Build and Deployment

### 🛠️ Prerequisites

-   ✅ **Python**: `3.11+` (For Streamlit UI).
-   ✅ **AWS Cloud Account**: You’ll need an AWS account to build and deploy this end-to-end application (excluding the streamlit UI, which can runs locally on your system).

    👉 [AWS Free Account](https://aws.amazon.com/free/?trk=ac458a05-be8b-40e0-8d01-a185a5530151&sc_channel=ps&ef_id=Cj0KCQjw0LDBBhCnARIsAMpYlAqh7SSXOWwVg9QY_kYOXcrrp4IH9FEnyCVj77ulQ2Bok0aCDHNxpTwaAl6jEALw_wcB:G:s&s_kwcid=AL!4422!3!733868005590!e!!g!!aws%20console!22269308134!170505082450&gad_campaignid=22269308134&gbraid=0AAAAADjHtp9YddvDzIUJZUVVM05tVvxI1&gclid=Cj0KCQjw0LDBBhCnARIsAMpYlAqh7SSXOWwVg9QY_kYOXcrrp4IH9FEnyCVj77ulQ2Bok0aCDHNxpTwaAl6jEALw_wcB&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all)

    ![aws-account](./images/aws-account.png)
-   ✅ **GitHub Account**: For forking the repository and using [GitHub](https://github.com/) Actions.
-   ✅ **Git installed on Local Machine**: Use [Git Bash](https://git-scm.com/downloads) or any preferred Git client to manage your repository.
-   ✅ **Google API Key**: For accessing Google's free-tier Gemini Pro and Gemini Embedding models.
    
    👉 [Get your API key from Google AI Studio](https://aistudio.google.com/apikey)

    ![gemini-api-key](./images/api_key.png)
-   ✅ Free SonarCloud Account for Code Quality Checks (Optional)

    Sign up at [SonarCloud](https://sonarcloud.io/login?return_to=%2Fexplore%2Fprojects) to enable automated quality gates and static analysis for your codebase.

    ![sonar-cloud](./images/sonar-cloud.png)
    
---
### 🌍 Environment Management

The repository supports multiple deployment environments, typically:
*   `dev`: For development and testing.
*   `staging`: For pre-production validation.
*   `prod`: For the live production environment.

Configuration for each environment (Terraform variables, backend configuration) is managed within its respective subfolder under the `environments/` directory (e.g., `environments/dev/`, `environments/staging/`).

---
### 🚀 Build and Deployment

#### 🔐 Set Up GitHub Repo for Build & Deployment

1. Fork the Repository
   
   👉 https://github.com/genieincodebottle/rag-app-on-aws
 
   ![github-fork](./images/github-fork.png)

2. Clone to Your Local Machine:

    ```
    git clone https://github.com/<your-github-username>/rag-app-on-aws.git
    ```
3. Customize Project Configuration:

   Update the following fields in environments/<stage>/terraform.tfvars:

   * `project_name = "<your-project-name>"` – to avoid global resource name conflicts (e.g., S3 buckets).
   * `github_repo = "<your-github-username>/rag-app-on-aws"` – for CI/CD pipeline setup.
   * `alert_email = "<your-email>"` – for receiving deployment alerts.

     <img src="./images/terraform.png" alt="security-credentials" width="600" height="400"/>


#### 🔐 Setting Up GitHub Secrets

1.  **AWS Access Keys**:
    *   Generate an Access Key for either an IAM user with sufficient permissions or the Root user (which has full access) to experiment and create resources defined in Terraform..
    
    *   If logged in as root user -> Go to the top-right dropdown menu after login and select 'Security Credentials' as shown below.

        <img src="./images/security-credential.png" alt="security-credentials" width="200"/>

    *   If IAM User -> Navigate to IAM > Users > [Your User] > Security credentials > Create access key.

    *   Add these as GitHub repository secrets:
        *   `AWS_ACCESS_KEY_ID`
        *   `AWS_SECRET_ACCESS_KEY`
2.  **SonarQube Token (Optional)**:
    * First, create an organization and import your GitHub project. 
    * Then, generate an access token and add it to your GitHub repository secrets as SONAR_TOKEN.
      
      <img src="./images/sonar-token.png" alt="sonar-token" width="600"/>
    * Also, update the following keys in the sonar-project.properties file at the project root:

      * `sonar.projectKey=<your-sonar-organization-name>_rag-app-on-aws`
      * `sonar.organization=<your-sonar-organization-name>`

      <img src="./images/sonar-properties.png" alt="sonar-properties" width="600"/>
3.  **Google API Key**:
    *   Although the GEMINI_API_KEY isn’t stored as a GitHub secret for deployment, it’s configured post-deployment via AWS Secrets Manager or as a Terraform variable. Terraform will create a placeholder secret in AWS Secrets Manager, which you must update manually. Go to the AWS Console, search for “Secrets Manager,” and update the secret with your actual Gemini API key (generated from Google’s Gemini AI Studio).

    🔑 Secret name format: <your-project-name>-<your-env>-gemini-api-key
    Example: rag-app-dev-gemini-api-key

    ![secret-manager](./images/secret-manager.png)

**To Add Secrets**:
Go to your forked GitHub repository → Settings → Secrets and variables → Actions → New repository secret.

![secrets](./images/secret-keys.png)

#### 🤖 Automated Deployment via GitHub Actions

The repository includes two primary GitHub Actions workflows:

1.  **Terraform AWS Deployment (`.github/workflows/deploy.yml`)**:
    *   Deploys the infrastructure based on the target environment.
    *   Triggered automatically on pushes to `develop` (for `dev` env), `main` (for `prod` env), and `staging` (for `staging` env).
    *   Can also be manually triggered from the GitHub Actions tab, allowing selection of environment and other parameters like `reset_db_password` or `bastion_allowed_cidr` (Keep everything default if running pipeline manually)
2.  **Manual AWS Cleanup (`.github/workflows/manual_cleanup.yml`)**:
    *   A manually triggered workflow to tear down all AWS resources for a specified environment.
    *   Uses the `scripts/cleanup.sh` script.

![github-action](./images/github-action.png)

📤 **Push to trigger CI/CD deployment**:
   *   **Dev**: `git push origin develop`
   *   **Staging**: `git push origin staging`
   *   **Production**: `git push origin main` 
        * It’s recommended to make changes directly in the main branch of your forked repository to deploy AWS resources.
        * SonarCloud offers free integration with GitHub’s main branch, and the AWS setup is configured similarly to the dev environment for easy experimentation.

🧑‍💻 **Manually trigger deployment from GitHub**:
   *   Go to your repository on GitHub.
   *   Click on the "Actions" tab.
   *   Select "Terraform AWS Deployment" from the list of workflows.
   *   Click "Run workflow", choose the branch, environment, and fill in any desired input parameters.

---

### 🚀 Running the Web Search MCP Server (Optional - enables live search if RAG results are insufficient)

```bash
git clone https://github.com/genieincodebottle/rag-app-on-aws.git

cd rag-app-on-aws/mcp_server

pip install uv # If uv doesn't exist in your system

uv venv

.venv\Scripts\activate   # Linux: source .venv/bin/activate

uv pip install -r requirements.txt
```

### 🛠️ Configuration

Create a `.env` file:

```env
SERPAPI_API_KEY=your_serpai_api_key
```
SerpAPI API Key (Free Quota) -> https://serpapi.com/dashboard

### 💡 Usage

Run the following command to start the MCP server on localhost at port 8000 (you can change the port if needed)

```bash
python web_search_mcp_server.py --host localhost --port 8000
```

To expose your local server to the internet (required because AWS Lambda cannot access localhost), choose one of the following methods. Be sure to update the port if you're not using 8000.

✅ Option 1 (Recommended): Use Cloudflare Tunnel (Free without login)
Run the following commands in Windows PowerShell to start a secure tunnel and get a public URL. This URL allows external access to your local MCP server in the RAG UI for testing purposes.

  ```bash
  iwr -useb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile cloudflared.exe
  
  cloudflared tunnel --url http://localhost:8000
  ```

✅ Option 2: Use Serveo (Quick SSH Tunnel)
Run this command in Windows PowerShell or Git Bash to open an SSH tunnel and expose your local server for testing purpose:

 ```bash
 ssh -R 80:localhost:8000 serveo.net
 ```

### 🔄 Start Streamlit UI to Try Out the RAG App

👉 UI Readme: https://github.com/genieincodebottle/rag-app-on-aws/rag_ui

Once the AWS resources are deployed via the GitHub Actions pipeline, follow these steps to launch the UI and test the application locally.

* Navigate to the rag-ui directory in your cloned repository using the terminal.

  ```bash
  cd rag-app-on-aws/rag_ui

  pip install uv # If uv doesn't exist in your system

  uv venv

  .venv\Scripts\activate # Linux: source .venv/bin/activate 
  
  uv pip install -r requirements.txt
  ```

* Configuration

    Create a `.env` file:

    ```env
    # RAG Application API Configuration
    API_ENDPOINT=https://your-api-gateway-url.amazonaws.com/stage
    UPLOAD_ENDPOINT=/upload
    QUERY_ENDPOINT=/query
    AUTH_ENDPOINT=/auth

    # Default user settings
    DEFAULT_USER_ID=test-user

    # Cognito Configuration
    COGNITO_CLIENT_ID=your_cognito_client_id

    # Enabling/disabling evaluation
    ENABLE_EVALUATION="true"
    ```

    Once the GitHub Action pipeline completes successfully, you can download the zipped environment variables file from the GitHub Artifact. Unzip it, open the file, and copy both API_ENDPOINT and COGNITO_CLIENT_ID into your .env file.

    ![env-variable](./rag_ui/images/env-variable.png)

*  Usage

    ```bash
    streamlit run app.py
    ```

    Visit `http://localhost:8501`, register or log in, upload documents, and start querying.

---

### 🔄 CI/CD Pipeline Highlights (`deploy.yml`)

The `deploy.yml` workflow automates the deployment process with the following key steps:

1.  **Determine Environment**: Identifies the target environment (`dev`, `staging`, `prod`) based on the Git branch or manual workflow input.
2.  **Code Quality (SonarQube)**: (Optional) If a `SONAR_TOKEN` secret is configured, it runs SonarQube analysis using `tox` for code quality checks.
3.  **Build Lambda Functions**:
    *   Sets up Python 3.11.
    *   Installs dependencies for each Lambda function.
    *   Packages the `auth_handler`, `db_init`, `document_processor`, `query_processor`, and `upload_handler` Lambda functions into ZIP artifacts.
    *   Uploads these artifacts to GitHub Actions.
4.  **Terraform Setup & Plan**:
    *   Configures AWS credentials using GitHub secrets.
    *   Dynamically creates `backend.tf` for S3 state storage.
    *   Creates the Terraform state S3 bucket (`{PROJECT_NAME}-terraform-state`) and DynamoDB lock table (`{PROJECT_NAME}-{STAGE}-terraform-state-lock`) if they don't already exist.
    *   Downloads Lambda artifacts and uploads them to the `{PROJECT_NAME}-{STAGE}-lambda-code` S3 bucket.
    *   Initializes Terraform (`terraform init`).
    *   Attempts to import existing AWS resources into the Terraform state using `scripts/import_resources.sh` (this helps adopt unmanaged resources).
    *   Generates a Terraform plan (`terraform plan`) using environment-specific variables (e.g., `reset_db_password`, `enable_lifecycle_rules`, `bastion_allowed_cidr`).
    *   Uploads the `tfplan` file as a GitHub artifact.
5.  **Terraform Apply** (Conditional - runs on push to specific branches or manual trigger):
    *   Downloads the `tfplan` artifact.
    *   Applies the Terraform configuration (`terraform apply -auto-approve tfplan`).
    *   Extracts outputs like `api_endpoint` and `cognito_app_client_id`.
    *   Uploads an `env_vars.env` file with these outputs for UI configuration.
6.  **Database Availability & Initialization**:
    *   (Optional via `wait_for_db` input) Waits for the RDS instance to become available.
    *   If `reset_db_password` was true, updates Lambda environment variables with the new DB secret ARN.
    *   Ensures `db_init` and `auth_handler` Lambda functions are updated with the latest code from S3 (as a safeguard).
    *   Invokes the `db_init` Lambda function to set up the PostgreSQL schema and `pgvector` extension. This step includes retries in case the database isn't immediately ready.
7.  **Verify Deployment**: Makes a health check call to the `upload_handler` Lambda via API Gateway.
8.  **Integration Tests**:
    *   Sets up Python and installs dependencies.
    *   Runs integration tests located in `src/tests/integration/run_integration_tests.py` against the deployed API Gateway endpoint.
    *   Uploads test results as a GitHub artifact.

---


### 🧰 Utilities

The `/scripts/` folder contains helpful shell scripts:

-   **`cleanup.sh`**: A comprehensive script to tear down all AWS resources created by Terraform for a specific environment. It requires `jq` to be installed. **Use with extreme caution as this is destructive.**
-   **`import_resources.sh`**: Aids in importing existing AWS resources into the Terraform state. This can be useful if some resources were created manually or outside of Terraform initially.
-   **`network-diagnostics.sh`**: A script to help troubleshoot network connectivity issues, particularly between Lambda functions and the RDS database within the VPC. It checks security groups, RDS status, and can test DNS resolution from a Lambda.

---

### 🧹 Uninstallation

To remove all AWS resources created by this project for a specific environment:

#### Using GitHub Actions (Recommended):
1.  Navigate to your repository's "Actions" tab on GitHub.
2.  Find the "Manual AWS Cleanup" workflow in the sidebar.
3.  Click "Run workflow".
4.  Select the branch (usually your main or develop branch).
5.  Enter the **environment name** (e.g., `dev`, `staging`, `prod`) you wish to clean up.
6.  Click "Run workflow". This will execute the `scripts/cleanup.sh` script with the necessary context.

#### Using the Cleanup Script Manually:
**Warning**: This script will delete resources. Ensure you have the correct AWS credentials and region configured for your AWS CLI, and that you are targeting the correct environment.
1.  Ensure `jq` is installed:
    ```bash
    # On Debian/Ubuntu
    sudo apt-get update && sudo apt-get install -y jq
    # On macOS (using Homebrew)
    brew install jq
    ```
2.  Navigate to the `scripts` directory:
    ```bash
    cd scripts
    ```
3.  Make the script executable:
    ```bash
    chmod +x cleanup.sh
    ```
4.  Run the script, providing the necessary environment variables. The script expects `PROJECT_NAME`, `STAGE`, and `AWS_REGION` to be set. You can set them inline:
    ```bash
    PROJECT_NAME="your-project-name" STAGE="dev" AWS_REGION="us-east-1" ./cleanup.sh
    ```
    (Replace with your actual project name, stage, and region).
    The script has built-in confirmations but destructive actions are significant.
---

### 🤝 Contributing

Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a new feature branch (e.g., `git checkout -b feature/new-ai-model-integration`).
3.  Make your changes and commit them with clear messages (e.g., `git commit -m 'feat: Add support for Claude 3 model'`).
4.  Push your changes to your forked repository (`git push origin feature/new-ai-model-integration`).
5.  Open a Pull Request to the `develop` branch of the original repository.

---

> **Note**: Deploying this infrastructure will incur AWS charges. Always review the output of `terraform plan` before applying changes to understand potential costs and resource modifications.
>
> **Security Best Practice**: Never commit secrets directly to your Git repository. Use GitHub Secrets for CI/CD variables and manage sensitive application configurations (like API keys) securely, for instance, through AWS Secrets Manager populated via secure Terraform variables or post-deployment steps.
