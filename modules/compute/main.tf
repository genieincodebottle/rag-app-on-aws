# ===============================
# Compute Module for RAG System
# ===============================

# =========================
# Locals
# =========================

locals {
  name                    = "${var.project_name}-${var.stage}" 
  document_processor_name = "${var.project_name}-${var.stage}-document-processor"
  query_processor_name    = "${var.project_name}-${var.stage}-query-processor"
  upload_handler_name     = "${var.project_name}-${var.stage}-upload-handler"
  db_init_name            = "${var.project_name}-${var.stage}-db-init"
  auth_handler_name       = "${var.project_name}-${var.stage}-auth-handler"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.stage
    ManagedBy   = "Terraform"
  }
}

# ===================================================================
# Store GEMINI_API_KEY credentials placeholder in AWS Secrets Manager
# ===================================================================

resource "aws_secretsmanager_secret" "gemini_api_credentials" {
  name        = "${local.name}-gemini-api-key"
  description = "Gemini API Key for ${local.name}"

  tags = {
    Name        = "${local.name}-gemini-api-key"
    Environment = var.stage
  }
}

resource "aws_secretsmanager_secret_version" "gemini_api_credentials" {
  secret_id = aws_secretsmanager_secret.gemini_api_credentials.id
  secret_string = jsonencode({
    GEMINI_API_KEY    = var.gemini_api_key
  })
}

# =========================
# IAM Role and Policy
# =========================

resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-${var.stage}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = { Service = "lambda.amazonaws.com" },
      Action = "sts:AssumeRole"
    }]
  })

  tags = {
    Name = "${var.project_name}-${var.stage}-lambda-role"
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_iam_policy" "lambda_policy" {
  name = "${var.project_name}-${var.stage}-lambda-policy"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:HeadObject"
        ],
        Resource = [
          "arn:aws:s3:::${var.documents_bucket}",
          "arn:aws:s3:::${var.documents_bucket}/*"
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem",
          "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan"
        ],
        Resource = [
          "arn:aws:dynamodb:*:*:table/${var.metadata_table}",
          "arn:aws:dynamodb:*:*:table/${var.metadata_table}/index/*"
        ]
      },
      {
        Effect = "Allow",
        Action = ["bedrock:InvokeModel"],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "ec2:CreateNetworkInterface", "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface", "ec2:AssignPrivateIpAddresses",
          "ec2:UnassignPrivateIpAddresses"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = ["secretsmanager:GetSecretValue"],
        Resource = [
          var.db_secret_arn,
          aws_secretsmanager_secret.gemini_api_credentials.arn
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "cognito-idp:AdminInitiateAuth",
          "cognito-idp:AdminCreateUser",
          "cognito-idp:AdminSetUserPassword",
          "cognito-idp:AdminUpdateUserAttributes",
          "cognito-idp:AdminGetUser",
          "cognito-idp:AdminConfirmSignUp",
          "cognito-idp:SignUp",
          "cognito-idp:ConfirmSignUp",
          "cognito-idp:InitiateAuth",
          "cognito-idp:ForgotPassword",
          "cognito-idp:ConfirmForgotPassword",
          "cognito-idp:RespondToAuthChallenge"
        ],
        Resource = var.cognito_user_pool_arn != "" ? [var.cognito_user_pool_arn] : ["*"]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}

# =========================
# Lambda Code from S3
# =========================

data "aws_s3_object" "auth_handler_code" {
  bucket = var.lambda_code_bucket
  key    = "lambda/auth_handler.zip"
}

data "aws_s3_object" "document_processor_code" {
  bucket = var.lambda_code_bucket
  key    = "lambda/document_processor.zip"
}

data "aws_s3_object" "query_processor_code" {
  bucket = var.lambda_code_bucket
  key    = "lambda/query_processor.zip"
}

data "aws_s3_object" "upload_handler_code" {
  bucket = var.lambda_code_bucket
  key    = "lambda/upload_handler.zip"
}

data "aws_s3_object" "db_init_code" {
  bucket = var.lambda_code_bucket
  key    = "lambda/db_init.zip"
}

# =========================
# Lambda Functions
# =========================

resource "aws_lambda_function" "auth_handler" {
  function_name = local.auth_handler_name
  role          = aws_iam_role.lambda_role.arn
  handler       = "auth_handler.handler"
  runtime       = "python3.11"
  memory_size   = var.lambda_memory_size
  timeout       = var.lambda_timeout

  environment {
    variables = {
      USER_POOL_ID = var.cognito_user_pool_id
      CLIENT_ID    = var.cognito_app_client_id
      STAGE        = var.stage
    }
  }

  s3_bucket        = var.lambda_code_bucket
  s3_key           = "lambda/auth_handler.zip"
  source_code_hash = data.aws_s3_object.auth_handler_code.etag
}

resource "aws_lambda_function" "document_processor" {
  function_name = local.document_processor_name
  role          = aws_iam_role.lambda_role.arn
  handler       = "document_processor.handler"
  runtime       = "python3.11"
  memory_size   = var.lambda_memory_size
  timeout       = var.lambda_timeout

  environment {
    variables = {
      DOCUMENTS_BUCKET         = var.documents_bucket
      METADATA_TABLE           = var.metadata_table
      STAGE                    = var.stage
      DB_SECRET_ARN            = var.db_secret_arn
      GEMINI_SECRET_ARN        = aws_secretsmanager_secret.gemini_api_credentials.arn
      GEMINI_EMBEDDING_MODEL   = var.gemini_embedding_model
      TEMPERATURE              = 0.2
      MAX_OUTPUT_TOKENS        = 1024
      TOP_K                    = 40
      TOP_P                    = 0.8
      SIMILARITY_THRESHOLD     = 0.7
    }
  }

  vpc_config {
    subnet_ids         = var.vpc_subnet_ids
    security_group_ids = [var.lambda_security_group_id]
  }

  s3_bucket        = var.lambda_code_bucket
  s3_key           = "lambda/document_processor.zip"
  source_code_hash = data.aws_s3_object.document_processor_code.etag

  tags = {
    Name = local.document_processor_name
  }
}

resource "aws_lambda_function" "query_processor" {
  function_name = local.query_processor_name
  role          = aws_iam_role.lambda_role.arn
  handler       = "query_processor.handler"
  runtime       = "python3.11"
  memory_size   = var.lambda_memory_size
  timeout       = var.lambda_timeout

  environment {
    variables = {
      DOCUMENTS_BUCKET         = var.documents_bucket
      METADATA_TABLE           = var.metadata_table
      STAGE                    = var.stage
      DB_SECRET_ARN            = var.db_secret_arn
      GEMINI_SECRET_ARN        = aws_secretsmanager_secret.gemini_api_credentials.arn
      GEMINI_EMBEDDING_MODEL   = var.gemini_embedding_model
      TEMPERATURE              = 0.2
      MAX_OUTPUT_TOKENS        = 1024
      TOP_K                    = 40
      TOP_P                    = 0.8
      SIMILARITY_THRESHOLD     = 0.7

      # MCP Configuration
      MCP_TIMEOUT              = var.mcp_timeout
      RAG_CONFIDENCE_THRESHOLD = var.rag_confidence_threshold
      MIN_CONTEXT_LENGTH       = var.min_context_length
    }
  }

  vpc_config {
    subnet_ids         = var.vpc_subnet_ids
    security_group_ids = [var.lambda_security_group_id]
  }

  s3_bucket        = var.lambda_code_bucket
  s3_key           = "lambda/query_processor.zip"
  source_code_hash = data.aws_s3_object.query_processor_code.etag

  tags = {
    Name = local.query_processor_name
  }
}

resource "aws_lambda_function" "upload_handler" {
  function_name = local.upload_handler_name
  role          = aws_iam_role.lambda_role.arn
  handler       = "upload_handler.handler"
  runtime       = "python3.11"
  memory_size   = var.lambda_memory_size
  timeout       = var.lambda_timeout

  environment {
    variables = {
      DOCUMENTS_BUCKET = var.documents_bucket
      METADATA_TABLE   = var.metadata_table
      STAGE            = var.stage
      DB_SECRET_ARN    = var.db_secret_arn
    }
  }

  vpc_config {
    subnet_ids         = var.vpc_subnet_ids
    security_group_ids = [var.lambda_security_group_id]
  }

  s3_bucket        = var.lambda_code_bucket
  s3_key           = "lambda/upload_handler.zip"
  source_code_hash = data.aws_s3_object.upload_handler_code.etag

  tags = {
    Name = local.upload_handler_name
  }
}

resource "aws_lambda_function" "db_init" {
  function_name = local.db_init_name
  role          = aws_iam_role.lambda_role.arn
  handler       = "db_init.handler"
  runtime       = "python3.11"
  memory_size   = var.lambda_memory_size
  timeout       = var.lambda_timeout

  environment {
    variables = {
      DOCUMENTS_BUCKET = var.documents_bucket
      METADATA_TABLE   = var.metadata_table
      STAGE            = var.stage
      DB_SECRET_ARN    = var.db_secret_arn
      MAX_RETRIES      = var.max_retries
      RETRY_DELAY      = var.retry_delay
    }
  }

  vpc_config {
    subnet_ids         = var.vpc_subnet_ids
    security_group_ids = [var.lambda_security_group_id]
  }

  s3_bucket        = var.lambda_code_bucket
  s3_key           = "lambda/db_init.zip"
  source_code_hash = data.aws_s3_object.db_init_code.etag

  tags = {
    Name = local.db_init_name
  }
}

# =========================
# Lambda Triggers
# =========================

resource "aws_lambda_permission" "s3_invoke_lambda_document_processor" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.document_processor.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = "arn:aws:s3:::${var.documents_bucket}"
}

resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket = var.documents_bucket

  lambda_function {
    lambda_function_arn = aws_lambda_function.document_processor.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "uploads/"
  }

  depends_on = [aws_lambda_permission.s3_invoke_lambda_document_processor]
}
