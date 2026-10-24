# One Lambda function with its own IAM role, encrypted log group and a "live"
# alias. Callers (API Gateway, the SQS event source) invoke the alias, so a
# rollback is an alias update to the previous published version.

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "this" {
  name              = "/aws/lambda/${var.name}"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_arn
}

resource "aws_iam_role" "this" {
  name = var.name
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = { StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id } }
    }]
  })
}

resource "aws_iam_role_policy" "this" {
  name = "${var.name}-permissions"
  role = aws_iam_role.this.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid      = "OwnLogGroup"
          Effect   = "Allow"
          Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
          Resource = "${aws_cloudwatch_log_group.this.arn}:*"
        },
        {
          Sid      = "XRay"
          Effect   = "Allow"
          Action   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
          Resource = "*" # X-Ray does not support resource-level permissions
        },
      ],
      var.statements,
    )
  })
}

resource "aws_lambda_function" "this" {
  #checkov:skip=CKV_AWS_116:The API and migrate functions are invoked synchronously; ingest failures go to the SQS DLQ, not a function DLQ
  #checkov:skip=CKV_AWS_117:Functions reach Aurora through the RDS Data API and run outside a VPC by design (docs/adr/0003)
  #checkov:skip=CKV_AWS_272:Code signing is not wired; provenance comes from GitHub artifact attestation and the SHA-256 pinned in Terraform (known gap, docs/security.md)
  function_name    = var.name
  role             = aws_iam_role.this.arn
  handler          = var.handler
  runtime          = "python3.13"
  architectures    = ["arm64"]
  memory_size      = var.memory_mb
  timeout          = var.timeout_s
  s3_bucket        = var.artifact_bucket
  s3_key           = var.artifact_key
  source_code_hash = var.artifact_sha256
  publish          = true
  kms_key_arn      = var.kms_key_arn # environment variables at rest

  reserved_concurrent_executions = var.reserved_concurrency

  environment {
    variables = merge(var.environment, {
      POWERTOOLS_SERVICE_NAME      = "ragplatform"
      POWERTOOLS_METRICS_NAMESPACE = "RagPlatform"
      POWERTOOLS_LOG_LEVEL         = var.log_level
    })
  }

  logging_config {
    log_format = "JSON"
    log_group  = aws_cloudwatch_log_group.this.name
  }

  tracing_config {
    mode = "Active"
  }

  depends_on = [aws_iam_role_policy.this, aws_cloudwatch_log_group.this]
}

resource "aws_lambda_alias" "live" {
  name             = "live"
  function_name    = aws_lambda_function.this.function_name
  function_version = aws_lambda_function.this.version
}

variable "name" {
  type = string
}

variable "handler" {
  type = string
}

variable "artifact_bucket" {
  type = string
}

variable "artifact_key" {
  type = string
}

variable "artifact_sha256" {
  type        = string
  description = "base64-encoded SHA-256 of the zip, so a new build always publishes a new version"
}

variable "memory_mb" {
  type    = number
  default = 1024
}

variable "timeout_s" {
  type    = number
  default = 30
}

variable "reserved_concurrency" {
  type    = number
  default = -1
}

variable "environment" {
  type    = map(string)
  default = {}
}

variable "statements" {
  # "any" rather than list(any): statements differ in shape (some carry a
  # Condition), and list(any) requires every element to have the same type.
  type        = any
  description = "IAM policy statements specific to this function"
}

variable "kms_key_arn" {
  type = string
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "log_level" {
  type    = string
  default = "INFO"
}

output "function_name" {
  value = aws_lambda_function.this.function_name
}

output "function_arn" {
  value = aws_lambda_function.this.arn
}

output "alias_arn" {
  value = aws_lambda_alias.live.arn
}

output "alias_invoke_arn" {
  value = aws_lambda_alias.live.invoke_arn
}

output "role_arn" {
  value = aws_iam_role.this.arn
}

output "version" {
  value = aws_lambda_function.this.version
}
