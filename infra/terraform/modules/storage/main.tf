# Documents bucket and the ingest queue it notifies.
#
# * Versioning is on, so an overwritten or deleted object can be recovered
#   during the noncurrent retention window. The application's delete path
#   removes every version explicitly; the lifecycle rule is the backstop.
# * Only TLS requests are accepted, and only KMS encryption with this key.
# * S3 sends ObjectCreated events for keys under raw/ to SQS. Five failed
#   receives move a message to the DLQ, which has an alarm.

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "documents" {
  #checkov:skip=CKV_AWS_144:Single-region design; cross-region recovery is described in docs/operations-runbook.md
  #checkov:skip=CKV_AWS_18:Object-level access is audited with CloudTrail S3 data events (docs/security.md); server access logs are not used
  bucket        = var.bucket_name
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket_ownership_controls" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = var.kms_key_arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    id     = "expire-noncurrent-and-abandoned-uploads"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_retention_days
    }
    expiration {
      expired_object_delete_marker = true
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

resource "aws_s3_bucket_cors_configuration" "documents" {
  count  = length(var.upload_origins) > 0 ? 1 : 0
  bucket = aws_s3_bucket.documents.id
  cors_rule {
    # Browser uploads use the presigned POST returned by the API.
    allowed_methods = ["POST"]
    allowed_origins = var.upload_origins
    allowed_headers = ["*"]
    max_age_seconds = 600
  }
}

resource "aws_s3_bucket_policy" "documents" {
  bucket = aws_s3_bucket.documents.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = [aws_s3_bucket.documents.arn, "${aws_s3_bucket.documents.arn}/*"]
        Condition = { Bool = { "aws:SecureTransport" = "false" } }
      },
      {
        Sid       = "DenyOtherEncryptionKeys"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:PutObject"
        Resource  = "${aws_s3_bucket.documents.arn}/*"
        Condition = {
          StringNotEqualsIfExists = {
            "s3:x-amz-server-side-encryption-aws-kms-key-id" = var.kms_key_arn
          }
        }
      },
    ]
  })
  depends_on = [aws_s3_bucket_public_access_block.documents]
}

# ---------------------------------------------------------------------- queue

resource "aws_sqs_queue" "dlq" {
  name                      = "${var.name}-ingest-dlq"
  kms_master_key_id         = var.kms_key_arn
  message_retention_seconds = 1209600 # 14 days, the maximum, to leave time to investigate
}

resource "aws_sqs_queue" "ingest" {
  name              = "${var.name}-ingest"
  kms_master_key_id = var.kms_key_arn
  # AWS recommends at least six times the function timeout for an SQS event
  # source, so a message is not redelivered while a slow attempt still runs.
  visibility_timeout_seconds = var.ingest_function_timeout_s * 6
  message_retention_seconds  = 345600 # 4 days
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = var.max_receive_count
  })
}

resource "aws_sqs_queue_redrive_allow_policy" "dlq" {
  queue_url = aws_sqs_queue.dlq.id
  redrive_allow_policy = jsonencode({
    redrivePermission = "byQueue"
    sourceQueueArns   = [aws_sqs_queue.ingest.arn]
  })
}

resource "aws_sqs_queue_policy" "ingest" {
  queue_url = aws_sqs_queue.ingest.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "AllowDocumentBucketNotifications"
      Effect    = "Allow"
      Principal = { Service = "s3.amazonaws.com" }
      Action    = "sqs:SendMessage"
      Resource  = aws_sqs_queue.ingest.arn
      Condition = {
        ArnEquals    = { "aws:SourceArn" = aws_s3_bucket.documents.arn }
        StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
      }
    }]
  })
}

resource "aws_s3_bucket_notification" "documents" {
  bucket = aws_s3_bucket.documents.id
  queue {
    queue_arn     = aws_sqs_queue.ingest.arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "raw/"
  }
  depends_on = [aws_sqs_queue_policy.ingest]
}

variable "name" {
  type = string
}

variable "bucket_name" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "force_destroy" {
  type    = bool
  default = false
}

variable "noncurrent_retention_days" {
  type    = number
  default = 30
}

variable "upload_origins" {
  type    = list(string)
  default = []
}

variable "ingest_function_timeout_s" {
  type    = number
  default = 300
}

variable "max_receive_count" {
  type    = number
  default = 5
}

output "bucket_name" {
  value = aws_s3_bucket.documents.bucket
}

output "bucket_arn" {
  value = aws_s3_bucket.documents.arn
}

output "queue_arn" {
  value = aws_sqs_queue.ingest.arn
}

output "queue_name" {
  value = aws_sqs_queue.ingest.name
}

output "dlq_arn" {
  value = aws_sqs_queue.dlq.arn
}

output "dlq_name" {
  value = aws_sqs_queue.dlq.name
}
