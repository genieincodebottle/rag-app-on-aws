# One customer-managed key per environment. It encrypts the documents bucket,
# the ingest queue and DLQ, Aurora storage, both database secrets and the
# CloudWatch log groups. Rotation is automatic (yearly).
#
# The key policy lets the account's IAM policies grant use of the key (the
# root statement), and adds the two service principals that act on the key
# without a role of ours in the loop: CloudWatch Logs (log group encryption)
# and S3 (sending encrypted event notifications to the SQS queue).

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

resource "aws_kms_key" "this" {
  description             = "${var.name} data key"
  enable_key_rotation     = true
  deletion_window_in_days = var.deletion_window_in_days

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AccountAdministration"
        Effect    = "Allow"
        Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid       = "CloudWatchLogs"
        Effect    = "Allow"
        Principal = { Service = "logs.${data.aws_region.current.region}.amazonaws.com" }
        Action    = ["kms:Encrypt*", "kms:Decrypt*", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:Describe*"]
        Resource  = "*"
        Condition = {
          ArnLike = {
            "kms:EncryptionContext:aws:logs:arn" = "arn:aws:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:*"
          }
        }
      },
      {
        Sid       = "S3EventNotificationsToEncryptedQueue"
        Effect    = "Allow"
        Principal = { Service = "s3.amazonaws.com" }
        Action    = ["kms:GenerateDataKey", "kms:Decrypt"]
        Resource  = "*"
        Condition = {
          StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
        }
      },
    ]
  })
}

resource "aws_kms_alias" "this" {
  name          = "alias/${var.name}"
  target_key_id = aws_kms_key.this.key_id
}

variable "name" {
  type = string
}

variable "deletion_window_in_days" {
  type    = number
  default = 30
}

output "key_arn" {
  value = aws_kms_key.this.arn
}

output "key_id" {
  value = aws_kms_key.this.key_id
}
