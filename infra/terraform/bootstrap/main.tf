# One-time account setup, applied by an administrator from a workstation:
#
#   cd infra/terraform/bootstrap
#   terraform init && terraform apply -var account_id=<id> -var github_repository=<owner/repo>
#
# It creates:
#   * the Terraform state bucket (versioned, encrypted, TLS-only; Terraform
#     1.10+ locks state with a lock object in this bucket);
#   * the Lambda artifacts bucket the deploy workflow uploads bundles to;
#   * the GitHub Actions OIDC identity provider;
#   * per environment, a read-only "plan" role and a "deploy" role that GitHub
#     Actions can assume only from that repository and that GitHub environment.
#
# No long-lived AWS keys are stored in GitHub.

terraform {
  required_version = ">= 1.10.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region              = var.region
  allowed_account_ids = [var.account_id]
  default_tags {
    tags = { Project = var.project, ManagedBy = "terraform-bootstrap" }
  }
}

locals {
  state_bucket    = "${var.project}-tfstate-${var.account_id}-${var.region}"
  artifact_bucket = "${var.project}-artifacts-${var.account_id}-${var.region}"
}

# ---------------------------------------------------------------- buckets

resource "aws_s3_bucket" "state" {
  #checkov:skip=CKV_AWS_144:Single-region bootstrap; versioning protects state history
  #checkov:skip=CKV_AWS_18:Access to state is audited through CloudTrail management and data events
  #checkov:skip=CKV2_AWS_62:Nothing consumes events from this bucket
  bucket = local.state_bucket
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket" "artifacts" {
  #checkov:skip=CKV_AWS_144:Single-region bootstrap; versioning protects state history
  #checkov:skip=CKV_AWS_18:Access to state is audited through CloudTrail management and data events
  #checkov:skip=CKV2_AWS_62:Nothing consumes events from this bucket
  bucket = local.artifact_bucket
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "both" {
  for_each = { state = aws_s3_bucket.state.id, artifacts = aws_s3_bucket.artifacts.id }
  bucket   = each.value
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "both" {
  for_each                = { state = aws_s3_bucket.state.id, artifacts = aws_s3_bucket.artifacts.id }
  bucket                  = each.value
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "tls_only" {
  for_each = { state = aws_s3_bucket.state, artifacts = aws_s3_bucket.artifacts }
  bucket   = each.value.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [each.value.arn, "${each.value.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.both]
}

resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    id     = "keep-state-history-90-days"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    id     = "expire-old-bundles"
    status = "Enabled"
    filter {}
    # Old bundles are kept long enough to roll back to.
    expiration {
      days = var.artifact_retention_days
    }
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# ---------------------------------------------------------------- GitHub OIDC

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

# Deploy roles trust GitHub environment "<env>" (prod has required reviewers).
# Plan roles trust "<env>-plan", which has no approval rule, so a plan for a
# pull request never waits for a reviewer and can never apply.
data "aws_iam_policy_document" "trust" {
  for_each = merge(
    { for e in var.environments : e => e },
    { for e in var.environments : "${e}-plan" => "${e}-plan" },
  )
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Only jobs that run in this repository's GitHub environment of the same
    # name. Production approval rules live on that GitHub environment.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:environment:${each.value}"]
    }
  }
}

resource "aws_iam_role" "plan" {
  for_each             = toset(var.environments)
  name                 = "${var.project}-${each.value}-plan"
  assume_role_policy   = data.aws_iam_policy_document.trust["${each.value}-plan"].json
  max_session_duration = 3600
}

resource "aws_iam_role_policy_attachment" "plan_readonly" {
  for_each   = toset(var.environments)
  role       = aws_iam_role.plan[each.value].name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

resource "aws_iam_role" "deploy" {
  for_each             = toset(var.environments)
  name                 = "${var.project}-${each.value}-deploy"
  assume_role_policy   = data.aws_iam_policy_document.trust[each.value].json
  max_session_duration = 3600
}

resource "aws_iam_role_policy" "state_access" {
  for_each = merge(
    { for e in var.environments : "plan-${e}" => aws_iam_role.plan[e].name },
    { for e in var.environments : "deploy-${e}" => aws_iam_role.deploy[e].name },
  )
  name = "terraform-state"
  role = each.value
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.state.arn
      },
      {
        # Each environment's state and lock object live under its own prefix.
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.state.arn}/${split("-", each.key)[1]}/*"
      },
    ]
  })
}

resource "aws_iam_role_policy" "deploy" {
  for_each = toset(var.environments)
  name     = "deploy-${var.project}"
  role     = aws_iam_role.deploy[each.value].name
  policy = templatefile("${path.module}/deploy-policy.json.tftpl", {
    project         = var.project
    environment     = each.value
    account_id      = var.account_id
    region          = var.region
    artifact_bucket = aws_s3_bucket.artifacts.arn
  })
}

variable "project" {
  type    = string
  default = "ragplatform"
}

variable "account_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "github_repository" {
  type        = string
  description = "owner/repo that may deploy, e.g. genieincodebottle/rag-app-on-aws"
}

variable "environments" {
  type    = list(string)
  default = ["dev", "test", "prod"]
}

variable "artifact_retention_days" {
  type    = number
  default = 90
}

output "state_bucket" {
  value = aws_s3_bucket.state.bucket
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "deploy_role_arns" {
  value = { for e, r in aws_iam_role.deploy : e => r.arn }
}

output "plan_role_arns" {
  value = { for e, r in aws_iam_role.plan : e => r.arn }
}
