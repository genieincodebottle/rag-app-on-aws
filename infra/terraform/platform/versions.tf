terraform {
  # 1.10+ for S3-native state locking (use_lockfile), so no DynamoDB lock table.
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # Filled in per environment: terraform init -backend-config=../envs/<env>.s3.tfbackend
  backend "s3" {}
}

provider "aws" {
  region = var.region

  # A plan run with the wrong credentials fails here instead of touching
  # resources in the wrong account.
  allowed_account_ids = [var.account_id]

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "terraform"
      Repository  = "genieincodebottle/rag-app-on-aws"
    }
  }
}
