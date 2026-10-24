variable "project" {
  type    = string
  default = "ragplatform"
}

variable "environment" {
  type = string
  validation {
    condition     = contains(["dev", "test", "prod"], var.environment)
    error_message = "environment must be dev, test or prod."
  }
}

variable "account_id" {
  type        = string
  description = "The only AWS account this configuration may change."
}

variable "region" {
  type    = string
  default = "us-east-1"
}

# ---------------------------------------------------------------- artifacts (set by CI)

variable "artifact_bucket" {
  type        = string
  description = "Bucket created by the bootstrap stack that holds Lambda bundles."
}

variable "artifact_key" {
  type        = string
  description = "Key of the bundle for this deploy, e.g. lambda/<git-sha>/ragplatform.zip"
}

variable "artifact_sha256" {
  type        = string
  description = "base64-encoded SHA-256 of the bundle"
}

# ---------------------------------------------------------------- models

variable "embedding_model_id" {
  type    = string
  default = "amazon.titan-embed-text-v2:0"
}

variable "embedding_dimensions" {
  type    = number
  default = 1024
}

variable "chat_model_id" {
  type    = string
  default = "us.amazon.nova-2-lite-v1:0"
}

variable "fallback_chat_model_id" {
  type    = string
  default = null
}

variable "rerank_model_id" {
  type    = string
  default = null
}

variable "guardrail_enabled" {
  type    = bool
  default = false
}

variable "guardrail_grounding_check" {
  type    = bool
  default = false
}

# ---------------------------------------------------------------- sizing and protection

variable "db_min_acu" {
  type    = number
  default = 0
}

variable "db_max_acu" {
  type    = number
  default = 4
}

variable "db_engine_version" {
  type    = string
  default = "16.13"
}

variable "db_backup_retention_days" {
  type    = number
  default = 7
}

variable "protect_data" {
  type        = bool
  default     = false
  description = "Deletion protection on Aurora, Cognito and secrets, a final snapshot, and no force_destroy on the bucket. true in prod."
}

variable "api_memory_mb" {
  type    = number
  default = 1024
}

variable "ingest_memory_mb" {
  type    = number
  default = 2048
}

variable "ingest_timeout_s" {
  type    = number
  default = 300
}

variable "ingest_max_concurrency" {
  type        = number
  default     = 5
  description = "Upper bound on concurrent ingest invocations. Protects Bedrock quotas and Aurora capacity."
}

variable "api_throttle_rate" {
  type    = number
  default = 20
}

variable "api_throttle_burst" {
  type    = number
  default = 40
}

variable "waf_enabled" {
  type    = bool
  default = false
}

variable "mfa_required" {
  type    = bool
  default = false
}

variable "allow_password_auth" {
  type    = bool
  default = true
}

variable "upload_origins" {
  type        = list(string)
  default     = []
  description = "Browser origins allowed to POST uploads straight to S3."
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "alert_emails" {
  type    = list(string)
  default = []
}

variable "monthly_budget_usd" {
  type    = number
  default = 0
}
