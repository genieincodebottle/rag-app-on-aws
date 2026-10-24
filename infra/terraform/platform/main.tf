data "aws_caller_identity" "current" {}

locals {
  name       = "${var.project}-${var.environment}"
  account_id = data.aws_caller_identity.current.account_id

  # A model id starting with a geo prefix is a cross-region inference profile.
  # Calling one needs permission on the profile and on the foundation model in
  # every region the profile can route to.
  profile_prefix = "^(us|eu|apac|global)\\."
  chat_models    = compact([var.chat_model_id, var.fallback_chat_model_id])
  chat_model_arns = flatten([
    for id in local.chat_models : can(regex(local.profile_prefix, id)) ? [
      "arn:aws:bedrock:${var.region}:${local.account_id}:inference-profile/${id}",
      "arn:aws:bedrock:*::foundation-model/${replace(id, "/${local.profile_prefix}/", "")}",
    ] : ["arn:aws:bedrock:${var.region}::foundation-model/${id}"]
  ])
  embedding_model_arn = "arn:aws:bedrock:${var.region}::foundation-model/${var.embedding_model_id}"

  # Settings shared by every function. Null values are dropped so an unset
  # optional setting stays unset instead of becoming an empty string.
  base_env = { for k, v in {
    RAG_ENVIRONMENT               = var.environment
    RAG_AWS_REGION                = var.region
    RAG_AUTH_MODE                 = "cognito"
    RAG_PROVIDER                  = "bedrock"
    RAG_DB_BACKEND                = "data_api"
    RAG_DB_CLUSTER_ARN            = module.database.cluster_arn
    RAG_DB_SECRET_ARN             = module.database.app_secret_arn
    RAG_DB_NAME                   = module.database.database_name
    RAG_BLOB_BACKEND              = "s3"
    RAG_DOCUMENTS_BUCKET          = module.storage.bucket_name
    RAG_INGEST_MODE               = "sqs"
    RAG_EMBEDDING_MODEL_ID        = var.embedding_model_id
    RAG_EMBEDDING_DIMENSIONS      = tostring(var.embedding_dimensions)
    RAG_CHAT_MODEL_ID             = var.chat_model_id
    RAG_FALLBACK_CHAT_MODEL_ID    = var.fallback_chat_model_id
    RAG_RERANK_MODEL_ID           = var.rerank_model_id
    RAG_GUARDRAIL_ID              = var.guardrail_enabled ? module.guardrail[0].guardrail_id : null
    RAG_GUARDRAIL_VERSION         = var.guardrail_enabled ? module.guardrail[0].version : null
    RAG_GUARDRAIL_GROUNDING_CHECK = var.guardrail_enabled ? tostring(var.guardrail_grounding_check) : null
    RAG_INGEST_LEASE_S            = tostring(var.ingest_timeout_s)
  } : k => v if v != null }

  data_api_statements = [
    {
      Sid    = "DataApi"
      Effect = "Allow"
      Action = [
        "rds-data:ExecuteStatement", "rds-data:BatchExecuteStatement",
        "rds-data:BeginTransaction", "rds-data:CommitTransaction", "rds-data:RollbackTransaction",
      ]
      Resource = module.database.cluster_arn
    },
    {
      Sid      = "RuntimeRoleSecret"
      Effect   = "Allow"
      Action   = "secretsmanager:GetSecretValue"
      Resource = module.database.app_secret_arn
    },
    {
      Sid      = "DecryptWithDataKey"
      Effect   = "Allow"
      Action   = "kms:Decrypt"
      Resource = module.kms.key_arn
    },
  ]
}

module "kms" {
  source = "../modules/kms"
  name   = local.name
}

module "network" {
  source      = "../modules/network"
  name        = local.name
  kms_key_arn = module.kms.key_arn
  flow_logs   = var.environment == "prod"
}

module "database" {
  source                = "../modules/database"
  name                  = local.name
  engine_version        = var.db_engine_version
  db_subnet_group_name  = module.network.db_subnet_group_name
  security_group_id     = module.network.database_security_group_id
  kms_key_arn           = module.kms.key_arn
  min_acu               = var.db_min_acu
  max_acu               = var.db_max_acu
  backup_retention_days = var.db_backup_retention_days
  deletion_protection   = var.protect_data
  final_snapshot        = var.protect_data
  performance_insights  = var.environment == "prod"
  enhanced_monitoring   = var.environment == "prod"
}

module "storage" {
  source                    = "../modules/storage"
  name                      = local.name
  bucket_name               = "${local.name}-documents-${local.account_id}"
  kms_key_arn               = module.kms.key_arn
  force_destroy             = !var.protect_data
  upload_origins            = var.upload_origins
  ingest_function_timeout_s = var.ingest_timeout_s
}

module "auth" {
  source              = "../modules/auth"
  name                = local.name
  deletion_protection = var.protect_data
  mfa_required        = var.mfa_required
  allow_password_auth = var.allow_password_auth
}

module "guardrail" {
  count       = var.guardrail_enabled ? 1 : 0
  source      = "../modules/guardrail"
  name        = local.name
  kms_key_arn = module.kms.key_arn
}

# ---------------------------------------------------------------------- functions

module "api_function" {
  source             = "../modules/lambda_function"
  name               = "${local.name}-api"
  handler            = "ragplatform.handlers.api.handler"
  artifact_bucket    = var.artifact_bucket
  artifact_key       = var.artifact_key
  artifact_sha256    = var.artifact_sha256
  memory_mb          = var.api_memory_mb
  timeout_s          = 29 # the API Gateway integration limit
  kms_key_arn        = module.kms.key_arn
  log_retention_days = var.log_retention_days
  environment        = local.base_env

  statements = concat(local.data_api_statements, [
    {
      Sid      = "InvokeModels"
      Effect   = "Allow"
      Action   = "bedrock:InvokeModel"
      Resource = concat([local.embedding_model_arn], local.chat_model_arns)
    },
    {
      # Presigned POST uploads are signed with this role's credentials, so the
      # role needs PutObject and permission to create the object's data key.
      Sid      = "PresignUploads"
      Effect   = "Allow"
      Action   = ["s3:PutObject"]
      Resource = "${module.storage.bucket_arn}/raw/*"
    },
    {
      Sid      = "PurgeDeletedDocuments"
      Effect   = "Allow"
      Action   = ["s3:DeleteObject", "s3:DeleteObjectVersion"]
      Resource = "${module.storage.bucket_arn}/raw/*"
    },
    {
      Sid       = "ListVersionsForPurge"
      Effect    = "Allow"
      Action    = "s3:ListBucketVersions"
      Resource  = module.storage.bucket_arn
      Condition = { StringLike = { "s3:prefix" = ["raw/*"] } }
    },
    {
      Sid      = "EncryptUploads"
      Effect   = "Allow"
      Action   = ["kms:GenerateDataKey"]
      Resource = module.kms.key_arn
    },
    ], var.rerank_model_id == null ? [] : [
    {
      Sid      = "Rerank"
      Effect   = "Allow"
      Action   = "bedrock:InvokeModel"
      Resource = "arn:aws:bedrock:${var.region}::foundation-model/${var.rerank_model_id}"
    },
    {
      Sid      = "RerankApi"
      Effect   = "Allow"
      Action   = "bedrock:Rerank"
      Resource = "*" # the Rerank action does not support resource-level permissions
    },
    ], var.guardrail_enabled ? [
    {
      Sid      = "Guardrail"
      Effect   = "Allow"
      Action   = "bedrock:ApplyGuardrail"
      Resource = module.guardrail[0].guardrail_arn
    },
  ] : [])
}

module "ingest_function" {
  source             = "../modules/lambda_function"
  name               = "${local.name}-ingest"
  handler            = "ragplatform.handlers.ingest.handler"
  artifact_bucket    = var.artifact_bucket
  artifact_key       = var.artifact_key
  artifact_sha256    = var.artifact_sha256
  memory_mb          = var.ingest_memory_mb
  timeout_s          = var.ingest_timeout_s
  kms_key_arn        = module.kms.key_arn
  log_retention_days = var.log_retention_days
  environment        = local.base_env

  statements = concat(local.data_api_statements, [
    {
      Sid      = "EmbedChunks"
      Effect   = "Allow"
      Action   = "bedrock:InvokeModel"
      Resource = local.embedding_model_arn
    },
    {
      Sid      = "ReadUploads"
      Effect   = "Allow"
      Action   = "s3:GetObject"
      Resource = "${module.storage.bucket_arn}/raw/*"
    },
    {
      Sid      = "ConsumeIngestQueue"
      Effect   = "Allow"
      Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"]
      Resource = module.storage.queue_arn
    },
  ])
}

module "migrate_function" {
  source             = "../modules/lambda_function"
  name               = "${local.name}-migrate"
  handler            = "ragplatform.handlers.migrate.handler"
  artifact_bucket    = var.artifact_bucket
  artifact_key       = var.artifact_key
  artifact_sha256    = var.artifact_sha256
  memory_mb          = 512
  timeout_s          = 300
  kms_key_arn        = module.kms.key_arn
  log_retention_days = var.log_retention_days
  environment = merge(local.base_env, {
    RAG_DB_ADMIN_SECRET_ARN = module.database.master_secret_arn
  })

  statements = [
    {
      Sid    = "DataApi"
      Effect = "Allow"
      Action = [
        "rds-data:ExecuteStatement", "rds-data:BatchExecuteStatement",
        "rds-data:BeginTransaction", "rds-data:CommitTransaction", "rds-data:RollbackTransaction",
      ]
      Resource = module.database.cluster_arn
    },
    {
      Sid      = "ReadSecrets"
      Effect   = "Allow"
      Action   = "secretsmanager:GetSecretValue"
      Resource = [module.database.master_secret_arn, module.database.app_secret_arn]
    },
    {
      Sid      = "WriteRuntimeRoleSecret"
      Effect   = "Allow"
      Action   = "secretsmanager:PutSecretValue"
      Resource = module.database.app_secret_arn
    },
    {
      Sid      = "UseDataKey"
      Effect   = "Allow"
      Action   = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey"]
      Resource = module.kms.key_arn
    },
  ]
}

resource "aws_lambda_event_source_mapping" "ingest" {
  event_source_arn                   = module.storage.queue_arn
  function_name                      = module.ingest_function.alias_arn
  batch_size                         = 5
  maximum_batching_window_in_seconds = 5
  function_response_types            = ["ReportBatchItemFailures"]

  # Caps concurrent invocations from this queue. Messages wait in the queue
  # rather than being throttled, which is the backpressure we want.
  scaling_config {
    maximum_concurrency = var.ingest_max_concurrency
  }
}

module "api" {
  source              = "../modules/api"
  name                = local.name
  stage_name          = var.environment
  user_pool_arn       = module.auth.user_pool_arn
  function_name       = module.api_function.function_name
  function_invoke_arn = module.api_function.alias_invoke_arn
  kms_key_arn         = module.kms.key_arn
  log_retention_days  = var.log_retention_days
  throttle_rate       = var.api_throttle_rate
  throttle_burst      = var.api_throttle_burst
  waf_enabled         = var.waf_enabled
}

module "observability" {
  source               = "../modules/observability"
  name                 = local.name
  region               = var.region
  kms_key_arn          = module.kms.key_arn
  alert_emails         = var.alert_emails
  api_name             = module.api.rest_api_name
  stage_name           = module.api.stage_name
  api_function_name    = module.api_function.function_name
  ingest_function_name = module.ingest_function.function_name
  queue_name           = module.storage.queue_name
  dlq_name             = module.storage.dlq_name
  db_cluster_id        = module.database.cluster_id
  monthly_budget_usd   = var.monthly_budget_usd
  project_tag          = var.project
}
