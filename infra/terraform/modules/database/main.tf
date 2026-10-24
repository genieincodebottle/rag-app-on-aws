# Aurora PostgreSQL Serverless v2 with the RDS Data API.
#
# * min_capacity 0 lets an idle dev or test cluster pause; the first request
#   after a pause waits about 15 seconds (longer after more than a day idle).
#   The application retries DatabaseResumingException and returns 503 with
#   Retry-After if the resume takes longer than its wait budget.
# * Pausing needs pg 16.3+ (or 15.7+, 14.12+, 13.15+) and is blocked by any open
#   connection, by RDS Proxy, and by traffic arriving more often than
#   seconds_until_auto_pause. A health check polling the database keeps it awake.
# * The master password is generated and stored by RDS in Secrets Manager
#   (manage_master_user_password), so it never appears in Terraform state.
# * The runtime role's secret is created empty here. The migrate function
#   creates the role and writes the secret value, so that password never
#   passes through Terraform either.

resource "aws_rds_cluster_parameter_group" "this" {
  name        = "${var.name}-pg16"
  family      = "aurora-postgresql16"
  description = "${var.name} cluster parameters"

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }
  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }
  # Statement text can include document content in literals. Only slow
  # statements are logged, and parameters are never logged.
  parameter {
    name  = "log_statement"
    value = "none"
  }
}

resource "aws_rds_cluster" "this" {
  #checkov:skip=CKV2_AWS_8:Aurora automated backups with point-in-time restore cover recovery; AWS Backup vaults are an optional addition (docs/operations-runbook.md)
  #checkov:skip=CKV_AWS_139:Deletion protection is driven by var.deletion_protection, true in prod
  #checkov:skip=CKV_AWS_162:Access is through the RDS Data API with Secrets Manager credentials; IAM database authentication is not used
  cluster_identifier = var.name
  engine             = "aurora-postgresql"
  engine_mode        = "provisioned"
  engine_version     = var.engine_version
  database_name      = var.database_name
  master_username    = "rag_owner"

  manage_master_user_password   = true
  master_user_secret_kms_key_id = var.kms_key_arn

  enable_http_endpoint = true # RDS Data API

  db_subnet_group_name            = var.db_subnet_group_name
  vpc_security_group_ids          = [var.security_group_id]
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.this.name

  storage_encrypted = true
  kms_key_id        = var.kms_key_arn

  backup_retention_period      = var.backup_retention_days
  preferred_backup_window      = "02:00-03:00"
  preferred_maintenance_window = "sun:03:30-sun:04:30"
  copy_tags_to_snapshot        = true

  deletion_protection       = var.deletion_protection
  skip_final_snapshot       = !var.final_snapshot
  final_snapshot_identifier = var.final_snapshot ? "${var.name}-final" : null

  enabled_cloudwatch_logs_exports = ["postgresql"]

  serverlessv2_scaling_configuration {
    min_capacity             = var.min_acu
    max_capacity             = var.max_acu
    seconds_until_auto_pause = var.min_acu == 0 ? var.seconds_until_auto_pause : null
  }

  lifecycle {
    # The final snapshot identifier must not change between applies, or a
    # destroy would try to create a second snapshot with a new name.
    ignore_changes = [final_snapshot_identifier]
  }
}

resource "aws_rds_cluster_instance" "writer" {
  identifier                   = "${var.name}-writer"
  cluster_identifier           = aws_rds_cluster.this.id
  instance_class               = "db.serverless"
  engine                       = aws_rds_cluster.this.engine
  engine_version               = aws_rds_cluster.this.engine_version
  db_subnet_group_name         = var.db_subnet_group_name
  performance_insights_enabled = var.performance_insights
  # Performance Insights keeps the instance's KMS key; reuse the data key.
  performance_insights_kms_key_id = var.performance_insights ? var.kms_key_arn : null
  auto_minor_version_upgrade      = true
  monitoring_interval             = var.enhanced_monitoring ? 60 : 0
  monitoring_role_arn             = var.enhanced_monitoring ? aws_iam_role.monitoring[0].arn : null
}

resource "aws_iam_role" "monitoring" {
  count = var.enhanced_monitoring ? 1 : 0
  name  = "${var.name}-rds-monitoring"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "monitoring.rds.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "monitoring" {
  count      = var.enhanced_monitoring ? 1 : 0
  role       = aws_iam_role.monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Runtime role credentials, written by the migrate function. No version is
# created here on purpose.
resource "aws_secretsmanager_secret" "app" {
  #checkov:skip=CKV2_AWS_57:Rotated by invoking the migrate function with rotate_app_password; scheduled rotation is a known gap (docs/security.md)
  name                    = "${var.name}/app-role"
  description             = "Aurora runtime role (rag_app). Written by the migrate function."
  kms_key_id              = var.kms_key_arn
  recovery_window_in_days = var.deletion_protection ? 30 : 0
}

variable "name" {
  type = string
}

variable "engine_version" {
  type = string
  # 16.13 ships pgvector 0.8.1, the version the local docker image runs.
  default = "16.13"
  validation {
    # Scale-to-zero needs 16.3 or later; the RDS Data API needs 16.1 or later.
    condition     = can(regex("^16\\.(([3-9])|([1-9][0-9]))$", var.engine_version))
    error_message = "Use Aurora PostgreSQL 16.3 or later (scale to zero and the Data API)."
  }
}

variable "database_name" {
  type    = string
  default = "ragplatform"
}

variable "db_subnet_group_name" {
  type = string
}

variable "security_group_id" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "min_acu" {
  type    = number
  default = 0
}

variable "max_acu" {
  type    = number
  default = 4
}

variable "seconds_until_auto_pause" {
  type    = number
  default = 900
  validation {
    condition     = var.seconds_until_auto_pause >= 300 && var.seconds_until_auto_pause <= 86400
    error_message = "seconds_until_auto_pause must be between 300 and 86400."
  }
}

variable "backup_retention_days" {
  type    = number
  default = 7
}

variable "deletion_protection" {
  type    = bool
  default = false
}

variable "final_snapshot" {
  type    = bool
  default = false
}

variable "enhanced_monitoring" {
  type    = bool
  default = false
}

variable "performance_insights" {
  type    = bool
  default = false
}

output "cluster_arn" {
  value = aws_rds_cluster.this.arn
}

output "cluster_id" {
  value = aws_rds_cluster.this.cluster_identifier
}

output "master_secret_arn" {
  value = aws_rds_cluster.this.master_user_secret[0].secret_arn
}

output "app_secret_arn" {
  value = aws_secretsmanager_secret.app.arn
}

output "database_name" {
  value = var.database_name
}
