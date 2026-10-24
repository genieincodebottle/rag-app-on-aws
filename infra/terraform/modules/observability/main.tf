# Alarms, dashboard and cost budget.
#
# Application metrics arrive in the RagPlatform namespace through CloudWatch
# Embedded Metric Format (Powertools Metrics), with dimension service=ragplatform.
# Every alarm here maps to a runbook entry in docs/operations-runbook.md.

locals {
  ns  = "RagPlatform"
  svc = { service = "ragplatform" }
  functions = {
    api    = var.api_function_name
    ingest = var.ingest_function_name
  }
}

resource "aws_sns_topic" "alerts" {
  name              = "${var.name}-alerts"
  kms_master_key_id = var.kms_key_arn
}

resource "aws_sns_topic_subscription" "email" {
  for_each  = toset(var.alert_emails)
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = each.value
}

# ------------------------------------------------------------------ API

resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "${var.name}-api-5xx"
  alarm_description   = "API returned 5xx responses. Runbook: API 5xx errors."
  namespace           = "AWS/ApiGateway"
  metric_name         = "5XXError"
  dimensions          = { ApiName = var.api_name, Stage = var.stage_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "api_latency" {
  alarm_name          = "${var.name}-api-latency-p95"
  alarm_description   = "p95 API latency above ${var.latency_p95_ms} ms for 15 minutes. Runbook: Slow answers."
  namespace           = "AWS/ApiGateway"
  metric_name         = "Latency"
  dimensions          = { ApiName = var.api_name, Stage = var.stage_name }
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 3
  threshold           = var.latency_p95_ms
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "internal_errors" {
  alarm_name          = "${var.name}-internal-errors"
  alarm_description   = "Unhandled exceptions in the API. Runbook: API 5xx errors."
  namespace           = local.ns
  metric_name         = "ApiError"
  dimensions          = merge(local.svc, { code = "internal_error" })
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# ------------------------------------------------------------------ Lambda

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  for_each            = local.functions
  alarm_name          = "${var.name}-${each.key}-errors"
  alarm_description   = "Errors in the ${each.key} function. Runbook: Lambda errors."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = each.value }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  for_each            = local.functions
  alarm_name          = "${var.name}-${each.key}-throttles"
  alarm_description   = "The ${each.key} function is being throttled. Runbook: Throttling."
  namespace           = "AWS/Lambda"
  metric_name         = "Throttles"
  dimensions          = { FunctionName = each.value }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# ------------------------------------------------------------------ ingestion

resource "aws_cloudwatch_metric_alarm" "dlq" {
  alarm_name          = "${var.name}-ingest-dlq-not-empty"
  alarm_description   = "Messages reached the ingest DLQ. Runbook: Messages in the ingest DLQ."
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = var.dlq_name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "queue_age" {
  alarm_name          = "${var.name}-ingest-backlog"
  alarm_description   = "Oldest ingest message is older than 15 minutes. Runbook: Ingestion backlog."
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateAgeOfOldestMessage"
  dimensions          = { QueueName = var.queue_name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 900
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "purge_failed" {
  alarm_name          = "${var.name}-blob-purge-failed"
  alarm_description   = "A deleted document's file could not be removed from S3. Runbook: Deletion purge failed."
  namespace           = local.ns
  metric_name         = "BlobPurgeFailed"
  dimensions          = local.svc
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# ------------------------------------------------------------------ answer quality

resource "aws_cloudwatch_metric_alarm" "insufficient_ratio" {
  alarm_name          = "${var.name}-insufficient-context-ratio"
  alarm_description   = "Over ${var.insufficient_ratio_threshold * 100}% of answers in the last hour were insufficient_context. Runbook: Answer quality drop."
  evaluation_periods  = 1
  threshold           = var.insufficient_ratio_threshold
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  metric_query {
    id          = "ratio"
    expression  = "IF(total > 20, insufficient / total, 0)"
    label       = "Insufficient-context share"
    return_data = true
  }
  metric_query {
    id = "insufficient"
    metric {
      namespace   = local.ns
      metric_name = "AnswerStatus"
      dimensions  = merge(local.svc, { status = "insufficient_context" })
      period      = 3600
      stat        = "Sum"
    }
  }
  metric_query {
    id         = "total"
    expression = "insufficient + answered + conflicting"
  }
  metric_query {
    id = "answered"
    metric {
      namespace   = local.ns
      metric_name = "AnswerStatus"
      dimensions  = merge(local.svc, { status = "answered" })
      period      = 3600
      stat        = "Sum"
    }
  }
  metric_query {
    id = "conflicting"
    metric {
      namespace   = local.ns
      metric_name = "AnswerStatus"
      dimensions  = merge(local.svc, { status = "conflicting_sources" })
      period      = 3600
      stat        = "Sum"
    }
  }
}

# ------------------------------------------------------------------ database

resource "aws_cloudwatch_metric_alarm" "aurora_capacity" {
  alarm_name          = "${var.name}-aurora-at-max-capacity"
  alarm_description   = "Aurora used over 90% of its maximum ACUs for 15 minutes. Runbook: Database at capacity."
  namespace           = "AWS/RDS"
  metric_name         = "ACUUtilization"
  dimensions          = { DBClusterIdentifier = var.db_cluster_id }
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 90
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# ------------------------------------------------------------------ cost

resource "aws_budgets_budget" "monthly" {
  count        = var.monthly_budget_usd > 0 ? 1 : 0
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name = "TagKeyValue"
    # The Project tag must be activated as a cost allocation tag in the
    # Billing console before this filter matches anything.
    values = [format("user:Project$%s", var.project_tag)]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_emails
  }
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = var.alert_emails
  }
}

# ------------------------------------------------------------------ dashboard

resource "aws_cloudwatch_dashboard" "this" {
  dashboard_name = var.name
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title = "API requests and errors", region = var.region, stat = "Sum", period = 300
          metrics = [
            ["AWS/ApiGateway", "Count", "ApiName", var.api_name, "Stage", var.stage_name],
            [".", "4XXError", ".", ".", ".", "."],
            [".", "5XXError", ".", ".", ".", "."],
          ]
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title = "Latency (ms)", region = var.region, period = 300
          metrics = [
            ["AWS/ApiGateway", "Latency", "ApiName", var.api_name, "Stage", var.stage_name, { stat = "p50" }],
            ["...", { stat = "p95" }],
            [local.ns, "RetrievalLatency", "service", "ragplatform", { stat = "p95" }],
            [local.ns, "ModelLatency", "service", "ragplatform", { stat = "p95" }],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 8, height = 6
        properties = {
          title = "Answer status", region = var.region, stat = "Sum", period = 3600
          metrics = [
            [local.ns, "AnswerStatus", "service", "ragplatform", "status", "answered"],
            ["...", "insufficient_context"],
            ["...", "conflicting_sources"],
          ]
        }
      },
      {
        type = "metric", x = 8, y = 6, width = 8, height = 6
        properties = {
          title = "Tokens and estimated cost (micro-USD)", region = var.region, stat = "Sum", period = 3600
          metrics = [
            [local.ns, "InputTokens", "service", "ragplatform"],
            [".", "OutputTokens", ".", "."],
            [".", "EstimatedCostMicroUSD", ".", "."],
          ]
        }
      },
      {
        type = "metric", x = 16, y = 6, width = 8, height = 6
        properties = {
          title = "Cache and safety signals", region = var.region, stat = "Sum", period = 3600
          metrics = [
            [local.ns, "CacheHit", "service", "ragplatform"],
            [".", "CacheMiss", ".", "."],
            [".", "InvalidCitationsRemoved", ".", "."],
            [".", "FlaggedSourcesInContext", ".", "."],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 12, width = 12, height = 6
        properties = {
          title = "Ingestion", region = var.region, stat = "Sum", period = 300
          metrics = [
            [local.ns, "ChunksWritten", "service", "ragplatform"],
            [".", "EmbeddingsReused", ".", "."],
            [".", "IngestRetried", ".", "."],
            ["AWS/SQS", "ApproximateNumberOfMessagesVisible", "QueueName", var.queue_name, { stat = "Maximum" }],
            ["AWS/SQS", "ApproximateNumberOfMessagesVisible", "QueueName", var.dlq_name, { stat = "Maximum" }],
          ]
        }
      },
      {
        type = "metric", x = 12, y = 12, width = 12, height = 6
        properties = {
          title = "Aurora capacity (ACU)", region = var.region, period = 300
          metrics = [
            ["AWS/RDS", "ServerlessDatabaseCapacity", "DBClusterIdentifier", var.db_cluster_id, { stat = "Average" }],
            [".", "ACUUtilization", ".", ".", { stat = "Average", yAxis = "right" }],
          ]
        }
      },
    ]
  })
}

variable "name" {
  type = string
}

variable "region" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "alert_emails" {
  type    = list(string)
  default = []
}

variable "api_name" {
  type = string
}

variable "stage_name" {
  type = string
}

variable "api_function_name" {
  type = string
}

variable "ingest_function_name" {
  type = string
}

variable "queue_name" {
  type = string
}

variable "dlq_name" {
  type = string
}

variable "db_cluster_id" {
  type = string
}

variable "latency_p95_ms" {
  type    = number
  default = 15000
}

variable "insufficient_ratio_threshold" {
  type    = number
  default = 0.5
}

variable "monthly_budget_usd" {
  type    = number
  default = 0
}

variable "project_tag" {
  type = string
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}

output "dashboard_name" {
  value = aws_cloudwatch_dashboard.this.dashboard_name
}
