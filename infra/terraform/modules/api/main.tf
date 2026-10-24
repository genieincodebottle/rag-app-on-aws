# API Gateway REST API in front of the api function's "live" alias.
#
# * GET /v1/health is open, for load balancers and smoke tests. It does not
#   touch the database.
# * Every other path goes through the Cognito authorizer, which validates the
#   ID token and passes its claims to the function. The function trusts only
#   those claims for identity.
# * Stage throttling caps the request rate for the whole API; the optional WAF
#   adds AWS managed rule groups and a per-IP rate limit.
# * Access logs record request metadata only, never bodies (questions and
#   documents would otherwise land in CloudWatch).

resource "aws_api_gateway_rest_api" "this" {
  name                         = var.name
  description                  = "RAG platform API"
  disable_execute_api_endpoint = false
  endpoint_configuration {
    types = ["REGIONAL"]
  }
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_authorizer" "cognito" {
  name            = "${var.name}-cognito"
  rest_api_id     = aws_api_gateway_rest_api.this.id
  type            = "COGNITO_USER_POOLS"
  provider_arns   = [var.user_pool_arn]
  identity_source = "method.request.header.Authorization"
}

# /v1/health -----------------------------------------------------------------
resource "aws_api_gateway_resource" "v1" {
  rest_api_id = aws_api_gateway_rest_api.this.id
  parent_id   = aws_api_gateway_rest_api.this.root_resource_id
  path_part   = "v1"
}

resource "aws_api_gateway_resource" "health" {
  rest_api_id = aws_api_gateway_rest_api.this.id
  parent_id   = aws_api_gateway_resource.v1.id
  path_part   = "health"
}

resource "aws_api_gateway_method" "health" {
  #checkov:skip=CKV_AWS_59:Intentionally public liveness check; returns a constant and touches no data
  #checkov:skip=CKV2_AWS_53:No request body; nothing to validate
  rest_api_id   = aws_api_gateway_rest_api.this.id
  resource_id   = aws_api_gateway_resource.health.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "health" {
  rest_api_id             = aws_api_gateway_rest_api.this.id
  resource_id             = aws_api_gateway_resource.health.id
  http_method             = aws_api_gateway_method.health.http_method
  type                    = "AWS_PROXY"
  integration_http_method = "POST"
  uri                     = var.function_invoke_arn
}

# /{proxy+} ------------------------------------------------------------------
resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.this.id
  parent_id   = aws_api_gateway_rest_api.this.root_resource_id
  path_part   = "{proxy+}"
}

resource "aws_api_gateway_method" "proxy" {
  #checkov:skip=CKV2_AWS_53:Request bodies are validated by the FastAPI models in the function; a proxy resource cannot carry per-route schemas
  rest_api_id   = aws_api_gateway_rest_api.this.id
  resource_id   = aws_api_gateway_resource.proxy.id
  http_method   = "ANY"
  authorization = "COGNITO_USER_POOLS"
  authorizer_id = aws_api_gateway_authorizer.cognito.id
}

resource "aws_api_gateway_integration" "proxy" {
  rest_api_id             = aws_api_gateway_rest_api.this.id
  resource_id             = aws_api_gateway_resource.proxy.id
  http_method             = aws_api_gateway_method.proxy.http_method
  type                    = "AWS_PROXY"
  integration_http_method = "POST"
  uri                     = var.function_invoke_arn
  timeout_milliseconds    = 29000
}

resource "aws_api_gateway_deployment" "this" {
  rest_api_id = aws_api_gateway_rest_api.this.id
  triggers = {
    redeploy = sha1(jsonencode([
      aws_api_gateway_method.health, aws_api_gateway_integration.health,
      aws_api_gateway_method.proxy, aws_api_gateway_integration.proxy,
      aws_api_gateway_authorizer.cognito,
    ]))
  }
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_cloudwatch_log_group" "access" {
  name              = "/aws/apigateway/${var.name}"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_arn
}

resource "aws_api_gateway_stage" "this" {
  #checkov:skip=CKV2_AWS_51:Clients authenticate with Cognito ID tokens; mutual TLS client certificates are not used
  #checkov:skip=CKV_AWS_120:Answers depend on the caller identity and ACL groups; a shared response cache would serve one user's answer to another
  #checkov:skip=CKV2_AWS_77:The WAF (on in test and prod) includes AWSManagedRulesKnownBadInputsRuleSet, which covers Log4j
  rest_api_id          = aws_api_gateway_rest_api.this.id
  deployment_id        = aws_api_gateway_deployment.this.id
  stage_name           = var.stage_name
  xray_tracing_enabled = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.access.arn
    format = jsonencode({
      requestId          = "$context.requestId"
      requestTime        = "$context.requestTime"
      httpMethod         = "$context.httpMethod"
      resourcePath       = "$context.resourcePath"
      status             = "$context.status"
      responseLength     = "$context.responseLength"
      integrationLatency = "$context.integrationLatency"
      latency            = "$context.responseLatency"
      sourceIp           = "$context.identity.sourceIp"
      userAgent          = "$context.identity.userAgent"
      sub                = "$context.authorizer.claims.sub"
      tenant             = "$context.authorizer.claims.custom:tenant_id"
      error              = "$context.error.message"
      integrationError   = "$context.integrationErrorMessage"
      wafStatus          = "$context.waf.status"
    })
  }
}

resource "aws_api_gateway_method_settings" "all" {
  #checkov:skip=CKV_AWS_225:Caching stays off for the same reason as the stage: responses are per user
  rest_api_id = aws_api_gateway_rest_api.this.id
  stage_name  = aws_api_gateway_stage.this.stage_name
  method_path = "*/*"
  settings {
    metrics_enabled        = true
    logging_level          = "ERROR"
    data_trace_enabled     = false # would log full request and response bodies
    throttling_rate_limit  = var.throttle_rate
    throttling_burst_limit = var.throttle_burst
  }
}

resource "aws_lambda_permission" "api" {
  statement_id  = "AllowApiGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = var.function_name
  qualifier     = "live"
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.this.execution_arn}/*/*"
}

# ---------------------------------------------------------------------- WAF

resource "aws_wafv2_web_acl" "this" {
  count = var.waf_enabled ? 1 : 0
  name  = var.name
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "aws-common"
    priority = 10
    override_action {
      none {}
    }
    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-common"
      sampled_requests_enabled   = false
    }
  }

  rule {
    name     = "aws-known-bad-inputs"
    priority = 20
    override_action {
      none {}
    }
    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-bad-inputs"
      sampled_requests_enabled   = false
    }
  }

  rule {
    name     = "per-ip-rate-limit"
    priority = 30
    action {
      block {}
    }
    statement {
      rate_based_statement {
        limit              = var.waf_rate_limit_per_5min
        aggregate_key_type = "IP"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name}-rate-limit"
      sampled_requests_enabled   = false
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = var.name
    sampled_requests_enabled   = false
  }
}

# WAF log group names must start with "aws-waf-logs-". The Authorization header
# is redacted so ID tokens never reach the logs.
resource "aws_cloudwatch_log_group" "waf" {
  count             = var.waf_enabled ? 1 : 0
  name              = "aws-waf-logs-${var.name}"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_arn
}

resource "aws_wafv2_web_acl_logging_configuration" "this" {
  count                   = var.waf_enabled ? 1 : 0
  resource_arn            = aws_wafv2_web_acl.this[0].arn
  log_destination_configs = [aws_cloudwatch_log_group.waf[0].arn]
  redacted_fields {
    single_header {
      name = "authorization"
    }
  }
}

resource "aws_wafv2_web_acl_association" "this" {
  count        = var.waf_enabled ? 1 : 0
  resource_arn = aws_api_gateway_stage.this.arn
  web_acl_arn  = aws_wafv2_web_acl.this[0].arn
}

variable "name" {
  type = string
}

variable "stage_name" {
  type = string
}

variable "user_pool_arn" {
  type = string
}

variable "function_name" {
  type = string
}

variable "function_invoke_arn" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "throttle_rate" {
  type    = number
  default = 20
}

variable "throttle_burst" {
  type    = number
  default = 40
}

variable "waf_enabled" {
  type    = bool
  default = false
}

variable "waf_rate_limit_per_5min" {
  type    = number
  default = 1000
}

output "invoke_url" {
  value = aws_api_gateway_stage.this.invoke_url
}

output "rest_api_name" {
  value = aws_api_gateway_rest_api.this.name
}

output "stage_name" {
  value = aws_api_gateway_stage.this.stage_name
}
