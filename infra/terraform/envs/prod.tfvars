# Production: never pauses (first-request latency), protected from deletion,
# WAF on, MFA required, password sign-in off (SRP or hosted UI only).
environment               = "prod"
region                    = "us-east-1"
db_min_acu                = 0.5
db_max_acu                = 16
db_backup_retention_days  = 35
protect_data              = true
waf_enabled               = true
mfa_required              = true
allow_password_auth       = false
fallback_chat_model_id    = "us.anthropic.claude-haiku-5-5"
guardrail_enabled         = true
guardrail_grounding_check = false
ingest_max_concurrency    = 10
api_throttle_rate         = 50
api_throttle_burst        = 100
log_retention_days        = 90
monthly_budget_usd        = 500
