# Development: smallest footprint. Aurora pauses at 0 ACU after 15 idle minutes.
# account_id, artifact_bucket, artifact_key and artifact_sha256 are passed by
# the deploy workflow with -var, so this file holds only settings.
environment            = "dev"
region                 = "us-east-1"
db_min_acu             = 0
db_max_acu             = 2
protect_data           = false
waf_enabled            = false
mfa_required           = false
allow_password_auth    = true
ingest_max_concurrency = 2
api_throttle_rate      = 5
api_throttle_burst     = 10
log_retention_days     = 14
monthly_budget_usd     = 50
