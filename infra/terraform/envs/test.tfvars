# Test: same shape as prod, but pauses when idle and can be destroyed.
environment            = "test"
region                 = "us-east-1"
db_min_acu             = 0
db_max_acu             = 4
protect_data           = false
waf_enabled            = true
mfa_required           = false
allow_password_auth    = true
ingest_max_concurrency = 5
log_retention_days     = 14
monthly_budget_usd     = 100
