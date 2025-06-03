# =========================
# Root terraform.tfvars
# =========================

# -------------------------
# Project Settings
# -------------------------
project_name = "rag-app"
stage        = "dev"
aws_region   = "us-east-1"

# -------------------------
# Lambda Settings
# -------------------------
lambda_memory_size = 512
lambda_timeout     = 150

# -------------------------
# GitHub Repository Settings
# -------------------------
github_repo   = "genieincodebottle/rag-app-on-aws"
github_branch = "develop"

# -------------------------
# Monitoring
# -------------------------
alert_email = "rajsrivastava2@gmail.com"

# -------------------------
# VPC Settings
# -------------------------
vpc_cidr              = "10.0.0.0/16"
az_count              = 2
single_nat_gateway    = true        # Cost saving for dev environment
enable_flow_logs      = false       # Only needed for prod
create_bastion_sg     = true        # Useful for dev environment
bastion_allowed_cidr  = ["0.0.0.0/0"]  # Restrict this in production

# -------------------------
# Storage Settings
# -------------------------
enable_lifecycle_rules       = false  # Only enable in prod
standard_ia_transition_days  = 90
glacier_transition_days      = 365

# -------------------------
# Database Settings
# -------------------------
db_instance_class     = "db.t3.micro"
db_allocated_storage  = 20
db_name               = "ragapp"
db_username           = "ragadmin"
import_db             = false
reset_db_password     = false  # Only set to true when you need to reset the password

# MCP Configuration
mcp_timeout = 60
rag_confidence_threshold = 0.7
min_context_length = 100
