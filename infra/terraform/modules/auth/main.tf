# Cognito user pool. Users are created by an administrator, never by sign-up,
# because every user must be assigned to a tenant before they can do anything.
#
# custom:tenant_id is readable by the app client but NOT writable: Cognito lets
# a signed-in user update their own attributes with their access token, and a
# writable tenant attribute would let any user move themselves into another
# tenant. Only an administrator (AdminUpdateUserAttributes) can set it.

resource "aws_cognito_user_pool" "this" {
  name                = var.name
  deletion_protection = var.deletion_protection ? "ACTIVE" : "INACTIVE"
  username_attributes = ["email"]
  mfa_configuration   = var.mfa_required ? "ON" : "OPTIONAL"

  admin_create_user_config {
    allow_admin_create_user_only = true
  }

  password_policy {
    minimum_length                   = 14
    require_lowercase                = true
    require_uppercase                = true
    require_numbers                  = true
    require_symbols                  = false
    temporary_password_validity_days = 3
  }

  software_token_mfa_configuration {
    enabled = true
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  schema {
    name                     = "tenant_id"
    attribute_data_type      = "String"
    mutable                  = true
    developer_only_attribute = false
    required                 = false
    string_attribute_constraints {
      min_length = 2
      max_length = 63
    }
  }

  schema {
    name                = "email"
    attribute_data_type = "String"
    mutable             = true
    required            = true
    string_attribute_constraints {
      min_length = 3
      max_length = 254
    }
  }
}

resource "aws_cognito_user_group" "tenant_admin" {
  name         = "tenant-admin"
  user_pool_id = aws_cognito_user_pool.this.id
  description  = "Can read every document in their tenant and manage any document's access"
}

resource "aws_cognito_user_pool_client" "app" {
  name         = "${var.name}-app"
  user_pool_id = aws_cognito_user_pool.this.id

  generate_secret               = false
  prevent_user_existence_errors = "ENABLED"
  enable_token_revocation       = true

  explicit_auth_flows = concat(
    ["ALLOW_USER_SRP_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"],
    # The Streamlit demo client signs in with username and password over TLS.
    # Turn this off in environments where only SRP or the hosted UI is used.
    var.allow_password_auth ? ["ALLOW_USER_PASSWORD_AUTH"] : [],
  )

  read_attributes  = ["email", "email_verified", "custom:tenant_id"]
  write_attributes = ["email"]

  access_token_validity  = 60
  id_token_validity      = 60
  refresh_token_validity = 12
  token_validity_units {
    access_token  = "minutes"
    id_token      = "minutes"
    refresh_token = "hours"
  }
}

variable "name" {
  type = string
}

variable "deletion_protection" {
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

output "user_pool_id" {
  value = aws_cognito_user_pool.this.id
}

output "user_pool_arn" {
  value = aws_cognito_user_pool.this.arn
}

output "client_id" {
  value = aws_cognito_user_pool_client.app.id
}
