# Bedrock Guardrail used by the api function (optional, off by default in dev).
#
# * The prompt-attack filter screens the user's question. The application puts
#   only the question in a guardContent block, so retrieved documents, which
#   legitimately contain imperative text, are not screened by this filter.
# * Credentials and card numbers in answers are blocked outright.
# * The contextual grounding filter is applied by the application through
#   ApplyGuardrail on the final answer when RAG_GUARDRAIL_GROUNDING_CHECK=true.
#
# Guardrails bill per 1,000 text units (1,000 characters each). Check the
# Bedrock pricing page before enabling it on a high-traffic environment.

resource "aws_bedrock_guardrail" "this" {
  name                      = var.name
  description               = "Prompt attack screening, secret blocking and grounding for the RAG platform"
  blocked_input_messaging   = "Your question was blocked by the content policy."
  blocked_outputs_messaging = "The answer was blocked by the content policy."
  kms_key_arn               = var.kms_key_arn

  content_policy_config {
    filters_config {
      type            = "PROMPT_ATTACK"
      input_strength  = "HIGH"
      output_strength = "NONE"
    }
    filters_config {
      type            = "HATE"
      input_strength  = "MEDIUM"
      output_strength = "MEDIUM"
    }
    filters_config {
      type            = "VIOLENCE"
      input_strength  = "MEDIUM"
      output_strength = "MEDIUM"
    }
  }

  sensitive_information_policy_config {
    pii_entities_config {
      type   = "AWS_ACCESS_KEY"
      action = "BLOCK"
    }
    pii_entities_config {
      type   = "AWS_SECRET_KEY"
      action = "BLOCK"
    }
    pii_entities_config {
      type   = "CREDIT_DEBIT_CARD_NUMBER"
      action = "BLOCK"
    }
    pii_entities_config {
      type   = "PASSWORD"
      action = "BLOCK"
    }
  }

  contextual_grounding_policy_config {
    filters_config {
      type      = "GROUNDING"
      threshold = var.grounding_threshold
    }
    filters_config {
      type      = "RELEVANCE"
      threshold = var.relevance_threshold
    }
  }
}

resource "aws_bedrock_guardrail_version" "this" {
  guardrail_arn = aws_bedrock_guardrail.this.guardrail_arn
  description   = "Managed by Terraform"
}

variable "name" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "grounding_threshold" {
  type    = number
  default = 0.75
}

variable "relevance_threshold" {
  type    = number
  default = 0.5
}

output "guardrail_id" {
  value = aws_bedrock_guardrail.this.guardrail_id
}

output "guardrail_arn" {
  value = aws_bedrock_guardrail.this.guardrail_arn
}

output "version" {
  value = aws_bedrock_guardrail_version.this.version
}
