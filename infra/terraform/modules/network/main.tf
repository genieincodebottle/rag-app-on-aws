# A VPC that exists only to hold Aurora. The subnets are isolated: no internet
# gateway, no NAT gateway, no route out. The Lambda functions do not run in this
# VPC; they reach Aurora through the RDS Data API, which is an IAM-authenticated
# AWS endpoint. The database security group therefore has no inbound rules.

data "aws_availability_zones" "available" {
  #checkov:skip=CKV_AWS_394:Only the first az_count zones by name are used; a new zone sorts after the existing ones in this region
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, var.az_count)
}

resource "aws_vpc" "this" {
  cidr_block           = var.cidr_block
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = var.name }
}

# The default security group of a new VPC allows all traffic between its
# members. It is emptied so nothing can be attached to it by accident.
resource "aws_default_security_group" "this" {
  vpc_id = aws_vpc.this.id
}

resource "aws_subnet" "database" {
  count             = var.az_count
  vpc_id            = aws_vpc.this.id
  availability_zone = local.azs[count.index]
  cidr_block        = cidrsubnet(var.cidr_block, 4, count.index)
  tags              = { Name = "${var.name}-db-${local.azs[count.index]}" }
}

resource "aws_db_subnet_group" "this" {
  name       = var.name
  subnet_ids = aws_subnet.database[*].id
}

resource "aws_security_group" "database" {
  #checkov:skip=CKV2_AWS_5:Attached to the Aurora cluster through vpc_security_group_ids in the database module
  name        = "${var.name}-aurora"
  description = "Aurora cluster. No inbound rules: access is through the RDS Data API only."
  vpc_id      = aws_vpc.this.id
}

resource "aws_flow_log" "this" {
  count                = var.flow_logs ? 1 : 0
  vpc_id               = aws_vpc.this.id
  traffic_type         = "REJECT"
  log_destination_type = "cloud-watch-logs"
  log_destination      = aws_cloudwatch_log_group.flow[0].arn
  iam_role_arn         = aws_iam_role.flow[0].arn
}

resource "aws_cloudwatch_log_group" "flow" {
  count             = var.flow_logs ? 1 : 0
  name              = "/vpc/${var.name}/flow"
  retention_in_days = 365 # network evidence for incident investigation
  kms_key_id        = var.kms_key_arn
}

resource "aws_iam_role" "flow" {
  count = var.flow_logs ? 1 : 0
  name  = "${var.name}-flow-logs"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "vpc-flow-logs.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "flow" {
  count = var.flow_logs ? 1 : 0
  role  = aws_iam_role.flow[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
      Resource = "${aws_cloudwatch_log_group.flow[0].arn}:*"
    }]
  })
}

variable "name" {
  type = string
}

variable "cidr_block" {
  type    = string
  default = "10.40.0.0/20"
}

variable "az_count" {
  type    = number
  default = 2
  validation {
    condition     = var.az_count >= 2
    error_message = "Aurora needs subnets in at least two availability zones."
  }
}

variable "flow_logs" {
  type    = bool
  default = false
}

variable "kms_key_arn" {
  type = string
}

variable "log_retention_days" {
  type    = number
  default = 30
}

output "db_subnet_group_name" {
  value = aws_db_subnet_group.this.name
}

output "database_security_group_id" {
  value = aws_security_group.database.id
}

output "vpc_id" {
  value = aws_vpc.this.id
}
