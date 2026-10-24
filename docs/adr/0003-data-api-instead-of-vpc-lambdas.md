# 0003. Lambdas outside the VPC, reaching Aurora through the RDS Data API

- Status: accepted, 2026-10-23

## Context

The previous design put Lambdas in private subnets and used a NAT gateway so they could reach Gemini. A VPC-attached Lambda that must reach Bedrock, S3, SQS and Secrets Manager needs a NAT gateway or interface endpoints for Bedrock, SQS and Secrets Manager in each availability zone (S3 can use a free gateway endpoint), and database connections from many short-lived functions need pooling (RDS Proxy), which also blocks Aurora's auto-pause.

## Decision

Run the functions outside any VPC. Access Aurora through the RDS Data API (`enable_http_endpoint = true`) with IAM permissions and a Secrets Manager secret. Keep Aurora in isolated subnets with a security group that has no inbound rules. Locally, the same SQL runs through psycopg behind the same `Database` interface.

## Alternatives

- **VPC Lambdas with a NAT gateway**: simplest network, highest idle cost, and a general egress path to restrict.
- **VPC Lambdas with interface endpoints**: private, no egress, but several endpoints in two availability zones cost more than the rest of an idle dev stack.
- **RDS Proxy**: solves pooling, prevents scale to zero.

## Consequences

- No NAT, no endpoints, no pooling, and Aurora can pause at 0 ACU.
- Each statement is an HTTPS call: a few milliseconds more than a pooled connection. A question makes 18 Data API calls; batching them is a possible later optimisation.
- Data API limits shaped the code: one statement per call (the migrator splits files), no array parameters (lists are JSON text cast in SQL), 1 MiB responses, and no session state you can rely on between calls (connections are pooled and reused), so row-level-security settings are applied per transaction with `set_config(..., true)` and every statement carries the transaction id.
- `CommitTransaction` and `RollbackTransaction` take no `database` parameter; the contract tests caught a version of the executor that sent one.
- The Data API path is contract-tested, not run against a live cluster by the test suite.
