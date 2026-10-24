# 0008. One Terraform root per platform, environments as variable files, OIDC deploys

- Status: accepted, 2026-10-23

## Context

The previous repository had three 249-line copies of the environment root, static IAM access keys in GitHub secrets, `terraform plan -lock=false`, and 2,000 lines of import and cleanup scripts.

## Decision

- One root (`infra/terraform/platform`) composed of modules; `envs/<env>.tfvars` and `envs/<env>.s3.tfbackend` are the only per-environment files.
- State in S3 with native lock files (Terraform 1.10+), no DynamoDB table.
- A bootstrap stack creates the state bucket, artifact bucket, GitHub OIDC provider, and per environment a read-only plan role and a deploy role, each trusted only from one GitHub environment of one repository.
- Deploys: build once, plan with the read-only role, apply the saved plan with the deploy role (prod waits for reviewers), migrate, smoke-test, roll back aliases on failure. The plan job refuses changes that delete or replace data-bearing resources.

## Alternatives

- **Terraform workspaces**: hide which environment a command targets; separate backend files are explicit.
- **AWS CDK**: a good fit too; Terraform was kept because the existing repository and its audience use it.
- **Long-lived deploy keys**: rejected; a leaked key works from anywhere until rotated.

## Consequences

- A fix in a module reaches every environment.
- Lambda rollback is an alias change, seconds rather than a redeploy; Terraform state must then be reconciled by deploying the previous commit.
- The bootstrap stack is applied by hand with administrator credentials, once.
