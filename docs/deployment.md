# Deployment

This guide provisions the platform in an AWS account and deploys it through GitHub Actions. Read the cost section of [cost.md](cost.md) first: an idle dev stack costs a few dollars a month, but every question calls Bedrock and every document upload stores data.

Nothing in this repository has been applied to a live AWS account by its test suite. The Terraform validates against provider 6.70.0, passes Checkov, and the workflows pass actionlint; the first apply in your account is where account-specific problems (quotas, model access, region availability) show up.

## What gets created

| Stack | Applied by | Resources |
|---|---|---|
| `infra/terraform/bootstrap` | an administrator, once per account | state bucket, Lambda artifact bucket, GitHub OIDC provider, a plan role and a deploy role per environment |
| `infra/terraform/platform` | the deploy workflow, per environment | KMS key, isolated VPC, Aurora cluster, documents bucket, ingest queue and DLQ, three Lambda functions, Cognito user pool, API Gateway, optional WAF and Guardrail, alarms, dashboard, budget |

One Terraform root serves every environment. `infra/terraform/envs/<env>.tfvars` holds the settings and `<env>.s3.tfbackend` the state location, so dev, test and prod differ only in those files.

| Setting | dev | test | prod |
|---|---|---|---|
| Aurora ACU range | 0 to 2 (pauses) | 0 to 4 (pauses) | 0.5 to 16 |
| Deletion protection, final snapshot | no | no | yes |
| WAF | off | on | on |
| MFA | optional | optional | required |
| Password sign-in (`USER_PASSWORD_AUTH`) | on | on | off |
| Guardrail | off | off | on |
| Fallback model | none | none | Claude Haiku 5.5 |
| Log retention | 14 days | 14 days | 90 days |

## Prerequisites

1. An AWS account, and administrator credentials for the one-time bootstrap.
2. **Bedrock model access** in the region for the models you use (Titan Text Embeddings V2, Nova 2 Lite, and any fallback, reranker or Guardrail). Check in the Bedrock console under Model access. Calls to a model without access fail with `AccessDeniedException`, which the API returns as a 502.
3. Terraform 1.10 or later (1.17.0 tested) and the AWS CLI v2.
4. A GitHub repository (your fork) with Actions enabled.

## 1. Bootstrap the account

```bash
cd infra/terraform/bootstrap
terraform init
terraform apply -var account_id=<12-digit id> -var github_repository=<owner>/<repo>
```

Note the outputs: `state_bucket`, `artifact_bucket`, `deploy_role_arns`, `plan_role_arns`. The bootstrap keeps its own state locally; store `terraform.tfstate` somewhere safe, or move it into the state bucket with a backend block after the first apply.

The deploy role can only manage resources named `ragplatform-<env>-*` (where the service supports name patterns), can pass roles only to Lambda, VPC Flow Logs and RDS monitoring, and is explicitly denied creating IAM users or access keys. Its policy is `bootstrap/deploy-policy.json.tftpl`.

## 2. Configure GitHub

Repository variables (Settings, Secrets and variables, Actions, Variables):

| Variable | Value |
|---|---|
| `AWS_ACCOUNT_ID` | your account id |
| `AWS_REGION` | `us-east-1` (or the region you bootstrapped) |
| `TF_STATE_BUCKET` | bootstrap output `state_bucket` |
| `ARTIFACT_BUCKET` | bootstrap output `artifact_bucket` |

Environments (Settings, Environments): create `dev`, `dev-plan`, `test`, `test-plan`, `prod`, `prod-plan`. On `prod`, add required reviewers. The plan environments need no rules: their role is read-only, so a plan can run without approval and can never change anything.

Optional secrets on the `dev` and `test` environments for the authenticated smoke test: `SMOKE_USERNAME`, `SMOKE_PASSWORD` (see step 5).

## 3. Deploy

- Push to `main` deploys `dev`.
- Actions, deploy, Run workflow, pick `test` or `prod`.

The workflow runs these jobs:

1. **build**: builds `build/ragplatform.zip` from `uv.lock` for arm64, records build provenance, and outputs the bundle's SHA-256.
2. **plan** (read-only role): `terraform plan` with the artifact key `lambda/<git sha>/ragplatform.zip`. The job fails if the plan deletes or replaces the Aurora cluster, the documents bucket, the user pool or the KMS key. Those changes need a person, not a pipeline.
3. **apply** (deploy role; waits for reviewers on prod): uploads the bundle, records the current `live` alias versions, applies the saved plan, invokes the migrate function, and runs the smoke test. If the job fails after the live versions are recorded, in the apply itself, migrate or the smoke test, the aliases are pointed back at the recorded versions.

The first deploy of an environment takes 10 to 15 minutes, most of it Aurora. Later deploys take 2 to 4 minutes.

## 4. Create tenants and users

Tenants live in the database. Create them through the migrate function:

```bash
aws lambda invoke --function-name ragplatform-dev-migrate:live --cli-binary-format raw-in-base64-out \
  --payload '{"tenants": [{"tenant_id": "acme", "name": "Acme Corp", "allow_agentic": true}]}' out.json && cat out.json
```

Users live in Cognito and must carry the tenant attribute. Only an administrator can set it:

```bash
POOL=$(terraform -chdir=infra/terraform/platform output -raw user_pool_id)
aws cognito-idp admin-create-user --user-pool-id "$POOL" --username alice@example.com \
  --user-attributes Name=email,Value=alice@example.com Name=email_verified,Value=true Name=custom:tenant_id,Value=acme
aws cognito-idp create-group --user-pool-id "$POOL" --group-name staff
aws cognito-idp admin-add-user-to-group --user-pool-id "$POOL" --username alice@example.com --group-name staff
```

Cognito emails a temporary password; the first sign-in asks for a new one (the UI handles that challenge). Add `tenant-admin` to give a user access to every document in their tenant.

## 5. Smoke test with a real question

Create a tenant called `smoke` and a user in it, upload one document, and store the user's credentials as `SMOKE_USERNAME` and `SMOKE_PASSWORD` on the environment. The deploy then signs in and asks a question after every apply, retrying while Aurora resumes. Without these secrets the smoke test only calls `/v1/health`, which does not touch the database or Bedrock. Prod always stops at `/v1/health`: it disables password sign-in and requires MFA, so the workflow cannot sign in there. Run the real-question check on `test` with the same build before promoting it.

## 6. Use it

```bash
URL=$(terraform -chdir=infra/terraform/platform output -raw api_url)
CLIENT=$(terraform -chdir=infra/terraform/platform output -raw user_pool_client_id)
TOKEN=$(aws cognito-idp initiate-auth --client-id "$CLIENT" --auth-flow USER_PASSWORD_AUTH \
  --auth-parameters USERNAME=alice@example.com,PASSWORD='...' --query AuthenticationResult.IdToken --output text)
curl -s "$URL/v1/query" -H "Authorization: $TOKEN" -H "Content-Type: application/json" \
  -d '{"question": "What is our expense approval limit?"}'
```

API Gateway's Cognito authorizer reads the raw ID token from the `Authorization` header. For the UI against a deployed stage:

```bash
RAG_UI_API_URL=$URL RAG_UI_AUTH_MODE=cognito RAG_UI_COGNITO_CLIENT_ID=$CLIENT uv run streamlit run ui/app.py
```

## Rollback

| Situation | Action |
|---|---|
| Smoke test failed during deploy | Automatic: aliases already point at the previous versions. Fix forward or re-run the deploy for the previous commit to bring Terraform state back in line |
| Bad release found later | `aws lambda update-alias --function-name ragplatform-<env>-<fn> --name live --function-version <previous>` for api and ingest, then re-run the deploy for the last good commit |
| Bad migration | Migrations are forward-only. Write a new migration that reverses the change; restore from a snapshot only for data loss ([operations-runbook.md](operations-runbook.md)) |

Previous versions stay published until you delete them, and bundles stay in the artifact bucket for 90 days.

## Teardown

Dev and test: Actions, destroy, choose the environment and type its name again. Prod is not offered, has deletion protection on Aurora, Cognito and the secrets, and the documents bucket refuses `force_destroy`.

After a destroy, the KMS key is scheduled for deletion (30-day window, cancellable with `aws kms cancel-key-deletion`). Log groups are deleted with the stack because Terraform creates them before the functions first write. To remove the bootstrap stack, empty the state and artifact buckets, remove `prevent_destroy` from the state bucket, and run `terraform destroy` in `bootstrap/`.
