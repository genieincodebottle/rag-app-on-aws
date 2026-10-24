# 0004. Bedrock in deployed environments, swappable providers behind ports

- Status: accepted, 2026-10-23

## Context

Enterprise deployments want model traffic to stay in the AWS account, IAM instead of API keys, and Guardrails. Learners and CI need to run everything without an AWS account or spend.

## Decision

Define `Embedder`, `ChatModel` and `Reranker` protocols. Provide three implementations: `bedrock` (Titan V2, Converse, Rerank, Guardrails), `gemini` (local real-model runs on the free tier) and `fake` (deterministic, offline). Configuration validation refuses anything but `bedrock` when deployed. Record the embedding model and width in the database and refuse queries when the configuration differs.

## Alternatives

- **Bedrock only**: no free local path; CI would need AWS credentials and spend.
- **A framework abstraction (LangChain, LlamaIndex)**: more surface than the three small interfaces this needs, and the previous implementation broke when LangChain removed modules it imported.

## Consequences

- The whole system, including evaluation, runs offline in CI.
- Live quality numbers so far come from Gemini; Bedrock quality must be measured separately.
- Changing the embedding model requires `ragplatform reembed`; the platform will not mix vectors from two models.
