"""Runtime configuration.

Every setting is read from an environment variable with the ``RAG_`` prefix
(``RAG_PROVIDER``, ``RAG_DATABASE_URL`` ...) or from a local ``.env`` file.
The validator at the bottom refuses combinations that are unsafe for the
environment the code is running in, so a misconfigured Lambda fails at cold
start instead of serving traffic with, say, local development auth.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class ChunkStrategy(StrEnum):
    RECURSIVE = "recursive"
    PARENT_CHILD = "parent_child"


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    VECTOR = "vector"
    KEYWORD = "keyword"


def running_in_lambda() -> bool:
    return "AWS_LAMBDA_FUNCTION_NAME" in os.environ


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: Environment = Environment.LOCAL
    aws_region: str = "us-east-1"

    # --- Authentication -------------------------------------------------
    # "cognito": identity comes only from claims verified by the API Gateway
    # Cognito authorizer. "local": HS256 tokens minted by `ragplatform token`.
    auth_mode: Literal["cognito", "local"] = "local"
    local_auth_secret: SecretStr | None = None
    tenant_admin_group: str = "tenant-admin"

    # --- Database -------------------------------------------------------
    db_backend: Literal["psycopg", "data_api"] = "psycopg"
    database_url: SecretStr | None = None  # application role, used at runtime
    admin_database_url: SecretStr | None = None  # owner role, migrations only
    db_cluster_arn: str | None = None
    db_secret_arn: str | None = None  # application role secret (Data API)
    db_admin_secret_arn: str | None = None  # master secret (migrations)
    db_name: str = "ragplatform"
    app_db_role: str = "rag_app"
    local_app_db_password: SecretStr | None = None  # `ragplatform migrate` on a laptop
    db_resume_max_wait_s: float = 25.0

    # --- Blob storage and queue -----------------------------------------
    blob_backend: Literal["local", "s3"] = "local"
    local_blob_dir: Path = Path(".data/blobs")
    documents_bucket: str | None = None
    upload_url_ttl_s: int = 900
    # local: documents are processed in-process after upload.
    # sqs: S3 notifies SQS and the ingest Lambda processes the message.
    ingest_mode: Literal["inline", "sqs"] = "inline"

    # --- Model providers ------------------------------------------------
    provider: Literal["bedrock", "gemini", "fake"] = "fake"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    chat_model_id: str = "us.amazon.nova-2-lite-v1:0"
    fallback_chat_model_id: str | None = None
    rerank_model_id: str | None = None  # e.g. cohere.rerank-v3-5:0
    guardrail_id: str | None = None
    guardrail_version: str = "DRAFT"
    guardrail_grounding_check: bool = False
    gemini_api_key: SecretStr | None = None
    model_timeout_s: float = 30.0
    model_max_attempts: int = 4
    embed_concurrency: int = 8
    embed_batch_size: int = 64

    # --- Ingestion ------------------------------------------------------
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 1500
    max_docx_uncompressed_bytes: int = 200 * 1024 * 1024
    chunk_strategy: ChunkStrategy = ChunkStrategy.PARENT_CHILD
    chunk_tokens: int = 220
    chunk_overlap_tokens: int = 30
    parent_chunk_tokens: int = 900
    contextual_headers: bool = True
    ingest_lease_s: int = 900

    # --- Retrieval and generation ---------------------------------------
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    vector_candidates: int = 40
    keyword_candidates: int = 40
    rrf_k: int = 60
    final_k: int = 6
    rerank_candidates: int = 20
    hnsw_ef_search: int = 100
    context_token_budget: int = 6000
    max_output_tokens: int = 1024
    temperature: float = 0.1
    max_question_chars: int = 2000

    # --- Agentic mode ---------------------------------------------------
    agent_max_steps: int = 4
    agent_max_tokens: int = 24000
    agent_deadline_s: float = 25.0
    web_search_url: str | None = None  # server-side allowlist of one MCP endpoint
    web_search_api_key: SecretStr | None = None

    # --- Caching, budgets, telemetry ------------------------------------
    answer_cache_ttl_s: int = 3600
    default_daily_query_limit: int = 2000
    default_daily_token_limit: int = 5_000_000
    log_prompts: bool = False
    metrics_namespace: str = "RagPlatform"
    service_name: str = "ragplatform"

    # --- Derived ----------------------------------------------------------
    allowed_cors_origins: list[str] = Field(default_factory=list)

    @property
    def in_lambda(self) -> bool:
        return running_in_lambda()

    @model_validator(mode="after")
    def _check_safety(self) -> Settings:
        problems: list[str] = []
        if self.embedding_dimensions not in (256, 512, 768, 1024):
            problems.append("RAG_EMBEDDING_DIMENSIONS must be one of 256, 512, 768, 1024")
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            problems.append("RAG_CHUNK_OVERLAP_TOKENS must be smaller than RAG_CHUNK_TOKENS")
        if self.parent_chunk_tokens < self.chunk_tokens:
            problems.append("RAG_PARENT_CHUNK_TOKENS must be at least RAG_CHUNK_TOKENS")

        if self.auth_mode == "local" and self.local_auth_secret is None:
            problems.append("RAG_AUTH_MODE=local needs RAG_LOCAL_AUTH_SECRET")

        deployed = self.in_lambda or self.environment != Environment.LOCAL
        if deployed:
            # These are the controls that keep a deployed stack honest. Each one
            # is a configuration that is fine on a laptop and wrong on AWS.
            if self.auth_mode != "cognito":
                problems.append("deployed environments must use RAG_AUTH_MODE=cognito")
            if self.provider != "bedrock":
                problems.append("deployed environments must use RAG_PROVIDER=bedrock")
            if self.db_backend != "data_api":
                problems.append("deployed environments must use RAG_DB_BACKEND=data_api")
            if self.blob_backend != "s3" or self.ingest_mode != "sqs":
                problems.append("deployed environments must use S3 blobs and SQS ingestion")
            if self.log_prompts and self.environment == Environment.PROD:
                problems.append("RAG_LOG_PROMPTS is not allowed in prod")
        if self.web_search_url and not (
            self.web_search_url.startswith("https://")
            or (not deployed and self.web_search_url.startswith("http://localhost"))
        ):
            problems.append("RAG_WEB_SEARCH_URL must be https (http://localhost only locally)")
        if problems:
            raise ValueError("; ".join(problems))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
