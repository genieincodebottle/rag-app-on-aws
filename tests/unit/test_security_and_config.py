"""Identity, local tokens and the configuration safety checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragplatform.config import Settings
from ragplatform.domain.errors import Forbidden, Unauthenticated
from ragplatform.security.auth import (
    context_from_claims,
    issue_local_token,
    parse_groups,
    verify_local_token,
)

SECRET = "a-test-secret-that-is-at-least-32-bytes"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings read RAG_* variables; a developer's shell must not change these tests."""
    import os

    for name in list(os.environ):
        if name.startswith("RAG_") or name == "AWS_LAMBDA_FUNCTION_NAME":
            monkeypatch.delenv(name)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["finance", "staff"], {"finance", "staff"}),
        ("finance,staff", {"finance", "staff"}),
        ("[finance staff]", {"finance", "staff"}),
        ("finance", {"finance"}),
        (None, set()),
        ("[]", set()),
        ("good,bad group;drop", {"good", "bad"}),
    ],
)
def test_group_claim_formats(value: object, expected: set[str]) -> None:
    assert parse_groups(value) == frozenset(expected)


def test_context_comes_from_claims() -> None:
    ctx = context_from_claims(
        {"sub": "u-1", "custom:tenant_id": "acme", "cognito:groups": "tenant-admin,staff"},
        admin_group="tenant-admin",
        request_id="r-1",
    )
    assert (ctx.user_id, ctx.tenant_id, ctx.is_tenant_admin) == ("u-1", "acme", True)
    assert ctx.groups == frozenset({"tenant-admin", "staff"})


def test_user_without_tenant_is_forbidden() -> None:
    with pytest.raises(Forbidden):
        context_from_claims({"sub": "u-1"}, admin_group="tenant-admin")


@pytest.mark.parametrize("tenant", ["ACME", "a", "acme;drop", "../acme", ""])
def test_malformed_tenant_claim_is_forbidden(tenant: str) -> None:
    with pytest.raises(Forbidden):
        context_from_claims({"sub": "u", "custom:tenant_id": tenant}, admin_group="x")


def test_missing_subject_is_unauthenticated() -> None:
    with pytest.raises(Unauthenticated):
        context_from_claims({"custom:tenant_id": "acme"}, admin_group="x")


def test_local_token_round_trip_and_tampering() -> None:
    token = issue_local_token(SECRET, user_id="alice", tenant_id="acme", groups=["staff"])
    claims = verify_local_token(SECRET, token)
    assert claims["custom:tenant_id"] == "acme"
    with pytest.raises(Unauthenticated):
        verify_local_token("another-secret-that-is-also-long-enough", token)
    header, payload, signature = token.split(".")
    with pytest.raises(Unauthenticated):
        verify_local_token(SECRET, f"{header}.{payload}x.{signature}")


def test_expired_local_token_is_rejected() -> None:
    token = issue_local_token(SECRET, user_id="a", tenant_id="acme", groups=[], ttl_s=-10)
    with pytest.raises(Unauthenticated):
        verify_local_token(SECRET, token)


def _settings(**values: object) -> Settings:
    base: dict[str, object] = {"local_auth_secret": "x"}
    base.update(values)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_local_defaults_are_valid() -> None:
    assert _settings().provider == "fake"


def test_local_auth_needs_a_secret() -> None:
    with pytest.raises(ValidationError, match="RAG_LOCAL_AUTH_SECRET"):
        Settings(_env_file=None, auth_mode="local")  # type: ignore[call-arg]


def test_lambda_refuses_local_development_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "rag-api")
    with pytest.raises(ValidationError) as err:
        _settings()
    message = str(err.value)
    for expected in (
        "RAG_AUTH_MODE=cognito",
        "RAG_PROVIDER=bedrock",
        "RAG_DB_BACKEND=data_api",
        "S3 blobs and SQS",
    ):
        assert expected in message


def test_deployed_configuration_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "rag-api")
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        environment="prod",
        auth_mode="cognito",
        provider="bedrock",
        db_backend="data_api",
        blob_backend="s3",
        ingest_mode="sqs",
    )
    assert settings.in_lambda


def test_prompt_logging_is_refused_in_prod() -> None:
    with pytest.raises(ValidationError, match="RAG_LOG_PROMPTS"):
        Settings(
            _env_file=None,
            environment="prod",
            auth_mode="cognito",  # type: ignore[call-arg]
            provider="bedrock",
            db_backend="data_api",
            blob_backend="s3",
            ingest_mode="sqs",
            log_prompts=True,
        )


@pytest.mark.parametrize("url", ["http://evil.example/mcp", "ftp://x", "http://169.254.169.254/"])
def test_web_search_url_must_be_https(url: str) -> None:
    with pytest.raises(ValidationError, match="RAG_WEB_SEARCH_URL"):
        _settings(web_search_url=url)


def test_chunk_settings_are_checked() -> None:
    with pytest.raises(ValidationError, match="OVERLAP"):
        _settings(chunk_tokens=50, chunk_overlap_tokens=50)
    with pytest.raises(ValidationError, match="DIMENSIONS"):
        _settings(embedding_dimensions=300)
