"""Shared fixtures.

Integration tests use the ``ragplatform_test`` database from docker compose,
migrated once per session by the non-superuser owner role and truncated
between tests. The application connects as ``rag_app_test``, a non-owner role like
the one the Lambdas use, so row-level security applies exactly as on Aurora.

If the database is unreachable, integration tests are skipped locally and fail
in CI (``RAG_REQUIRE_DB=1``), so a missing database can never turn the gate green.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ragplatform.adapters.blobstore import LocalBlobStore
from ragplatform.config import Settings
from ragplatform.container import Container, build_container
from ragplatform.domain.models import AuthContext
from ragplatform.observability.telemetry import Telemetry
from ragplatform.services.document_service import RegisterRequest
from ragplatform.store.db import PsycopgDatabase
from ragplatform.store.migrate import (
    apply_migrations,
    ensure_app_role,
    ensure_tenant,
    record_embedding_space,
)

ADMIN_URL = os.environ.get(
    "RAG_TEST_ADMIN_DATABASE_URL",
    "postgresql://rag_owner:rag_owner_local@localhost:55432/ragplatform_test",
)
APP_PASSWORD = "746573742d6f6e6c792d6170702d70617373776f7264"  # hex, test database only
APP_URL = os.environ.get(
    "RAG_TEST_DATABASE_URL",
    f"postgresql://rag_app_test:{APP_PASSWORD}@localhost:55432/ragplatform_test",
)
# Roles are cluster-wide. A separate role keeps tests from rotating the
# password the local dev database uses.
APP_ROLE = "rag_app_test"
DIMENSIONS = 256  # small vectors keep the fake embedder fast; any supported width works

_TABLES = "audit_events, usage_daily, answer_cache, chunks, document_versions, documents, tenants"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "local",
        "auth_mode": "local",
        "local_auth_secret": "test-secret-for-local-tokens-only",
        "provider": "fake",
        "embedding_dimensions": DIMENSIONS,
        "db_backend": "psycopg",
        "database_url": APP_URL,
        "admin_database_url": ADMIN_URL,
        "blob_backend": "local",
        "ingest_mode": "inline",
        "chunk_tokens": 60,
        "chunk_overlap_tokens": 10,
        "parent_chunk_tokens": 200,
        "app_db_role": APP_ROLE,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def _database_available() -> bool:
    import psycopg

    try:
        psycopg.connect(ADMIN_URL, connect_timeout=3).close()
    except psycopg.Error:
        return False
    return True


@pytest.fixture(scope="session")
def admin_db() -> Iterator[PsycopgDatabase]:
    if not _database_available():
        if os.environ.get("RAG_REQUIRE_DB") == "1":
            pytest.fail(f"test database not reachable at {ADMIN_URL}")
        pytest.skip("test database not running (docker compose up -d db)")
    db = PsycopgDatabase(ADMIN_URL, max_size=4)
    with db.transaction() as tx:
        # Start from an empty schema on every session.
        tx.execute(f"DROP TABLE IF EXISTS {_TABLES}, embedding_space, schema_migrations CASCADE")
        tx.execute(
            "DROP FUNCTION IF EXISTS app_groups(), app_can_read(text, text[]),"
            " text_array(jsonb) CASCADE"
        )
    ensure_app_role(db, APP_ROLE, APP_PASSWORD, rotate=True)
    apply_migrations(db, dimensions=DIMENSIONS, app_role=APP_ROLE)
    record_embedding_space(db, provider="fake", model_id="hashing-v1", dimensions=DIMENSIONS)
    yield db
    db.close()


@pytest.fixture(scope="session")
def app_db(admin_db: PsycopgDatabase) -> Iterator[PsycopgDatabase]:
    db = PsycopgDatabase(APP_URL, max_size=8)
    yield db
    db.close()


@pytest.fixture
def clean_db(admin_db: PsycopgDatabase) -> PsycopgDatabase:
    with admin_db.transaction() as tx:
        tx.execute(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE")
    for tenant, name in (("acme", "Acme Corp"), ("globex", "Globex Inc")):
        ensure_tenant(admin_db, tenant, name, allow_agentic=True)
    return admin_db


@pytest.fixture
def container(clean_db: PsycopgDatabase, app_db: PsycopgDatabase, tmp_path: Path) -> Container:
    return build_container(
        make_settings(),
        db=app_db,
        blobs=LocalBlobStore(tmp_path / "blobs"),
        telemetry=Telemetry("Test", "test", emit=False),
    )


def user(
    user_id: str = "alice",
    tenant: str = "acme",
    groups: tuple[str, ...] = ("staff",),
    admin: bool = False,
) -> AuthContext:
    return AuthContext(
        user_id=user_id,
        tenant_id=tenant,
        groups=frozenset(groups),
        is_tenant_admin=admin,
        request_id="test",
    )


def ingest_text(
    container: Container,
    ctx: AuthContext,
    name: str,
    text: str,
    *,
    acl: tuple[str, ...] = ("staff",),
    title: str | None = None,
) -> tuple[Any, Any]:
    """Register, upload and process one document the way the API does."""
    data = text.encode()
    registered = container.documents.register(
        ctx,
        RegisterRequest(
            source_name=name,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            acl_groups=list(acl),
            title=title,
        ),
    )
    assert registered.version is not None, "unexpected duplicate"
    document_id, version = registered.document.document_id, registered.version.version
    container.documents.accept_local_upload(ctx, document_id, version, data)
    result = container.pipeline.process(ctx.tenant_id, document_id, version)
    return document_id, result


TRAVEL_POLICY = """# Travel Policy

## Hotels

The nightly hotel limit in London is 220 GBP. Bookings above the limit need approval
from a finance manager before travel.

## Meals

The daily meal allowance for business travel is 60 GBP, including tips.

## Flights

Economy class is required for flights under six hours. Business class needs director approval.
"""

SECURITY_POLICY = """# Information Security Policy

## Passwords

Passwords must be at least 14 characters and are rotated only after a suspected compromise.

## Laptops

Company laptops must use full-disk encryption and lock after 5 minutes of inactivity.
"""
