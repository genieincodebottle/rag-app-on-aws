"""Tenant isolation and document access control, proven against PostgreSQL.

These tests query as the runtime role, the same role the Lambdas use, and try
to read across the boundaries in every way the code could get wrong.
"""

from __future__ import annotations

import pytest

from ragplatform.container import Container
from ragplatform.domain.errors import Forbidden, NotFound
from ragplatform.domain.models import AccessScope, AnswerStatus
from ragplatform.services.query_service import QueryRequest
from tests.conftest import SECURITY_POLICY, TRAVEL_POLICY, ingest_text, user

pytestmark = pytest.mark.integration


def _ask(container: Container, ctx: object, question: str) -> object:
    return container.queries.answer(ctx, QueryRequest(question=question, use_cache=False))  # type: ignore[arg-type]


def test_no_scope_sees_no_rows(container: Container) -> None:
    ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    with container.db.transaction() as tx:  # runtime role, no tenant set
        assert tx.execute("SELECT count(*) AS n FROM chunks")[0]["n"] == 0
        assert tx.execute("SELECT count(*) AS n FROM documents")[0]["n"] == 0
        assert tx.execute("SELECT count(*) AS n FROM tenants")[0]["n"] == 0


def test_other_tenant_sees_nothing_even_with_raw_sql(container: Container) -> None:
    ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    scope = AccessScope(tenant_id="globex", user_id="mallory", acl_bypass=True)
    with container.db.transaction(scope) as tx:
        # An unfiltered query: only row-level security stands in the way.
        assert tx.execute("SELECT count(*) AS n FROM chunks")[0]["n"] == 0
        assert tx.execute("SELECT count(*) AS n FROM document_versions")[0]["n"] == 0
        assert tx.execute("SELECT count(*) AS n FROM audit_events")[0]["n"] == 0


def test_cross_tenant_question_gets_no_answer(container: Container) -> None:
    ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    response = _ask(
        container, user("mallory", tenant="globex"), "What is the hotel limit in London?"
    )
    assert response.status == AnswerStatus.INSUFFICIENT_CONTEXT  # type: ignore[attr-defined]
    assert response.citations == []  # type: ignore[attr-defined]


def test_cross_tenant_document_id_is_not_found(container: Container) -> None:
    doc_id, _ = ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    with pytest.raises(NotFound):
        container.documents.get(user("mallory", tenant="globex"), doc_id)
    with pytest.raises(NotFound):
        container.documents.delete(user("mallory", tenant="globex", admin=True), doc_id)


def test_cannot_write_rows_into_another_tenant(container: Container) -> None:
    import psycopg

    scope = AccessScope(tenant_id="acme", user_id="alice")
    with pytest.raises(psycopg.errors.InsufficientPrivilege), container.db.transaction(scope) as tx:
        tx.execute(
            "INSERT INTO audit_events (tenant_id, actor_id, action, resource_type)"
            " VALUES ('globex', 'alice', 'x', 'y')"
        )


def test_acl_hides_documents_from_users_outside_the_group(container: Container) -> None:
    ingest_text(
        container,
        user("alice", groups=("security",)),
        "security.md",
        SECURITY_POLICY,
        acl=("security",),
    )
    question = "How long must passwords be?"
    outsider = _ask(container, user("bob", groups=("staff",)), question)
    insider = _ask(container, user("carol", groups=("security",)), question)
    assert outsider.status == AnswerStatus.INSUFFICIENT_CONTEXT  # type: ignore[attr-defined]
    assert insider.status == AnswerStatus.ANSWERED  # type: ignore[attr-defined]
    assert "14 characters" in insider.answer  # type: ignore[attr-defined]


def test_owner_reads_own_private_document_and_admin_reads_all(container: Container) -> None:
    doc_id, _ = ingest_text(
        container,
        user("alice", groups=()),
        "notes.md",
        "# Notes\n\nThe project codename is BLUEHERON.",
        acl=(),
    )
    assert container.documents.get(user("alice", groups=()), doc_id).document.title == "Notes"
    with pytest.raises(NotFound):
        container.documents.get(user("bob"), doc_id)
    admin = user("root", groups=("tenant-admin",), admin=True)
    assert container.documents.get(admin, doc_id).document.document_id == doc_id


def test_listing_respects_acl(container: Container) -> None:
    ingest_text(container, user("alice", groups=("staff",)), "travel.md", TRAVEL_POLICY)
    ingest_text(
        container, user("alice", groups=("security",)), "sec.md", SECURITY_POLICY, acl=("security",)
    )
    titles = {
        d.title
        for d in container.documents.list(user("bob"), limit=10, cursor=None, status=None).items
    }
    assert titles == {"Travel Policy"}


def test_users_can_only_share_with_their_own_groups(container: Container) -> None:
    with pytest.raises(Forbidden):
        ingest_text(
            container,
            user("alice", groups=("staff",)),
            "x.md",
            "# X\n\nhello there",
            acl=("finance",),
        )


def test_only_owner_or_admin_can_delete(container: Container) -> None:
    doc_id, _ = ingest_text(container, user("alice"), "travel.md", TRAVEL_POLICY)
    with pytest.raises(Forbidden):
        container.documents.delete(user("bob"), doc_id)


def test_runtime_role_cannot_change_tenant_limits_or_audit_history(container: Container) -> None:
    import psycopg

    scope = AccessScope(tenant_id="acme", user_id="alice")
    with pytest.raises(psycopg.errors.InsufficientPrivilege), container.db.transaction(scope) as tx:
        tx.execute("UPDATE tenants SET daily_query_limit = 999999 WHERE tenant_id = 'acme'")
    ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    with pytest.raises(psycopg.errors.InsufficientPrivilege), container.db.transaction(scope) as tx:
        tx.execute("DELETE FROM audit_events")


def test_body_supplied_identity_is_ignored_by_the_api(container: Container) -> None:
    from fastapi.testclient import TestClient

    from ragplatform.api.app import create_app
    from ragplatform.security.auth import issue_local_token

    ingest_text(
        container, user("alice", groups=("security",)), "sec.md", SECURITY_POLICY, acl=("security",)
    )
    client = TestClient(create_app(container))
    secret = container.settings.local_auth_secret.get_secret_value()  # type: ignore[union-attr]
    token = issue_local_token(secret, user_id="bob", tenant_id="acme", groups=["staff"])
    response = client.post(
        "/v1/query",
        headers={"Authorization": f"Bearer {token}"},
        # The previous implementation trusted these fields.
        json={
            "question": "How long must passwords be?",
            "user_id": "alice",
            "tenant_id": "acme",
            "groups": ["security"],
            "mcp_server_url": "http://evil",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "insufficient_context"
