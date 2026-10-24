"""Document lifecycle: idempotency, validation, versions, deletion, re-embedding reuse."""

from __future__ import annotations

import hashlib

import pytest

from ragplatform.container import Container
from ragplatform.domain.errors import (
    Conflict,
    NotFound,
    PayloadTooLarge,
    UnsupportedMediaType,
    ValidationFailed,
)
from ragplatform.domain.models import AccessScope, AnswerStatus
from ragplatform.services.document_service import NewVersionRequest, RegisterRequest
from ragplatform.services.query_service import QueryRequest
from tests.conftest import TRAVEL_POLICY, ingest_text, user

pytestmark = pytest.mark.integration

QUESTION = "What is the hotel limit in London?"


def _count(container: Container, sql: str, tenant: str = "acme") -> int:
    scope = AccessScope.system(tenant)
    with container.db.transaction(scope) as tx:
        return int(tx.execute(sql)[0]["n"])


def _ask(container: Container, question: str = QUESTION) -> object:
    return container.queries.answer(user("bob"), QueryRequest(question=question, use_cache=False))


def test_ingest_writes_chunks_and_audit(container: Container) -> None:
    doc_id, result = ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    assert result.outcome == "ready"
    assert result.chunks > 0
    detail = container.documents.get(user(), doc_id)
    assert detail.document.status == "ready"
    assert detail.document.title == "Travel Policy"  # from the Markdown H1, not the file name
    assert (
        _count(
            container, "SELECT count(*) AS n FROM audit_events WHERE action = 'document.ingested'"
        )
        == 1
    )


def test_duplicate_delivery_is_skipped(container: Container) -> None:
    doc_id, _ = ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    before = _count(container, "SELECT count(*) AS n FROM chunks")
    again = container.pipeline.process("acme", doc_id, 1)
    assert again.outcome == "skipped"
    assert _count(container, "SELECT count(*) AS n FROM chunks") == before


def test_same_file_twice_returns_the_existing_document(container: Container) -> None:
    doc_id, _ = ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    data = TRAVEL_POLICY.encode()
    again = container.documents.register(
        user(),
        RegisterRequest(
            source_name="copy.md",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            acl_groups=["staff"],
        ),
    )
    assert again.duplicate
    assert again.document.document_id == doc_id
    assert again.upload is None


def test_checksum_mismatch_fails_the_version_permanently(container: Container) -> None:
    data = TRAVEL_POLICY.encode()
    registered = container.documents.register(
        user(),
        RegisterRequest(
            source_name="travel.md", size_bytes=len(data), sha256="0" * 64, acl_groups=["staff"]
        ),
    )
    doc_id = registered.document.document_id
    # Simulate a client that registered one file and uploaded another straight to storage.
    container.blobs.put(registered.version.storage_key, data, "text/markdown")  # type: ignore[union-attr]
    result = container.pipeline.process("acme", doc_id, 1)
    assert result.outcome == "failed"
    assert result.error_code == "checksum_mismatch"
    detail = container.documents.get(user(), doc_id)
    assert detail.document.status == "failed"
    assert detail.versions[0].error_code == "checksum_mismatch"


def test_local_upload_rejects_wrong_bytes(container: Container) -> None:
    data = TRAVEL_POLICY.encode()
    registered = container.documents.register(
        user(),
        RegisterRequest(
            source_name="travel.md",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            acl_groups=["staff"],
        ),
    )
    with pytest.raises(ValidationFailed):
        container.documents.accept_local_upload(
            user(), registered.document.document_id, 1, data[:-1] + b"X"
        )


def test_registration_validates_type_and_size(container: Container) -> None:
    with pytest.raises(UnsupportedMediaType):
        container.documents.register(
            user(), RegisterRequest(source_name="malware.exe", size_bytes=10, sha256="a" * 64)
        )
    with pytest.raises(PayloadTooLarge):
        container.documents.register(
            user(), RegisterRequest(source_name="big.pdf", size_bytes=10**10, sha256="a" * 64)
        )


def test_new_version_replaces_old_chunks_atomically(container: Container) -> None:
    doc_id, _ = ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    assert "220 GBP" in _ask(container).answer  # type: ignore[attr-defined]

    updated = TRAVEL_POLICY.replace("220 GBP", "250 GBP").encode()
    response = container.documents.new_version(
        user(),
        doc_id,
        NewVersionRequest(size_bytes=len(updated), sha256=hashlib.sha256(updated).hexdigest()),
    )
    assert response.version is not None
    assert response.version.version == 2
    container.documents.accept_local_upload(user(), doc_id, 2, updated)
    result = container.pipeline.process("acme", doc_id, 2)
    assert result.outcome == "ready"
    # Unchanged chunks reuse their stored vectors instead of calling the embedder.
    assert result.reused_embeddings > 0

    answer = _ask(container)
    assert "250 GBP" in answer.answer  # type: ignore[attr-defined]
    assert "220 GBP" not in answer.answer  # type: ignore[attr-defined]
    assert _count(container, "SELECT count(*) AS n FROM chunks WHERE version = 1") == 0
    versions = container.documents.get(user(), doc_id).versions
    assert [v.status for v in versions] == ["superseded", "ready"]


def test_identical_new_version_is_rejected(container: Container) -> None:
    doc_id, _ = ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    data = TRAVEL_POLICY.encode()
    with pytest.raises(Conflict):
        container.documents.new_version(
            user(),
            doc_id,
            NewVersionRequest(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),
        )


def test_older_version_finishing_late_never_rolls_back(container: Container) -> None:
    doc_id, _ = ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    v2 = TRAVEL_POLICY.replace("220", "230").encode()
    v3 = TRAVEL_POLICY.replace("220", "240").encode()
    for data in (v2, v3):
        container.documents.new_version(
            user(),
            doc_id,
            NewVersionRequest(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),
        )
    container.documents.accept_local_upload(user(), doc_id, 2, v2)
    container.documents.accept_local_upload(user(), doc_id, 3, v3)
    assert container.pipeline.process("acme", doc_id, 3).outcome == "ready"
    assert container.pipeline.process("acme", doc_id, 2).outcome == "ready"  # finishes late
    detail = container.documents.get(user(), doc_id)
    assert detail.document.current_version == 3
    assert {v.version: v.status for v in detail.versions}[2] == "superseded"
    assert "240 GBP" in _ask(container).answer  # type: ignore[attr-defined]


def test_delete_removes_chunks_bytes_and_cached_answers(container: Container) -> None:
    doc_id, _ = ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    cached_request = QueryRequest(question=QUESTION)
    assert container.queries.answer(user("bob"), cached_request).status == AnswerStatus.ANSWERED
    assert container.queries.answer(user("bob"), cached_request).cached

    outcome = container.documents.delete(user(), doc_id)
    assert outcome["objects_purged"] == 1
    assert _count(container, "SELECT count(*) AS n FROM chunks") == 0
    after = container.queries.answer(user("bob"), cached_request)
    assert not after.cached
    assert after.status == AnswerStatus.INSUFFICIENT_CONTEXT
    with pytest.raises(NotFound):
        container.documents.get(user(), doc_id)
    # The tombstone keeps the id for the audit trail but no personal metadata.
    with container.db.transaction(AccessScope.system("acme")) as tx:
        row = tx.execute("SELECT status, title, source_name FROM documents")[0]
    assert (row["status"], row["title"], row["source_name"]) == (
        "deleted",
        "deleted document",
        "deleted",
    )


def test_acl_change_invalidates_cached_answers(container: Container) -> None:
    doc_id, _ = ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    request = QueryRequest(question=QUESTION)
    assert container.queries.answer(user("bob"), request).status == AnswerStatus.ANSWERED
    from ragplatform.services.document_service import AccessUpdate

    container.documents.update_access(user(), doc_id, AccessUpdate(acl_groups=[]))
    after = container.queries.answer(user("bob"), request)
    assert after.status == AnswerStatus.INSUFFICIENT_CONTEXT
    assert not after.cached
