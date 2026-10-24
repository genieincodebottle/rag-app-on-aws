"""Document lifecycle: register, upload, version, list, change access, delete."""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from ragplatform.adapters.blobstore import BlobStore, storage_key
from ragplatform.config import Settings
from ragplatform.domain.errors import (
    Conflict,
    Forbidden,
    NotFound,
    PayloadTooLarge,
    RagError,
    UnsupportedMediaType,
    ValidationFailed,
)
from ragplatform.domain.models import (
    AccessScope,
    AuthContext,
    Document,
    DocumentVersion,
    UploadInstruction,
)
from ragplatform.ingestion.chunking import default_title
from ragplatform.ingestion.parsers import EXTENSION_TYPES, SUPPORTED_TYPES
from ragplatform.observability.telemetry import Telemetry, logger
from ragplatform.store.db import Database, UniqueViolation
from ragplatform.store.repository import Repository

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _valid_sha(value: str) -> str:
    value = value.lower()
    if not _SHA256.match(value):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    return value


def _valid_labels(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    for value in values:
        if not _LABEL.match(value):
            raise ValueError(f"invalid label {value!r}")
    return sorted(set(values))


class RegisterRequest(BaseModel):
    source_name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    sha256: str
    content_type: str | None = None
    title: str | None = Field(default=None, max_length=300)
    acl_groups: list[str] = Field(default_factory=list, max_length=20)
    tags: list[str] = Field(default_factory=list, max_length=20)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=20)

    @field_validator("sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        return _valid_sha(value)

    @field_validator("acl_groups", "tags")
    @classmethod
    def _labels(cls, values: list[str]) -> list[str]:
        return _valid_labels(values) or []

    @field_validator("source_name")
    @classmethod
    def _name(cls, value: str) -> str:
        return value.replace("\\", "/").rsplit("/", 1)[-1].strip() or "upload"


class NewVersionRequest(BaseModel):
    size_bytes: int = Field(gt=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        return _valid_sha(value)


class AccessUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    acl_groups: list[str] | None = Field(default=None, max_length=20)
    tags: list[str] | None = Field(default=None, max_length=20)

    @field_validator("acl_groups", "tags")
    @classmethod
    def _labels(cls, values: list[str] | None) -> list[str] | None:
        return _valid_labels(values)


class RegisterResponse(BaseModel):
    document: Document
    version: DocumentVersion | None
    upload: UploadInstruction | None
    duplicate: bool = False


class DocumentDetail(BaseModel):
    document: Document
    versions: list[DocumentVersion]


class DocumentPage(BaseModel):
    items: list[Document]
    next_cursor: str | None


def encode_cursor(updated_at: datetime, document_id: UUID) -> str:
    return base64.urlsafe_b64encode(f"{updated_at.isoformat()}|{document_id}".encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        at, doc = base64.urlsafe_b64decode(cursor.encode()).decode().split("|")
        return datetime.fromisoformat(at), UUID(doc)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValidationFailed("invalid cursor") from exc


class DocumentService:
    def __init__(
        self,
        *,
        db: Database,
        repo: Repository,
        blobs: BlobStore,
        settings: Settings,
        telemetry: Telemetry,
    ) -> None:
        self.db = db
        self.repo = repo
        self.blobs = blobs
        self.settings = settings
        self.telemetry = telemetry

    # ------------------------------------------------------------------ helpers

    def _content_type(self, request: RegisterRequest) -> str:
        content_type = (request.content_type or "").split(";")[0].strip().lower()
        if not content_type:
            extension = (
                "." + request.source_name.rsplit(".", 1)[-1].lower()
                if "." in request.source_name
                else ""
            )
            content_type = EXTENSION_TYPES.get(extension, "")
        if content_type not in SUPPORTED_TYPES:
            raise UnsupportedMediaType(
                f"unsupported type {content_type or 'unknown'}",
                public_message="Supported types are PDF, DOCX, HTML, Markdown, plain text and CSV.",
            )
        return content_type

    def _check_size(self, size: int) -> None:
        if size > self.settings.max_upload_bytes:
            raise PayloadTooLarge(f"{size} bytes exceeds {self.settings.max_upload_bytes}")

    def _check_groups(self, ctx: AuthContext, groups: list[str]) -> None:
        # A user can share only with groups they belong to. Otherwise a user
        # could plant a document in a group's search results that the group
        # never chose to receive.
        if ctx.is_tenant_admin:
            return
        outside = set(groups) - set(ctx.groups)
        if outside:
            raise Forbidden(
                f"not a member of {sorted(outside)}",
                public_message="You can only share documents with groups you belong to.",
            )

    def _local_url(self, document_id: UUID, version: int) -> str:
        return f"/v1/documents/{document_id}/versions/{version}/content"

    def _instruction(
        self, key: str, document_id: UUID, version: DocumentVersion, content_type: str
    ) -> UploadInstruction:
        return self.blobs.upload_instruction(
            key,
            size_bytes=version.size_bytes,
            content_type=content_type,
            local_url=self._local_url(document_id, version.version),
        )

    def _load_owned(self, tx: Any, ctx: AuthContext, document_id: UUID) -> Document:
        document = self.repo.get_document(tx, document_id)
        if document is None:
            raise NotFound(str(document_id))
        if document.owner_id != ctx.user_id and not ctx.is_tenant_admin:
            raise Forbidden("only the owner or a tenant admin can change this document")
        return document

    # ------------------------------------------------------------------ operations

    def register(self, ctx: AuthContext, request: RegisterRequest) -> RegisterResponse:
        content_type = self._content_type(request)
        self._check_size(request.size_bytes)
        self._check_groups(ctx, request.acl_groups)
        scope = AccessScope.for_user(ctx)
        title = request.title or default_title(request.source_name)
        # Without a caller-supplied title, the file name stands in until the
        # parser finds a real one (a PDF title, a DOCX title, a Markdown H1).
        metadata = {k: v for k, v in request.metadata.items() if k != "title_source"}
        metadata["title_source"] = "user" if request.title else "filename"
        with self.db.transaction(scope) as tx:
            existing = self.repo.find_document_by_sha(
                tx, ctx.tenant_id, ctx.user_id, request.sha256
            )
            if existing is not None:
                document = self.repo.get_document(tx, existing)
                if document is not None:
                    return RegisterResponse(
                        document=document, version=None, upload=None, duplicate=True
                    )
            try:
                document, version = self.repo.create_document(
                    tx,
                    tenant_id=ctx.tenant_id,
                    owner_id=ctx.user_id,
                    title=title,
                    source_name=request.source_name,
                    content_type=content_type,
                    acl_groups=request.acl_groups,
                    tags=request.tags,
                    metadata=metadata,
                    sha256=request.sha256,
                    size_bytes=request.size_bytes,
                    storage_key_for=lambda d, v: storage_key(ctx.tenant_id, d, v),
                )
            except UniqueViolation as exc:
                raise Conflict("the same file is being registered concurrently") from exc
            self.repo.audit(
                tx,
                tenant_id=ctx.tenant_id,
                actor_id=ctx.user_id,
                action="document.registered",
                resource_type="document",
                resource_id=str(document.document_id),
                request_id=ctx.request_id,
                detail={"content_type": content_type, "size_bytes": request.size_bytes},
            )
        self.telemetry.add("DocumentsRegistered", 1)
        upload = self._instruction(version.storage_key, document.document_id, version, content_type)
        return RegisterResponse(document=document, version=version, upload=upload)

    def new_version(
        self, ctx: AuthContext, document_id: UUID, request: NewVersionRequest
    ) -> RegisterResponse:
        self._check_size(request.size_bytes)
        scope = AccessScope.for_user(ctx)
        with self.db.transaction(scope) as tx:
            document = self._load_owned(tx, ctx, document_id)
            try:
                version = self.repo.add_version(
                    tx,
                    tenant_id=ctx.tenant_id,
                    owner_id=document.owner_id,
                    document_id=document_id,
                    sha256=request.sha256,
                    size_bytes=request.size_bytes,
                    storage_key_for=lambda d, v: storage_key(ctx.tenant_id, d, v),
                )
            except UniqueViolation as exc:
                raise Conflict(
                    "this exact file is already a live version",
                    public_message="This file is identical to an existing version.",
                ) from exc
            self.repo.audit(
                tx,
                tenant_id=ctx.tenant_id,
                actor_id=ctx.user_id,
                action="document.version_added",
                resource_type="document",
                resource_id=str(document_id),
                request_id=ctx.request_id,
                detail={"version": version.version},
            )
        upload = self._instruction(version.storage_key, document_id, version, document.content_type)
        return RegisterResponse(document=document, version=version, upload=upload)

    def accept_local_upload(
        self, ctx: AuthContext, document_id: UUID, version: int, data: bytes
    ) -> DocumentVersion:
        """Local mode only: the API receives the bytes that S3 would receive on AWS."""
        scope = AccessScope.for_user(ctx)
        with self.db.transaction(scope) as tx:
            document = self._load_owned(tx, ctx, document_id)
            record = self.repo.get_version(tx, document_id, version)
        if record is None:
            raise NotFound(f"{document_id} v{version}")
        if record.status != "pending_upload":
            raise Conflict(f"version is {record.status}, not awaiting upload")
        if len(data) != record.size_bytes:
            raise ValidationFailed(f"expected {record.size_bytes} bytes, received {len(data)}")
        if hashlib.sha256(data).hexdigest() != record.sha256:
            raise ValidationFailed("the uploaded bytes do not match the registered SHA-256")
        self.blobs.put(record.storage_key, data, document.content_type)
        with self.db.transaction(scope) as tx:
            self.repo.mark_uploaded(tx, document_id, version)
        return record

    def get(self, ctx: AuthContext, document_id: UUID) -> DocumentDetail:
        with self.db.transaction(AccessScope.for_user(ctx)) as tx:
            document = self.repo.get_document(tx, document_id)
            if document is None:
                # Same response whether the document does not exist or the
                # caller's ACL hides it, so ids cannot be probed.
                raise NotFound(str(document_id))
            versions = self.repo.list_versions(tx, document_id)
        return DocumentDetail(document=document, versions=versions)

    def list(
        self, ctx: AuthContext, *, limit: int, cursor: str | None, status: str | None
    ) -> DocumentPage:
        limit = max(1, min(limit, 100))
        before = decode_cursor(cursor) if cursor else None
        with self.db.transaction(AccessScope.for_user(ctx)) as tx:
            items = self.repo.list_documents(tx, limit=limit + 1, before=before, status=status)
        next_cursor = None
        if len(items) > limit:
            items = items[:limit]
            next_cursor = encode_cursor(items[-1].updated_at, items[-1].document_id)
        return DocumentPage(items=items, next_cursor=next_cursor)

    def update_access(self, ctx: AuthContext, document_id: UUID, update: AccessUpdate) -> Document:
        if update.acl_groups is not None:
            self._check_groups(ctx, update.acl_groups)
        scope = AccessScope.for_user(ctx)
        with self.db.transaction(scope) as tx:
            self._load_owned(tx, ctx, document_id)
            document = self.repo.update_access(
                tx, document_id, title=update.title, acl_groups=update.acl_groups, tags=update.tags
            )
            if document is None:
                raise NotFound(str(document_id))
            # Visibility changed, so cached answers may now cite text the caller
            # can no longer read, or miss text they now can.
            self.repo.bump_corpus_version(tx, ctx.tenant_id)
            self.repo.audit(
                tx,
                tenant_id=ctx.tenant_id,
                actor_id=ctx.user_id,
                action="document.access_changed",
                resource_type="document",
                resource_id=str(document_id),
                request_id=ctx.request_id,
                detail=update.model_dump(exclude_none=True, exclude={"title"}),
            )
        return document

    def delete(self, ctx: AuthContext, document_id: UUID) -> dict[str, Any]:
        scope = AccessScope.for_user(ctx)
        with self.db.transaction(scope) as tx:
            self._load_owned(tx, ctx, document_id)
            keys = self.repo.delete_document(tx, document_id)
            self.repo.bump_corpus_version(tx, ctx.tenant_id)
            self.repo.audit(
                tx,
                tenant_id=ctx.tenant_id,
                actor_id=ctx.user_id,
                action="document.deleted",
                resource_type="document",
                resource_id=str(document_id),
                request_id=ctx.request_id,
                detail={"versions": len(keys)},
            )
        # The index no longer holds the document once the transaction commits.
        # Byte removal runs after; a failure here is logged and alarmed, and the
        # bucket's noncurrent-version expiry rule is the backstop.
        purged, failed = 0, 0
        for key in keys:
            try:
                purged += self.blobs.delete_all_versions(key)
            except RagError:
                failed += 1
                logger.exception("blob purge failed", extra={"key": key})
        if failed:
            self.telemetry.add("BlobPurgeFailed", failed)
        self.telemetry.add("DocumentsDeleted", 1)
        return {
            "document_id": str(document_id),
            "deleted": True,
            "objects_purged": purged,
            "purge_failures": failed,
        }
