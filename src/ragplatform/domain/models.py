"""Typed domain models shared by every layer."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- identity


class AuthContext(Frozen):
    """Who is calling. Built only from verified token claims, never from a request body."""

    user_id: str
    tenant_id: str
    groups: frozenset[str] = frozenset()
    is_tenant_admin: bool = False
    request_id: str = "-"


class AccessScope(Frozen):
    """What the database session may see. Applied with row-level security."""

    tenant_id: str
    user_id: str
    groups: frozenset[str] = frozenset()
    # Skip document ACL checks inside the tenant. Set only by trusted code paths
    # (the ingest worker and tenant admins). Tenant isolation is never skipped.
    acl_bypass: bool = False

    @classmethod
    def for_user(cls, ctx: AuthContext) -> AccessScope:
        return cls(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            groups=ctx.groups,
            acl_bypass=ctx.is_tenant_admin,
        )

    @classmethod
    def system(cls, tenant_id: str) -> AccessScope:
        return cls(tenant_id=tenant_id, user_id="system:ingest", acl_bypass=True)


# --------------------------------------------------------------------------- documents


class DocumentStatus(StrEnum):
    PENDING_UPLOAD = "pending_upload"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class VersionStatus(StrEnum):
    PENDING_UPLOAD = "pending_upload"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class Document(BaseModel):
    document_id: UUID
    tenant_id: str
    owner_id: str
    title: str
    source_name: str
    content_type: str
    acl_groups: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    status: DocumentStatus
    current_version: int | None = None
    latest_version: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DocumentVersion(BaseModel):
    document_id: UUID
    version: int
    sha256: str
    size_bytes: int
    storage_key: str
    status: VersionStatus
    chunk_count: int = 0
    attempts: int = 0
    error_code: str | None = None
    error_detail: str | None = None
    created_at: datetime
    processed_at: datetime | None = None


class UploadInstruction(Frozen):
    """How the client sends the file bytes. S3 presigned POST on AWS, a PUT locally."""

    method: str
    url: str
    fields: dict[str, str] = Field(default_factory=dict)
    expires_in_s: int
    max_bytes: int


# --------------------------------------------------------------------------- parsing and chunks


class Section(Frozen):
    """A contiguous run of text that shares a heading path and page range."""

    text: str
    heading_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None


class ParsedDocument(Frozen):
    title: str | None
    sections: tuple[Section, ...]
    page_count: int | None = None


class ChunkKind(StrEnum):
    FLAT = "flat"  # recursive strategy: retrieved and sent to the model
    CHILD = "child"  # parent-child strategy: retrieved, then expanded to its parent
    PARENT = "parent"  # parent-child strategy: sent to the model, never embedded


class ChunkDraft(Frozen):
    chunk_index: int
    kind: ChunkKind
    text: str
    embed_text: str | None  # text sent to the embedder (may carry a contextual header)
    section_path: str
    page_start: int | None
    page_end: int | None
    content_hash: str
    token_count: int
    parent_index: int | None = None
    risk_flags: tuple[str, ...] = ()


# --------------------------------------------------------------------------- retrieval


class SearchFilters(BaseModel):
    document_ids: list[UUID] | None = None
    tags: list[str] | None = None
    content_types: list[str] | None = None


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    version: int
    title: str
    section_path: str
    page_start: int | None
    page_end: int | None
    text: str  # what the model sees (the parent text when expanded)
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None
    parent_chunk_id: UUID | None = None
    risk_flags: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


# --------------------------------------------------------------------------- generation


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    CONFLICTING_SOURCES = "conflicting_sources"


class Citation(BaseModel):
    marker: str
    chunk_id: UUID
    document_id: UUID
    version: int
    title: str
    section_path: str
    page_start: int | None = None
    page_end: int | None = None
    snippet: str


class Answer(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    model_id: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    # Set when the server had to change what the model returned.
    invalid_citations_removed: int = 0
    downgraded_reason: str | None = None
