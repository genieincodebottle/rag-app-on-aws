"""Processing one uploaded document version.

Called by the SQS-triggered ingest Lambda on AWS and in-process locally. The
outcome is always one of:

* ``ready``: chunks written and the version made live in one transaction;
* ``failed``: the file can never be processed (wrong type, corrupt, no text,
  checksum mismatch). Recorded on the version and not retried;
* ``skipped``: another delivery of the same event already owns or finished
  the version (S3 and SQS both deliver at least once);
* a raised ``UpstreamUnavailable``: a dependency is down or throttling. The
  claim is released and the message returns to the queue for a retry.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from ragplatform.adapters.blobstore import BlobStore
from ragplatform.config import Settings
from ragplatform.domain.errors import PermanentIngestError, RagError
from ragplatform.domain.models import AccessScope, ChunkKind
from ragplatform.ingestion.chunking import build_chunks
from ragplatform.ingestion.parsers import ParseLimits, parse
from ragplatform.observability.telemetry import Telemetry, logger, tracer
from ragplatform.providers.base import Embedder
from ragplatform.store.db import Database
from ragplatform.store.repository import Repository

Outcome = Literal["ready", "failed", "skipped"]


@dataclass(frozen=True)
class IngestResult:
    outcome: Outcome
    document_id: UUID
    version: int
    chunks: int = 0
    reused_embeddings: int = 0
    error_code: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        *,
        db: Database,
        repo: Repository,
        blobs: BlobStore,
        embedder: Embedder,
        settings: Settings,
        telemetry: Telemetry,
    ) -> None:
        self.db = db
        self.repo = repo
        self.blobs = blobs
        self.embedder = embedder
        self.settings = settings
        self.telemetry = telemetry

    @tracer.capture_method(capture_response=False)
    def process(self, tenant_id: str, document_id: UUID, version: int) -> IngestResult:
        scope = AccessScope.system(tenant_id)
        started = time.perf_counter()
        with self.db.transaction(scope) as tx:
            claimed = self.repo.claim_version(
                tx, document_id, version, self.settings.ingest_lease_s
            )
            document = self.repo.get_document(tx, document_id) if claimed else None
        if claimed is None or document is None:
            logger.info(
                "version not claimable, skipping",
                extra={"document_id": str(document_id), "version": version},
            )
            self.telemetry.add("IngestSkipped", 1)
            return IngestResult("skipped", document_id, version)

        log_extra = {"document_id": str(document_id), "version": version, "tenant_id": tenant_id}
        try:
            data = self.blobs.get(claimed.storage_key, max_bytes=self.settings.max_upload_bytes)
            if len(data) != claimed.size_bytes:
                raise PermanentIngestError(
                    "size_mismatch", f"declared {claimed.size_bytes} bytes, received {len(data)}"
                )
            if hashlib.sha256(data).hexdigest() != claimed.sha256:
                raise PermanentIngestError(
                    "checksum_mismatch", "the uploaded bytes do not match the declared SHA-256"
                )
            try:
                parsed = parse(
                    data,
                    document.content_type,
                    ParseLimits(
                        self.settings.max_pdf_pages, self.settings.max_docx_uncompressed_bytes
                    ),
                )
            except PermanentIngestError:
                raise
            except Exception as exc:
                # Parser libraries raise many exception types on malformed
                # files. The same bytes fail the same way on every retry.
                raise PermanentIngestError(
                    "parse_error", f"the file could not be parsed ({type(exc).__name__})"
                ) from exc
            title = document.title
            if document.metadata.get("title_source") == "filename" and parsed.title:
                title = parsed.title[:300]
            chunks = build_chunks(
                parsed,
                title=title,
                strategy=self.settings.chunk_strategy,
                chunk_tokens=self.settings.chunk_tokens,
                overlap_tokens=self.settings.chunk_overlap_tokens,
                parent_tokens=self.settings.parent_chunk_tokens,
                contextual_headers=self.settings.contextual_headers,
            )
            if not any(c.kind != ChunkKind.PARENT for c in chunks):
                raise PermanentIngestError("no_text", "the document produced no searchable text")

            embeddings, reused = self._embed(tenant_id, chunks)
            with self.db.transaction(scope) as tx:
                written = self.repo.activate_version(
                    tx,
                    document=document,
                    version=version,
                    chunks=chunks,
                    embeddings=embeddings,
                    title=title,
                )
                self.repo.audit(
                    tx,
                    tenant_id=tenant_id,
                    actor_id=scope.user_id,
                    action="document.ingested",
                    resource_type="document",
                    resource_id=str(document_id),
                    request_id=None,
                    detail={
                        "version": version,
                        "chunks": written,
                        "reused_embeddings": reused,
                        "flagged_chunks": sum(
                            1 for c in chunks if c.risk_flags and c.kind != ChunkKind.PARENT
                        ),
                    },
                )
        except PermanentIngestError as exc:
            with self.db.transaction(scope) as tx:
                self.repo.fail_version(tx, document_id, version, exc.reason_code, exc.detail)
            self.telemetry.add("IngestFailed", 1, reason=exc.reason_code)
            logger.warning("ingest rejected", extra={**log_extra, "reason": exc.reason_code})
            return IngestResult("failed", document_id, version, error_code=exc.reason_code)
        except Exception:
            # Hand the version back so the redelivered message can claim it.
            # Without this a crash would leave it "processing" and the retry
            # would skip it until the lease expired.
            try:
                with self.db.transaction(scope) as tx:
                    self.repo.release_claim(tx, document_id, version)
            except RagError:
                logger.exception(
                    "could not release claim; it frees when the lease expires", extra=log_extra
                )
            self.telemetry.add("IngestRetried", 1)
            logger.exception(
                "ingest hit a transient error; message will be retried", extra=log_extra
            )
            raise

        elapsed = (time.perf_counter() - started) * 1000
        flagged = sum(1 for c in chunks if c.risk_flags and c.kind != ChunkKind.PARENT)
        self.telemetry.add("IngestDuration", elapsed, "ms")
        self.telemetry.add("ChunksWritten", written)
        self.telemetry.add("EmbeddingsReused", reused)
        if flagged:
            self.telemetry.add("ChunksFlaggedForInjection", flagged)
        logger.info(
            "ingest complete",
            extra={
                **log_extra,
                "chunks": written,
                "reused": reused,
                "flagged": flagged,
                "ms": round(elapsed),
            },
        )
        return IngestResult("ready", document_id, version, chunks=written, reused_embeddings=reused)

    def _embed(self, tenant_id: str, chunks: list) -> tuple[dict[int, list[float]], int]:  # type: ignore[type-arg]
        targets = [c for c in chunks if c.kind != ChunkKind.PARENT]
        with self.db.transaction(AccessScope.system(tenant_id)) as tx:
            cached = self.repo.embeddings_for_hashes(
                tx, tenant_id, (c.content_hash for c in targets)
            )
        embeddings: dict[int, list[float]] = {}
        missing = []
        for chunk in targets:
            if chunk.content_hash in cached:
                embeddings[chunk.chunk_index] = cached[chunk.content_hash]
            else:
                missing.append(chunk)
        batch = self.settings.embed_batch_size
        for start in range(0, len(missing), batch):
            group = missing[start : start + batch]
            vectors = self.embedder.embed_documents([c.embed_text or c.text for c in group])
            for chunk, vector in zip(group, vectors, strict=True):
                embeddings[chunk.chunk_index] = vector
        return embeddings, len(targets) - len(missing)
