"""Typed errors. Each carries a stable code and the HTTP status the API maps it to.

Services raise these and never return error strings in place of results, so a
failure cannot be mistaken for an answer (a failed model call is a 502 or 503,
not a 200 with an apology in the body).
"""

from __future__ import annotations


class RagError(Exception):
    code = "internal_error"
    http_status = 500
    # Safe to show to the caller. Internal detail goes to logs only.
    public_message = "The request could not be completed."

    def __init__(self, detail: str | None = None, *, public_message: str | None = None) -> None:
        super().__init__(detail or self.public_message)
        self.detail = detail or self.public_message
        if public_message is not None:
            self.public_message = public_message


class ValidationFailed(RagError):
    code = "validation_failed"
    http_status = 422
    public_message = "The request is not valid."


class Unauthenticated(RagError):
    code = "unauthenticated"
    http_status = 401
    public_message = "Authentication is required."


class Forbidden(RagError):
    code = "forbidden"
    http_status = 403
    public_message = "You do not have access to this resource."


class NotFound(RagError):
    code = "not_found"
    http_status = 404
    public_message = "The resource was not found."


class Conflict(RagError):
    code = "conflict"
    http_status = 409
    public_message = "The request conflicts with the current state of the resource."


class PayloadTooLarge(RagError):
    code = "payload_too_large"
    http_status = 413
    public_message = "The file is larger than the configured limit."


class UnsupportedMediaType(RagError):
    code = "unsupported_media_type"
    http_status = 415
    public_message = "The file type is not supported."


class BudgetExceeded(RagError):
    code = "budget_exceeded"
    http_status = 429
    public_message = "The daily usage budget for this tenant has been reached."


class GuardrailBlocked(RagError):
    code = "guardrail_blocked"
    http_status = 422
    public_message = "The request was blocked by the content policy."


class UpstreamError(RagError):
    """A dependency answered with an error we will not retry (bad request, format)."""

    code = "upstream_error"
    http_status = 502
    public_message = "A downstream service returned an error."


class UpstreamUnavailable(RagError):
    """A dependency is throttling or down. Safe to retry later."""

    code = "upstream_unavailable"
    http_status = 503
    public_message = "A downstream service is temporarily unavailable. Retry shortly."
    retry_after_s = 5


class DatabaseResuming(UpstreamUnavailable):
    """Aurora Serverless v2 is resuming from 0 ACU (typically about 15 seconds)."""

    code = "database_resuming"
    public_message = "The database is starting up. Retry in about 15 seconds."
    retry_after_s = 15


class EmbeddingSpaceMismatch(RagError):
    """The configured embedding model or dimension differs from the indexed one."""

    code = "embedding_space_mismatch"
    http_status = 500
    public_message = "The search index was built with a different embedding model."


class PermanentIngestError(RagError):
    """The file can never be ingested as uploaded (corrupt, wrong type, too big)."""

    code = "ingest_rejected"
    http_status = 422

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail, public_message=detail)
        self.reason_code = reason_code
