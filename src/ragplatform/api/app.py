"""HTTP API. The same app runs under uvicorn locally and under Mangum in Lambda."""

from __future__ import annotations

import uuid
from typing import Annotated, Any
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ragplatform import __version__
from ragplatform.container import Container
from ragplatform.domain.errors import (
    NotFound,
    PayloadTooLarge,
    RagError,
    Unauthenticated,
    UpstreamUnavailable,
)
from ragplatform.domain.models import AccessScope, AuthContext, Document, DocumentVersion
from ragplatform.observability.telemetry import logger
from ragplatform.security.auth import context_from_claims, verify_local_token
from ragplatform.services.document_service import (
    AccessUpdate,
    DocumentDetail,
    DocumentPage,
    NewVersionRequest,
    RegisterRequest,
    RegisterResponse,
)
from ragplatform.services.query_service import QueryRequest, QueryResponse


def _request_id(request: Request) -> str:
    event = request.scope.get("aws.event") or {}
    return str((event.get("requestContext") or {}).get("requestId") or uuid.uuid4())


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = _request_id(request)
        logger.append_keys(request_id=request.state.request_id)
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


def _error(
    request: Request, status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers=headers,
    )


def auth(request: Request) -> AuthContext:
    """Identity for this request. Never read from the request body."""
    settings = request.app.state.container.settings
    request_id = getattr(request.state, "request_id", "-")
    if settings.auth_mode == "cognito":
        event = request.scope.get("aws.event") or {}
        # Any level can be missing or null; all of them mean "not authenticated".
        authorizer = (event.get("requestContext") or {}).get("authorizer") or {}
        claims = authorizer.get("claims")
        if not claims:
            raise Unauthenticated("no authorizer claims on the request")
    else:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise Unauthenticated("missing bearer token")
        secret = settings.local_auth_secret
        if secret is None:  # config validation makes this unreachable
            raise Unauthenticated("local auth is not configured")
        claims = verify_local_token(secret.get_secret_value(), header[7:].strip())
    ctx = context_from_claims(
        claims, admin_group=settings.tenant_admin_group, request_id=request_id
    )
    logger.append_keys(tenant_id=ctx.tenant_id)
    return ctx


Auth = Annotated[AuthContext, Depends(auth)]


def create_app(container: Container) -> FastAPI:
    settings = container.settings
    app = FastAPI(
        title="RAG Platform API",
        version=__version__,
        docs_url="/docs" if not settings.in_lambda else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.in_lambda else None,
    )
    app.state.container = container
    app.add_middleware(RequestIdMiddleware)
    if settings.allowed_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_cors_origins,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "PUT"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.exception_handler(RagError)
    async def rag_error(request: Request, exc: RagError) -> JSONResponse:
        headers = None
        if isinstance(exc, UpstreamUnavailable):
            headers = {"Retry-After": str(exc.retry_after_s)}
        log = logger.warning if exc.http_status < 500 else logger.error
        log(
            "request failed",
            extra={"code": exc.code, "status": exc.http_status, "detail": exc.detail[:500]},
        )
        container.telemetry.add("ApiError", 1, code=exc.code)
        return _error(request, exc.http_status, exc.code, exc.public_message, headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in e["loc"][1:]) for e in exc.errors()})
        return _error(request, 422, "validation_failed", f"Invalid fields: {', '.join(fields)}")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        container.telemetry.add("ApiError", 1, code="internal_error")
        return _error(request, 500, "internal_error", "The request could not be completed.")

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/health/ready")
    def ready(request: Request) -> JSONResponse:
        # Touches the database. Do not poll this on a schedule against a dev
        # stack: every call wakes Aurora Serverless from 0 ACU.
        try:
            container.retriever.check_embedding_space()
        except RagError as exc:
            return _error(request, 503, exc.code, "not ready")
        return JSONResponse({"status": "ready"})

    @app.post("/v1/query", response_model=QueryResponse)
    def query(body: QueryRequest, ctx: Auth) -> QueryResponse:
        return container.queries.answer(ctx, body)

    @app.post("/v1/documents", response_model=RegisterResponse, status_code=201)
    def register(body: RegisterRequest, ctx: Auth, response: Response) -> RegisterResponse:
        result = container.documents.register(ctx, body)
        if result.duplicate:
            response.status_code = 200  # nothing was created
        return result

    @app.get("/v1/documents", response_model=DocumentPage)
    def list_documents(
        ctx: Auth,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
        cursor: str | None = None,
        status: str | None = None,
    ) -> DocumentPage:
        return container.documents.list(ctx, limit=limit, cursor=cursor, status=status)

    @app.get("/v1/documents/{document_id}", response_model=DocumentDetail)
    def get_document(document_id: UUID, ctx: Auth) -> DocumentDetail:
        return container.documents.get(ctx, document_id)

    @app.patch("/v1/documents/{document_id}", response_model=Document)
    def update_document(document_id: UUID, body: AccessUpdate, ctx: Auth) -> Document:
        return container.documents.update_access(ctx, document_id, body)

    @app.delete("/v1/documents/{document_id}")
    def delete_document(document_id: UUID, ctx: Auth) -> dict[str, Any]:
        return container.documents.delete(ctx, document_id)

    @app.post(
        "/v1/documents/{document_id}/versions", response_model=RegisterResponse, status_code=201
    )
    def new_version(document_id: UUID, body: NewVersionRequest, ctx: Auth) -> RegisterResponse:
        return container.documents.new_version(ctx, document_id, body)

    @app.put(
        "/v1/documents/{document_id}/versions/{version}/content",
        response_model=DocumentVersion,
        status_code=202,
    )
    async def upload_content(
        document_id: UUID, version: int, request: Request, background: BackgroundTasks, ctx: Auth
    ) -> DocumentVersion:
        if settings.ingest_mode != "inline":
            raise NotFound("direct upload is only available in local mode")
        declared = int(request.headers.get("content-length") or 0)
        if declared > settings.max_upload_bytes:
            raise PayloadTooLarge(f"{declared} bytes")
        data = await request.body()
        record = container.documents.accept_local_upload(ctx, document_id, version, data)
        background.add_task(container.pipeline.process, ctx.tenant_id, document_id, version)
        return record

    @app.get("/v1/usage")
    def usage(ctx: Auth) -> dict[str, Any]:
        with container.db.transaction(AccessScope.for_user(ctx)) as tx:
            today = container.repo.usage_today(tx, ctx.tenant_id)
            tenant = container.repo.get_tenant(tx, ctx.tenant_id) or {}
        return {
            "tenant_id": ctx.tenant_id,
            "today": {k: float(v) if k == "cost_usd" else int(v) for k, v in today.items()},
            "limits": {
                "daily_query_limit": tenant.get("daily_query_limit"),
                "daily_token_limit": tenant.get("daily_token_limit"),
            },
        }

    return app
