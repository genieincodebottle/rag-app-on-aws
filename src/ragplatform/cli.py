"""Command-line tools for local development and operations.

ragplatform migrate                         apply migrations, create the runtime role
ragplatform tenant acme "Acme Corp" --agentic
ragplatform token --user alice --tenant acme --groups finance,staff
ragplatform ingest docs/*.md --tenant acme --user alice --groups staff
ragplatform ask "What is the hotel limit?" --tenant acme --user alice --groups staff
ragplatform reembed                         re-embed every chunk with the configured model
ragplatform serve                           run the API on http://localhost:8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ragplatform.config import get_settings
from ragplatform.domain.models import AuthContext


def _ctx(args: argparse.Namespace) -> AuthContext:
    groups = frozenset(g for g in (args.groups or "").split(",") if g)
    return AuthContext(
        user_id=args.user,
        tenant_id=args.tenant,
        groups=groups,
        is_tenant_admin="tenant-admin" in groups,
        request_id="cli",
    )


def cmd_migrate(args: argparse.Namespace) -> int:
    from ragplatform.container import build_database
    from ragplatform.store.migrate import apply_migrations, ensure_app_role, record_embedding_space

    settings = get_settings()
    db = build_database(settings, admin=True)
    password = (
        settings.local_app_db_password.get_secret_value()
        if settings.local_app_db_password
        else None
    )
    ensure_app_role(db, settings.app_db_role, password, rotate=args.rotate)
    applied = apply_migrations(
        db, dimensions=settings.embedding_dimensions, app_role=settings.app_db_role
    )
    from ragplatform.providers.factory import build_providers

    space = build_providers(settings).embedder.space
    stored = record_embedding_space(
        db, provider=space.provider, model_id=space.model_id, dimensions=space.dimensions
    )
    print(json.dumps({"applied": applied, "embedding_space": stored}, indent=2))
    if (stored["model_id"], int(stored["dimensions"])) != (space.model_id, space.dimensions):
        print(
            "WARNING: the index was built with a different embedding model. "
            "Run `ragplatform reembed`.",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_tenant(args: argparse.Namespace) -> int:
    from ragplatform.container import build_database
    from ragplatform.store.migrate import ensure_tenant

    db = build_database(get_settings(), admin=True)
    ensure_tenant(
        db, args.tenant_id, args.name, allow_agentic=args.agentic, allow_web_search=args.web
    )
    print(f"tenant {args.tenant_id} ready")
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    from ragplatform.security.auth import issue_local_token

    settings = get_settings()
    if settings.local_auth_secret is None:
        print("RAG_LOCAL_AUTH_SECRET is not set", file=sys.stderr)
        return 1
    groups = [g for g in (args.groups or "").split(",") if g]
    print(
        issue_local_token(
            settings.local_auth_secret.get_secret_value(),
            user_id=args.user,
            tenant_id=args.tenant,
            groups=groups,
        )
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from ragplatform.container import build_container
    from ragplatform.services.document_service import RegisterRequest

    container = build_container()
    ctx = _ctx(args)
    acl = [g for g in (args.acl or "").split(",") if g]
    failures = 0
    for name in args.paths:
        path = Path(name)
        data = path.read_bytes()
        registered = container.documents.register(
            ctx,
            RegisterRequest(
                source_name=path.name,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                acl_groups=acl,
            ),
        )
        if registered.duplicate or registered.version is None:
            print(f"{path.name}: already ingested as {registered.document.document_id}")
            continue
        doc_id, version = registered.document.document_id, registered.version.version
        container.documents.accept_local_upload(ctx, doc_id, version, data)
        result = container.pipeline.process(ctx.tenant_id, doc_id, version)
        print(
            f"{path.name}: {result.outcome} ({result.chunks} chunks, "
            f"{result.reused_embeddings} embeddings reused) {doc_id}"
        )
        failures += result.outcome == "failed"
    return 1 if failures else 0


def cmd_ask(args: argparse.Namespace) -> int:
    from ragplatform.container import build_container
    from ragplatform.services.query_service import QueryRequest

    container = build_container()
    response = container.queries.answer(
        _ctx(args),
        QueryRequest(question=args.question, mode=args.mode, use_cache=False, debug=args.debug),
    )
    print(response.model_dump_json(indent=2, exclude_none=True))
    return 0


def cmd_reembed(args: argparse.Namespace) -> int:
    from ragplatform.container import build_database
    from ragplatform.providers.factory import build_providers
    from ragplatform.store.reembed import reembed_all

    settings = get_settings()
    summary = reembed_all(
        build_database(settings, admin=True),
        build_providers(settings).embedder,
        contextual_headers=settings.contextual_headers,
        batch_size=settings.embed_batch_size,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from ragplatform.api.app import create_app
    from ragplatform.container import build_container

    uvicorn.run(create_app(build_container()), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragplatform")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("migrate", help="apply migrations and create the runtime role")
    p.add_argument("--rotate", action="store_true", help="reset the runtime role password")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("tenant", help="create or update a tenant")
    p.add_argument("tenant_id")
    p.add_argument("name")
    p.add_argument("--agentic", action="store_true")
    p.add_argument("--web", action="store_true")
    p.set_defaults(func=cmd_tenant)

    for name, func in (("token", cmd_token), ("ingest", cmd_ingest), ("ask", cmd_ask)):
        p = sub.add_parser(name)
        p.add_argument("--user", required=True)
        p.add_argument("--tenant", required=True)
        p.add_argument("--groups", default="")
        if name == "ingest":
            p.add_argument("paths", nargs="+")
            p.add_argument("--acl", default="", help="comma-separated groups that may read")
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--mode", choices=["standard", "agentic"], default="standard")
            p.add_argument("--debug", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("reembed", help="re-embed all chunks with the configured embedder")
    p.set_defaults(func=cmd_reembed)

    p = sub.add_parser("serve", help="run the API locally")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
