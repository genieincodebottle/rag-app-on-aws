"""Forward-only SQL migrations and runtime-role setup.

Run by ``ragplatform migrate`` locally and by the ``migrate`` Lambda on AWS,
always with the owner (master) credentials. The runtime role is created here,
not in Terraform, so its password never appears in Terraform state.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

from ragplatform.domain.models import AccessScope
from ragplatform.observability.telemetry import logger
from ragplatform.store.db import Database

_ROLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")
_FILE_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()


def load_migrations() -> list[Migration]:
    found = []
    for entry in resources.files("ragplatform.store.migrations").iterdir():
        match = _FILE_NAME.match(entry.name)
        if match:
            found.append(Migration(int(match.group(1)), entry.name, entry.read_text("utf-8")))
    found.sort(key=lambda m: m.version)
    versions = [m.version for m in found]
    if versions != list(range(1, len(found) + 1)):
        raise RuntimeError(f"migration versions must be contiguous from 1, got {versions}")
    return found


def render(sql: str, *, dimensions: int, app_role: str) -> str:
    if not _ROLE_NAME.match(app_role):
        raise ValueError(f"invalid role name {app_role!r}")
    if dimensions not in (256, 512, 768, 1024, 1536, 3072):
        raise ValueError(f"unsupported embedding dimension {dimensions}")
    return sql.replace("{{EMBEDDING_DIMENSIONS}}", str(dimensions)).replace(
        "{{APP_ROLE}}", app_role
    )


def split_statements(sql: str) -> list[str]:
    """Split on top-level semicolons, respecting quotes, dollar quotes and comments.

    The RDS Data API runs one statement per call, so a migration file cannot be
    sent as a single string the way psycopg would accept it.
    """
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql)
    dollar_tag: str | None = None
    in_single = False
    while i < n:
        ch = sql[i]
        if dollar_tag is not None:
            if sql.startswith(dollar_tag, i):
                buf.append(dollar_tag)
                i += len(dollar_tag)
                dollar_tag = None
                continue
            buf.append(ch)
            i += 1
            continue
        if in_single:
            buf.append(ch)
            if ch == "'":
                in_single = False
            i += 1
            continue
        if ch == "-" and sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end == -1 else end + 1
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == "$":
            match = re.match(r"\$[A-Za-z_]*\$", sql[i:])
            if match:
                dollar_tag = match.group(0)
                buf.append(dollar_tag)
                i += len(dollar_tag)
                continue
        if ch == ";":
            statement = "".join(buf).strip()
            if statement:
                statements.append(statement)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def ensure_app_role(db: Database, role: str, password: str | None, *, rotate: bool = False) -> bool:
    """Create the runtime role if missing. Returns True when a password was set.

    ``password`` must come from ``secrets.token_hex``; it is inlined because
    ``CREATE ROLE`` does not accept bind parameters, so it is validated here.
    """
    if not _ROLE_NAME.match(role):
        raise ValueError(f"invalid role name {role!r}")
    if password is not None and not re.fullmatch(r"[0-9a-f]{32,128}", password):
        raise ValueError("password must be lowercase hex from secrets.token_hex")
    with db.transaction() as tx:
        exists = tx.execute("SELECT 1 AS present FROM pg_roles WHERE rolname = :r", {"r": role})
        if not exists:
            if password is None:
                raise ValueError("a password is required to create the runtime role")
            tx.execute(
                f"CREATE ROLE {role} LOGIN PASSWORD '{password}' "
                "NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS"
            )
            logger.info("created runtime role", extra={"role": role})
            return True
        if rotate and password is not None:
            tx.execute(f"ALTER ROLE {role} PASSWORD '{password}'")
            logger.info("rotated runtime role password", extra={"role": role})
            return True
    return False


def apply_migrations(db: Database, *, dimensions: int, app_role: str) -> list[str]:
    with db.transaction() as tx:
        tx.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version integer PRIMARY KEY, name text NOT NULL, checksum text NOT NULL,"
            " applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = {
            int(r["version"]): r["checksum"]
            for r in tx.execute("SELECT version, checksum FROM schema_migrations")
        }

    newly_applied: list[str] = []
    for migration in load_migrations():
        if migration.version in applied:
            if applied[migration.version] != migration.checksum:
                raise RuntimeError(
                    f"{migration.name} changed after it was applied. "
                    "Add a new migration instead of editing an applied one."
                )
            continue
        rendered = render(migration.sql, dimensions=dimensions, app_role=app_role)
        with db.transaction() as tx:
            for statement in split_statements(rendered):
                tx.execute(statement)
            tx.execute(
                "INSERT INTO schema_migrations (version, name, checksum) VALUES (:v, :n, :c)",
                {"v": migration.version, "n": migration.name, "c": migration.checksum},
            )
        logger.info("applied migration", extra={"migration": migration.name})
        newly_applied.append(migration.name)
    return newly_applied


def record_embedding_space(
    db: Database, *, provider: str, model_id: str, dimensions: int
) -> dict[str, Any]:
    """Write the embedding space on first run. Never overwrite a different one."""
    with db.transaction() as tx:
        rows = tx.execute("SELECT provider, model_id, dimensions FROM embedding_space")
        if not rows:
            tx.execute(
                "INSERT INTO embedding_space (provider, model_id, dimensions) VALUES (:p, :m, :d)",
                {"p": provider, "m": model_id, "d": dimensions},
            )
            return {"provider": provider, "model_id": model_id, "dimensions": dimensions}
        return dict(rows[0])


def ensure_tenant(db: Database, tenant_id: str, name: str, **limits: Any) -> None:
    allowed = {"daily_query_limit", "daily_token_limit", "allow_agentic", "allow_web_search"}
    unknown = set(limits) - allowed
    if unknown:
        raise ValueError(f"unknown tenant fields {sorted(unknown)}")
    columns = ["tenant_id", "name", *limits]
    values = ", ".join(f":{c}" for c in columns)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in ["name", *limits])
    # The owner role bypasses row-level security, so the scope is not required
    # here. It is set anyway so this also works when called with a non-owner
    # role that holds the needed grants.
    with db.transaction(AccessScope.system(tenant_id)) as tx:
        tx.execute(
            f"INSERT INTO tenants ({', '.join(columns)}) VALUES ({values})"
            f" ON CONFLICT (tenant_id) DO UPDATE SET {updates}",
            {"tenant_id": tenant_id, "name": name, **limits},
        )
