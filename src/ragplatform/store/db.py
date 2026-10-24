"""SQL execution behind one small interface, with two implementations.

* ``PsycopgDatabase`` talks to PostgreSQL directly. Used locally and in tests.
* ``DataApiDatabase`` talks to Aurora through the RDS Data API. Used in Lambda,
  which then needs no VPC attachment, NAT gateway or connection pooling.

Both run the same SQL. Parameters use ``:name`` placeholders, lists and dicts
are sent as JSON text, and SQL casts them (``CAST(:x AS jsonb)``). That keeps
the repository free of driver-specific code.

Every transaction can carry an ``AccessScope``. Its first statement sets the
row-level-security variables with ``set_config(..., true)``, which scopes them
to that transaction. The Data API pools and reuses connections and keeps no
session state you can rely on, so a session-level ``SET`` would not survive between calls, while
transaction-scoped settings do as long as every statement carries the same
``transactionId``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID

from ragplatform.domain.errors import DatabaseResuming, RagError, UpstreamUnavailable
from ragplatform.domain.models import AccessScope
from ragplatform.observability.telemetry import logger

Row = dict[str, Any]


class UniqueViolation(Exception):
    def __init__(self, constraint: str | None) -> None:
        super().__init__(constraint or "unique violation")
        self.constraint = constraint


class Transaction(Protocol):
    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> list[Row]: ...


class Database(Protocol):
    def transaction(
        self, scope: AccessScope | None = None, *, search_tuning: int | None = None
    ) -> Any:  # a context manager yielding a Transaction
        ...


def to_param(value: Any) -> Any:
    """Normalise a Python value into something both drivers send the same way."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list | tuple | dict | frozenset | set):
        if isinstance(value, frozenset | set):
            value = sorted(value)
        return json.dumps(value, separators=(",", ":"))
    return value


def scope_statement(scope: AccessScope) -> tuple[str, dict[str, Any]]:
    return (
        "SELECT set_config('app.tenant_id', :tenant_id, true),"
        " set_config('app.user_id', :user_id, true),"
        " set_config('app.groups', :groups, true),"
        " set_config('app.acl_bypass', :acl_bypass, true)",
        {
            "tenant_id": scope.tenant_id,
            "user_id": scope.user_id,
            "groups": to_param(scope.groups),
            "acl_bypass": "on" if scope.acl_bypass else "off",
        },
    )


def search_statement(ef_search: int) -> tuple[str, dict[str, Any]]:
    # relaxed_order lets HNSW keep scanning when filters (tenant, ACL, tags)
    # remove candidates, instead of returning fewer than LIMIT rows (pgvector 0.8+).
    return (
        "SELECT set_config('hnsw.ef_search', :ef, true),"
        " set_config('hnsw.iterative_scan', 'relaxed_order', true)",
        {"ef": str(int(ef_search))},
    )


# --------------------------------------------------------------------------- psycopg

_PLACEHOLDER = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")


def to_pyformat(sql: str) -> str:
    """``:name`` to ``%(name)s``. Leaves ``::type`` casts alone."""
    return _PLACEHOLDER.sub(r"%(\1)s", sql.replace("%", "%%"))


class PsycopgDatabase:
    def __init__(
        self, conninfo: str, *, min_size: int = 1, max_size: int = 10, timeout_s: float = 5.0
    ) -> None:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self._timeout_s = timeout_s
        self._conninfo = conninfo
        self._pool = ConnectionPool(
            conninfo,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row, "autocommit": False, "connect_timeout": 5},
            timeout=timeout_s,
            # Opened lazily so a wrong URL fails on first use with one clear
            # error, not with a background thread retrying in the log.
            open=False,
        )

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def transaction(
        self, scope: AccessScope | None = None, *, search_tuning: int | None = None
    ) -> Iterator[_PsycopgTx]:
        import psycopg
        from psycopg_pool import PoolTimeout

        if self._pool.closed:
            self._pool.open(wait=False)
        try:
            with self._pool.connection() as conn, conn.transaction():
                tx = _PsycopgTx(conn)
                if scope is not None:
                    tx.execute(*scope_statement(scope))
                if search_tuning is not None:
                    tx.execute(*search_statement(search_tuning))
                yield tx
        except psycopg.errors.UniqueViolation as exc:
            raise UniqueViolation(exc.diag.constraint_name) from exc
        except psycopg.OperationalError as exc:
            raise UpstreamUnavailable(f"database unavailable: {exc}") from exc
        except PoolTimeout as exc:
            raise self._diagnose() from exc

    def _diagnose(self) -> UpstreamUnavailable:
        """The pool hides why it could not connect; one direct attempt shows it."""
        import psycopg

        try:
            psycopg.connect(self._conninfo, connect_timeout=5).close()
        except psycopg.Error as direct:
            return UpstreamUnavailable(f"database unavailable: {direct}")
        return UpstreamUnavailable(
            f"no free database connection within {self._timeout_s} s (pool exhausted)"
        )


class _PsycopgTx:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> list[Row]:
        import psycopg

        bound = {k: to_param(v) for k, v in (params or {}).items()}
        with self._conn.cursor() as cur:
            try:
                cur.execute(to_pyformat(sql), bound)
            except psycopg.errors.UniqueViolation as exc:
                # Raised at the statement, as the Data API does, so a service
                # can turn it into a 409 inside the transaction block. The
                # transaction is aborted either way and rolls back on exit.
                raise UniqueViolation(exc.diag.constraint_name) from exc
            if cur.description is None:
                return []
            return list(cur.fetchall())


# --------------------------------------------------------------------------- RDS Data API

_CONSTRAINT = re.compile(r'constraint "([^"]+)"')


def _data_api_value(value: Any) -> dict[str, Any]:
    value = to_param(value)
    if value is None:
        return {"isNull": True}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"longValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


class DataApiDatabase:
    """Aurora PostgreSQL through the RDS Data API.

    Not exercised against a live cluster by this repository's test suite: the
    request and response shapes are contract-tested with botocore's Stubber.
    """

    def __init__(
        self,
        *,
        cluster_arn: str,
        secret_arn: str,
        database: str,
        client: Any | None = None,
        resume_max_wait_s: float = 25.0,
        sleep: Any = time.sleep,
    ) -> None:
        if client is None:
            import boto3
            from botocore.config import Config

            # Statements can run up to 45 s by default; a shorter SDK read
            # timeout would retry a statement that is still running.
            client = boto3.client(
                "rds-data", config=Config(read_timeout=60, retries={"mode": "standard"})
            )
        self._client = client
        self._base = {"resourceArn": cluster_arn, "secretArn": secret_arn}
        self._database = database
        self._resume_max_wait_s = resume_max_wait_s
        self._sleep = sleep

    def _call(self, op: str, **kwargs: Any) -> dict[str, Any]:
        from botocore.exceptions import ClientError

        deadline = time.monotonic() + self._resume_max_wait_s
        delay = 1.0
        while True:
            try:
                params = dict(self._base, **kwargs)
                # Commit and rollback take only the ARNs and the transaction id;
                # the API model rejects a "database" parameter on them.
                if op in {"begin_transaction", "execute_statement"}:
                    params["database"] = self._database
                result: dict[str, Any] = getattr(self._client, op)(**params)
                return result
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                message = exc.response.get("Error", {}).get("Message", "")
                if code == "DatabaseResumingException":
                    if time.monotonic() + delay > deadline:
                        raise DatabaseResuming(message) from exc
                    logger.info("database resuming, waiting", extra={"wait_s": delay})
                    self._sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                if code == "BadRequestException" and "duplicate key value" in message:
                    match = _CONSTRAINT.search(message)
                    raise UniqueViolation(match.group(1) if match else None) from exc
                if code in {
                    "ThrottlingException",
                    "ServiceUnavailableError",
                    "StatementTimeoutException",
                    "InternalServerErrorException",
                }:
                    raise UpstreamUnavailable(f"data api {code}: {message}") from exc
                raise RagError(f"data api {code}: {message}") from exc

    @contextmanager
    def transaction(
        self, scope: AccessScope | None = None, *, search_tuning: int | None = None
    ) -> Iterator[_DataApiTx]:
        tx_id = self._call("begin_transaction")["transactionId"]
        tx = _DataApiTx(self, tx_id)
        try:
            if scope is not None:
                tx.execute(*scope_statement(scope))
            if search_tuning is not None:
                tx.execute(*search_statement(search_tuning))
            yield tx
        except BaseException:
            try:
                self._call("rollback_transaction", transactionId=tx_id)
            except RagError:
                logger.exception("rollback failed", extra={"transaction_id": tx_id})
            raise
        else:
            self._call("commit_transaction", transactionId=tx_id)


class _DataApiTx:
    def __init__(self, db: DataApiDatabase, tx_id: str) -> None:
        self._db = db
        self._tx_id = tx_id

    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> list[Row]:
        parameters: Sequence[dict[str, Any]] = [
            {"name": k, "value": _data_api_value(v)} for k, v in (params or {}).items()
        ]
        result = self._db._call(
            "execute_statement",
            sql=sql,
            parameters=list(parameters),
            transactionId=self._tx_id,
            formatRecordsAs="JSON",
        )
        records = result.get("formattedRecords")
        if not records:
            return []
        rows: list[Row] = json.loads(records)
        return rows
