"""Lambda entry point for database migrations. Invoked by the deploy pipeline.

Runs with the Aurora master secret (managed by RDS in Secrets Manager). It:

1. creates the runtime role with a random password if it does not exist and
   writes host, port, user and password to the runtime-role secret, which the
   API and ingest functions use with the Data API;
2. applies pending migrations;
3. records the embedding space;
4. creates any tenants listed in the event, for example
   ``{"tenants": [{"tenant_id": "acme", "name": "Acme"}]}``.

Event ``{"rotate_app_password": true}`` rotates the runtime role's password.
The function is idempotent: running it twice changes nothing the second time.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

import boto3
from aws_lambda_powertools.utilities.typing import LambdaContext

from ragplatform.config import get_settings
from ragplatform.container import build_database
from ragplatform.observability.telemetry import logger
from ragplatform.store.migrate import (
    apply_migrations,
    ensure_app_role,
    ensure_tenant,
    record_embedding_space,
)


def _secret_has_value(client: Any, secret_id: str | None) -> bool:
    """False when the runtime secret exists but has never been written.

    Covers a previous run that created the role and then failed before writing
    the secret: without this, the role would exist with a password nobody has.
    """
    from botocore.exceptions import ClientError

    try:
        client.get_secret_value(SecretId=secret_id)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            return False
        raise
    return True


@logger.inject_lambda_context(clear_state=True)
def handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    settings = get_settings()
    admin_db = build_database(settings, admin=True)
    secrets_client = boto3.client("secretsmanager", region_name=settings.aws_region)

    rotate = bool(event.get("rotate_app_password")) or not _secret_has_value(
        secrets_client, settings.db_secret_arn
    )
    password = secrets.token_hex(32)
    password_set = ensure_app_role(admin_db, settings.app_db_role, password, rotate=rotate)
    if password_set:
        # The runtime secret is written only after the role holds the password,
        # so a reader never sees a password the database does not accept. The
        # Data API needs only username and password in the secret; the cluster
        # is named by its ARN on every call.
        secrets_client.put_secret_value(
            SecretId=settings.db_secret_arn,
            SecretString=json.dumps(
                {
                    "engine": "postgres",
                    "dbname": settings.db_name,
                    "username": settings.app_db_role,
                    "password": password,
                }
            ),
        )

    applied = apply_migrations(
        admin_db, dimensions=settings.embedding_dimensions, app_role=settings.app_db_role
    )
    space = record_embedding_space(
        admin_db,
        provider=settings.provider,
        model_id=settings.embedding_model_id,
        dimensions=settings.embedding_dimensions,
    )
    for tenant in event.get("tenants", []):
        ensure_tenant(
            admin_db,
            tenant["tenant_id"],
            tenant["name"],
            **{k: v for k, v in tenant.items() if k not in {"tenant_id", "name"}},
        )

    result = {
        "applied": applied,
        "app_password_set": password_set,
        "embedding_space": space,
        "tenants": [t["tenant_id"] for t in event.get("tenants", [])],
    }
    logger.info("migration complete", extra=result)
    return result
