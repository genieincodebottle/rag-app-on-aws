"""Building the caller's identity.

In AWS the API Gateway Cognito authorizer validates the ID token (signature,
expiry, audience) before the Lambda runs, and puts the claims in
``requestContext.authorizer.claims``. Only API Gateway can invoke the API
function (its resource policy names the API), so those claims are trusted and
nothing in the request body or headers can override them.

Locally, ``RAG_AUTH_MODE=local`` verifies HS256 tokens minted by
``ragplatform token``. Config validation refuses local mode in any deployed
environment.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import Any

from ragplatform.domain.errors import Forbidden, Unauthenticated
from ragplatform.domain.models import AuthContext

TENANT_CLAIM = "custom:tenant_id"
GROUPS_CLAIM = "cognito:groups"
_TENANT = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_GROUP = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def parse_groups(value: Any) -> frozenset[str]:
    """Cognito groups arrive as a list, a comma-separated string, or "[a b]".

    The REST API authorizer flattens claims to strings, and the exact format
    has varied, so all three shapes are accepted. Anything that is not a valid
    group name is dropped.
    """
    if value is None:
        return frozenset()
    if isinstance(value, list | tuple | set | frozenset):
        items = [str(v) for v in value]
    else:
        items = re.split(r"[\s,]+", str(value).strip().strip("[]"))
    return frozenset(item for item in items if item and _GROUP.match(item))


def context_from_claims(
    claims: Mapping[str, Any], *, admin_group: str, request_id: str = "-"
) -> AuthContext:
    subject = claims.get("sub")
    if not subject:
        raise Unauthenticated("token has no subject")
    tenant = claims.get(TENANT_CLAIM)
    if not tenant or not _TENANT.match(str(tenant)):
        # A user without a tenant assignment can authenticate but reach nothing.
        raise Forbidden("user is not assigned to a tenant")
    groups = parse_groups(claims.get(GROUPS_CLAIM))
    return AuthContext(
        user_id=str(subject),
        tenant_id=str(tenant),
        groups=groups,
        is_tenant_admin=admin_group in groups,
        request_id=request_id,
    )


def issue_local_token(
    secret: str, *, user_id: str, tenant_id: str, groups: list[str], ttl_s: int = 12 * 3600
) -> str:
    import jwt

    now = int(time.time())
    payload = {
        "sub": user_id,
        TENANT_CLAIM: tenant_id,
        GROUPS_CLAIM: groups,
        "iss": "ragplatform-local",
        "iat": now,
        "exp": now + ttl_s,
    }
    token: str = jwt.encode(payload, secret, algorithm="HS256")
    return token


def verify_local_token(secret: str, token: str) -> dict[str, Any]:
    import jwt

    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            issuer="ragplatform-local",
            options={"require": ["exp", "sub", "iss"]},
        )
    except jwt.PyJWTError as exc:
        raise Unauthenticated(f"invalid local token: {exc}") from exc
    return claims
