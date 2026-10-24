"""The HTTP API end to end, and both Lambda entry points with real event shapes."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mangum import Mangum

from ragplatform.adapters.blobstore import LocalBlobStore
from ragplatform.api.app import create_app
from ragplatform.container import Container, build_container
from ragplatform.domain.errors import UpstreamUnavailable
from ragplatform.observability.telemetry import Telemetry
from ragplatform.security.auth import issue_local_token
from tests.conftest import TRAVEL_POLICY, ingest_text, make_settings, user

pytestmark = pytest.mark.integration


def token(
    container: Container,
    user_id: str = "alice",
    tenant: str = "acme",
    groups: tuple[str, ...] = ("staff",),
) -> dict[str, str]:
    secret = container.settings.local_auth_secret.get_secret_value()  # type: ignore[union-attr]
    return {
        "Authorization": "Bearer "
        + issue_local_token(secret, user_id=user_id, tenant_id=tenant, groups=list(groups))
    }


def wait_ready(client: TestClient, headers: dict[str, str], document_id: str) -> dict[str, Any]:
    for _ in range(50):
        body: dict[str, Any] = client.get(f"/v1/documents/{document_id}", headers=headers).json()
        if body["document"]["status"] in {"ready", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError("document never finished processing")


def test_full_document_and_question_flow(container: Container) -> None:
    client = TestClient(create_app(container))
    alice = token(container)
    data = TRAVEL_POLICY.encode()

    registered = client.post(
        "/v1/documents",
        headers=alice,
        json={
            "source_name": "travel-policy.md",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "acl_groups": ["staff"],
            "tags": ["policy"],
        },
    )
    assert registered.status_code == 201, registered.text
    body = registered.json()
    document_id = body["document"]["document_id"]
    upload = body["upload"]
    assert upload["method"] == "PUT"
    assert upload["max_bytes"] == len(data)

    uploaded = client.put(
        upload["url"], headers={**alice, "Content-Type": "text/markdown"}, content=data
    )
    assert uploaded.status_code == 202, uploaded.text
    detail = wait_ready(client, alice, document_id)
    assert detail["document"]["status"] == "ready"
    assert detail["document"]["title"] == "Travel Policy"

    answer = client.post(
        "/v1/query",
        headers=token(container, "bob"),
        json={"question": "What is the hotel limit in London?"},
    )
    assert answer.status_code == 200, answer.text
    payload = answer.json()
    assert payload["status"] == "answered"
    assert payload["citations"][0]["document_id"] == document_id
    assert answer.headers["X-Request-Id"] == payload["request_id"]
    assert answer.headers["Cache-Control"] == "no-store"

    again = client.post(
        "/v1/documents",
        headers=alice,
        json={
            "source_name": "copy.md",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "acl_groups": ["staff"],
        },
    )
    assert again.status_code == 200
    assert again.json()["duplicate"] is True
    assert again.json()["upload"] is None

    listing = client.get("/v1/documents?limit=1", headers=alice).json()
    assert [d["document_id"] for d in listing["items"]] == [document_id]

    usage = client.get("/v1/usage", headers=alice).json()
    assert usage["today"]["queries"] == 1

    deleted = client.delete(f"/v1/documents/{document_id}", headers=alice)
    assert deleted.status_code == 200
    gone = client.post(
        "/v1/query",
        headers=token(container, "bob"),
        json={"question": "What is the hotel limit in London?"},
    )
    assert gone.json()["status"] == "insufficient_context"


def test_errors_have_one_shape_and_hide_internals(container: Container) -> None:
    client = TestClient(create_app(container))
    unauthenticated = client.post("/v1/query", json={"question": "hi"})
    assert unauthenticated.status_code == 401
    assert set(unauthenticated.json()["error"]) == {"code", "message", "request_id"}

    forged = client.post(
        "/v1/query", headers={"Authorization": "Bearer not-a-token"}, json={"question": "hi"}
    )
    assert forged.status_code == 401
    assert "PyJWT" not in forged.text
    assert "Traceback" not in forged.text

    invalid = client.post("/v1/query", headers=token(container), json={"question": ""})
    assert invalid.status_code == 422
    missing = client.get(f"/v1/documents/{uuid.uuid4()}", headers=token(container))
    assert missing.status_code == 404


def test_upstream_outage_returns_503_with_retry_after(container: Container, tmp_path: Any) -> None:
    ingest_text(container, user(), "travel.md", TRAVEL_POLICY)

    class Down:
        model_id = "down"

        def generate(self, request: Any) -> Any:
            raise UpstreamUnavailable("ServiceUnavailableException")

    from ragplatform.providers.factory import Providers
    from ragplatform.providers.fake import HashingEmbedder

    broken = build_container(
        container.settings,
        db=container.db,
        blobs=LocalBlobStore(tmp_path / "b"),
        telemetry=Telemetry("T", "t", emit=False),
        providers=Providers(
            embedder=HashingEmbedder(256), chat=Down(), reranker=None, grounding=None
        ),
    )
    response = TestClient(create_app(broken)).post(
        "/v1/query", headers=token(container), json={"question": "What is the hotel limit?"}
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert response.json()["error"]["code"] == "upstream_unavailable"


def test_local_upload_endpoint_is_disabled_in_sqs_mode(container: Container, tmp_path: Any) -> None:
    sqs = build_container(
        make_settings(ingest_mode="sqs"),
        db=container.db,
        blobs=LocalBlobStore(tmp_path / "b"),
        telemetry=Telemetry("T", "t", emit=False),
    )
    response = TestClient(create_app(sqs)).put(
        f"/v1/documents/{uuid.uuid4()}/versions/1/content", headers=token(container), content=b"x"
    )
    assert response.status_code == 404


def api_gateway_event(
    method: str, path: str, claims: dict[str, str] | None, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A REST API (v1) Lambda proxy event, as API Gateway sends it after the Cognito authorizer."""
    return {
        "resource": "/{proxy+}",
        "path": path,
        "httpMethod": method,
        "headers": {"content-type": "application/json", "host": "api.example.com"},
        "multiValueHeaders": {},
        "queryStringParameters": None,
        "multiValueQueryStringParameters": None,
        "pathParameters": {"proxy": path[1:]},
        "stageVariables": None,
        "isBase64Encoded": False,
        "body": json.dumps(body) if body is not None else None,
        "requestContext": {
            "resourcePath": "/{proxy+}",
            "httpMethod": method,
            "path": f"/dev{path}",
            "requestId": "apigw-request-1",
            "stage": "dev",
            "accountId": "111122223333",
            "identity": {"sourceIp": "203.0.113.10"},
            "authorizer": {"claims": claims} if claims else None,
        },
    }


class FakeLambdaContext:
    function_name = "rag-api"
    aws_request_id = "lambda-request-1"
    invoked_function_arn = "arn:aws:lambda:us-east-1:111122223333:function:rag-api"
    memory_limit_in_mb = 1024

    def get_remaining_time_in_millis(self) -> int:
        return 29_000


def test_lambda_proxy_path_trusts_only_authorizer_claims(
    container: Container, tmp_path: Any
) -> None:
    ingest_text(container, user(), "travel.md", TRAVEL_POLICY)
    cognito = build_container(
        make_settings(auth_mode="cognito"),
        db=container.db,
        blobs=LocalBlobStore(tmp_path / "b"),
        telemetry=Telemetry("T", "t", emit=False),
    )
    handler = Mangum(create_app(cognito), lifespan="off")
    claims = {"sub": "bob", "custom:tenant_id": "acme", "cognito:groups": "staff"}

    ok = handler(
        api_gateway_event(
            "POST", "/v1/query", claims, {"question": "What is the hotel limit in London?"}
        ),
        FakeLambdaContext(),
    )
    assert ok["statusCode"] == 200
    assert json.loads(ok["body"])["status"] == "answered"
    assert ok["headers"]["x-request-id"] == "apigw-request-1"

    # A bearer header is ignored in cognito mode: only authorizer claims count.
    no_claims = api_gateway_event("POST", "/v1/query", None, {"question": "hotel limit?"})
    no_claims["headers"]["authorization"] = "Bearer anything"
    assert handler(no_claims, FakeLambdaContext())["statusCode"] == 401

    other_tenant = {"sub": "mallory", "custom:tenant_id": "globex", "cognito:groups": "staff"}
    leak = handler(
        api_gateway_event(
            "POST", "/v1/query", other_tenant, {"question": "What is the hotel limit in London?"}
        ),
        FakeLambdaContext(),
    )
    assert json.loads(leak["body"])["status"] == "insufficient_context"


def test_ingest_handler_reports_partial_batch_failures(
    container: Container, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ragplatform.handlers import ingest as ingest_handler
    from ragplatform.services.document_service import RegisterRequest

    settings = make_settings(documents_bucket="docs")
    good = build_container(
        settings, db=container.db, blobs=container.blobs, telemetry=Telemetry("T", "t", emit=False)
    )
    monkeypatch.setattr(ingest_handler, "get_container", lambda: good)

    data = TRAVEL_POLICY.encode()
    registered = good.documents.register(
        user(),
        RegisterRequest(
            source_name="t.md",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            acl_groups=["staff"],
        ),
    )
    assert registered.version is not None
    good.blobs.put(registered.version.storage_key, data, "text/markdown")

    def message(message_id: str, key: str) -> dict[str, Any]:
        return {
            "messageId": message_id,
            "receiptHandle": "r",
            "attributes": {},
            "messageAttributes": {},
            "md5OfBody": "",
            "eventSource": "aws:sqs",
            "eventSourceARN": "arn:aws:sqs:us-east-1:111122223333:ingest",
            "awsRegion": "us-east-1",
            "body": json.dumps(
                {
                    "Records": [
                        {
                            "eventName": "ObjectCreated:Post",
                            "s3": {"bucket": {"name": "docs"}, "object": {"key": key}},
                        }
                    ]
                }
            ),
        }

    def failing_process(*args: Any) -> Any:
        raise UpstreamUnavailable("throttled")

    event = {"Records": [message("ok-1", registered.version.storage_key)]}
    result = ingest_handler.handler(event, FakeLambdaContext())
    assert result == {"batchItemFailures": []}
    assert good.documents.get(user(), registered.document.document_id).document.status == "ready"

    monkeypatch.setattr(good.pipeline, "process", failing_process)
    retry = ingest_handler.handler(
        {
            "Records": [
                message("bad-1", f"raw/acme/{uuid.uuid4()}/v1"),
                message("ignored-1", "uploads/old/style/key.pdf"),
            ]
        },
        FakeLambdaContext(),
    )
    assert retry == {"batchItemFailures": [{"itemIdentifier": "bad-1"}]}
