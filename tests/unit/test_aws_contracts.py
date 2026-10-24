"""Contract tests for the AWS adapters.

botocore's Stubber checks each request against the service's API model, so a
misspelt parameter, a wrong nesting level or a missing required field fails
here. These tests do not prove the services behave as stubbed; they prove the
requests are well formed and the responses are read correctly.
"""

from __future__ import annotations

import io
import json
from typing import Any

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber

from ragplatform.domain.errors import (
    DatabaseResuming,
    GuardrailBlocked,
    UpstreamError,
    UpstreamUnavailable,
)
from ragplatform.domain.models import AccessScope
from ragplatform.providers.base import ChatRequest
from ragplatform.providers.bedrock import (
    BedrockReranker,
    ConverseChatModel,
    GuardrailGroundingCheck,
    TitanEmbedder,
)
from ragplatform.store.db import DataApiDatabase, UniqueViolation

REGION = "us-east-1"
CLUSTER = "arn:aws:rds:us-east-1:111122223333:cluster:rag-dev"
SECRET = "arn:aws:secretsmanager:us-east-1:111122223333:secret:rag-app-AbCdEf"


def body(payload: dict[str, Any]) -> StreamingBody:
    raw = json.dumps(payload).encode()
    return StreamingBody(io.BytesIO(raw), len(raw))


# --------------------------------------------------------------------------- RDS Data API


def data_api() -> tuple[Any, Stubber, DataApiDatabase]:
    client = boto3.client("rds-data", region_name=REGION)
    stub = Stubber(client)
    db = DataApiDatabase(
        cluster_arn=CLUSTER,
        secret_arn=SECRET,
        database="ragplatform",
        client=client,
        sleep=lambda _s: None,
    )
    return client, stub, db


ARNS = {"resourceArn": CLUSTER, "secretArn": SECRET}
BASE = {**ARNS, "database": "ragplatform"}


def test_data_api_transaction_sets_scope_first_and_commits() -> None:
    _, stub, db = data_api()
    stub.add_response("begin_transaction", {"transactionId": "tx-1"}, BASE)
    stub.add_response(
        "execute_statement",
        {"formattedRecords": "[{}]"},
        {
            **BASE,
            "sql": ANY,
            "transactionId": "tx-1",
            "formatRecordsAs": "JSON",
            "parameters": [
                {"name": "tenant_id", "value": {"stringValue": "acme"}},
                {"name": "user_id", "value": {"stringValue": "alice"}},
                {"name": "groups", "value": {"stringValue": '["staff"]'}},
                {"name": "acl_bypass", "value": {"stringValue": "off"}},
            ],
        },
    )
    stub.add_response(
        "execute_statement",
        {"formattedRecords": '[{"n": 1}]'},
        {
            **BASE,
            "sql": "SELECT :n AS n",
            "transactionId": "tx-1",
            "formatRecordsAs": "JSON",
            "parameters": [{"name": "n", "value": {"longValue": 1}}],
        },
    )
    stub.add_response(
        "commit_transaction",
        {"transactionStatus": "Transaction Committed"},
        {**ARNS, "transactionId": "tx-1"},
    )
    with stub:
        scope = AccessScope(tenant_id="acme", user_id="alice", groups=frozenset({"staff"}))
        with db.transaction(scope) as tx:
            assert tx.execute("SELECT :n AS n", {"n": 1}) == [{"n": 1}]
    stub.assert_no_pending_responses()


def test_data_api_rolls_back_on_error() -> None:
    _, stub, db = data_api()
    stub.add_response("begin_transaction", {"transactionId": "tx-2"}, BASE)
    stub.add_response(
        "rollback_transaction",
        {"transactionStatus": "Rollback Complete"},
        {**ARNS, "transactionId": "tx-2"},
    )
    with stub, pytest.raises(RuntimeError), db.transaction():
        raise RuntimeError("boom")
    stub.assert_no_pending_responses()


def test_data_api_waits_for_aurora_resume_then_succeeds() -> None:
    _, stub, db = data_api()
    stub.add_client_error(
        "begin_transaction",
        service_error_code="DatabaseResumingException",
        service_message="Database is resuming",
        http_status_code=400,
    )
    stub.add_response("begin_transaction", {"transactionId": "tx-3"}, BASE)
    stub.add_response(
        "commit_transaction", {"transactionStatus": "ok"}, {**ARNS, "transactionId": "tx-3"}
    )
    with stub, db.transaction():
        pass
    stub.assert_no_pending_responses()


def test_data_api_gives_up_with_retry_after_when_resume_takes_too_long() -> None:
    client = boto3.client("rds-data", region_name=REGION)
    stub = Stubber(client)
    db = DataApiDatabase(
        cluster_arn=CLUSTER,
        secret_arn=SECRET,
        database="ragplatform",
        client=client,
        resume_max_wait_s=0.0,
        sleep=lambda _s: None,
    )
    stub.add_client_error(
        "begin_transaction",
        service_error_code="DatabaseResumingException",
        service_message="resuming",
        http_status_code=400,
    )
    with stub, pytest.raises(DatabaseResuming) as err, db.transaction():
        pass
    assert err.value.retry_after_s == 15


def test_data_api_maps_duplicate_key_to_unique_violation() -> None:
    _, stub, db = data_api()
    stub.add_response("begin_transaction", {"transactionId": "tx-4"}, BASE)
    stub.add_client_error(
        "execute_statement",
        service_error_code="BadRequestException",
        http_status_code=400,
        service_message='ERROR: duplicate key value violates unique constraint "documents_pkey"',
    )
    stub.add_response(
        "rollback_transaction", {"transactionStatus": "ok"}, {**ARNS, "transactionId": "tx-4"}
    )
    with stub, pytest.raises(UniqueViolation) as err, db.transaction() as tx:
        tx.execute("INSERT INTO documents VALUES (1)")
    assert err.value.constraint == "documents_pkey"


def test_data_api_value_types() -> None:
    from ragplatform.store.db import _data_api_value

    assert _data_api_value(None) == {"isNull": True}
    assert _data_api_value(True) == {"booleanValue": True}
    assert _data_api_value(3) == {"longValue": 3}
    assert _data_api_value(0.5) == {"doubleValue": 0.5}
    assert _data_api_value(["a", "b"]) == {"stringValue": '["a","b"]'}


# --------------------------------------------------------------------------- Bedrock embeddings


def runtime() -> tuple[Any, Stubber]:
    client = boto3.client("bedrock-runtime", region_name=REGION)
    return client, Stubber(client)


def titan(client: Any) -> TitanEmbedder:
    return TitanEmbedder(
        model_id="amazon.titan-embed-text-v2:0",
        dimensions=256,
        region=REGION,
        timeout_s=5,
        attempts=1,
        concurrency=1,
        client=client,
    )


def test_titan_request_shape() -> None:
    client, stub = runtime()
    vector = [1.0] + [0.0] * 255
    stub.add_response(
        "invoke_model",
        {
            "body": body({"embedding": vector, "inputTextTokenCount": 7}),
            "contentType": "application/json",
        },
        {
            "modelId": "amazon.titan-embed-text-v2:0",
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({"inputText": "hello", "dimensions": 256, "normalize": True}),
        },
    )
    embedder = titan(client)
    with stub:
        assert embedder.embed_query("hello") == vector
    assert embedder.input_tokens == 7


def test_titan_wrong_width_is_rejected() -> None:
    client, stub = runtime()
    stub.add_response(
        "invoke_model", {"body": body({"embedding": [0.6, 0.8]}), "contentType": "application/json"}
    )
    with stub, pytest.raises(UpstreamError, match="2 dimensions, expected 256"):
        titan(client).embed_query("hello")


def test_titan_zero_vector_is_rejected() -> None:
    client, stub = runtime()
    stub.add_response(
        "invoke_model",
        {"body": body({"embedding": [0.0] * 256}), "contentType": "application/json"},
    )
    with stub, pytest.raises(UpstreamError, match="zero vector"):
        titan(client).embed_query("hello")


def test_titan_throttling_is_retryable_error() -> None:
    client, stub = runtime()
    stub.add_client_error(
        "invoke_model", service_error_code="ThrottlingException", http_status_code=429
    )
    with stub, pytest.raises(UpstreamUnavailable):
        titan(client).embed_query("hello")


def test_titan_rejects_unsupported_dimension() -> None:
    with pytest.raises(ValueError, match="256, 512 or 1024"):
        TitanEmbedder(
            model_id="amazon.titan-embed-text-v2:0",
            dimensions=768,
            region=REGION,
            timeout_s=5,
            attempts=1,
            concurrency=1,
            client=object(),
        )


# --------------------------------------------------------------------------- Converse


REQUEST = ChatRequest(
    system="sys",
    prompt="prompt",
    max_tokens=100,
    temperature=0.1,
    purpose="answer",
    question="What is the limit?",
)


def test_converse_request_with_guardrail_screens_only_the_question() -> None:
    client, stub = runtime()
    stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": '{"status":"x"}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "metrics": {"latencyMs": 12},
        },
        {
            "modelId": "us.amazon.nova-2-lite-v1:0",
            "system": [{"text": "sys"}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"guardContent": {"text": {"text": "What is the limit?"}}},
                        {"text": "prompt"},
                    ],
                }
            ],
            "inferenceConfig": {"maxTokens": 100, "temperature": 0.1},
            "guardrailConfig": {
                "guardrailIdentifier": "gr-123",
                "guardrailVersion": "1",
                "trace": "disabled",
            },
        },
    )
    model = ConverseChatModel(
        model_id="us.amazon.nova-2-lite-v1:0",
        region=REGION,
        timeout_s=5,
        attempts=1,
        guardrail_id="gr-123",
        guardrail_version="1",
        client=client,
    )
    with stub:
        result = model.generate(REQUEST)
    assert result.text == '{"status":"x"}'
    assert (result.usage.input_tokens, result.usage.output_tokens) == (10, 5)


def test_converse_guardrail_intervention_raises() -> None:
    client, stub = runtime()
    stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "blocked"}]}},
            "stopReason": "guardrail_intervened",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        },
    )
    model = ConverseChatModel(
        model_id="m", region=REGION, timeout_s=5, attempts=1, guardrail_id="g", client=client
    )
    with stub, pytest.raises(GuardrailBlocked):
        model.generate(REQUEST)


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("ThrottlingException", UpstreamUnavailable),
        ("ServiceUnavailableException", UpstreamUnavailable),
        ("ValidationException", UpstreamError),
        ("AccessDeniedException", UpstreamError),
    ],
)
def test_converse_error_mapping(code: str, error: type[Exception]) -> None:
    client, stub = runtime()
    stub.add_client_error("converse", service_error_code=code, http_status_code=400)
    model = ConverseChatModel(model_id="m", region=REGION, timeout_s=5, attempts=1, client=client)
    with stub, pytest.raises(error):
        model.generate(REQUEST)


# --------------------------------------------------------------------------- Rerank, ApplyGuardrail


def test_rerank_request_shape() -> None:
    client = boto3.client("bedrock-agent-runtime", region_name=REGION)
    stub = Stubber(client)
    stub.add_response(
        "rerank",
        {"results": [{"index": 1, "relevanceScore": 0.9}, {"index": 0, "relevanceScore": 0.2}]},
        {
            "queries": [{"type": "TEXT", "textQuery": {"text": "q"}}],
            "sources": [
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": "a"}},
                },
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": "b"}},
                },
            ],
            "rerankingConfiguration": {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": 2,
                    "modelConfiguration": {
                        "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
                    },
                },
            },
        },
    )
    reranker = BedrockReranker(
        model_id="cohere.rerank-v3-5:0", region=REGION, timeout_s=5, attempts=1, client=client
    )
    with stub:
        assert reranker.rerank("q", ["a", "b"], top_n=5) == [(1, 0.9), (0, 0.2)]


def test_apply_guardrail_grounding_request_shape() -> None:
    client, stub = runtime()
    stub.add_response(
        "apply_guardrail",
        {
            "usage": {
                "topicPolicyUnits": 0,
                "contentPolicyUnits": 0,
                "wordPolicyUnits": 0,
                "sensitiveInformationPolicyUnits": 0,
                "sensitiveInformationPolicyFreeUnits": 0,
                "contextualGroundingPolicyUnits": 1,
            },
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [],
            "assessments": [],
        },
        {
            "guardrailIdentifier": "g",
            "guardrailVersion": "1",
            "source": "OUTPUT",
            "content": [
                {"text": {"text": "sources", "qualifiers": ["grounding_source"]}},
                {"text": {"text": "q", "qualifiers": ["query"]}},
                {"text": {"text": "answer", "qualifiers": ["guard_content"]}},
            ],
        },
    )
    check = GuardrailGroundingCheck(
        guardrail_id="g",
        guardrail_version="1",
        region=REGION,
        timeout_s=5,
        attempts=1,
        client=client,
    )
    with stub:
        assert check.is_grounded(sources="sources", question="q", answer="answer") is False
