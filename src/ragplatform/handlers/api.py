"""Lambda entry point for the HTTP API (API Gateway REST, Lambda proxy integration)."""

from __future__ import annotations

from typing import Any, cast

from aws_lambda_powertools.utilities.typing import LambdaContext
from mangum import Mangum

from ragplatform.api.app import create_app
from ragplatform.container import get_container
from ragplatform.observability.telemetry import logger, tracer

# Built at cold start and reused by every invocation in this execution
# environment: clients, connection settings and the embedding-space check.
_container = get_container()
_asgi = Mangum(create_app(_container), lifespan="off", api_gateway_base_path="/")


@logger.inject_lambda_context(correlation_id_path="requestContext.requestId", clear_state=True)
@tracer.capture_lambda_handler(capture_response=False)
def handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    if _container.telemetry.metrics is None:
        response: dict[str, Any] = _asgi(event, cast(Any, context))
        return response
    # log_metrics flushes the EMF buffer once per invocation.
    wrapped = _container.telemetry.metrics.log_metrics(lambda e, c: _asgi(e, cast(Any, c)))
    response = wrapped(event, context)
    return response
