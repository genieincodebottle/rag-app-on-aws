"""Lambda entry point for ingestion. Triggered by SQS, which S3 notifies on upload.

Each SQS message carries one S3 event notification. Messages are processed
independently and failures are reported per message (``ReportBatchItemFailures``),
so one bad message does not make SQS redeliver the whole batch.

* Permanent failures (bad file) are recorded on the document version and the
  message is acknowledged.
* Transient failures raise, the message is reported as failed, SQS redelivers
  it after the visibility timeout, and after ``maxReceiveCount`` it moves to the DLQ.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote_plus

from aws_lambda_powertools.utilities.batch import (
    BatchProcessor,
    EventType,
    process_partial_response,
)
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext

from ragplatform.adapters.blobstore import parse_storage_key
from ragplatform.container import get_container
from ragplatform.observability.telemetry import logger, tracer

_processor = BatchProcessor(event_type=EventType.SQS)


def s3_keys_from_message(body: str) -> list[tuple[str, str]]:
    """Extract (bucket, key) pairs from an S3 notification delivered through SQS."""
    payload = json.loads(body)
    if payload.get("Event") == "s3:TestEvent":
        return []
    pairs = []
    for record in payload.get("Records", []):
        if not str(record.get("eventName", "")).startswith("ObjectCreated:"):
            continue
        s3 = record["s3"]
        # S3 URL-encodes keys in notifications. Document keys contain only
        # [a-z0-9-/v], so decoding is a no-op for them, but it is done anyway.
        pairs.append((s3["bucket"]["name"], unquote_plus(s3["object"]["key"])))
    return pairs


def process_record(record: SQSRecord) -> None:
    for bucket, key in s3_keys_from_message(record.body):
        if bucket != get_container().settings.documents_bucket:
            logger.warning("event for an unexpected bucket ignored", extra={"bucket": bucket})
            continue
        try:
            tenant_id, document_id, version = parse_storage_key(key)
        except ValueError:
            logger.warning("object key is not a document key; ignored", extra={"key": key[:200]})
            continue
        logger.append_keys(document_id=str(document_id), tenant_id=tenant_id)
        result = get_container().pipeline.process(tenant_id, document_id, version)
        logger.info("ingest message handled", extra={"outcome": result.outcome})


@logger.inject_lambda_context(clear_state=True)
@tracer.capture_lambda_handler(capture_response=False)
def handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    def run(e: dict[str, Any], c: LambdaContext) -> dict[str, Any]:
        return dict(
            process_partial_response(
                event=e, record_handler=process_record, processor=_processor, context=c
            )
        )

    metrics = get_container().telemetry.metrics
    if metrics is None:
        return run(event, context)
    result: dict[str, Any] = metrics.log_metrics(run)(event, context)
    return result
