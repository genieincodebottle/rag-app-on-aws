"""Metrics, logging and tracing in one place.

In Lambda, metrics go out as CloudWatch Embedded Metric Format through
Powertools (one JSON log line per flush, no PutMetricData calls), logs are
structured JSON with the request id as correlation id, and X-Ray traces are on.
Locally the same calls record metrics in memory, so tests and the evaluation
runner can read them.

Prompts, document text and answers are never logged unless
``RAG_LOG_PROMPTS=true``, which config validation refuses in prod. Questions
are logged as a SHA-256 prefix so repeated questions can be correlated without
storing their text.
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit, single_metric

from ragplatform.config import running_in_lambda

# Lambda keeps the Powertools default (stdout, which CloudWatch captures). Locally,
# logs go to stderr so the CLI's JSON on stdout can be piped into jq.
logger = Logger(service="ragplatform", stream=None if running_in_lambda() else sys.stderr)
tracer = Tracer(service="ragplatform", disabled=not running_in_lambda())

UNITS = {
    "count": MetricUnit.Count,
    "ms": MetricUnit.Milliseconds,
    "none": MetricUnit.NoUnit,
    "bytes": MetricUnit.Bytes,
}


@dataclass(frozen=True)
class Recorded:
    name: str
    value: float
    unit: str
    dimensions: tuple[tuple[str, str], ...]


class Telemetry:
    def __init__(self, namespace: str, service: str, *, emit: bool | None = None) -> None:
        self.namespace = namespace
        self.service = service
        self.emit = running_in_lambda() if emit is None else emit
        self.metrics = Metrics(namespace=namespace, service=service) if self.emit else None
        # Bounded, so a long-running local process cannot grow it without limit.
        self.recorded: deque[Recorded] = deque(maxlen=5000)

    def add(self, name: str, value: float, unit: str = "count", **dimensions: str) -> None:
        self.recorded.append(Recorded(name, value, unit, tuple(sorted(dimensions.items()))))
        if self.metrics is None:
            return
        if dimensions:
            with single_metric(
                name=name,
                unit=UNITS[unit],
                value=value,
                namespace=self.namespace,
                default_dimensions=self.metrics.default_dimensions,
            ) as metric:
                for key, dim_value in dimensions.items():
                    metric.add_dimension(name=key, value=dim_value)
        else:
            self.metrics.add_metric(name=name, unit=UNITS[unit], value=value)

    @contextmanager
    def timer(self, name: str, **dimensions: str) -> Iterator[dict[str, float]]:
        holder: dict[str, float] = {}
        started = time.perf_counter()
        try:
            yield holder
        finally:
            holder["ms"] = (time.perf_counter() - started) * 1000
            self.add(name, holder["ms"], "ms", **dimensions)

    def values(self, name: str) -> list[float]:
        return [r.value for r in self.recorded if r.name == name]


def question_fingerprint(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]


def safe_text(text: str, *, allowed: bool, limit: int = 500) -> dict[str, Any]:
    """Log-safe representation of user or document text."""
    if allowed:
        return {"text": text[:limit]}
    return {"sha256_16": question_fingerprint(text), "chars": len(text)}
