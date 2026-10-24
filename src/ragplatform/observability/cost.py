"""Per-request cost estimates from token counts.

Prices are USD per million tokens (rerank: per 1,000 queries) for us-east-1.
``verified`` records whether the number was read on the official AWS pricing
page when it was last checked; unverified prices come from secondary sources
and must be confirmed before they are used for budgeting. Override or add
models with ``RAG_PRICE_OVERRIDES`` (JSON: {"model-id": [input, output]}).

Cost estimates are for dashboards and budgets. The AWS bill is the record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ragplatform.domain.models import TokenUsage


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float
    verified: bool
    source: str


CHECKED_ON = "2026-10-24"

PRICES: dict[str, Price] = {
    # AWS Price List API (AmazonBedrock, AmazonBedrockFoundationModels), us-east-1.
    "amazon.titan-embed-text-v2:0": Price(0.02, 0.0, True, "AWS Price List API"),
    "amazon.nova-micro-v1:0": Price(0.035, 0.14, True, "AWS Price List API"),
    "us.amazon.nova-micro-v1:0": Price(0.035, 0.14, True, "AWS Price List API"),
    # Geo (us.) profiles are assumed to bill at the regional rate; the global
    # profile has its own, lower rate in the price list.
    "us.amazon.nova-2-lite-v1:0": Price(0.33, 2.75, True, "AWS Price List API, regional rate"),
    "global.amazon.nova-2-lite-v1:0": Price(0.30, 2.50, True, "AWS Price List API, global rate"),
    "us.anthropic.claude-haiku-5-5": Price(0.11, 0.55, True, "AWS Price List API, regional rate"),
    "global.anthropic.claude-haiku-5-5": Price(0.10, 0.50, True, "AWS Price List API, global rate"),
    # Local and test providers cost nothing to the AWS account.
    "hashing-v1": Price(0.0, 0.0, True, "offline"),
    "fake-extractive-v1": Price(0.0, 0.0, True, "offline"),
}

RERANK_PER_1K_QUERIES: dict[str, Price] = {
    # Billed per search unit (one query with up to 100 documents).
    "cohere.rerank-v3-5:0": Price(2.00, 0.0, True, "AWS Price List API, per 1,000 search units"),
}


def _overrides() -> dict[str, Price]:
    raw = os.environ.get("RAG_PRICE_OVERRIDES")
    if not raw:
        return {}
    parsed = json.loads(raw)
    return {k: Price(float(v[0]), float(v[1]), True, "override") for k, v in parsed.items()}


def chat_cost(model_id: str, usage: TokenUsage) -> float | None:
    price = _overrides().get(model_id) or PRICES.get(model_id)
    if price is None:
        return None
    return (usage.input_tokens * price.input_per_m + usage.output_tokens * price.output_per_m) / 1e6


def embedding_cost(model_id: str, tokens: int) -> float | None:
    price = _overrides().get(model_id) or PRICES.get(model_id)
    return None if price is None else tokens * price.input_per_m / 1e6


def rerank_cost(model_id: str, queries: int) -> float | None:
    price = RERANK_PER_1K_QUERIES.get(model_id)
    return None if price is None else queries * price.input_per_m / 1000
