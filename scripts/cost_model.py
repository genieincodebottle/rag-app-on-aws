"""Monthly cost model for the platform, by scenario and line item.

    uv run python scripts/cost_model.py              # markdown tables
    uv run python scripts/cost_model.py --json       # machine-readable

Unit prices are us-east-1 on-demand, read from the AWS Price List API on
2026-10-24 unless marked otherwise. Per-request quantities come from the code
(how many Data API calls a query makes) or from measurement (tokens per query
in the live evaluation). Everything else is an assumption, listed in
ASSUMPTIONS, and the output is only as good as those. Compare with Cost
Explorer after a month of real traffic and correct the assumptions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

# ------------------------------------------------------------------ unit prices (USD)
PRICES = {
    "aurora_acu_hour": (0.12, "Price List API, Aurora Serverless v2 Standard"),
    "aurora_storage_gb_month": (0.10, "Price List API"),
    "aurora_io_per_million": (0.20, "Price List API"),
    "data_api_per_million": (0.35, "Price List API; each extra 32 KB counts as a request"),
    "lambda_gb_second_arm": (0.0000133334, "Price List API, arm64 first tier"),
    "lambda_requests_per_million": (0.20, "Price List API"),
    "apigw_rest_per_million": (3.50, "Price List API, first 333M"),
    "sqs_per_million": (0.40, "Price List API"),
    "kms_key_month": (1.00, "Price List API"),
    "kms_per_10k": (0.03, "Price List API"),
    "secret_month": (0.40, "Price List API"),
    "secrets_per_10k": (0.05, "Price List API"),
    "logs_ingest_gb": (0.50, "Price List API"),
    "logs_storage_gb_month": (0.03, "Price List API"),
    "custom_metric_month": (0.30, "Price List API, first 10,000 metrics"),
    "alarm_metric_month": (0.10, "Price List API, standard resolution"),
    "dashboard_month": (
        3.00,
        "Price List API; first 3 dashboards per account are free (unverified)",
    ),
    "waf_acl_month": (5.00, "AWS WAF pricing page example"),
    "waf_rule_month": (1.00, "Price List API"),
    "waf_per_million": (0.60, "Price List API"),
    "s3_storage_gb_month": (0.023, "Price List API"),
    "s3_put_per_1k": (0.005, "Price List API"),
    "s3_get_per_1k": (0.0004, "Price List API"),
    "xray_per_million_traces": (5.00, "Price List API; a free tier exists (amount unverified)"),
    "cognito_essentials_mau": (0.015, "Price List API; first 10,000 MAU free"),
    "titan_embed_per_m_tokens": (0.02, "Price List API"),
    "nova2lite_in_per_m": (0.33, "Price List API, regional rate"),
    "nova2lite_out_per_m": (2.75, "Price List API, regional rate"),
    "rerank_per_1k": (2.00, "Price List API, per 1,000 search units"),
    "guardrail_per_1k_text_units": (
        0.15 + 0.10,
        "Bedrock pricing page: content filter + sensitive info",
    ),
}


def p(name: str) -> float:
    return PRICES[name][0]


# ------------------------------------------------------------------ per-request quantities
PER_QUERY = {
    # Measured in the live evaluation (Gemini Flash, 45 questions): 31,464 input
    # and 17,810 output tokens. Gemini counts reasoning tokens as output; Nova's
    # answers are expected to be shorter, so this is conservative.
    "input_tokens": 700,
    "output_tokens": 400,
    "query_embed_tokens": 20,
    # From the code: three transactions (budget and cache, search, usage and
    # audit), each begin + scope + work + commit through the Data API.
    "data_api_calls": 18,
    "api_lambda_seconds": 4.0,  # most of it waiting for the model; measured p50 3.8 s
    "api_lambda_gb": 1.0,
    "log_bytes": 3_000,
}
PER_DOCUMENT = {
    "chunks": 40,  # about 30 pages of prose with the default chunk sizes
    "embed_tokens_per_chunk": 240,  # chunk plus contextual header
    "ingest_lambda_seconds": 15.0,
    "ingest_lambda_gb": 2.0,
    "data_api_calls": 12,
    "storage_mb": 1.5,  # file plus chunk rows and vectors
}
CUSTOM_METRICS = 40  # distinct metric name and dimension sets the code emits
ALARM_METRICS = 17  # 14 alarms, one of them a metric-math alarm over 3 metrics


@dataclass
class Scenario:
    name: str
    description: str
    queries_per_month: int
    documents_per_month: int
    stored_gb: float
    monthly_active_users: int
    aurora_acu_hours: float
    waf: bool
    guardrail: bool
    rerank: bool
    cache_hit_rate: float


SCENARIOS = [
    Scenario(
        "idle-dev",
        "dev stack, no traffic, Aurora paused all month",
        0,
        0,
        0.2,
        2,
        0,
        False,
        False,
        False,
        0.0,
    ),
    Scenario(
        "pilot",
        "dev settings, 50 users, 2,000 questions and 200 documents a month",
        2_000,
        200,
        1.0,
        50,
        4 * 22 * 0.5,
        False,
        False,
        False,
        0.10,
    ),
    Scenario(
        "department",
        "prod settings, 500 users, 30,000 questions and 2,000 documents a month",
        30_000,
        2_000,
        10.0,
        500,
        730 * 0.5 + 10 * 22 * 1.0,
        True,
        True,
        False,
        0.15,
    ),
    Scenario(
        "organisation",
        "prod settings with reranking, 5,000 users, 300,000 questions a month",
        300_000,
        10_000,
        60.0,
        5_000,
        730 * 2.0,
        True,
        True,
        True,
        0.20,
    ),
]
ASSUMPTIONS = [
    "us-east-1, on-demand prices, no Savings Plans or free tier except where noted",
    "Generation with Nova 2 Lite through the us. geo inference profile, billed at the regional rate",
    "Cached answers skip embedding and generation but still pay API Gateway, Lambda and Data API",
    "Aurora ACU-hours: pilot 4 active hours on 22 working days at 0.5 ACU; department 0.5 ACU "
    "always plus 1 ACU for 10 hours on 22 days; organisation 2 ACU average",
    "Guardrail evaluates the question only (about 1 text unit per query)",
    "Custom metrics are billed per hour they are published, so an idle stack pays almost nothing for them",
]


def monthly_cost(s: Scenario) -> dict[str, float]:
    q = s.queries_per_month
    uncached = q * (1 - s.cache_hit_rate)
    d = s.documents_per_month
    items: dict[str, float] = {}

    items["Bedrock generation"] = (
        uncached
        * (
            PER_QUERY["input_tokens"] * p("nova2lite_in_per_m")
            + PER_QUERY["output_tokens"] * p("nova2lite_out_per_m")
        )
        / 1e6
    )
    items["Bedrock embeddings"] = (
        (
            uncached * PER_QUERY["query_embed_tokens"]
            + d * PER_DOCUMENT["chunks"] * PER_DOCUMENT["embed_tokens_per_chunk"]
        )
        * p("titan_embed_per_m_tokens")
        / 1e6
    )
    if s.rerank:
        items["Bedrock rerank"] = uncached / 1000 * p("rerank_per_1k")
    if s.guardrail:
        items["Bedrock Guardrails"] = q / 1000 * p("guardrail_per_1k_text_units")

    items["Aurora compute"] = s.aurora_acu_hours * p("aurora_acu_hour")
    items["Aurora storage and I/O"] = s.stored_gb * p("aurora_storage_gb_month") + (
        q * 40 + d * 200
    ) / 1e6 * p("aurora_io_per_million")
    items["RDS Data API"] = (
        (q * PER_QUERY["data_api_calls"] + d * PER_DOCUMENT["data_api_calls"])
        / 1e6
        * p("data_api_per_million")
    )

    gb_seconds = (
        q * PER_QUERY["api_lambda_seconds"] * PER_QUERY["api_lambda_gb"]
        + d * PER_DOCUMENT["ingest_lambda_seconds"] * PER_DOCUMENT["ingest_lambda_gb"]
    )
    items["Lambda"] = gb_seconds * p("lambda_gb_second_arm") + (q + d) / 1e6 * p(
        "lambda_requests_per_million"
    )
    items["API Gateway"] = (
        (q * 1.2) / 1e6 * p("apigw_rest_per_million")
    )  # +20% list and upload calls
    if s.waf:
        items["WAF"] = (
            p("waf_acl_month") + 3 * p("waf_rule_month") + q * 1.2 / 1e6 * p("waf_per_million")
        )
    items["S3 and SQS"] = (
        s.stored_gb * p("s3_storage_gb_month")
        + d / 1000 * p("s3_put_per_1k")
        + d / 1000 * p("s3_get_per_1k")
        + d * 3 / 1e6 * p("sqs_per_million")
    )
    items["KMS and Secrets Manager"] = (
        p("kms_key_month")
        + 2 * p("secret_month")
        + (q * 2 + d * 4) / 1e4 * p("kms_per_10k")
        + (q * 0.05) / 1e4 * p("secrets_per_10k")
    )
    log_gb = q * PER_QUERY["log_bytes"] / 1e9
    active_share = 0.0 if q == 0 else 1.0
    items["CloudWatch"] = (
        log_gb * p("logs_ingest_gb")
        + log_gb * 3 * p("logs_storage_gb_month")
        + CUSTOM_METRICS * p("custom_metric_month") * active_share
        + ALARM_METRICS * p("alarm_metric_month")
    )
    items["X-Ray"] = max(0.0, q - 100_000) / 1e6 * p("xray_per_million_traces")
    items["Cognito"] = max(0, s.monthly_active_users - 10_000) * p("cognito_essentials_mau")
    return {k: round(v, 2) for k, v in items.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = {s.name: monthly_cost(s) for s in SCENARIOS}
    if args.json:
        print(
            json.dumps(
                {
                    "prices": PRICES,
                    "per_query": PER_QUERY,
                    "per_document": PER_DOCUMENT,
                    "assumptions": ASSUMPTIONS,
                    "scenarios": [asdict(s) for s in SCENARIOS],
                    "monthly_usd": results,
                },
                indent=2,
            )
        )
        return 0
    lines = sorted({k for r in results.values() for k in r})
    header = "| Line item | " + " | ".join(s.name for s in SCENARIOS) + " |"
    print(header)
    print("|---|" + "---:|" * len(SCENARIOS))
    for line in lines:
        cells = [f"{results[s.name].get(line, 0.0):,.2f}" for s in SCENARIOS]
        print(f"| {line} | " + " | ".join(cells) + " |")
    totals = [f"**{sum(results[s.name].values()):,.2f}**" for s in SCENARIOS]
    print("| **Total per month (USD)** | " + " | ".join(totals) + " |")
    per_query = [
        f"{sum(results[s.name].values()) / s.queries_per_month:.4f}"
        if s.queries_per_month
        else "n/a"
        for s in SCENARIOS
    ]
    print("| Cost per question (USD) | " + " | ".join(per_query) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
