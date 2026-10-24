"""Evaluation runner: rebuild the index from the corpus, then score retrieval and answers.

    python -m eval.run_eval                                 # offline, fake providers
    python -m eval.run_eval --check eval/thresholds.json    # CI gate
    python -m eval.run_eval --provider gemini --retrieval vector --label vector-only
    python -m eval.run_eval --provider gemini --retrieval-only   # no generation calls

Retrieval and answer quality are scored separately, so a drop can be traced to
the stage that caused it. Retrieval metrics are document-level over the
sources sent to the model (after reranking, parent expansion and packing):

* hit@1     share of answerable questions whose top source is an expected document
* hit@k     share of answerable questions with at least one expected document in the sources
* recall@3  share of expected documents among the top three sources
* recall@k  share of expected documents found, averaged over questions
* MRR       mean of 1 / rank of the first expected document (0 when absent)
* nDCG@k    rank-weighted recall with binary relevance

Answer metrics:

* status accuracy    the answer's status is one the question allows
* fact accuracy      every ``must_contain`` fact appears (case-insensitive)
* citation precision share of cited documents that are expected documents
* refusal accuracy   unanswerable, ACL-denied and cross-tenant questions are refused
* leaks              forbidden documents retrieved or cited, or forbidden facts stated; must be 0
* injection obeyed   answers containing a fact planted by an injected instruction; must be 0

The runner resets the schema of the target database, so it refuses any
database whose name does not end in ``_test`` or ``_eval``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from ragplatform.adapters.blobstore import LocalBlobStore
from ragplatform.config import ChunkStrategy, RetrievalMode, Settings
from ragplatform.container import Container, build_container
from ragplatform.domain.models import AccessScope, AuthContext
from ragplatform.ingestion.parsers import EXTENSION_TYPES
from ragplatform.observability.telemetry import Telemetry
from ragplatform.services.document_service import RegisterRequest
from ragplatform.services.query_service import QueryRequest
from ragplatform.store.db import PsycopgDatabase
from ragplatform.store.migrate import (
    apply_migrations,
    ensure_app_role,
    ensure_tenant,
    record_embedding_space,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_ADMIN = "postgresql://rag_owner:rag_owner_local@localhost:55432/ragplatform_test"
EVAL_ROLE = "rag_app_eval"
EVAL_PASSWORD = "6576616c2d6f6e6c792d6170702d70617373776f7264"  # noqa: S105 - local eval DB only
TABLES = (
    "audit_events, usage_daily, answer_cache, chunks, document_versions, documents,"
    " tenants, embedding_space, schema_migrations"
)


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    retrieved_docs: list[str]
    hit: bool | None = None
    hit_at_1: bool | None = None
    recall: float | None = None
    recall_at_3: float | None = None
    reciprocal_rank: float | None = None
    ndcg: float | None = None
    status: str | None = None
    status_ok: bool | None = None
    facts_ok: bool | None = None
    cited_docs: list[str] = field(default_factory=list)
    citation_precision: float | None = None
    leak: bool = False
    injection_obeyed: bool = False
    latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    answer: str = ""
    agent_steps: int | None = None


def _app_url(admin_url: str) -> str:
    host = admin_url.split("@", 1)[1]
    return f"postgresql://{EVAL_ROLE}:{EVAL_PASSWORD}@{host}"


def reset_and_migrate(admin_url: str, dimensions: int, space: Any) -> None:
    database = admin_url.rsplit("/", 1)[-1].split("?")[0]
    if not database.endswith(("_test", "_eval")):
        raise SystemExit(
            f"refusing to reset database {database!r}: name must end in _test or _eval"
        )
    admin = PsycopgDatabase(admin_url, max_size=2)
    with admin.transaction() as tx:
        tx.execute(f"DROP TABLE IF EXISTS {TABLES} CASCADE")
        tx.execute(
            "DROP FUNCTION IF EXISTS app_groups(), app_can_read(text, text[]),"
            " text_array(jsonb) CASCADE"
        )
    ensure_app_role(admin, EVAL_ROLE, EVAL_PASSWORD, rotate=True)
    apply_migrations(admin, dimensions=dimensions, app_role=EVAL_ROLE)
    record_embedding_space(
        admin, provider=space.provider, model_id=space.model_id, dimensions=space.dimensions
    )
    ensure_tenant(
        admin,
        "northwind",
        "Northwind Robotics",
        allow_agentic=True,
        daily_query_limit=100000,
        daily_token_limit=100_000_000,
    )
    ensure_tenant(
        admin,
        "helios",
        "Helios Health",
        allow_agentic=True,
        daily_query_limit=100000,
        daily_token_limit=100_000_000,
    )
    admin.close()


def ingest_corpus(container: Container, dataset: dict[str, Any]) -> dict[UUID, str]:
    keys: dict[UUID, str] = {}
    for doc in dataset["documents"]:
        path = ROOT / "corpus" / doc["path"]
        data = path.read_bytes()
        admin = AuthContext(
            user_id=f"{doc['tenant']}-admin",
            tenant_id=doc["tenant"],
            groups=frozenset({"tenant-admin"}),
            is_tenant_admin=True,
            request_id="eval",
        )
        registered = container.documents.register(
            admin,
            RegisterRequest(
                source_name=path.name,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                acl_groups=doc["acl"],
                content_type=EXTENSION_TYPES[path.suffix],
            ),
        )
        assert registered.version is not None, f"{doc['key']} registered twice"
        document_id = registered.document.document_id
        container.documents.accept_local_upload(admin, document_id, 1, data)
        result = container.pipeline.process(doc["tenant"], document_id, 1)
        if result.outcome != "ready":
            raise SystemExit(f"ingest of {doc['key']} ended {result.outcome}: {result.error_code}")
        keys[document_id] = doc["key"]
    return keys


def _ndcg(ranked: list[str], expected: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in expected)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(expected), k)))
    return dcg / ideal if ideal else 0.0


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def evaluate(
    container: Container,
    dataset: dict[str, Any],
    doc_keys: dict[UUID, str],
    *,
    retrieval: RetrievalMode,
    mode: str,
    answers: bool,
) -> list[QuestionResult]:
    users = dataset["users"]
    results = []
    for q in dataset["questions"]:
        spec = users[q["user"]]
        ctx = AuthContext(
            user_id=q["user"],
            tenant_id=spec["tenant"],
            groups=frozenset(spec["groups"]),
            request_id=q["id"],
        )
        expected = set(q.get("expected_docs", []))
        forbidden = set(q.get("forbidden_docs", []))

        retrieved = container.retriever.retrieve(
            AccessScope.for_user(ctx), q["question"], mode=retrieval
        )
        ranked: list[str] = []
        for chunk in retrieved.chunks:
            key = doc_keys.get(chunk.document_id, "?")
            if key not in ranked:
                ranked.append(key)
        result = QuestionResult(
            id=q["id"], category=q["category"], question=q["question"], retrieved_docs=ranked
        )
        if forbidden & set(ranked):
            result.leak = True
        if expected:
            k = container.settings.final_k
            found = expected & set(ranked[:k])
            result.hit = bool(found)
            result.recall = len(found) / len(expected)
            result.hit_at_1 = bool(ranked) and ranked[0] in expected
            result.recall_at_3 = len(expected & set(ranked[:3])) / len(expected)
            first = next((i for i, d in enumerate(ranked) if d in expected), None)
            result.reciprocal_rank = 0.0 if first is None else 1 / (first + 1)
            result.ndcg = _ndcg(ranked, expected, k)

        if answers:
            started = time.perf_counter()
            response = container.queries.answer(
                ctx,
                QueryRequest(
                    question=q["question"], mode=mode, retrieval_mode=retrieval, use_cache=False
                ),
            )
            result.latency_ms = (time.perf_counter() - started) * 1000
            result.status = response.status.value
            result.answer = response.answer
            result.status_ok = result.status in q["statuses"]
            result.input_tokens = response.usage.input_tokens
            result.output_tokens = response.usage.output_tokens
            result.agent_steps = len(response.agent_trace) if response.agent_trace else None
            result.cited_docs = sorted(
                {doc_keys.get(c.document_id, "?") for c in response.citations}
            )
            text = response.answer
            must = q.get("must_contain", [])
            any_of = q.get("must_contain_any", [])
            if must or any_of:
                result.facts_ok = all(_contains(text, m) for m in must) and (
                    not any_of or any(_contains(text, m) for m in any_of)
                )
            if result.cited_docs and expected:
                result.citation_precision = len(set(result.cited_docs) & expected) / len(
                    result.cited_docs
                )
            if forbidden & set(result.cited_docs):
                result.leak = True
            if any(_contains(text, m) for m in q.get("must_not_contain", [])):
                if q["category"] == "injection":
                    result.injection_obeyed = True
                elif q["category"] in {"acl_denied", "cross_tenant"}:
                    result.leak = True
                else:
                    result.facts_ok = False
        results.append(result)
        print(
            f"  {q['id']} {q['category']:<14} docs={','.join(ranked[:4]):<40} "
            f"status={result.status} facts={result.facts_ok}",
            flush=True,
        )
    return results


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarise(results: list[QuestionResult]) -> dict[str, Any]:
    answerable = [r for r in results if r.hit is not None]
    refusals = [
        r for r in results if r.category in {"unanswerable", "acl_denied"} and r.status is not None
    ]
    answered = [r for r in results if r.status is not None]
    latencies = sorted(r.latency_ms for r in answered if r.latency_ms is not None)
    summary: dict[str, Any] = {
        "questions": len(results),
        "retrieval": {
            "hit_at_1": _mean([float(bool(r.hit_at_1)) for r in answerable]),
            "hit_at_k": _mean([float(r.hit) for r in answerable]),
            "recall_at_3": _mean([r.recall_at_3 for r in answerable if r.recall_at_3 is not None]),
            "recall_at_k": _mean([r.recall for r in answerable if r.recall is not None]),
            "mrr": _mean([r.reciprocal_rank for r in answerable if r.reciprocal_rank is not None]),
            "ndcg_at_k": _mean([r.ndcg for r in answerable if r.ndcg is not None]),
        },
        "leaks": sum(r.leak for r in results),
        "injection_obeyed": sum(r.injection_obeyed for r in results),
    }
    if answered:
        summary["answers"] = {
            "status_accuracy": _mean([float(bool(r.status_ok)) for r in answered]),
            "fact_accuracy": _mean(
                [float(bool(r.facts_ok)) for r in answered if r.facts_ok is not None]
            ),
            "citation_precision": _mean(
                [r.citation_precision for r in answered if r.citation_precision is not None]
            ),
            "refusal_accuracy": _mean(
                [float(r.status == "insufficient_context") for r in refusals]
            ),
            "latency_ms_p50": round(latencies[len(latencies) // 2], 1) if latencies else None,
            "latency_ms_p95": round(latencies[int(len(latencies) * 0.95) - 1], 1)
            if latencies
            else None,
            "input_tokens_total": sum(r.input_tokens for r in answered),
            "output_tokens_total": sum(r.output_tokens for r in answered),
        }
    by_category: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)
    summary["by_category"] = {
        name: {
            "n": len(items),
            "hit_at_k": _mean([float(r.hit) for r in items if r.hit is not None]),
            "fact_accuracy": _mean(
                [float(bool(r.facts_ok)) for r in items if r.facts_ok is not None]
            ),
            "status_accuracy": _mean(
                [float(bool(r.status_ok)) for r in items if r.status_ok is not None]
            ),
        }
        for name, items in sorted(by_category.items())
    }
    return summary


def check(summary: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    failures = []
    for dotted, rule in thresholds["gates"].items():
        node: Any = summary
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            failures.append(f"{dotted}: missing from the report")
        elif "min" in rule and node < rule["min"]:
            failures.append(f"{dotted} = {node} is below {rule['min']}")
        elif "max" in rule and node > rule["max"]:
            failures.append(f"{dotted} = {node} is above {rule['max']}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=["fake", "gemini", "bedrock"], default="fake")
    parser.add_argument("--retrieval", choices=[m.value for m in RetrievalMode], default="hybrid")
    parser.add_argument("--mode", choices=["standard", "agentic"], default="standard")
    parser.add_argument(
        "--chunk-strategy", choices=[s.value for s in ChunkStrategy], default="parent_child"
    )
    parser.add_argument("--no-contextual-headers", action="store_true")
    parser.add_argument("--rerank", help="rerank model id, e.g. cohere.rerank-v3-5:0")
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer generation")
    parser.add_argument(
        "--admin-url", default=os.environ.get("RAG_EVAL_ADMIN_DATABASE_URL", DEFAULT_ADMIN)
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--check", type=Path, default=None, help="thresholds file; exit 1 on failure"
    )
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {
        "environment": "local",
        "auth_mode": "local",
        "local_auth_secret": "eval-only-secret-not-used-for-auth-here",
        "provider": args.provider,
        "db_backend": "psycopg",
        "database_url": _app_url(args.admin_url),
        "admin_database_url": args.admin_url,
        "app_db_role": EVAL_ROLE,
        "blob_backend": "local",
        "ingest_mode": "inline",
        "chunk_strategy": args.chunk_strategy,
        "contextual_headers": not args.no_contextual_headers,
        "rerank_model_id": args.rerank,
        "embedding_dimensions": 1024,
    }
    if args.provider == "gemini":
        overrides.setdefault("embedding_model_id", "gemini-embedding-001")
        overrides["embedding_model_id"] = os.environ.get(
            "RAG_EVAL_EMBEDDING_MODEL_ID", "gemini-embedding-001"
        )
        overrides["chat_model_id"] = os.environ.get("RAG_EVAL_CHAT_MODEL_ID", "gemini-flash-latest")
    settings = Settings(
        _env_file=".env" if args.provider == "gemini" else None,  # type: ignore[call-arg]
        **overrides,
    )
    dataset = json.loads((ROOT / "dataset.json").read_text("utf-8"))

    from ragplatform.providers.factory import build_providers

    providers = build_providers(settings)
    print(f"resetting and migrating {args.admin_url.rsplit('@', 1)[-1]}", flush=True)
    reset_and_migrate(args.admin_url, settings.embedding_dimensions, providers.embedder.space)
    blob_dir = Path(".data/eval-blobs") / datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    container = build_container(
        settings,
        providers=providers,
        blobs=LocalBlobStore(blob_dir),
        telemetry=Telemetry("Eval", "eval", emit=False),
    )

    started = time.perf_counter()
    doc_keys = ingest_corpus(container, dataset)
    ingest_s = time.perf_counter() - started
    print(f"ingested {len(doc_keys)} documents in {ingest_s:.1f} s", flush=True)

    results = evaluate(
        container,
        dataset,
        doc_keys,
        retrieval=RetrievalMode(args.retrieval),
        mode=args.mode,
        answers=not args.retrieval_only,
    )
    summary = summarise(results)
    report = {
        "label": args.label or f"{args.provider}-{args.retrieval}-{args.mode}",
        "dataset_version": dataset["version"],
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "provider": args.provider,
            "embedding_model": settings.embedding_model_id
            if args.provider != "fake"
            else "hashing-v1",
            "chat_model": settings.chat_model_id
            if args.provider != "fake"
            else "fake-extractive-v1",
            "retrieval": args.retrieval,
            "mode": args.mode,
            "chunk_strategy": args.chunk_strategy,
            "contextual_headers": not args.no_contextual_headers,
            "rerank": args.rerank,
            "final_k": settings.final_k,
            "chunk_tokens": settings.chunk_tokens,
            "parent_chunk_tokens": settings.parent_chunk_tokens,
        },
        "ingest_seconds": round(ingest_s, 1),
        "summary": summary,
        "results": [r.__dict__ for r in results],
    }
    output = args.output or ROOT / "reports" / f"{report['label']}.local.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), "utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report written to {output}")

    if args.check:
        failures = check(summary, json.loads(args.check.read_text("utf-8")))
        for failure in failures:
            print(f"GATE FAILED: {failure}", file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
