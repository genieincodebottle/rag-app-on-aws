"""Load test for the API.

Locally, with the fake providers, this measures the platform's own cost per
request: authentication, budget check, hybrid SQL under row-level security,
citation validation and usage writes. Model latency is not included; the live
evaluation reports it separately. Against a deployed stage it measures the
whole path, including Bedrock.

    # terminal 1
    uv run ragplatform serve
    # terminal 2
    uv run locust -f tests/load/locustfile.py --headless -u 20 -r 5 -t 60s \
        --host http://127.0.0.1:8000 --csv build/load

Environment:
    LOAD_AUTH_HEADER full Authorization header value (for a deployed stage, the raw Cognito ID token)
    RAG_LOCAL_AUTH_SECRET  local mode: mint tokens for the northwind tenant instead
    LOAD_USE_CACHE   "true" to allow cached answers (default false: measure the full path)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [
    q["question"]
    for q in json.loads(
        (Path(__file__).resolve().parents[2] / "eval" / "dataset.json").read_text("utf-8")
    )["questions"]
    if q["user"] == "nw-staff"
]
USE_CACHE = os.environ.get("LOAD_USE_CACHE", "false").lower() == "true"


def _auth_header(index: int) -> str:
    if os.environ.get("LOAD_AUTH_HEADER"):
        return os.environ["LOAD_AUTH_HEADER"]
    from dotenv import dotenv_values

    secret = os.environ.get("RAG_LOCAL_AUTH_SECRET") or dotenv_values(".env").get(
        "RAG_LOCAL_AUTH_SECRET"
    )
    if not secret:
        raise RuntimeError("set LOAD_AUTH_HEADER or RAG_LOCAL_AUTH_SECRET")
    from ragplatform.security.auth import issue_local_token

    return "Bearer " + issue_local_token(
        secret, user_id=f"load-user-{index}", tenant_id="northwind", groups=["staff"]
    )


class ApiUser(HttpUser):
    wait_time = between(0.5, 1.5)
    counter = 0

    def on_start(self) -> None:
        ApiUser.counter += 1
        self.headers = {"Authorization": _auth_header(ApiUser.counter)}

    @task(8)
    def ask(self) -> None:
        with self.client.post(
            "/v1/query",
            headers=self.headers,
            name="POST /v1/query",
            json={"question": random.choice(QUESTIONS), "use_cache": USE_CACHE},
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
            elif response.json()["status"] not in {
                "answered",
                "insufficient_context",
                "conflicting_sources",
            }:
                response.failure("unexpected status")

    @task(1)
    def list_documents(self) -> None:
        self.client.get("/v1/documents?limit=20", headers=self.headers, name="GET /v1/documents")

    @task(1)
    def health(self) -> None:
        self.client.get("/v1/health", name="GET /v1/health")
