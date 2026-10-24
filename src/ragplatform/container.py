"""Object wiring. Built once per process (per Lambda execution environment).

Tests build their own container with a real PostgreSQL and fake providers, or
pass replacements for any part.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ragplatform.adapters.blobstore import BlobStore, LocalBlobStore, S3BlobStore
from ragplatform.agent.runner import AgentRunner
from ragplatform.agent.web_search import WebSearchClient
from ragplatform.config import Settings, get_settings
from ragplatform.generation.generator import Generator
from ragplatform.ingestion.pipeline import IngestionPipeline
from ragplatform.observability.telemetry import Telemetry
from ragplatform.providers.factory import Providers, build_providers
from ragplatform.retrieval.retriever import Retriever
from ragplatform.services.document_service import DocumentService
from ragplatform.services.query_service import QueryService
from ragplatform.store.db import DataApiDatabase, Database, PsycopgDatabase
from ragplatform.store.repository import Repository


def build_database(settings: Settings, *, admin: bool = False, client: Any = None) -> Database:
    if settings.db_backend == "data_api":
        secret = settings.db_admin_secret_arn if admin else settings.db_secret_arn
        if not settings.db_cluster_arn or not secret:
            raise ValueError("data_api backend needs RAG_DB_CLUSTER_ARN and the secret ARN")
        return DataApiDatabase(
            cluster_arn=settings.db_cluster_arn,
            secret_arn=secret,
            database=settings.db_name,
            client=client,
            resume_max_wait_s=settings.db_resume_max_wait_s,
        )
    url = settings.admin_database_url if admin else settings.database_url
    if url is None:
        raise ValueError("psycopg backend needs RAG_DATABASE_URL (and RAG_ADMIN_DATABASE_URL)")
    return PsycopgDatabase(url.get_secret_value())


def build_blobstore(settings: Settings) -> BlobStore:
    if settings.blob_backend == "s3":
        if not settings.documents_bucket:
            raise ValueError("s3 blob backend needs RAG_DOCUMENTS_BUCKET")
        return S3BlobStore(
            bucket=settings.documents_bucket,
            region=settings.aws_region,
            url_ttl_s=settings.upload_url_ttl_s,
        )
    return LocalBlobStore(settings.local_blob_dir)


@dataclass
class Container:
    settings: Settings
    telemetry: Telemetry
    db: Database
    repo: Repository
    providers: Providers
    blobs: BlobStore
    retriever: Retriever
    generator: Generator
    agent: AgentRunner
    queries: QueryService
    documents: DocumentService
    pipeline: IngestionPipeline


def build_container(
    settings: Settings | None = None,
    *,
    db: Database | None = None,
    blobs: BlobStore | None = None,
    providers: Providers | None = None,
    telemetry: Telemetry | None = None,
) -> Container:
    settings = settings or get_settings()
    telemetry = telemetry or Telemetry(settings.metrics_namespace, settings.service_name)
    db = db or build_database(settings)
    blobs = blobs or build_blobstore(settings)
    providers = providers or build_providers(settings)
    repo = Repository()
    retriever = Retriever(
        db=db,
        repo=repo,
        embedder=providers.embedder,
        reranker=providers.reranker,
        settings=settings,
        telemetry=telemetry,
    )
    generator = Generator(
        chat=providers.chat, grounding=providers.grounding, settings=settings, telemetry=telemetry
    )
    web = None
    if settings.web_search_url:
        key = (
            settings.web_search_api_key.get_secret_value() if settings.web_search_api_key else None
        )
        web = WebSearchClient(
            settings.web_search_url, api_key=key, timeout_s=min(10.0, settings.agent_deadline_s)
        )
    agent = AgentRunner(
        chat=providers.chat,
        retriever=retriever,
        generator=generator,
        settings=settings,
        telemetry=telemetry,
        web_search=web,
    )
    return Container(
        settings=settings,
        telemetry=telemetry,
        db=db,
        repo=repo,
        providers=providers,
        blobs=blobs,
        retriever=retriever,
        generator=generator,
        agent=agent,
        queries=QueryService(
            db=db,
            repo=repo,
            retriever=retriever,
            generator=generator,
            agent=agent,
            settings=settings,
            telemetry=telemetry,
        ),
        documents=DocumentService(
            db=db, repo=repo, blobs=blobs, settings=settings, telemetry=telemetry
        ),
        pipeline=IngestionPipeline(
            db=db,
            repo=repo,
            blobs=blobs,
            embedder=providers.embedder,
            settings=settings,
            telemetry=telemetry,
        ),
    )


@lru_cache(maxsize=1)
def get_container() -> Container:
    return build_container()
