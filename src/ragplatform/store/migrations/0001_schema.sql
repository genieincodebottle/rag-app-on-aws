-- Core schema. Rendered by ragplatform.store.migrate before it runs:
--   {{EMBEDDING_DIMENSIONS}}  vector width, from RAG_EMBEDDING_DIMENSIONS
--   {{APP_ROLE}}              runtime role (non-owner, so row-level security applies)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- One embedding space per deployment. Queries refuse to run when the configured
-- model or dimension differs from what the index was built with, because
-- vectors from two models are not comparable and the failure is otherwise silent.
CREATE TABLE embedding_space (
    id          smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    provider    text        NOT NULL,
    model_id    text        NOT NULL,
    dimensions  integer     NOT NULL CHECK (dimensions > 0),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE tenants (
    tenant_id          text PRIMARY KEY CHECK (tenant_id ~ '^[a-z0-9][a-z0-9-]{1,62}$'),
    name               text        NOT NULL,
    corpus_version     bigint      NOT NULL DEFAULT 0,
    daily_query_limit  integer     NOT NULL DEFAULT 2000 CHECK (daily_query_limit >= 0),
    daily_token_limit  bigint      NOT NULL DEFAULT 5000000 CHECK (daily_token_limit >= 0),
    allow_agentic      boolean     NOT NULL DEFAULT false,
    allow_web_search   boolean     NOT NULL DEFAULT false,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE documents (
    document_id      uuid PRIMARY KEY,
    tenant_id        text        NOT NULL REFERENCES tenants (tenant_id),
    owner_id         text        NOT NULL,
    title            text        NOT NULL CHECK (length(title) BETWEEN 1 AND 300),
    source_name      text        NOT NULL CHECK (length(source_name) BETWEEN 1 AND 255),
    content_type     text        NOT NULL,
    acl_groups       text[]      NOT NULL DEFAULT '{}',
    tags             text[]      NOT NULL DEFAULT '{}',
    status           text        NOT NULL CHECK (status IN
                         ('pending_upload', 'processing', 'ready', 'failed', 'deleted')),
    current_version  integer,
    latest_version   integer     NOT NULL DEFAULT 1,
    metadata         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    deleted_at       timestamptz
);
CREATE INDEX documents_tenant_status_idx ON documents (tenant_id, status, updated_at DESC);

CREATE TABLE document_versions (
    document_id   uuid        NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
    version       integer     NOT NULL CHECK (version > 0),
    tenant_id     text        NOT NULL,
    owner_id      text        NOT NULL,
    sha256       text        NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes    bigint      NOT NULL CHECK (size_bytes > 0),
    storage_key   text        NOT NULL UNIQUE,
    status        text        NOT NULL CHECK (status IN
                      ('pending_upload', 'uploaded', 'processing', 'ready', 'failed',
                       'superseded', 'deleted')),
    chunk_count   integer     NOT NULL DEFAULT 0,
    attempts      integer     NOT NULL DEFAULT 0,
    error_code    text,
    error_detail  text,
    claimed_at    timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now(),
    processed_at  timestamptz,
    PRIMARY KEY (document_id, version)
);
-- The same bytes uploaded twice by one user resolve to the existing document.
-- Deduplication is per owner, not per tenant: a tenant-wide check would tell a
-- user that a file exists even when its ACL hides it from them. Two owners
-- uploading the same file still share embeddings through the chunk hash.
CREATE UNIQUE INDEX document_versions_owner_sha_live_idx
    ON document_versions (tenant_id, owner_id, sha256)
    WHERE status NOT IN ('failed', 'superseded', 'deleted');

CREATE TABLE chunks (
    chunk_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id        text        NOT NULL,
    document_id      uuid        NOT NULL,
    version          integer     NOT NULL,
    chunk_index      integer     NOT NULL,
    kind             text        NOT NULL CHECK (kind IN ('flat', 'child', 'parent')),
    parent_chunk_id  uuid        REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    section_path     text        NOT NULL DEFAULT '',
    page_start       integer,
    page_end         integer,
    text             text        NOT NULL,
    content_hash     text        NOT NULL,
    token_count      integer     NOT NULL,
    embedding        vector({{EMBEDDING_DIMENSIONS}}),
    risk_flags       text[]      NOT NULL DEFAULT '{}',
    -- Copied from the document so row-level security and filters are evaluated
    -- on the chunk row itself, without a join inside every vector scan.
    owner_id         text        NOT NULL,
    acl_groups       text[]      NOT NULL DEFAULT '{}',
    tags             text[]      NOT NULL DEFAULT '{}',
    content_type     text        NOT NULL,
    is_active        boolean     NOT NULL DEFAULT false,
    created_at       timestamptz NOT NULL DEFAULT now(),
    tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(section_path, '') || ' ' || text)) STORED,
    FOREIGN KEY (document_id, version)
        REFERENCES document_versions (document_id, version) ON DELETE CASCADE,
    UNIQUE (document_id, version, chunk_index),
    CONSTRAINT chunks_parent_has_no_vector CHECK ((kind = 'parent') = (embedding IS NULL))
);
-- HNSW builds incrementally, so unlike ivfflat it does not need data at
-- creation time to choose good partitions.
CREATE INDEX chunks_embedding_hnsw_idx ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_tenant_active_idx ON chunks (tenant_id, is_active);
CREATE INDEX chunks_document_idx ON chunks (document_id, version);
CREATE INDEX chunks_hash_idx ON chunks (tenant_id, content_hash) WHERE embedding IS NOT NULL;

CREATE TABLE answer_cache (
    tenant_id       text        NOT NULL,
    cache_key       text        NOT NULL,
    corpus_version  bigint      NOT NULL,
    response        jsonb       NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, cache_key)
);

CREATE TABLE usage_daily (
    tenant_id      text          NOT NULL,
    day            date          NOT NULL,
    queries        integer       NOT NULL DEFAULT 0,
    input_tokens   bigint        NOT NULL DEFAULT 0,
    output_tokens  bigint        NOT NULL DEFAULT 0,
    cost_usd       numeric(14, 6) NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant_id, day)
);

-- Append-only. The runtime role gets INSERT and SELECT, never UPDATE or DELETE.
CREATE TABLE audit_events (
    event_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id      text        NOT NULL,
    actor_id       text        NOT NULL,
    action         text        NOT NULL,
    resource_type  text        NOT NULL,
    resource_id    text,
    request_id     text,
    detail         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX audit_events_tenant_time_idx ON audit_events (tenant_id, created_at DESC);

-- --------------------------------------------------------------------------
-- Row-level security
--
-- Policies bind every role except the table owner. The runtime role is never
-- the owner, so every query the API and the ingest worker make is filtered.
-- The owner (the Aurora master user) is used only by migrations and admin
-- tooling, which need to see all tenants. FORCE ROW LEVEL SECURITY is
-- deliberately not used: it would filter the owner too, and the only way to
-- let maintenance through would be a setting the runtime role could also set.
--
-- The session variables are set with set_config(..., true), which scopes them
-- to the current transaction. When app.tenant_id is unset, current_setting
-- returns NULL, the comparison is NULL, and no row is visible.
-- --------------------------------------------------------------------------

-- Lists travel as JSON text so the same SQL works with psycopg and with the
-- RDS Data API, which does not accept array parameters.
CREATE FUNCTION text_array(j jsonb) RETURNS text[]
    LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce(ARRAY(SELECT jsonb_array_elements_text(j)), '{}'::text[])
$$;

CREATE FUNCTION app_groups() RETURNS text[]
    LANGUAGE sql STABLE AS $$
    SELECT coalesce(
        ARRAY(SELECT jsonb_array_elements_text(
            nullif(current_setting('app.groups', true), '')::jsonb)),
        '{}'::text[])
$$;

CREATE FUNCTION app_can_read(owner text, groups text[]) RETURNS boolean
    LANGUAGE sql STABLE AS $$
    SELECT current_setting('app.acl_bypass', true) = 'on'
        OR owner = current_setting('app.user_id', true)
        OR groups && app_groups()
$$;

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_self ON tenants
    USING (tenant_id = current_setting('app.tenant_id', true));

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY documents_tenant ON documents
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
CREATE POLICY documents_acl ON documents AS RESTRICTIVE FOR SELECT
    USING (app_can_read(owner_id, acl_groups));

ALTER TABLE document_versions ENABLE ROW LEVEL SECURITY;
CREATE POLICY versions_tenant ON document_versions
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY chunks_tenant ON chunks
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
CREATE POLICY chunks_acl ON chunks AS RESTRICTIVE FOR SELECT
    USING (app_can_read(owner_id, acl_groups));

ALTER TABLE answer_cache ENABLE ROW LEVEL SECURITY;
CREATE POLICY cache_tenant ON answer_cache
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

ALTER TABLE usage_daily ENABLE ROW LEVEL SECURITY;
CREATE POLICY usage_tenant ON usage_daily
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY audit_tenant ON audit_events
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

-- --------------------------------------------------------------------------
-- Grants for the runtime role
-- --------------------------------------------------------------------------
GRANT USAGE ON SCHEMA public TO {{APP_ROLE}};
GRANT SELECT ON embedding_space TO {{APP_ROLE}};
GRANT SELECT ON tenants TO {{APP_ROLE}};
-- Only the corpus counter. Limits and feature flags change through migrations
-- or the admin CLI, never through the runtime role.
GRANT UPDATE (corpus_version) ON tenants TO {{APP_ROLE}};
GRANT SELECT, INSERT, UPDATE ON documents, document_versions TO {{APP_ROLE}};
GRANT SELECT, INSERT, UPDATE, DELETE ON chunks, answer_cache TO {{APP_ROLE}};
GRANT SELECT, INSERT, UPDATE ON usage_daily TO {{APP_ROLE}};
GRANT SELECT, INSERT ON audit_events TO {{APP_ROLE}};
GRANT EXECUTE ON FUNCTION app_groups(), app_can_read(text, text[]), text_array(jsonb)
    TO {{APP_ROLE}};
