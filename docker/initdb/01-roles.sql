-- Local PostgreSQL bootstrap, run once by the official image on first start.
--
-- rag_owner mirrors the Aurora master user: it owns the database and can
-- create roles, but it is NOT a superuser, so FORCE ROW LEVEL SECURITY binds
-- it exactly as it binds Aurora's rds_superuser-based master. Running tests as
-- a true superuser would hide RLS mistakes that only show up on Aurora.
-- Development-only credentials; never reuse them anywhere else.

CREATE ROLE rag_owner LOGIN PASSWORD 'rag_owner_local' NOSUPERUSER CREATEROLE NOBYPASSRLS;

CREATE DATABASE ragplatform OWNER rag_owner;
CREATE DATABASE ragplatform_test OWNER rag_owner;

\connect ragplatform
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

\connect ragplatform_test
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
