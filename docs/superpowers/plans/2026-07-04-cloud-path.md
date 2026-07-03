# Cloud Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Backblaze B2 + Supabase Postgres + Polars cloud path while keeping local SQLite/local filesystem behavior unchanged.

**Architecture:** Keep `WorkStore` and `CloudStorage` as the stable seams. Implement Postgres with the same contract as SQLite, keep S3 writes shard-oriented and immutable, and make cloud/no-compact poller output write canonical per-work-item delta shards registered in the workstore.

**Tech Stack:** Python 3.14, Polars, PyArrow S3 filesystem, psycopg 3, pytest, Supabase Postgres, Backblaze B2 S3-compatible storage.

---

### Task 1: Postgres WorkStore Contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/biominer/workstore/postgres.py`
- Test: `tests/test_workstore.py`

- [ ] **Step 1: Write the failing test**

Add a fake DB-API/psycopg-style connection test that exercises `get_or_create_run`, `enqueue_work`, `claim_next_batch`, `mark_completed`, `completed_keys`, `register_shard`, and `list_committed_shards` through `PostgresWorkStore`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_workstore.py::test_postgres_workstore_contract_with_injected_connection`

Expected: FAIL because `PostgresWorkStore` is still a placeholder.

- [ ] **Step 3: Implement minimal code**

Add `psycopg[binary]` as an optional postgres dependency. Implement schema init and the minimal required methods against psycopg connections, using `POSTGRES_CLAIM_SQL` for claims and row mappers compatible with SQLite semantics.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest -q tests/test_workstore.py tests/test_storage_config.py tests/test_provider_config.py`

- [ ] **Step 5: Commit and push**

Commit message: `infra: implement postgres workstore`

### Task 2: Cloud CLI Init and Doctor

**Files:**
- Modify: `src/biominer/cli.py`
- Test: `tests/test_cli_dry_run.py`
- Test: `tests/test_provider_config.py`

- [ ] **Step 1: Write failing parser and function-level tests**

Verify `biominer cloud init` and `biominer cloud doctor` parse and use redacted config without printing secrets.

- [ ] **Step 2: Implement commands**

`cloud init` initializes the configured workstore schema. `cloud doctor` writes/reads/deletes tiny JSON in S3, writes/scans tiny Parquet in S3, initializes Postgres schema, runs one enqueue/claim/complete cycle, and registers/lists one shard.

- [ ] **Step 3: Verify**

Run: `uv run pytest -q tests/test_cli_dry_run.py tests/test_provider_config.py tests/test_workstore.py`

- [ ] **Step 4: Commit and push**

Commit message: `infra: add cloud doctor`

### Task 3: S3 Streaming Parquet Writes

**Files:**
- Modify: `src/biominer/storage/s3.py`
- Test: `tests/test_storage_backends.py`

- [ ] **Step 1: Write failing test**

Patch `polars.DataFrame.write_parquet` or a fake filesystem stream to prove `write_parquet_shard` writes through a temporary local file/stream path and does not materialize `BytesIO.getvalue()`.

- [ ] **Step 2: Implement streaming upload**

Write Parquet to a temp file, then upload chunks through PyArrow output stream. Preserve URI return semantics.

- [ ] **Step 3: Verify**

Run: `uv run pytest -q tests/test_storage_backends.py`

- [ ] **Step 4: Commit and push**

Commit message: `infra: stream s3 parquet shards`

### Task 4: Poller Canonical Delta Shards

**Files:**
- Modify: `src/biominer/flickr_fetch/metadata_poller.py`
- Test: `tests/test_metadata_poller.py`

- [ ] **Step 1: Write failing tests**

Cover cloud/no-compact mode where two work items return the same Flickr photo ID. Assert one canonical evidence row in the written shard, folded text/tag provenance arrays, and one registered shard per work item batch.

- [ ] **Step 2: Implement canonical delta output**

Derive shard rows from canonical `source_records` after insert/update, not raw payloads. Keep raw JSON responses and image URL records. Keep legacy query-hit rows only as migration fallback.

- [ ] **Step 3: Verify**

Run: `uv run pytest -q tests/test_metadata_poller.py`

- [ ] **Step 4: Commit and push**

Commit message: `step1: write canonical cloud delta shards`

### Task 5: Cleanup and Full Verification

**Files:**
- Modify only files proven by call-site search to contain obsolete duplicate helpers or redundant parser tests.

- [ ] **Step 1: Remove unused metadata poller legacy shard helpers**

Use `rg` to prove no live production callers remain before deletion.

- [ ] **Step 2: Reduce duplicate CLI/parser tests**

Move duplicate parser checks into function-level contract tests without weakening coverage.

- [ ] **Step 3: Full verification**

Run: `uv run pytest -q`

- [ ] **Step 4: Real cloud doctor**

Load secrets from `/home/toffe/.config/agent-env/secrets.env` without printing them and run `uv run biominer cloud doctor`.

- [ ] **Step 5: Commit and push**

Commit message: `infra: verify cloud path`
