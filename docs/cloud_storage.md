# Cloud Storage Integration

Phase 1 added interfaces and local-compatible implementations. Phase 2 starts routing Step 1 `poll_once` raw responses and evidence rows onto immutable local/cloud-compatible paths. The default BioMiner runtime remains local filesystem storage plus SQLite operational state.

## Split

`CloudStorage` owns durable artifacts:

- immutable Parquet shards;
- compact JSON manifests and reports;
- raw Flickr JSON payloads if retained;
- registry, filtered, and classified evidence outputs.

`WorkStore` owns operational state:

- work queue rows;
- completed work keys;
- shard inventory;
- API-call ledgers in later phases;
- run manifests and resume state in later phases;

Local mode uses `LocalStorageBackend` and `SQLiteWorkStore`. Backblaze B2 is represented by `S3StorageBackend` through the S3-compatible API, using `s3://...` URIs plus `BIOMINER_S3_ENDPOINT_URL`. Supabase Postgres is represented by `PostgresWorkStore` scaffolding and schema SQL.

## Configuration

Local mode is the default:

```text
BIOMINER_STORAGE_BACKEND=local
BIOMINER_WORKSTORE_BACKEND=sqlite
BIOMINER_WORKSTORE_SQLITE_PATH=data/state/biominer.sqlite
```

S3-compatible object storage uses:

```text
BIOMINER_STORAGE_BACKEND=s3
BIOMINER_S3_ENDPOINT_URL=https://s3.<region>.backblazeb2.com
BIOMINER_S3_ACCESS_KEY_ID=...
BIOMINER_S3_SECRET_ACCESS_KEY=...
BIOMINER_S3_REGION=<region>
BIOMINER_S3_BUCKET=biominer
BIOMINER_S3_PREFIX=biominer
```

Supabase Postgres scaffolding uses:

```text
BIOMINER_WORKSTORE_BACKEND=postgres
BIOMINER_WORKSTORE_DSN=postgresql://...
```

`psycopg` is intentionally optional. Importing BioMiner does not require it; Postgres methods raise a clear runtime error if it is absent.

## Shard Invariant

Workers must not append into one shared cloud Parquet file. Each worker writes immutable shard objects using unique paths and then registers the shard in the control store when available.

Preferred path shape:

```text
evidence/stage=poll_once/run_id=<run_id>/worker=<worker_id>/batch=<batch_id>.parquet
```

For an S3-compatible bucket:

```text
s3://biominer/biominer/evidence/stage=poll_once/run_id=run-1/worker=w1/batch=000001.parquet
```

Compaction remains a later phase using Polars and DuckDB over shard sets.

## Phase 2 Paths

`poll_once` writes raw Flickr responses through `CloudStorage.write_json`:

```text
raw/source=flickr/method=photos_search/run_id=<run_id>/field=<text|tags>/term=<safe_term>/lane=<lane>/page=<page>/work_item_id=<work_item_id>.json
```

It writes per-work-item evidence shards through `CloudStorage.write_parquet_shard`:

```text
evidence/stage=poll_once/run_id=<run_id>/worker=<worker_id>/batch=<work_item_id>.parquet
```

Local compatibility remains: old-style `--raw-root data/raw --evidence-output staging/evidence/poll_once_evidence.parquet` still writes the compacted local evidence output after the run. Passing `--no-compact` skips that compatibility output and leaves only immutable shards.

Reports and registry pointers use cloud-safe helper paths:

```text
reports/run_id=<run_id>/<report_name>.json
registry/version=<registry_version>/<filename>
registry/current/manifest.json
```

For object storage, registry `current` should be represented by a JSON pointer payload instead of POSIX symlink semantics.

## Phase 3 Resume Model

Phase 3 makes resume state durable in `WorkStore` instead of deriving it from local files alone. A run starts by reading three control-plane views:

- `biominer_runs`: one logical execution for a job/stage/run ID;
- `biominer_work_items`: pending, claimed, completed, and failed units of work;
- `biominer_parquet_shards`: immutable output objects already committed by workers.

`prepare_resume_plan` creates or reuses the run row, requeues stale claims, reads completed keys, enqueues only missing planned work, optionally repairs shard manifest rows from existing shard objects, and claims only pending work for the current worker.

Completed work keys prevent duplicate processing. For Flickr `poll_once`, keys should be based on the query payload identity. For BioCLIP screening, the resume identity is exactly:

```text
source
flickr_photo_id
image_url
model_id
model_version
model_checkpoint
```

Scores, bins, review state, and local cache paths are mutable outputs and are intentionally excluded from the BioCLIP resume key.

Stale claims are rows with `status='claimed'` and an old `claimed_at`. Requeueing changes them back to `pending` and clears `claimed_by`/`claimed_at`; `attempt_count` increments only when the item is claimed again.

Shard manifest repair is optional. It lists shard objects under a prefix, compares them with `biominer_parquet_shards`, and registers missing rows. Local tests may read Parquet metadata for row counts. Cloud repair does not download large objects for checksums in this phase.

Postgres/Supabase remains scaffolded for Phase 3, with claim SQL using `FOR UPDATE SKIP LOCKED` for safe multi-worker claiming. Tests are local-only and do not require Backblaze B2, Supabase, network access, Docker, Flickr credentials, CUDA, or BioCLIP weights.
