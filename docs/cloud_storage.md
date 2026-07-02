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
