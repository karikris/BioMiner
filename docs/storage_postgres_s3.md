# Storage, Postgres, And S3

Production BioMiner separates durable artifacts from operational state.

## Artifact Storage

S3-compatible object storage holds:

```text
registry artifacts
raw Flickr JSON audit payloads when retained
canonical source record Parquet shards
object detection Parquet shards
BioCLIP score Parquet shards
joined evidence Parquet shards
JSON and Markdown reports
run manifests
```

Required S3-style variables:

```text
BIOMINER_S3_ENDPOINT_URL
BIOMINER_S3_ACCESS_KEY_ID
BIOMINER_S3_SECRET_ACCESS_KEY
BIOMINER_S3_REGION
BIOMINER_S3_BUCKET
BIOMINER_S3_PREFIX
```

Backblaze B2 and similar stores use the S3-compatible API. Do not introduce a custom provider URI scheme when `s3://` plus `BIOMINER_S3_ENDPOINT_URL` is enough.

## Workstore

Postgres holds operational control-plane state:

```text
work queue rows
claimed/completed/failed status
completed work keys
API-call ledgers
run manifests and resume state
Parquet shard inventory
compaction manifests
```

Required workstore variables:

```text
BIOMINER_WORKSTORE_DSN
BIOMINER_WORKER_ID
```

Postgres claiming should use transactional row claims and `FOR UPDATE SKIP LOCKED` where supported. Operational failures remain retryable and should return to the queue rather than becoming biological negatives.

## Local Override

Local filesystem and SQLite are for tests, parser smoke checks, and isolated development. They must be selected explicitly:

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir data/registry/current \
  --output-prefix staging/runs/papilio_demoleus \
  --storage-backend local \
  --workstore-backend sqlite \
  --dry-run
```

Useful local environment variables:

```text
BIOMINER_STORAGE_BACKEND=local
BIOMINER_WORKSTORE_BACKEND=sqlite
BIOMINER_WORKSTORE_SQLITE_PATH=data/state/biominer.sqlite
```

## Diagnostics

Validate configured backends with:

```bash
uv run biominer storage doctor
uv run biominer workstore doctor
```

See also `docs/cloud_storage.md` and `docs/cloud_provider_config.md` for lower-level implementation details.
