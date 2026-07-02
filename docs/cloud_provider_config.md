# Cloud Provider Configuration

BioMiner can be configured for local development or cloud-ready execution without changing pipeline code.

Local defaults require no environment variables:

```toml
[biominer.storage]
backend = "local"
prefix = "."

[biominer.workstore]
backend = "sqlite"
sqlite_path = "data/state/biominer.sqlite"

[biominer.runtime]
worker_id_env = "BIOMINER_WORKER_ID"
default_batch_rows = 50000
target_parquet_mb = 128
```

Backblaze B2 uses the S3-compatible API. Do not use a custom `b2://` URI scheme.

```toml
[biominer.storage]
backend = "s3"
bucket = "biominer"
prefix = "biominer"
endpoint_url_env = "BIOMINER_S3_ENDPOINT_URL"
access_key_id_env = "BIOMINER_S3_ACCESS_KEY_ID"
secret_access_key_env = "BIOMINER_S3_SECRET_ACCESS_KEY"
region = "auto"

[biominer.workstore]
backend = "postgres"
dsn_env = "BIOMINER_WORKSTORE_DSN"
```

Required environment variables for a Backblaze B2 plus Supabase-style setup:

```text
BIOMINER_S3_ENDPOINT_URL=https://s3.<region>.backblazeb2.com
BIOMINER_S3_ACCESS_KEY_ID=<from environment>
BIOMINER_S3_SECRET_ACCESS_KEY=<from environment>
BIOMINER_WORKSTORE_DSN=postgresql://user:password@host:5432/postgres
BIOMINER_WORKER_ID=worker-001
```

Supabase Postgres stores control-plane state: work queue rows, API-call ledger rows, completed keys, run state, shard inventory, resume state, and compaction manifests. It does not store Parquet payloads.

`default_batch_rows = 50000` is a Parquet/object-storage flush threshold. It is not a Flickr API request size and must not be used as Flickr `per_page`. Flickr paging remains controlled by the query planner constants and Flickr API limits.

Shard writers should flush when either `default_batch_rows` is reached, approximate staged bytes reach `target_parquet_mb`, or the run ends. Byte estimation is configuration-ready; exact byte enforcement can be improved later.

Worker and compacted outputs remain immutable:

```text
evidence/stage=poll_once/run_id=<run_id>/worker=<worker_id>/batch=<batch_id>.parquet
evidence/stage=poll_once_compacted/registry_version=<registry_version>/run_id=<run_id>/part=000001.parquet
```

Example local parser smoke:

```text
biominer --config config/biominer.local.example.toml --version
```

Example cloud-compatible compaction dry run:

```text
biominer --config config/biominer.cloud.example.toml compact-parquet \
  --input-prefix s3://biominer/biominer/evidence/stage=poll_once/run_id=... \
  --output-prefix s3://biominer/biominer \
  --source-stage poll_once \
  --registry-version butterflies-v1 \
  --compaction-run-id compact-1 \
  --dry-run
```

Cloud integration tests are offline/local only. They validate configuration parsing, validation, redaction, and local factories without Backblaze, Supabase, Docker, network access, Flickr credentials, CUDA, or BioCLIP weights.
