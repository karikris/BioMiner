# Cloud-Centric Cleanup Completion

Date: 2026-07-05

## Summary

BioMiner production execution is now centered on S3-compatible object storage for durable artifacts and Postgres for queue, run, and shard state. Local filesystem and SQLite paths remain available only as explicit dev/test overrides or short-lived worker scratch.

The cleanup phase removed the remaining production fallback from `source_record_query_hits` and reduced duplicated CLI stage tests now covered by direct cloud workflow contract tests.

## Verification

- Full test suite: `uv run pytest -q` -> `546 passed in 7.72s`.
- Postgres doctor: `uv run biominer workstore doctor` -> `status=ok`, schema initialized, one test work item inserted/claimed/completed, one shard registered.
- S3 doctor: `uv run biominer storage doctor` could not run against real object storage because the environment is missing `BIOMINER_S3_BUCKET`, `BIOMINER_S3_PREFIX`, `BIOMINER_S3_ENDPOINT_URL`, and `BIOMINER_S3_REGION`.

## Durable Data Locations

S3-compatible storage is the durable data plane:

- `registry/version=<registry_version>/`: registry source snapshot, canonical registry Parquet artifacts, QA output, manifest.
- `registry/current/manifest.json`: JSON pointer to the promoted registry version.
- `raw/source=flickr/...`: raw Flickr JSON responses when retained.
- `evidence/stage=poll_once/...`: canonical source-record shards.
- `evidence/stage=object_detections/...`: YOLOE/YOLO26 object proposal shards.
- `evidence/stage=object_bioclip_scores/...`: BioCLIP 2.5 crop-level score shards.
- `evidence/stage=join_evidence/...`: joined source/detection/score evidence shards.
- `evidence/stage=photo_summary/...`: one-row-per-photo summary shards.
- `evidence/stage=review_queue/...`: review queue shards.
- `reports/` and `manifests/`: compact JSON/Markdown reports and run manifests.
- `compaction/` and compacted evidence stages: immutable compaction outputs and lineage.

Postgres is the durable control plane:

- `biominer_runs`: run identity, status, config, and summaries.
- `biominer_work_items`: pending/claimed/completed/failed work.
- `biominer_api_call_ledger`: API call reservations and telemetry.
- `biominer_parquet_shards`: committed shard inventory.
- `biominer_compaction_inputs`: compaction input-to-output lineage.

Large evidence tables stay in Parquet shards, not Postgres.

## Data Integrity Invariants

- Workers write immutable shard objects; they do not append into shared Parquet files.
- A work item is completed only after its durable output shard has been written and registered.
- Flickr source records deduplicate on `source + flickr_photo_id`.
- Duplicate Flickr discoveries fold into provenance arrays on the canonical source record.
- Detection and BioCLIP evidence preserve object identity with `source + flickr_photo_id + detection_id + crop_hash`.
- Only `detection_status="detected"` and `detector_label="butterfly_like"` are eligible for BioCLIP 2.5 scoring.
- Photo summaries deduplicate back to one row per `source + flickr_photo_id`.
- Registry promotion uses a JSON pointer instead of local symlinks or object-store directory mutation.

## Remaining Local-Only Exceptions

- `LocalStorageBackend` and `SQLiteWorkStore` remain for tests, local smoke runs, and explicit dev overrides.
- Local registry builds still write to local paths when a user explicitly selects a local `--output-dir`.
- Dev commands under `biominer dev ...` may read/write local fixtures, SQLite state, or temporary Parquet files.
- Comment review execution still has a local SQLite implementation; cloud review queue shards exist, but cloud-backed comment review state/promotion remains future work.
- Model execution may use worker-local temporary image/crop files and model caches. These are runtime scratch, not durable BioMiner outputs.
- Real S3 doctor verification is pending until the missing S3 configuration variables are supplied.

## Cleanup Commits

- `4a7a4ee` `refactor(cleanup): remove obsolete local-first production helpers`
- `e09b8b8` `test(cleanup): minimize cloud workflow coverage`

