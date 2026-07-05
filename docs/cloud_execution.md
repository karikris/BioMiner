# Cloud Execution Boundary

BioMiner production cloud mode is intended to be durable-cloud-only and batch/shard-driven. The production data plane is S3-compatible object storage; the production control plane is Postgres. Local filesystem and SQLite are allowed only for explicit tests, development overrides, and worker-local ephemeral scratch that is deleted before the stage completes.

## Production Defaults

Production runs should use:

```text
BIOMINER_STORAGE_BACKEND=s3
BIOMINER_WORKSTORE_BACKEND=postgres
```

The CLI equivalent is:

```text
uv run biominer run \
  --storage-backend s3 \
  --workstore-backend postgres
```

Local filesystem storage and SQLite workstores are dev/test overrides. They must not be selected implicitly by production cloud stages.

## Durable Cloud Data Plane: S3

S3-compatible object storage stores all durable artifacts:

- registry version artifacts and `registry/current` JSON pointer payloads;
- scoped Flickr query definitions;
- raw Flickr JSON audit payloads;
- canonical source-record Parquet shards;
- metadata flag and evidence Parquet shards;
- object detection Parquet shards;
- BioCLIP 2.5 score Parquet shards;
- joined object evidence Parquet shards;
- photo summary Parquet shards;
- review queue Parquet shards;
- reviewed evidence Parquet shards;
- compact JSON/Markdown reports;
- run manifests;
- compaction outputs.

Production workers must write immutable objects. They must not append into a shared Parquet file, rewrite source shards in place, or depend on POSIX symlink semantics in object storage.

Preferred path families:

```text
registry/version=<registry_version>/<artifact>
registry/current/pointer.json
raw/source=flickr/method=photos_search/run_id=<run_id>/field=<field>/term=<term>/lane=<lane>/page=<page>/work_item_id=<work_key>.json
evidence/stage=<stage>/run_id=<run_id>/worker=<worker_id>/batch=<batch_id>.parquet
evidence/stage=<stage>_compacted/registry_version=<registry_version>/run_id=<compaction_run_id>/part=<part>.parquet
reports/run_id=<run_id>/<report>.json
manifests/<run_id>.json
```

## Durable Cloud Control Plane: Postgres

Postgres stores operational state only. It should not store large biological evidence tables.

Current control-plane tables:

- `biominer_runs`: run identity, job/stage, registry version, status, config JSON, summary JSON.
- `biominer_work_items`: planned work payloads, pending/claimed/completed/failed state, claim owner/time, attempts, output URI, checksum, row count, errors.
- `biominer_api_call_ledger`: API request reservation and outcome telemetry.
- `biominer_parquet_shards`: committed shard inventory with URI, worker, run, registry version, stage, row count, byte count, checksum, metadata.
- `biominer_compaction_inputs`: source-to-output lineage for compaction.

Claiming must be transactional. For Postgres this means row claims using `FOR UPDATE SKIP LOCKED` or equivalent semantics so multiple workers can claim independent batches without duplicate processing.

## Allowed Local Worker State

Cloud workers may use local state only for ephemeral runtime scratch:

- downloaded Flickr image bytes used for object proposal or BioCLIP scoring;
- temporary crop files passed to model runtimes;
- temporary process-local buffers;
- model weights and runtime caches installed outside durable BioMiner output areas;
- local temporary upload buffers only until Phase 2 replaces them with streaming writes.

Allowed local scratch must be under OS temporary/cache locations or explicit model cache directories. It must not be written under durable project output roots such as:

```text
data/
staging/
reports/
runs/
*.sqlite
```

in production cloud mode.

## Disallowed Production Cloud Local State

Production cloud mode must not create durable local:

- Parquet outputs;
- SQLite databases;
- JSON reports/manifests;
- registry outputs;
- raw API dumps;
- staging files;
- pipeline caches;
- run directories.

If a cloud stage cannot use S3/Postgres for its durable outputs and control state, it should fail clearly or be marked experimental rather than silently falling back to local filesystem/SQLite.

## Stage Contracts

### Registry

Cloud registry builds write checkpoints, source snapshots, canonical registry artifacts, QA outputs, and manifests to S3 version prefixes. Promotion to current uses a JSON pointer payload rather than a local symlink or in-place directory swap.

Postgres may record run state and shard/artifact inventory. Taxonomy/name tables remain Parquet artifacts, not Postgres tables.

### Query Compilation

Scoped query definitions are Parquet artifacts in S3. Query rows preserve:

- `registry_version`;
- `query_definition_id`;
- accepted taxon key and accepted scientific name;
- family/genus/species keys;
- source term and normalized term;
- language/script/region/bbox metadata;
- source and trust metadata;
- search field and priority.

### Flickr Polling

Cloud polling claims query work from `biominer_work_items`. Raw Flickr JSON is written directly to S3. Canonical source-record output is written as immutable per-work or bounded-batch Parquet shards.

A work item is completed only after:

1. raw payload write succeeds when raw retention is enabled;
2. canonical source-record shard write succeeds;
3. shard inventory registration succeeds when a workstore is available.

Duplicate Flickr discoveries fold into canonical provenance arrays. They must not create duplicate canonical photo rows.

### Detection

Detection work is planned from committed source-record shard inventory. Workers claim bounded batches, download images temporarily, run YOLOE/YOLO26 object proposal, and write detection shards to S3.

YOLOE/YOLO26 remain object proposal backends only. Their boxes are evidence for crop selection and negative routing. BioMiner must not persist reviewed YOLO boxes as training data.

### BioCLIP 2.5

BioCLIP work is planned from detection shards. Only rows with:

```text
detection_status = "detected"
detector_label = "butterfly_like"
```

may proceed to BioCLIP 2.5 scoring.

BioCLIP 2.5 remains the crop-level species scorer. Output score shards carry object keys, model IDs, checkpoints, candidate-set IDs, visual mode, ranked family/genus/species candidates, target species score/rank, and bucket decision.

### Evidence Join And Summary

Cloud joins consume committed source-record, detection, and score shard inventory using lazy scans or bounded shard groups. They write immutable joined evidence shards and photo summary shards. They do not eager-read the complete run into one worker-local frame except in explicit dev/test mode.

Photo summaries remain one row per `source + flickr_photo_id`. Object evidence remains keyed by `source + flickr_photo_id + detection_id + crop_hash` plus model/candidate/mode metadata.

### Comment Review

Comment review should be cloud-backed before it is considered production cloud-native. The target state is:

- review queue state in Postgres or another durable control-plane table;
- comment-derived terms and missing-data requests in durable control state;
- reviewed evidence written to S3;
- promotions applied only after reviewed comment evidence supports the same species and bucket policy rules still pass.

The current local SQLite comment review state is a dev/local implementation until this target is implemented.

### Compaction

Compaction reads candidate shard inventory from Postgres, scans source shards from S3, writes new immutable compacted shards to S3, and records source-to-output lineage in `biominer_compaction_inputs`. It must not delete or rewrite source shards as part of normal execution.

## Current Audit Findings

Phase 1 guard tests intentionally track current local-first implementation leaks that later phases must remove:

- cloud Flickr polling bridges through a temporary local `MetadataPollState` SQLite database;
- cloud detection and BioCLIP stages use local temporary final files before uploading;
- cloud detection and BioCLIP stages eager-read full Parquet outputs and materialize rows into Python dicts;
- cloud join and summary stages eager-read full run artifacts rather than consuming shard inventory;
- cloud comment review state is not implemented and currently exists only as local SQLite.

These findings define the cleanup backlog. Until they are fixed, production cloud mode should be treated as partially cloud-backed rather than fully cloud-native.
