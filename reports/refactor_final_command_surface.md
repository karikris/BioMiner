# BioMiner Refactor Command Surface Audit

Recorded: 2026-07-05

Current branch: `main`

Latest audited base before this report edit: `ab0d354`

Note: the original refactor plan named branch `cleanup/production-workflow-postgres-s3`; a later operator instruction changed the working branch to `main`, and the cleanup commits are being pushed directly to `main`.

## Public Command Surface

Root commands from `uv run --extra test biominer --help`:

```text
biominer vision ...
biominer evidence ...
biominer registry ...
biominer dev ...
biominer storage ...
biominer workstore ...
biominer run ...
```

Production-level commands:

```text
biominer registry build
biominer registry audit
biominer run
biominer vision detect
biominer vision score
biominer vision ablate
biominer evidence join
biominer storage doctor
biominer workstore doctor
```

Vision runtime and QA/debug commands moved under `dev vision`:

```text
biominer dev vision bioclip-runtime-check
biominer dev vision bioclip-prefetch-model
biominer dev vision yoloe26-runtime-check
biominer dev vision yoloe26-prefetch
biominer dev vision yoloe26-smoke
biominer dev vision yoloe26-prototype-run
biominer dev vision crop-preview
biominer dev vision eval
```

Dev-only internal command groups:

```text
biominer dev registry compile-fixture
biominer dev registry compile-enriched
biominer dev registry fetch-taxonomy
biominer dev registry enrich-sources
biominer dev registry seed-flickr-queries
biominer dev comments fetch
biominer dev comments queue
biominer dev comments review-once
biominer dev comments apply-decisions
biominer dev flickr poll-once
biominer dev vision ...
```

## Removed Public Commands

These are no longer on the public command surface:

```text
biominer apply-rules
biominer compact-parquet
biominer gc-cache
biominer qa-rate-limit
biominer qa-summary
biominer export-bucket-views
biominer report-name-evidence
biominer species ...
biominer bioclip screen
biominer species bioclip-funnel
biominer species detect
biominer species bioclip-objects
biominer species ablate-objects
biominer species join-object-evidence
biominer registry fetch-taxonomy
biominer registry compile-fixture
biominer registry compile-enriched
biominer registry enrich-sources
biominer registry seed-flickr-queries
```

Low-level registry/Flickr/comment utilities remain under `biominer dev ...` for tests and controlled debug workflows.

## Production Run Interface

Current `biominer run` arguments:

```text
--taxon TAXON
--rank auto|family|genus|species
--registry-dir REGISTRY_DIR
--output-prefix OUTPUT_PREFIX
--storage-backend s3|local
--workstore-backend postgres|sqlite
--vision-backend VISION_BACKEND
--bioclip-model BIOCLIP_MODEL
--stages STAGES
--dry-run
--limit-species LIMIT_SPECIES
--limit-records LIMIT_RECORDS
```

Production defaults are S3-compatible storage and Postgres workstore. Local filesystem and SQLite are accepted only as an explicit dev/test pair:

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir data/registry/current \
  --output-prefix s3://example-bucket/biominer/runs/papilio-demoleus
```

Explicit local/dev mode:

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir data/registry/current \
  --output-prefix runs/papilio-demoleus \
  --storage-backend local \
  --workstore-backend sqlite \
  --dry-run
```

## Storage And Workstore Defaults

Default production config values in `src/biominer/config/__init__.py`:

```text
StorageConfig.backend = "s3"
WorkStoreConfig.backend = "postgres"
StorageConfig.endpoint_url_env = "BIOMINER_S3_ENDPOINT_URL"
StorageConfig.access_key_id_env = "BIOMINER_S3_ACCESS_KEY_ID"
StorageConfig.secret_access_key_env = "BIOMINER_S3_SECRET_ACCESS_KEY"
WorkStoreConfig.dsn_env = "BIOMINER_WORKSTORE_DSN"
RuntimeConfig.worker_id_env = "BIOMINER_WORKER_ID"
```

Production validation requires:

```text
BIOMINER_S3_ENDPOINT_URL
BIOMINER_S3_ACCESS_KEY_ID
BIOMINER_S3_SECRET_ACCESS_KEY
BIOMINER_S3_REGION
BIOMINER_S3_BUCKET
BIOMINER_S3_PREFIX
BIOMINER_WORKSTORE_DSN
BIOMINER_WORKER_ID
```

Validation redacts configured secrets and rejects mixed local/cloud modes such as local storage with Postgres or S3 storage with SQLite.

## Workflow State

Confirmed current behavior:

- The public workflow is rank-aware through `biominer run --rank auto|family|genus|species`.
- Registry build/audit are the only public registry commands.
- `biominer registry build` defaults to enrichment sources:
  `col`, `inaturalist`, `itis`, `tmd_de`, and `wikidata`.
- Registry-derived Flickr query definitions are the production discovery input.
- The implicit broad seed fallback is removed; `MetadataPollState` now exposes only explicit `enqueue_initial_work_items(queries)` for registry-derived query definitions.
- Built-in multilingual seed terms and the broad discovery seed planner were removed from `flickr_fetch.query_planner`.
- Metadata text matching is a flag/review signal, not a public hard-drop path.
- Object detection emits BioMiner coarse labels only.
- YOLOE-26 and YOLO26 adapters are object proposal backends, not species classifiers.
- BioCLIP object scoring remains the species scorer.
- Production `score_bioclip` now requests the full object visual mode set by default:
  `whole_image`, `detector_crop`, and `detector_crop_segmentation`.
- When segmentation masks are unavailable, the production scorer records
  `detector_crop_segmentation` as unavailable and continues with whole-image
  plus detector-crop scores.
- Production registry reuse now requires `flickr_query_definitions.parquet` in
  addition to `taxa.parquet`, `names.parquet`, and `manifest.json`; missing
  registry-derived Flickr queries fail the run stage clearly instead of falling
  back to broad seed search.
- Production run polling uses the validated `RuntimeConfig.worker_id` / `BIOMINER_WORKER_ID` instead of falling back to an ambient local worker ID.
- Production validation treats the default `local` worker id as missing for Postgres-backed runs; `BIOMINER_WORKER_ID` must be configured outside explicit local/SQLite dev mode.
- Vision runtime checks, prefetch, smoke/prototype, crop preview, and evaluation utilities are dev-only.

Known remaining debug surfaces:

- Shared `biominer.species.context` remains in use as the data model for species context across run, BioCLIP, evidence, and comment review. It is not a public `biominer species` workflow command.

## Verification Commands

Commands used for this audit:

```bash
uv run --extra test biominer --help
uv run --extra test biominer registry --help
uv run --extra test biominer run --help
uv run --extra test biominer vision --help
uv run --extra test biominer dev vision --help
uv run --extra test biominer evidence --help
uv run --extra test biominer storage --help
uv run --extra test biominer workstore --help
uv run --extra test biominer dev --help
uv run --extra test biominer dev registry --help
uv run --extra test biominer dev comments --help
uv run --extra test biominer dev flickr --help
uv run --extra test pytest -q
```

Latest full-suite result:

```text
514 passed
```
