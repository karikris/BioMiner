# BioMiner Production Workflow Refactor Initial Audit

Recorded on branch `cleanup/production-workflow-postgres-s3` from current HEAD `27d9f0d`.

## Repository State

- Git branch before cleanup work: `feature/yoloe26-prototype`.
- Cleanup branch created from current worktree: `cleanup/production-workflow-postgres-s3`.
- GitHub repository: `karikris/BioMiner`.
- GitHub default branch: `main`.
- GitHub access via MCP: push/admin-level permissions reported.
- Python package: `biominer`, Python `>=3.14`.
- Core dependencies: `duckdb`, `httpx`, `polars`, `pyarrow`, `pydantic`.
- Optional extras: `postgres`, `test`.

## Test Baseline

Command run:

```bash
uv run --extra test python -m pytest -q
```

Result:

```text
431 passed
```

The moved checkout does not expose a `pytest` executable shim reliably through `uv run`; `python -m pytest` is the stable invocation in this environment.

## Current Public Command Surface

Root commands currently exposed:

```text
bioclip
detect
fetch-comments
registry
cloud
species
build-comment-review-queue
review-comments-once
apply-comment-review-decisions
poll-once
apply-rules
filter
gc-cache
compact-parquet
qa-rate-limit
qa-summary
export-bucket-views
report-name-evidence
```

`registry` subcommands:

```text
compile-fixture
compile-enriched
fetch-taxonomy
enrich-sources
build
seed-flickr-queries
audit
```

`bioclip` subcommands:

```text
runtime-check
prefetch-model
screen
screen-objects
ablate-objects
join-object-evidence
```

`detect` subcommands:

```text
boxes
yoloe26-runtime-check
yoloe26-prefetch
yoloe26-smoke
yoloe26-prototype-run
crop-preview
eval
```

`species` subcommands:

```text
resolve
refresh-registry
compile-flickr-queries
fetch-flickr
bioclip-funnel
detect
bioclip-objects
ablate-objects
join-object-evidence
review-comments
run
```

`cloud` subcommands:

```text
init
doctor
```

## Current Storage / Workstore Defaults

Current default configuration is still development-local, not production cloud:

| Component | Current default | Source |
| --- | --- | --- |
| Storage backend | `local` | `StorageConfig.backend` and `BIOMINER_STORAGE_BACKEND` fallback |
| Storage prefix | `.` for local | `_load_storage_config` |
| Workstore backend | `sqlite` | `WorkStoreConfig.backend` and `BIOMINER_WORKSTORE_BACKEND` fallback |
| Workstore SQLite path | `data/state/biominer.sqlite` | `WorkStoreConfig.sqlite_path` |
| Worker ID | `local` | `RuntimeConfig.worker_id` |

Production target requires changing these defaults to S3 + Postgres for production commands, while preserving explicit local/sqlite test-dev fallback.

## Current Modules By Area

### Registry

```text
registry/audit.py
registry/build.py
registry/compiler.py
registry/enrichment.py
registry/enrichment_sources.py
registry/gbif.py
registry/gbif_production.py
registry/gbif_source.py
registry/normalize.py
registry/scope.py
```

Missing target modules:

```text
registry/translation_sources.py
registry/trust_policy.py
```

### Flickr Fetch

```text
flickr_fetch/endpoints.py
flickr_fetch/metadata_poller.py
flickr_fetch/query_planner.py
```

Needs audit for seed fallback and broad multilingual seed behavior.

### Detection

```text
detection/detector_base.py
detection/yoloe26_detector.py
detection/yolo_detector.py
detection/cropper.py
detection/segmentation.py
detection/pipeline.py
detection/schema.py
```

Target layout wants `yolo26_detector.py`; current code has `yolo_detector.py` and YOLOE-26 sidecar support.

### BioCLIP

```text
bioclip/object_runner.py
bioclip/candidate_sets.py
bioclip/embedding_cache.py
bioclip/model_registry.py
bioclip/register_runner.py
bioclip/triage.py
```

Whole-image/register path still exists and is public through `bioclip screen` and `species bioclip-funnel`.

### Evidence

No `src/biominer/evidence/` package currently exists. Evidence logic is split across:

```text
filter/rules.py
reports/buckets.py
bioclip/object_runner.py
```

Target modules `evidence/join.py`, `evidence/buckets.py`, and `evidence/metrics.py` are not present.

### Storage / Workstore

Current storage modules:

```text
storage/cloud.py
storage/compaction.py
storage/config.py
storage/factory.py
storage/local.py
storage/parquet.py
storage/paths.py
storage/s3.py
storage/shard_paths.py
```

Current workstore modules:

```text
workstore/base.py
workstore/factory.py
workstore/keys.py
workstore/postgres.py
workstore/resume.py
workstore/schema.py
workstore/sqlite.py
```

Target storage layout wants `storage/config.py`, `storage/s3.py`, `storage/cloud.py`, and `storage/factory.py`; local storage can remain as explicit dev/test fallback but should not be production default.

## Suspected Redundant Or Legacy Public Paths

These are directly named in the refactor objective or are implicated by current command surface:

| Current public path | Reason to remove/demote |
| --- | --- |
| `registry fetch-taxonomy` | Low-level registry internals exposed as normal workflow. |
| `registry compile-fixture` | Fixture/internal path, not production workflow. |
| `registry compile-enriched` | Low-level internal registry stage exposed publicly. |
| `registry enrich-sources` | Low-level enrichment internals exposed publicly. |
| `registry seed-flickr-queries` | Seed workflow conflicts with production registry-name query definitions. |
| `bioclip screen` | Old whole-image primary BioCLIP command. |
| `species bioclip-funnel` | Old whole-image species funnel alias. |
| `species detect` | Duplicate alias for object detection. |
| `species bioclip-objects` | Duplicate alias for object BioCLIP. |
| `species ablate-objects` | Duplicate alias for object ablations. |
| `species join-object-evidence` | Duplicate alias for object evidence join. |
| `apply-rules` | Legacy bucket source separate from object evidence bucket logic. |
| `filter` | Metadata anti-keyword hard-drop command; should become metadata flags only. |
| `compact-parquet` | Legacy local compaction exposed publicly. |
| `report-name-evidence` | Needs removal if dependent on ad hoc keywords JSON rather than registry evidence. |

## Current Risks For Target Workflow

- Public command surface still has multiple competing workflows instead of one `run`-oriented production path.
- Storage/workstore defaults remain `local` + `sqlite`.
- Local state defaults appear in public commands, especially `poll-once`, comment review, and registry seed commands.
- Metadata anti-keywords are still exposed as `filter` and `apply-rules` workflow stages.
- No `run/` orchestration package exists yet.
- No `evidence/` package exists yet.
- Translation/trust-policy modules are not present under `registry/`.
- `reports/` is currently ignored wholesale, so intentional audit markdown files need explicit `.gitignore` exceptions or forced adds.

## Phase 0 Recommendations

1. Add `.gitignore` exceptions for intentional `reports/*.md` refactor/audit documents while continuing to ignore generated run artifacts.
2. Add `docs/refactor_guardrails.md` to document artifact and workflow cleanup guardrails.
3. In Phase 1, add `src/biominer/run/` skeleton without live API calls.
4. In later phases, change production config defaults and then remove legacy public commands in small commits with tests updated immediately.

