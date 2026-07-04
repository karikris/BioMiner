# BioMiner Production Workflow Refactor Completion Audit

Recorded: 2026-07-05

Audited base: `a03d495`

Branch used for current cleanup commits: `main`

This report records current evidence for the long-running production workflow
refactor. It does not replace the source files, tests, or command output as
authority; it points to the current evidence that should be rechecked before
the active goal is marked complete.

## Current Evidence Summary

| Requirement area | Current evidence | Status |
| --- | --- | --- |
| Public production workflow is rank-aware | `uv run --extra test biominer run --help` exposes `--taxon` and `--rank auto|family|genus|species`; `tests/test_production_run_skeleton.py` covers species, genus, and family scope expansion. | Implemented |
| Public registry surface is production-only | `uv run --extra test biominer registry --help` exposes only `build` and `audit`. Low-level registry commands are under `uv run --extra test biominer dev registry --help`. | Implemented |
| Old public command families removed | `tests/test_cli_dry_run.py` asserts removed commands do not parse, including `species ...`, `bioclip screen`, `apply-rules`, `compact-parquet`, `gc-cache`, and low-level public registry commands. | Implemented |
| Production storage/workstore defaults | `src/biominer/config/__init__.py` defaults to `StorageConfig.backend = "s3"` and `WorkStoreConfig.backend = "postgres"`; `tests/test_storage_config.py` covers defaults and explicit local overrides. | Implemented |
| Local filesystem/SQLite are explicit dev/test overrides | `src/biominer/cli.py` validates `--storage-backend local --workstore-backend sqlite` as a paired local mode; mixed local/cloud modes fail. | Implemented |
| Broad multilingual seed production path removed | `tests/test_query_planner.py` asserts legacy broad seed helpers are not exported and that `query_planner.py` contains no legacy broad-seed planner symbols; `MetadataPollState` has no `ensure_seed_work_items`; `tests/test_metadata_poller.py` verifies an empty state does not seed broad probes. | Implemented |
| T5 generated translations are enabled as accepted registry name evidence and enter Flickr retrieval | `tests/test_registry_enrichment.py` verifies T5 translations enter `names.parquet`, produce normal query definitions, and expose `enabled_t5_name_rows` / `t5_query_definition_rows` manifest counters; `tests/test_query_planner.py`, `tests/test_production_run_skeleton.py`, and `tests/test_metadata_poller.py` verify T5 query definitions become Flickr retrieval work and API search params. | Implemented |
| Metadata keyword logic is flags, not hard pre-visual drop | `src/biominer/filter/metadata_flags.py` is retained; `src/biominer/filter/rules.py` is removed; `tests/test_metadata_flags.py`, `tests/test_evidence_rules.py`, and `tests/test_image_triage.py` cover soft metadata behavior. | Implemented |
| Object evidence buckets are the rule engine | `src/biominer/evidence/buckets.py` and `src/biominer/evidence/join.py` are retained; legacy `apply-rules` no longer parses. | Implemented |
| YOLOE/YOLO26 are object proposal backends only | `src/biominer/detection/yoloe26_detector.py` and `src/biominer/detection/yolo26_detector.py` map detector prompts/classes to coarse labels; detector tests cover the coarse-label contract. | Implemented |
| Reviewed YOLOE/YOLO26 boxes are not stored for later training | `tests/test_detection_pipeline.py::test_detection_and_run_sources_do_not_create_reviewed_box_training_artifacts` scans detection, run, object-scoring, and CLI source for reviewed-box/training artifact paths. | Implemented |
| BioCLIP remains the species scorer | `src/biominer/bioclip/object_runner.py` is used for object scoring; `tests/test_object_bioclip_pipeline.py` and production run tests cover object scoring and joined evidence. | Implemented |
| Third visual mode is segmentation, not enhancement | `tests/test_object_bioclip_pipeline.py::test_object_visual_modes_are_segmentation_not_enhancement` asserts the production mode tuple is `whole_image`, `detector_crop`, `detector_crop_segmentation` and scans core scoring/CLI sources for enhancement-mode terms; production scoring records unavailable segmentation when masks are absent. | Implemented |
| Generated artifacts are not tracked | `.gitignore` excludes caches, virtualenvs, Parquet, SQLite, model weights, and `__pycache__/`; `git ls-files '*__pycache__*' '*.pyc' '.venv*' 'data/*' 'staging/*' '*.sqlite' '*.parquet' '*.pt' '*.safetensors'` returned no tracked generated artifacts. | Implemented |

## Retained Non-Public Compatibility

These files remain because current code or tests still use them, but they are
not public production commands:

```text
src/biominer/bioclip/register_runner.py
src/biominer/bioclip/bioclip.py compatibility output columns
src/biominer/storage/compaction.py internal immutable shard compaction helpers
src/biominer/species/context.py shared SpeciesContext data model
src/biominer/flickr_fetch/metadata_poller.py legacy query-hit migration fallback
```

Current command help confirms the related old commands are not public.

## Verification Commands

Latest verification for this audit used:

```bash
uv run --extra test biominer --help
uv run --extra test biominer registry --help
uv run --extra test biominer run --help
uv run --extra test biominer vision --help
uv run --extra test biominer evidence --help
uv run --extra test biominer storage --help
uv run --extra test biominer workstore --help
uv run --extra test biominer dev registry --help
uv run --extra test biominer dev vision --help
git ls-files src/biominer/config/keywords.py src/biominer/filter/rules.py src/biominer/reports/name_evidence.py src/biominer/detection/yolo_detector.py src/biominer/species/registry_refresh.py src/biominer/species/query_compile.py src/biominer/species/workflow.py tests/test_species_workflow.py tests/test_name_evidence_report.py
git ls-files '*__pycache__*' '*.pyc' '.venv*' 'data/*' 'staging/*' '*.sqlite' '*.parquet' '*.pt' '*.safetensors'
```

The removed-file checks returned no tracked paths.

Latest full-suite result:

```text
uv run --extra test pytest -q
528 passed
```

## Remaining Caution

- The branch named in the original plan was superseded by later operator
  instruction: current cleanup commits are on `main`.
- Real S3/Postgres, Flickr, YOLOE/YOLO26, and BioCLIP model work is intentionally
  not required by unit tests. The current test suite uses fakes for those
  external systems.
- Before marking the active goal complete, rerun the full suite and recheck
  command help from the current `HEAD`, because this report is point-in-time
  evidence.
