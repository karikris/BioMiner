# BioMiner Removed Legacy Workflow Paths

Recorded: 2026-07-05

Branch used for current cleanup commits: `main`

Latest audited base before this report edit: `36ec19b`

## Removed Files

Recently removed workflow modules and tests:

```text
src/biominer/config/keywords.py
src/biominer/filter/rules.py
src/biominer/reports/buckets.py
src/biominer/reports/name_evidence.py
src/biominer/detection/yolo_detector.py
src/biominer/species/registry_refresh.py
src/biominer/species/query_compile.py
src/biominer/species/workflow.py
tests/test_species_workflow.py
tests/test_name_evidence_report.py
```

Older cleanup history also removed the pre-BioMiner package namespace and obsolete benchmark/publication/storage paths under `src/flickr_bio_occurrence/`.

## Removed Or Demoted Commands

Removed from the public command surface:

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
biominer registry fetch-taxonomy
biominer registry compile-fixture
biominer registry compile-enriched
biominer registry enrich-sources
biominer registry seed-flickr-queries
```

Demoted to dev/debug:

```text
biominer dev registry fetch-taxonomy
biominer dev registry compile-fixture
biominer dev registry compile-enriched
biominer dev registry enrich-sources
biominer dev registry seed-flickr-queries
biominer dev flickr poll-once
biominer dev comments fetch
biominer dev comments queue
biominer dev comments review-once
biominer dev comments apply-decisions
```

## Workflow Replacements

Legacy path:

```text
species run / species aliases / bioclip screen / apply-rules
```

Replacement:

```text
biominer run --taxon <name> --rank auto|family|genus|species ...
biominer vision detect
biominer vision score
biominer evidence join
```

Legacy metadata hard-drop path:

```text
filter/rules.py
apply-rules
metadata text-list path helpers
```

Replacement:

```text
filter/metadata_flags.py
evidence/buckets.py
evidence/join.py
```

Legacy detector path:

```text
detection/yolo_detector.py
YOLOv8 default backend
```

Replacement:

```text
detection/yoloe26_detector.py
detection/yolo26_detector.py
detection/detector_base.py coarse-label contract
```

Legacy broad seed path:

```text
MULTILINGUAL_SEED_TERMS
multilingual_seed_terms()
build_count_probes()
build_worldwide_discovery_plan()
config/papilio_demoleus_multilingual_keywords.json
src/biominer/config/keywords.py
```

Replacement:

```text
registry compiler output: flickr_query_definitions.parquet
MetadataPollState.enqueue_initial_work_items(explicit_registry_queries)
biominer dev registry seed-flickr-queries --query-definitions ...
```

Production polling no longer creates broad Flickr work items from an empty state.
Production run polling now uses the validated runtime worker ID from config rather than an ambient fallback.
Production Flickr work items preserve the compiled query `trust_tier`, including retrieval-only
T5 generated-translation terms, so query provenance remains auditable at API polling time.
Production BioCLIP scoring now emits combined object score outputs for available visual modes and records
`detector_crop_segmentation` as unavailable when masks are absent, instead of treating the third visual mode
as an enhancement or silently dropping it from manifest metrics.

## Tests Removed Or Rewritten

Tests now assert removed public commands do not parse:

```text
tests/test_cli_dry_run.py
tests/test_compaction.py
tests/test_evidence_rules.py
tests/test_query_planner.py
tests/test_metadata_flags.py
```

Tests cover replacement behavior:

```text
tests/test_production_run_skeleton.py
tests/test_metadata_poller.py
tests/test_object_bioclip_pipeline.py
tests/test_detection_pipeline.py
tests/test_yoloe26_detector.py
tests/test_yolo26_detector.py
tests/test_registry_enrichment.py
tests/test_storage_config.py
tests/test_workstore.py
```

Current full-suite result after the last code cleanup:

```text
uv run --extra test pytest -q
524 passed
```

The latest command-surface and workflow cleanup was additionally checked with:

```text
uv run --extra test biominer --help
uv run --extra test biominer run --help
uv run --extra test biominer registry --help
uv run --extra test biominer vision --help
uv run --extra test biominer evidence --help
```

## Remaining Non-Production Compatibility Surfaces

These are intentionally retained for internal implementation, tests, or debug workflows:

```text
src/biominer/bioclip/register_runner.py
src/biominer/bioclip/bioclip.py compatibility output columns
src/biominer/storage/compaction.py internal immutable shard compaction helpers
src/biominer/species/context.py shared SpeciesContext data model
src/biominer/flickr_fetch/metadata_poller.py legacy query-hit migration fallback
```

These are not public production commands. The public production entry point is `biominer run`, and registry-backed species context resolution now lives inside `run.taxon_scope`.

## Known Follow-Up Tasks

- Consider moving `SpeciesContext` from `biominer.species.context` into `biominer.run` or a neutral common schema module if the package name remains confusing after the public `species` CLI removal.
- Continue exercising the cloud-backed production stages with fake S3/Postgres fixtures as new stage behavior is added.
- Keep `bioclip/register_runner.py` non-public unless a later benchmark/debug command needs it explicitly.
- Keep top-level README examples focused on `biominer run`; local stage examples belong in docs and examples under explicit debug paths.

## MCP Limitations Encountered

Recorded in `reports/refactor_mcp_environment.md`:

```text
Morph MCP was unavailable/rate-limited during this cleanup and was recorded once.
GitHub and local shell tooling were used for repo inspection, commits, pushes, and verification.
```
