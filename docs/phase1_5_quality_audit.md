# Phase 1-5 Quality Audit

Status: initial implementation map and audit plan.
Date: 2026-07-10.
Branch: `main`.

This report tracks the post-implementation audit of the YOLOE-26 plus BioCLIP 2.5 hierarchical butterfly pipeline. It is intentionally evidence-based: each phase is mapped to code, tests, and remaining audit risk before deeper fix tasks proceed.

## Task Addendum

The older species-prompt variant aggregator is now an explicit audit-phase task:

`Fix/verify older species-prompt variant aggregation: taxon ranking must use the mean over all prompt templates by default, not the best single prompt. Preserve best-label evidence only as audit metadata.`

Initial finding: the current checkout already contains this fix in `src/biominer/bioclip/prompt_templates.py` and `src/biominer/bioclip/bioclip.py`, introduced in commit `a6c3ca3 step3: default species prompt aggregation to mean`. Focused regression tests pass. The older historical superpowers plan still contains max-based example text and must not be treated as the current contract.

## Evidence Collected

GitHits:

- `github:kubernetes/community`, docs search for architecture audit traceability matrix: indexing/no usable hits.
- `github:open-mmlab/mmdetection`, docs search for model evaluation audit checklist: indexing/no usable hits.
- `pypi:great-expectations`, docs search for validation checklist data pipeline audit report: indexing/no usable hits.
- `github:openai/clip`, code search for prompt ensembling and mean class embeddings: returned unrelated model internals, no usable exact prompt-ensemble implementation.
- `github:python-pillow/Pillow` and `github:ultralytics/ultralytics`, code search for crop/resize/LANCZOS patterns: returned relevant Pillow `thumbnail`, `ImageOps.pad`/`fit`, and Ultralytics crop/scale references after indexing completed.
- `get_example` for CLIP prompt ensembling was unavailable because the daily generated-example limit was reached.

External primary docs:

- Pillow `Image.resize(...)` documentation confirms `Resampling.LANCZOS` is a supported resampling filter and that `box` can provide a float source region: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize

Morph:

- Tool discovery found Morph MCP.
- `codebase_search` for pipeline tracing returned HTTP 429. Morph is unavailable for this initial map; local `rg`/focused reads were used instead.

Headroom:

- Headroom skill and MCP tools are available. Later large or repetitive outputs should be compressed before reasoning. Initial searches after loading Headroom were kept narrow enough not to require compression.

Commands run:

- `git status --short`
- `git log --oneline --decorate -15`
- focused `rg` searches for classification modes, prompt aggregation, stale markers, heavy imports, and docs inventory
- focused `sed` reads of BioCLIP, registry, detection, run, and evaluation modules
- `uv run pytest -q tests/test_prompt_templates.py tests/test_bioclip_prediction.py::test_bioclip_classifier_ranks_species_prompt_variants_by_mean_not_best_prompt tests/test_phase4_m5pro_acceptance.py::test_phase4_species_prompt_variants_rank_by_mean_not_best_single_prompt`
- `uv run pytest -q tests/test_cloud_bioclip_work.py`
- `uv run pytest -q tests/test_object_bioclip_pipeline.py::test_object_bioclip_hierarchical_mode_requires_taxonomy_store tests/test_object_bioclip_pipeline.py::test_object_bioclip_scores_local_hierarchical_mode_with_fake_taxonomy tests/test_object_bioclip_pipeline.py::test_object_bioclip_hierarchical_mode_uses_taxonomy_text_embedding_cache`
- `uv run pytest -q tests/test_production_run_skeleton.py::test_orchestrator_runs_fake_hierarchical_vision_pipeline_end_to_end tests/test_production_run_skeleton.py::test_production_run_hierarchical_score_stage_requires_taxonomy_table tests/test_production_run_skeleton.py::test_production_run_hierarchical_score_stage_validates_table_then_requires_score_inputs`
- `uv run pytest -q tests/test_embedding_cache.py`
- `uv run pytest -q tests/test_hierarchical_classifier.py::test_hierarchical_batch_uses_taxonomy_text_embedding_cache_for_species_first_pass tests/test_hierarchical_classifier.py::test_hierarchical_batch_taxonomy_text_embedding_cache_rejects_model_mismatch tests/test_hierarchical_classifier.py::test_hierarchical_batch_taxonomy_text_embedding_cache_rejects_missing_labels tests/test_hierarchical_classifier.py::test_hierarchical_batch_taxonomy_text_embedding_cache_rejects_stale_label_hash tests/test_hierarchical_classifier.py::test_rank_species_with_cached_text_embeddings_rejects_mixed_model_cache`
- `uv run pytest -q tests/test_detection_pipeline.py`
- `uv run pytest -q tests/test_object_bioclip_pipeline.py::test_materialized_detector_crop_batches_default_to_24_and_clean_between_batches tests/test_object_bioclip_pipeline.py::test_materialized_detector_crop_batches_skip_noneligible_without_image_load tests/test_object_bioclip_pipeline.py::test_materialized_detector_crop_batches_reuse_duplicate_crop_hash_within_batch tests/test_object_bioclip_pipeline.py::test_materialized_detector_crop_batches_retain_debug_crops_when_requested tests/test_object_bioclip_pipeline.py::test_screen_object_detections_reuses_materialized_detector_crop_paths_and_cleans_after_success tests/test_object_bioclip_pipeline.py::test_screen_object_detections_keeps_materialized_detector_crops_after_scorer_error tests/test_object_bioclip_pipeline.py::test_screen_object_detections_keeps_materialized_detector_crops_after_parquet_commit_failure tests/test_object_bioclip_pipeline.py::test_screen_object_detections_retains_materialized_detector_crops_when_debug_requested tests/test_object_bioclip_pipeline.py::test_screen_object_detections_materialized_path_skips_noneligible_without_image_load`
- `uv run pytest -q tests/test_cloud_detection_work.py`
- `uv run ruff check src/biominer/bioclip/cloud_work.py tests/test_cloud_bioclip_work.py`

Focused test result:

- `5 passed in 0.53s`
- `tests/test_cloud_bioclip_work.py`: `12 passed`
- local hierarchical object tests: `3 passed`
- production hierarchical skeleton tests: `3 passed`
- `tests/test_embedding_cache.py`: `13 passed`
- cached hierarchical classifier tests: `5 passed`
- `tests/test_detection_pipeline.py`: `41 passed`
- materialized object crop tests: `9 passed`
- `tests/test_cloud_detection_work.py`: `6 passed`
- Ruff could not run because the `ruff` executable is not installed in this environment.

## Files Inspected

- `README.md`
- `pyproject.toml`
- `src/biominer/cli.py`
- `src/biominer/bioclip/bioclip.py`
- `src/biominer/bioclip/classification_modes.py`
- `src/biominer/bioclip/cloud_work.py`
- `src/biominer/bioclip/hierarchical_classifier.py`
- `src/biominer/bioclip/object_runner.py`
- `src/biominer/bioclip/prompt_templates.py`
- `src/biominer/detection/pipeline.py`
- `src/biominer/detection/cropper.py`
- `src/biominer/detection/policy.py`
- `src/biominer/registry/build.py`
- `src/biominer/registry/classification_table.py`
- `src/biominer/run/manifest.py`
- `src/biominer/run/orchestrator.py`
- `src/biominer/evaluation/labels.py`
- `src/biominer/evaluation/metrics.py`
- `src/biominer/evaluation/qa.py`
- `src/biominer/evaluation/reports.py`
- `src/biominer/evaluation/review_queue.py`
- `src/biominer/evaluation/xie_style.py`
- `tests/test_bioclip_prediction.py`
- `tests/test_cloud_bioclip_work.py`
- `tests/test_embedding_cache.py`
- `tests/test_hierarchical_classifier.py`
- `tests/test_hierarchical_golden_evaluation.py`
- `tests/test_object_bioclip_pipeline.py`
- `tests/test_phase4_m5pro_acceptance.py`
- `tests/test_prompt_templates.py`
- `tests/test_production_run_skeleton.py`
- `tests/test_registry_classification_table.py`

## Phase Map

### Phase 1: Classifier Modes And Target Separation

Implemented evidence:

- `classification_modes.py` defines `target_scope_object_screening` and `hierarchical_butterfly_classification`.
- `DEFAULT_CLASSIFICATION_MODE` remains `target_scope_object_screening`.
- CLI and run plumbing expose `--classification-mode`, family/species top-k values, taxonomy candidate table, and optional taxonomy text embedding cache.
- `object_runner.py` rejects hierarchical classification through target-scope-only helpers and requires a taxonomy store for hierarchical mode.

Tests located:

- `tests/test_bioclip_classification_modes.py`
- `tests/test_object_bioclip_pipeline.py`
- `tests/test_production_run_skeleton.py`
- `tests/test_cloud_bioclip_work.py`
- `tests/test_phase4_m5pro_acceptance.py`

Remaining audit risk:

- Later tasks must compare every local/cloud path to ensure no helper silently falls back from hierarchical mode to target-scope screening.

### Phase 2: GBIF Classification Tables

Implemented evidence:

- `registry/classification_table.py` builds `butterfly_classification_taxa.parquet`, `butterfly_family_labels.parquet`, `butterfly_species_labels.parquet`, manifest, and QA findings.
- The table is derived from the accepted registry and is not a second taxonomy authority.
- Family prompts include butterfly context, for example `a photo of a butterfly in the family {family}`.
- `ButterflyTaxonomyStore` loads, validates, and serves family/species label rows plus lookup caches.

Tests located:

- `tests/test_registry_classification_table.py`
- `tests/test_registry_build.py`
- `tests/test_embedding_cache.py`

Remaining audit risk:

- Later tasks must check schema parity, manifest completeness, and stale-cache failure behavior across CLI/run paths.

### Phase 3: Hierarchical Family-First Classification

Implemented evidence:

- `hierarchical_classifier.py` scores family prompts, selects the top family, scores species only inside that family, records species top 20, and reranks all first-pass top-20 candidates into top 5.
- `aggregate_taxon_prompt_scores(...)` defaults to `mean`.
- Cached text-embedding ranking groups prompt labels by accepted taxon and mean-aggregates prompt-template similarities.
- `hierarchical_result_to_object_score_row(...)` writes conservative review-oriented rows with target fields null and `is_target_positive=False`.

Tests located:

- `tests/test_hierarchical_classifier.py`
- `tests/test_hierarchical_golden_evaluation.py`
- `tests/test_object_bioclip_pipeline.py`
- `tests/test_cloud_bioclip_work.py`

Remaining audit risk:

- Later tasks must stress cached and uncached ranking equivalence, output schema stability, and cloud scoring parity.

### Phase 4: YOLOE-26, Mac Profile, Crops, And Work Keys

Implemented evidence:

- `detection/policy.py` defines `mac_m5pro_64gb` settings: `mps`, `yoloe-26s-seg.pt`, `imgsz=768`, detector batch 16, crop batch 24, crop target 336, crop padding 0.08, zstd output, and delete-after-commit.
- `detection/pipeline.py` skips crop metadata generation for non-detected and non-eligible objects unless debug crop writing is explicit.
- `bioclip/cloud_work.py` includes crop, model, checkpoint, candidate set, classification mode, taxonomy table, prompt version, top-k, and ablation settings in score work keys.
- Heavy vision dependencies are not top-level project dependencies and are imported lazily in adapter/runtime paths.

Tests located:

- `tests/test_phase4_m5pro_acceptance.py`
- `tests/test_detection_pipeline.py`
- `tests/test_cloud_detection_work.py`
- `tests/test_vision_stage_reports.py`
- `tests/test_vision_plumbing_benchmark.py`

Remaining audit risk:

- Later tasks must inspect every prototype/dev command for hardcoded crop defaults and verify resize quality behavior.

### Phase 5: Evaluation, Review Queues, QA, And Reports

Implemented evidence:

- `evaluation/labels.py`, `metrics.py`, `calibration.py`, `reports.py`, `charts.py`, `review_queue.py`, `qa.py`, and `xie_style.py` implement model-free evaluation and QA plumbing.
- `run/orchestrator.py` emits hierarchical review queues and visual QA findings during summarize stages.
- Xie-style is documented and implemented as a metrics profile, not a production architecture change.

Tests located:

- `tests/test_evaluation_*.py`
- `tests/test_hierarchical_golden_evaluation.py`
- `tests/test_production_run_skeleton.py`

Remaining audit risk:

- Later tasks must verify report inputs across local filesystem and storage-backed paths and confirm docs/help parity.

## Compatibility Risks Found So Far

- GitHits and Morph evidence were limited by tool availability, so Task 1 depends primarily on local source and tests.
- Historical plan docs still describe an older max-based species prompt aggregation task. Current code and tests disagree with that stale historical text.
- `src/biominer/registry/build.py` still contains an explicit `cloud_registry_enrichment_not_implemented` path. This may be acceptable if out of scope, but later audit tasks must confirm no production phase 1-5 command depends on it.
- CLI contains lazy heavy imports for live benchmark/dev commands. Later audit tasks must confirm these do not break dry-run/help/import-only use.
- The initial map did not fully verify crop resize quality or all docs/help parity.
- Cloud BioCLIP work items store output-affecting scoring settings in the payload and work key. Before Task 2, the cloud worker accepted independent batch-level mode and top-k arguments without validating them against the payload. That could silently score replayed work under stale semantics.
- Before Task 3, `prepare_taxonomy_text_embedding_cache(... embedding_dtype=...)` could silently reuse a taxonomy text cache generated with a different recorded dtype. Runtime validation checked dtype presence and consistency, but not the requested dtype.
- Before Task 4, BioCLIP detector crops still used custom nearest-neighbor resize in `crop_with_padding(...)`. `_resize_image_to_max_side(...)` already used Pillow/LANCZOS, but the crop bytes fed to BioCLIP did not.

## Fixes Made Or Verified So Far

- Added this audit report.
- Added the species-prompt mean aggregation bug to the current phase task list.
- Verified the older species-prompt variant aggregator ranks taxa by mean by default:
  - `SPECIES_PROMPT_AGGREGATION_DEFAULT == "mean"`
  - `aggregate_prompt_scores(...)` writes the aggregated mean to `score`
  - `BioClipClassifier.classify_images_with_label_sets(...)` ranks species prompt variants using aggregated rows
  - focused regression tests pass
- Fixed cloud BioCLIP work contract validation:
  - cloud score workers now reject work-item classification-mode drift
  - cloud score workers now reject top-k settings drift
  - hierarchical cloud scoring validates payload taxonomy table and prompt-variant versions against the supplied taxonomy store
  - existing hierarchical cloud tests now create work items with top-k settings matching the score run
- Fixed taxonomy text embedding cache dtype invalidation:
  - taxonomy cache preparation validates the final reused/appended cache before returning
  - callers may pass an expected `embedding_dtype` to validation
  - reusing a float16 taxonomy cache for a float32 preparation request now fails clearly
- Fixed detector crop resize quality:
  - `crop_with_padding(...)` now uses Pillow `Image.resize(..., Resampling.LANCZOS, box=...)` when Pillow is available
  - the fallback nearest-neighbor byte resize remains for environments without Pillow
  - float padded crop boxes are preserved so adjacent detections do not collapse to identical integer crop boxes
  - cropper tests now assert LANCZOS interpolation when Pillow is available

## Core Invariant Checklist

Initial status is not the final audit verdict. `Supported` means evidence was located in code/tests during Task 1; later tasks may still tighten, fix, or revise it.

| # | Invariant | Initial status |
| -: | - | - |
| 1 | Target-scope screening remains the default classification mode. | Supported |
| 2 | Hierarchical mode is open classification, not target validation. | Supported |
| 3 | Hierarchical mode scores families before species. | Supported |
| 4 | Species top-20 is constrained by selected family top-1. | Supported |
| 5 | Reranking covers all first-pass top-20 species. | Supported |
| 6 | Hierarchical classification does not inject the run target species. | Supported |
| 7 | Family prompt variants are mean-aggregated by taxon by default. | Supported |
| 8 | Older species-prompt variants are mean-aggregated by taxon by default, not ranked by best single prompt. | Supported and focused-tested |
| 9 | GBIF-backed classification tables are derived candidate artifacts, not taxonomy authority. | Supported |
| 10 | Classification table outputs include taxa, family labels, species labels, manifest, and QA. | Supported |
| 11 | Taxonomy store validates classification table inputs before hierarchical scoring. | Supported |
| 12 | Optional text embedding caches validate taxonomy/model/prompt metadata. | Supported and dtype invalidation fixed |
| 13 | Mac M5 Pro profile settings flow into vision runtime policy. | Supported |
| 14 | Non-eligible detections retain metadata and are not cropped in production paths. | Supported and focused-tested |
| 15 | Crop/image lifecycle deletes staged files only after successful score writes. | Supported, deeper audit pending |
| 16 | Work keys include output-affecting detector, crop, model, taxonomy, mode, prompt, and top-k settings. | Supported and payload contract fixed |
| 17 | Heavy vision runtimes are optional/lazy and not required for model-free tests. | Supported, deeper audit pending |
| 18 | Evaluation/reporting writes review queues and visual QA findings without claiming validated occurrences. | Supported |
| 19 | Xie-style behavior is metrics-only and does not score every image with BioCLIP. | Supported |
| 20 | BioCLIP output remains screening evidence, not taxonomic validation or verified Darwin Core occurrence evidence. | Supported |

## Next Audit Tasks

- Continue tracing local/cloud hierarchical output schemas beyond the fixed work-payload contract.
- Continue classification table manifest/QA review beyond the fixed embedding-cache dtype invalidation.
- Continue docs/help review for crop/profile defaults beyond the fixed crop resize path.
- Audit CLI help, docs, examples, and deprecated-command docs against parser behavior.
- Run focused model-free tests for each fix, then full `pytest -q` before final push.
