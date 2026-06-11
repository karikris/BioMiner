# Code Cleanup Report

Generated: 2026-06-07T11:11:00.135875+00:00

## Phase 2: Metadata Anti-Keyword Filter

- Files changed: `src/biominer/filter/`, `src/biominer/cli.py`, filter/evidence tests, and this cleanup report.
- Tests added or updated: added anti-keyword tests for dropping non-biodiversity records, keeping butterfly life stages, grouped JSON loading, and kept/dropped parquet outputs.
- Redundant code removed: none; this phase adds the previously missing explicit filter stage.
- Compatibility shims retained: none.
- Generated artifacts excluded: no operator keyword lists, live-run data, parquet outputs, caches, images, model weights, or virtual environments were added.
- Remaining cleanup recommendation: replace any future hard-coded anti terms with operator-provided JSON fixtures or config files.

## Phase 1: Flickr Fetch Restructure

- Files changed: `src/biominer/flickr_fetch/`, `src/biominer/cli.py`, fetch/rate-limit/CLI tests, README, and this cleanup report.
- Tests added or updated: added coverage for stale claimed-work recovery and parallel `poll_once` workers; updated the Flickr API soft cap to 3,500.
- Redundant code removed: replaced serial fetch-only execution with one bounded worker-pool path and routed evidence parquet writes through shared storage.
- Compatibility shims retained: none.
- Generated artifacts excluded: no raw Flickr payloads, parquet outputs, caches, images, model weights, or virtual environments were added.
- Remaining cleanup recommendation: move the standalone `FlickrRateLimiter` to `biominer.common` if non-Flickr API stages begin sharing it.

## Phase 0: BioMiner Identity And Namespace

- Files changed: `pyproject.toml`, `README.md`, `src/biominer/`, tests, and this cleanup report.
- Tests added or updated: added `tests/test_project_identity.py`; updated imports and CLI expectations for the `biominer` namespace.
- Redundant code removed: removed the tracked `src/flickr_bio_occurrence` package files after moving active modules under `src/biominer`.
- Compatibility shims retained: none; the active package and CLI identity are now BioMiner-specific.
- Generated artifacts excluded: no live-run data, parquet outputs, caches, images, model weights, or virtual environments were added.
- Remaining cleanup recommendation: remove stale local egg-info/cache directories if they appear as untracked artifacts after editable installs.

## README Repository Rewrite

- Files changed: `README.md` and this cleanup report.
- Tests added or updated: none; this was a documentation-only change.
- Redundant code removed: none; no code paths were changed.
- Compatibility shims retained: existing compatibility-only code remains documented as out of scope for the active workflow.
- Generated artifacts excluded: no generated data, cache, image, parquet, DuckDB, report output, or model files were added.
- Remaining cleanup recommendation: keep README command examples aligned with the public CLI when a register-runner CLI wrapper is added.

## Geo-First Flickr Pagination

- Files changed: `config/pipeline.toml`, `src/flickr_bio_occurrence/flickr/query_planner.py`, `src/flickr_bio_occurrence/cli.py`, and focused planner/poller/CLI tests.
- Tests added or updated: geo page-size selection, `has_geo` propagation, Papilio query-plan CLI reporting, poller page enqueue counts, and dry-run planned record counts.
- Redundant code removed: removed `url_o` from active pipeline config extras; no parallel legacy pagination path was retained.
- Compatibility shims retained: non-geo Flickr search can still use `per_page=500`, while geotagged/bbox Papilio discovery uses `per_page=250` to match Flickr `photos.search` behaviour.
- Generated artifacts excluded: raw Flickr data, local config helper files, caches, virtual environments, image files, parquet, and DuckDB outputs remain ignored.
- Remaining cleanup recommendation: keep the ignored local Papilio keyword JSON as an operator input unless it is deliberately converted into a reviewed, tracked seed asset.

## Persistent BioCLIP Register Runner

- Files changed: `src/flickr_bio_occurrence/vision/bioclip.py`, `src/flickr_bio_occurrence/vision/bioclip_worker.py`, `src/flickr_bio_occurrence/vision/triage.py`, `src/flickr_bio_occurrence/evidence/rules.py`, plus new register-runner and species-candidate modules.
- Tests added or updated: multi-label BioCLIP request protocol, persistent label-set scoring, global species-candidate loading, four-register image staging, immediate image deletion, other-species Bronze routing, and generic butterfly non-Gold routing.
- Redundant code removed: no legacy runner was removed because the previous full-run controller lived in `/tmp`, not tracked repo code; the tracked single-image and legacy batch helpers are retained for tested compatibility.
- Compatibility shims retained: `DEFAULT_BIOCLIP_LABELS` and single-label `classify_images` remain available for older tests and callers, while the new runner uses named `species` and `triage` label sets.
- Generated artifacts excluded: global species-candidate parquet, raw Flickr metadata, BioCLIP outputs, temporary images, Hugging Face/model cache, and visual reports remain ignored.
- Remaining cleanup recommendation: once the register runner is the only active production path, remove or explicitly mark older one-record triage helpers as compatibility-only.

## Batched BioCLIP Image Encoding

- Files changed: `src/flickr_bio_occurrence/vision/bioclip_worker.py` and `tests/test_bioclip_worker_batching.py`.
- Tests added or updated: local fake-worker test proving a multi-image label-set request stacks all images and calls `encode_image` once for the batch.
- Redundant code removed: replaced per-image `encode_image` loops with tensor-batched image encoding for both single-label and named-label-set scoring.
- Compatibility shims retained: worker JSON request and response shapes are unchanged for existing single-label and persistent callers.
- Generated artifacts excluded: full-run parquet outputs, temporary images, model cache, and candidate parquet remain ignored.
- Remaining cleanup recommendation: add streaming checkpoints to the register runner before using it for very large production runs where mid-run recovery matters.

## Papilio Demoleus BioCLIP Visual Report

- Files changed: `scripts/generate_bioclip_species_visual_report.py`, `src/flickr_bio_occurrence/vision/triage.py`, `src/flickr_bio_occurrence/evidence/rules.py`, focused tests, and this cleanup report.
- Tests added or updated: focused local tests for top-k species-per-image counting, numeric summary statistics, report filters, and the `below_50` Bronze reason; no network, Flickr credentials, CUDA, model weights, or real parquet fixture required.
- Redundant code removed: replaced the narrower visual-report generator with one report path that writes the HTML report, PDF deck, PNG figures, CSV tables, JSON summary, optional filtered parquet, and bin-reason diagnostics from the same dataframe; replaced future `not_target_species` output with `below_50`.
- Compatibility shims retained: none.
- Generated artifacts excluded: the generated visual report outputs under `data/live_runs/.../visual_report*/` are run artifacts and should not be committed as source.
- Remaining cleanup recommendation: if this report becomes part of routine operations, add a CLI entry point and document optional plotting dependencies in a report/dev extras group.

## Species Candidate Loader

- Files changed: `src/flickr_bio_occurrence/vision/species_candidates.py`, `tests/test_species_candidates.py`, and this cleanup report.
- Tests added or updated: local CSV/parquet candidate-loading tests covering target-species pinning, species-only filtering, dedupe, and BioCLIP label generation.
- Redundant code removed: none; no tracked equivalent species-candidate loader existed.
- Compatibility shims retained: none.
- Generated artifacts excluded: candidate parquet inputs remain data artifacts and are not committed as source.
- Remaining cleanup recommendation: wire the loader into a CLI or runner entry point once the register runner is committed.

## BioCLIP Worker Label-Set Batching Test

- Files changed: `tests/test_bioclip_worker_batching.py` and this cleanup report.
- Tests added or updated: fake-model worker test proving label-set scoring stacks multiple images and calls `encode_image` once for the batch.
- Redundant code removed: none; this adds coverage for the existing batched worker path.
- Compatibility shims retained: none.
- Generated artifacts excluded: no generated artifacts created.
- Remaining cleanup recommendation: keep this as a lightweight guard against accidental per-image encoding regressions.

## BioCLIP Register Runner

- Files changed: `src/flickr_bio_occurrence/vision/register_runner.py` and this cleanup report.
- Tests added or updated: existing register-runner test was run before commit; the test itself is committed in the next phase.
- Redundant code removed: removed redundant final cleanup pass from the draft runner and replaced ambiguous staged-image accounting with an explicit counter.
- Compatibility shims retained: none after cleanup; the register runner is the only image download/classification runner.
- Generated artifacts excluded: no generated parquet, images, model cache, or reports were committed.
- Remaining cleanup recommendation: add a public CLI wrapper once the register runner is ready for routine operation.

## Phase 1 Codebase Minimization

- Files changed: CLI, dry-run/work-item helpers, triage tests, config tests, and this cleanup report.
- Tests added or updated: rewrote CLI and image-triage tests around the surviving lean pipeline; removed tests for deleted benchmark, Darwin Core, queue/service, storage, geo, and legacy vision paths.
- Redundant code removed: deleted Darwin Core export/mapping, benchmark runners/estimators/checkpoints, DuckDB/parquet storage helpers, geo/taxonomy helpers, queue/sharded-fetch/service code, legacy single-image vision pipeline, static report-pack generator, and standalone fetch/post-fetch scripts.
- Compatibility shims retained: none.
- Generated artifacts excluded: no generated data, cache, image, parquet, DuckDB, or model files were added.
- Remaining cleanup recommendation: run a follow-up import audit after the full suite passes and remove any now-unused report JSON fixtures if they are not consumed.

## Phase 2 Codebase Minimization

- Files changed: CLI, `src/flickr_bio_occurrence/pipeline/metadata_poller.py`, CLI tests, and this cleanup report.
- Tests added or updated: CLI command-surface test now asserts removed `fetch`, `fetch-live`, and benchmark commands are absent.
- Redundant code removed: deleted old monthly dry-run planning, dashboard butterfly-term loading, `FlickrClient`, and `WorkItem` helpers that duplicated the active metadata poller/query planner path.
- Compatibility shims retained: none; raw-response slugging is local to the metadata poller.
- Generated artifacts excluded: no generated artifacts were created.
- Remaining cleanup recommendation: inspect report fixture files for stale references to removed benchmark/DWC concepts.

## Phase 3 Codebase Minimization

- Files changed: this cleanup report.
- Tests added or updated: removed tests for deleted image-selection and prefetch helpers.
- Redundant code removed: deleted unused `vision/image_selection.py`, `vision/prefetch.py`, and empty review package files.
- Compatibility shims retained: none.
- Generated artifacts excluded: no generated artifacts were created.
- Remaining cleanup recommendation: keep URL choice in metadata/evidence input and avoid adding a second image-selection layer.

## Phase 4 Codebase Minimization

- Files changed: config asset tests and this cleanup report.
- Tests added or updated: config tests now require only `.env.example` and ignore rules for local operator inputs.
- Redundant code removed: deleted unused tracked config fixtures and static generated report JSON/Markdown artifacts.
- Compatibility shims retained: none.
- Generated artifacts excluded: only `reports/code_cleanup_report.md` remains tracked under reports.
- Remaining cleanup recommendation: generate operational reports from current runs rather than committing placeholder report artifacts.

## BioCLIP Register Runner Tests

- Files changed: `tests/test_register_runner.py` and this cleanup report.
- Tests added or updated: local fake-cache/fake-classifier tests for register sizing, temporary image deletion, Gold routing, other-species Bronze routing, and staged-image bounds.
- Redundant code removed: none.
- Compatibility shims retained: none.
- Generated artifacts excluded: test parquet and cached images are created under pytest temporary directories only.
- Remaining cleanup recommendation: add an end-to-end CLI-level dry-run test if a public register-runner command is introduced.

## Comment Review Scope

- Added a separate targeted comment-review phase after BioCLIP triage.
- Comment review creates its own queue, results table, derived terms, and missing-data request table.
- Comment review is not part of initial Flickr `photos.search` and must not run for every record by default.
- Retained BioCLIP as screening evidence only; no report claims taxonomic validation.
- Required metrics are present in JSON reports as null or `not_instrumented` when no bounded run data is available.

## Removed Or Superseded Report Paths

- Superseded `bioclip_run_summary.*`, `quality_profile.json`, `image_triage_profile.json`, `cache_profile.json`, `gpu_profile.json`, and `idempotency_profile.json` in the active report pack.
- Removed stale report text that described dedicated comments fetching as unavailable; comments enrichment now exists for selected candidates.

## Compatibility-Only Code

- `src/flickr_bio_occurrence/dwc/mapper.py` and `src/flickr_bio_occurrence/dwc/exporter.py` are retained only for existing tested public API compatibility.
- Darwin Core compatibility code must not be used as the active image-triage or occurrence-publication path in this phase.
- Removal condition: retire these shims when `tests/test_dwc_mapper.py` and any downstream public API expectations are removed or replaced.

## Still Out Of Scope

- Validated Darwin Core occurrence publication.
- `identificationVerificationStatus` expansion.
- Human verification as a gate for Gold/Silver.
- Network/CUDA/BioCLIP/Flickr-dependent tests.
