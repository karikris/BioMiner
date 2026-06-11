# Code Cleanup Report

Generated: 2026-06-07T11:11:00.135875+00:00

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
