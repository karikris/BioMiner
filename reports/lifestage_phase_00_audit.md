# Phase 0 Lifestage Audit

This audit is based on the current working tree. I did not inspect large data folders, image caches, model weights, generated parquet, DuckDB files, or virtual environments.

The working tree already contains pre-existing uncommitted Phase 8 implementation/test edits in:

- `scripts/generate_report_pack.py`
- `src/flickr_bio_occurrence/evidence/rules.py`
- `tests/test_evidence_rules.py`
- `tests/test_report_pack.py`

Those edits are not Phase 0 report changes and should not be included in the Phase 0 commit.

## Current State

Python target is `>=3.14` in `pyproject.toml`.

Publication-state logic currently lives in `src/flickr_bio_occurrence/evidence/rules.py`, mainly `classify_evidence_row`, `review_reasons_for_evidence`, `target_signal_is_positive`, and `_negative_material_reason`. In the current working tree, the states are `gold`, `silver`, `bronze`, and `in_review`, with `publication_state`, `publication_state_reason`, and `review_reason` columns.

Image triage logic lives in `src/flickr_bio_occurrence/vision/triage.py`, mainly `process_image_triage_records`, `classify_bioclip_triage`, `_base_row`, and `_negative_reason`. It writes metadata, URL fields, geo/time fields, hashes, model metadata, top-k output, `triage_bin`, `triage_reason`, and status fields. The dedupe key is source, Flickr photo ID, image URL, model ID, model version, and model checkpoint.

`image_category` does not currently exist as a stored column in the inspected active source/test paths.

`life_stage` does not currently exist as a stored column in the inspected active source/test paths.

## Life-Stage Handling

Caterpillar handling exists but is not normalized into a `life_stage` column.

- `src/flickr_bio_occurrence/vision/triage.py` has `VERIFIED_LIFE_STAGE_LABELS` for caterpillar and pupa/chrysalis.
- `src/flickr_bio_occurrence/vision/triage.py` maps caterpillar and pupa/chrysalis BioCLIP labels to negative reasons unless `human_verification_detected` and `species_text_match` allow verified life-stage evidence.
- `src/flickr_bio_occurrence/vision/bioclip.py` includes caterpillar and pupa/chrysalis labels.
- `src/flickr_bio_occurrence/evidence/review_flags.py` currently treats `caterpillar`, `pupa`, and `chrysalis` as `NON_TARGET_ORDER_TERMS`.

Current status:

- Caterpillar: present; usually negative/bronze unless verified life-stage evidence path applies in image triage.
- Pupa/chrysalis: present; usually negative/bronze unless verified life-stage evidence path applies in image triage.
- Egg: absent from inspected BioCLIP labels, negative-label maps, and review text terms.

## Bronze Logic

Image triage Bronze is in `src/flickr_bio_occurrence/vision/triage.py::classify_bioclip_triage`. It means negative/non-occurrence visual material such as museum specimen, pinned specimen, artwork, moth, caterpillar, pupa/chrysalis, other insect, not butterfly, object/background, or AI-generated image.

Publication-state Bronze is in `src/flickr_bio_occurrence/evidence/rules.py::classify_evidence_row`. In the current working tree this is Phase 8 WIP and maps negative material to Bronze.

Legacy storage naming still uses Bronze/Silver/Gold as ETL folders:

- `src/flickr_bio_occurrence/pipeline/transforms.py`
- `src/flickr_bio_occurrence/pipeline/dry_run.py`
- `src/flickr_bio_occurrence/storage/duckdb_index.py`

## Image Deletion

Image deletion logic lives in:

- `src/flickr_bio_occurrence/vision/temp_image_store.py::cleanup_cached_image`
- `src/flickr_bio_occurrence/vision/pipeline.py::classify_bronze_photo_row`
- `src/flickr_bio_occurrence/vision/triage.py::process_image_triage_records`

Successful cached images are deleted by default after classification. Failed images are deleted by default in `classify_bronze_photo_row` unless `keep_failed_images=True`. `cleanup_cached_image` refuses to delete outside `cache_root` and refuses protected cache path parts such as `huggingface`, `.hf-cache`, `model`, and `models`.

## Flickr Query Lanes

Flickr query planning lives in `src/flickr_bio_occurrence/flickr/work_items.py`.

Default query variants:

- `scientific_name`
- `lime_butterfly`
- `chequered_swallowtail`
- `citrus_swallowtail`
- `swallowtail`

Broad query variants add:

- `papilio`
- `butterfly`
- `citrusbutterfly`
- `limebutterfly`

Flickr search execution lives in `src/flickr_bio_occurrence/flickr/client.py::FlickrClient.search_photos`. The search method is `flickr.photos.search`; default extras include `url_m`, `url_l`, `url_o`, and `o_dims`. Default search parameters include geo-only photos, photo media, content type `0`, safe search, and up to 250 records per page.

Image URL selection lives in `src/flickr_bio_occurrence/vision/image_selection.py::select_flickr_image_url`. Default preference is `url_l -> url_m`. `url_o` is used only in `original_diagnostic` mode with a pixel guardrail.

## Comment Fetch Logic

`src/flickr_bio_occurrence/flickr/endpoints.py` allow-lists `flickr.photos.comments.getList`.

`src/flickr_bio_occurrence/cli.py` exposes `fetch-comments`, but it returns `implemented=false`; no real comments API fetch is implemented.

Embedded comments already present in raw payloads can be parsed by `src/flickr_bio_occurrence/evidence/extractor.py`.

## Reports Present

Current report files:

- `reports/agents_update_recommendations.json`
- `reports/bioclip_run_summary.json`
- `reports/bioclip_run_summary.md`
- `reports/cache_profile.json`
- `reports/code_cleanup_report.md`
- `reports/gpu_profile.json`
- `reports/idempotency_profile.json`
- `reports/image_triage_profile.json`

## Files To Modify By Later Phase

Lifestage/category columns:

- `src/flickr_bio_occurrence/vision/triage.py`
- `src/flickr_bio_occurrence/vision/bioclip.py`
- `tests/test_image_triage.py`

Evidence text/category terms:

- `src/flickr_bio_occurrence/evidence/review_flags.py`
- `src/flickr_bio_occurrence/evidence/extractor.py`
- `src/flickr_bio_occurrence/evidence/rules.py`
- `tests/test_evidence_rules.py`
- `tests/test_evidence_extractor.py`

Reports:

- `scripts/generate_report_pack.py`
- `reports/`
- `tests/test_report_pack.py`

CLI/docs:

- `src/flickr_bio_occurrence/cli.py`
- `README.md`
- `tests/test_cli_dry_run.py`

Legacy storage cleanup:

- `src/flickr_bio_occurrence/pipeline/transforms.py`
- `src/flickr_bio_occurrence/pipeline/dry_run.py`
- `src/flickr_bio_occurrence/storage/duckdb_index.py`
- `tests/test_pipeline_transforms.py`
- `tests/test_storage_outputs.py`

## Redundant Code Candidates

- `src/flickr_bio_occurrence/pipeline/transforms.py`: Bronze/Silver/Gold ETL naming is legacy and may be superseded by image-triage output, but compatibility tests still cover it.
- `src/flickr_bio_occurrence/storage/duckdb_index.py`: Bronze/Silver/Gold DuckDB view names are inconsistent with current image-triage semantics.
- `src/flickr_bio_occurrence/pipeline/dry_run.py`: output paths still describe occurrence-publication tiers.
- `src/flickr_bio_occurrence/vision/pipeline.py::classify_bronze_photo_row`: function name carries old Bronze tier wording although behavior is temporary image classification.

## Acceptance Findings

- Exact files/functions to modify are identified.
- `image_category` does not already exist.
- `life_stage` does not already exist.
- Caterpillar and pupa/chrysalis are present but generally treated as negative material unless the verified life-stage path applies; egg is absent.
- No Phase 0 implementation changes were made; only audit report files were created.
