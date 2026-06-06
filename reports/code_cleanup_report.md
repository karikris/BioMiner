# Code Cleanup Report

Generated: 2026-06-06T19:24:11.376784+00:00

## Final Integration Findings

- The active flow is now a lean image-triage pipeline centered on `image_triage.parquet`.
- Image selection defaults to `url_l -> url_m`; originals are not selected by default.
- BioCLIP output is stored as model evidence only, not taxonomic validation.
- Removed the legacy human-verification gate from the active evidence rule engine.
- Removed legacy publication reasons: `human_verified_bioclip_positive`, `human_verified_bioclip_missing`, `human_verified_bioclip_low_confidence`, `human_verified_bioclip_conflict`, and `bioclip_positive_without_human_verification`.
- Successful cached images are deleted by default after prediction writes.
- Darwin Core export remains compatibility-only and is not expanded in the active triage flow.

## Retained Compatibility Shims

- `src/flickr_bio_occurrence/dwc/mapper.py` is retained for existing tested public API behavior; remove when Darwin Core compatibility tests are retired.
- `human_verification_detected` extraction fields remain as metadata from source text; they no longer gate Gold/Silver/Bronze classification.

## Tests Added Or Updated

- `test_gold_score_gte_050_target_positive`
- `test_silver_score_lt_050_target_positive`
- `test_bronze_negative_material_museum_art_ai_other_insect`
- `test_cli_help_does_not_describe_old_gold_silver_bronze_logic`
- `test_no_legacy_human_verification_gold_gate_remains`
- `test_code_cleanup_report_lists_removed_legacy_rule_paths`

## Remaining Explicit Gaps

- Dedicated Flickr comment API fetching remains unavailable and reported through `fetch-comments`.
- Validated Darwin Core occurrence publication, multi-GPU, and dashboard workflows remain out of scope.
