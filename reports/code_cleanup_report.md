# Code Cleanup Report

Generated: 2026-06-06T20:02:48.575407+00:00

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
