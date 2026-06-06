# Code Cleanup Report

Generated: 2026-06-06T16:34:13.636500+00:00

## Final Integration Findings

- The active flow is now a lean image-triage pipeline centered on `image_triage.parquet`.
- Image selection defaults to `url_l -> url_m`; originals are not selected by default.
- BioCLIP output is stored as model evidence only, not taxonomic validation.
- Successful cached images are deleted by default after prediction writes.
- Darwin Core export remains compatibility-only and is not expanded in the active triage flow.

## Remaining Explicit Gaps

- Dedicated Flickr comment API fetching remains unavailable and reported through `fetch-comments`.
- Validated Darwin Core occurrence publication, multi-GPU, and dashboard workflows remain out of scope.
