# BioCLIP Run Summary

Generated: 2026-06-06T16:34:13.636500+00:00

## Final Integration Findings

- BioCLIPMiner local path exists: `True`.
- Flickr comments fetched: `False`.
- Comments stored in raw payloads: `False`.
- Comments transformed to parquet: `False`.
- Comments searched for scientific names/verification phrases: `False`.
- Image selection order: `['url_l', 'url_m']`.
- Image triage output: `image_triage.parquet`.
- Triage bins: `['gold', 'silver', 'bronze', 'in_review']`.
- Geo/time fields retained: `['latitude', 'longitude', 'date_taken', 'date_upload', 'captured_at', 'year', 'month']`.
- Cache cleanup handled: `True`.

## Notes

- BioCLIP output is screening evidence only, not taxonomic proof.
- Dedicated comments API fetching and validated Darwin Core publication remain explicitly unavailable in this phase.
- No network, CUDA, real BioCLIP weights, or real Flickr downloads are required to generate this report pack.
