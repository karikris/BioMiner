# BioCLIP Run Summary

Generated: 2026-06-06T19:20:03.680062+00:00

## Final Integration Findings

- BioCLIPMiner local path exists: `True`.
- Flickr comments fetched: `False`.
- Comments stored in raw payloads: `False`.
- Comments transformed to parquet: `False`.
- Comments searched for scientific names/verification phrases: `False`.
- Image selection order: `['url_l', 'url_m']`.
- Image triage output: `image_triage.parquet`.
- Triage bins: `['gold', 'silver', 'bronze', 'in_review']`.
- Publication state semantics: `{'file': 'src/flickr_bio_occurrence/evidence/rules.py', 'symbol': 'classify_evidence_row', 'states': ['gold', 'silver', 'bronze', 'in_review'], 'gold': 'BioCLIP target-positive score >= 0.50', 'silver': 'BioCLIP target-positive score < 0.50', 'bronze': 'negative/non-occurrence visual material', 'in_review': 'missing, unresolved, invalid, or operational failure records', 'screening_evidence_only': True}`.
- Publication state counts: `not_instrumented`.
- Review reason counts: `not_instrumented`.
- Bronze reason counts: `not_instrumented`.
- Gold score distribution: `None`.
- Silver score distribution: `None`.
- Missing image count: `None`.
- Missing BioCLIP count: `None`.
- Geo/time fields retained: `['latitude', 'longitude', 'date_taken', 'date_upload', 'captured_at', 'year', 'month']`.
- Cache cleanup handled: `True`.

## Notes

- BioCLIP output is screening evidence only, not taxonomic proof.
- Dedicated comments API fetching and validated Darwin Core publication remain explicitly unavailable in this phase.
- No network, CUDA, real BioCLIP weights, or real Flickr downloads are required to generate this report pack.
