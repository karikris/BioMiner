# BioCLIP Run Summary

Generated: 2026-06-06T15:34:36.597146+00:00

## Final Integration Findings

- BioCLIPMiner local path exists: `True`.
- Flickr comments fetched: `False`.
- Comments stored in raw payloads: `False`.
- Comments transformed to parquet: `False`.
- Comments searched for scientific names/verification phrases: `False`.
- Image selection order: `['url_l', 'url_m']`.
- CLI commands: `['fetch-live', 'fetch-comments', 'build-evidence', 'classify-once', 'classify-watch', 'apply-rules', 'gc-cache', 'compact-parquet', 'qa-summary']`.
- Evidence-first pipeline present: `True`.
- One publication_state per record: `True`.
- Prediction checkpoint layout: `silver/silver_vision_prediction/model_version=<model_id>/run_id=<run_id>/shard_id=<shard_id>/part-00000.parquet`.
- Cache cleanup handled: `True`.

## Notes

- BioCLIP output is screening evidence only, not taxonomic proof.
- Dedicated comments API fetching remains explicitly unavailable.
- No network, CUDA, real BioCLIP weights, or real Flickr downloads are required to generate this report pack.
