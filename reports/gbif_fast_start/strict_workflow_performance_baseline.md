# Strict reference workflow performance baseline

- Status: `strict_live_baseline_unavailable_with_prototype_proxy_evidence`
- Source SHA: `247b42f3206d48bb79e2dbf97c5a92e4f207ae71`
- Generated: `2026-07-17T10:19:41Z`
- Strict live run executed: `false`
- Fingerprint: `sha256:461881da39ca54b0cc4f6fc9393bf19e97c4b9d4222dec4af7d7fafdc0f6481c`

The committed evidence is a prototype-only proxy. It does not provide a strict
time-to-first-score result. Missing values remain unavailable or not instrumented.

| Metric | Status | Value | Unit | Evidence or reason |
|---|---|---:|---|---|
| `bioclip_model_cache_hits` | `measured` | 6 | `hits` | phase15.resume_and_cache.bioclip_model_cache_hits |
| `bioclip_persistent_model_loads` | `measured` | 1 | `loads` | phase15.resume_and_cache.bioclip_persistent_model_loads |
| `flickr_candidate_score_rows` | `measured` | 634312 | `rows` | phase14.staged_flickr_inference.candidate_score_rows |
| `flickr_records_per_second` | `measured` | 2.274524 | `rows_per_second` | phase14.staged_flickr_inference.performance.records_per_second |
| `flickr_records_scored` | `measured` | 13496 | `rows` | phase14.staged_flickr_inference.classified |
| `full_rerun_work_avoided` | `not_instrumented` | — | `rows` | the current resume report records reuse but not a common work-unit denominator |
| `manual_reference_reviews_completed` | `measured` | 0 | `rows` | phase14.reference_bank.human_verified |
| `peak_rss_memory` | `measured` | 1765261312 | `bytes` | phase14.staged_flickr_inference.performance.rss_peak_memory_bytes |
| `provisional_support_frozen` | `measured` | 81 | `rows` | phase14.reference_bank.prototype_support |
| `reference_candidates_acquired` | `unavailable` | — | `rows` | committed reports expose selected media but not the total acquired-candidate count |
| `reference_embedding_cache_hits` | `measured` | 81 | `rows` | phase15.resume_and_cache.support_embedding_resume_reused |
| `reference_embeddings_recomputed_on_resume` | `measured` | 0 | `rows` | phase15.resume_and_cache.support_embedding_resume_recomputed |
| `reference_media_downloaded` | `unavailable` | — | `rows` | selected-media count does not prove a complete download-attempt denominator |
| `reference_media_selected` | `measured` | 93 | `rows` | phase14.reference_bank.selected_media |
| `references_awaiting_human_review` | `derived` | 81 | `rows` | phase14.reference_bank.prototype_support - phase14.reference_bank.human_verified |
| `selective_rerun_records` | `unavailable` | — | `rows` | the baseline predates reference-revision impact analysis and selective rescoring |
| `strict_time_blocked_before_readiness` | `unavailable` | — | `seconds` | no committed end-to-end strict reference run records manual-wait start and completion |
| `strict_time_to_first_flickr_score` | `unavailable` | — | `seconds` | no committed strict run reached a first Flickr score |
| `strict_time_to_prototypes` | `not_instrumented` | — | `seconds` | the committed prototype report does not record prototype-ready elapsed time |
| `strict_time_to_reference_embeddings` | `not_instrumented` | — | `seconds` | the committed prototype report records embedding counts but not elapsed time from run start |

## Interpretation

- Prototype evidence proves persistent model and embedding reuse mechanisms.
- It does not measure the strict manual-review wait or strict time to first score.
- Adaptive and strict paired benchmarks must use this same metric contract.
- Unavailable values are not zero and must not be used in speedup calculations.
