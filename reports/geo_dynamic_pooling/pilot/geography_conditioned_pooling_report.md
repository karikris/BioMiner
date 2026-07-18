# Geography-conditioned dynamic pooling pilot

Decision: **insufficient production evidence; no default selected or changed**.

## Evidence boundary

The current run is a deterministic 7-case fixture pilot over 5 taxa, with 6 located cases and 1 no-geography case. It made no network call, ran no BioCLIP image encoder, and contains no source-bound human label. Historical real-execution manifests are inventory only and are not counted as current results.

## Complete ablation

The report covers 24 candidate/pool/fusion variants and 168 case-variant rows. All strategies retain the complete five-taxon union and every target. Their order metrics are fixture structural recall, not classification accuracy.

Across 72 located global/dynamic pairs, 36 target raw scores change and zero top candidates change. All 12 no-geography pairs retain exact global fallback. Raw values are not probabilities.

## Computation and review

The shared run uses 14 cached-vector work items, 7 unique query vectors, 7 query reuse events, 100 pool-matrix references, and 65 within-batch matrix reuses. Runtime savings and MPS peak memory were not measured.

Seven representative and seven targeted fixture work items are planned, but completed real reviews remain 0. The effective-review shortfall is therefore 86 of 86. Reviewed precision, confidence bounds, and family/geographic subgroup estimates are unavailable.

## Production decision

All nine selection criteria were evaluated. Six remain blocking: target_candidate_recall, reviewed_precision_and_confidence_bounds, family_and_geographic_subgroup_behavior, review_workload, computation, mps_memory. Zero variants are eligible. This is insufficient evidence, not rejection of measured production performance.

Current and resulting runtime settings have the same fingerprint: `sha256:0fd197b2650a79d99970cada3dcbabe9980c5a265d9d71f929bbcf6f51e13e7d`. Candidate strategy, pool variant, fusion method, production authority, and release authority remain unset.

## Claims allowed

- The frozen fixture pilot exercises all 24 declared candidate, pool and fusion variants through production contracts.
- All candidate strategies preserve the complete five-taxon union and target in every fixture case.
- Observed cached-vector embedding and matrix reuse counts describe the complete fixture execution.
- The no-geography fixture preserves exact global fallback without implying biological absence.
- The production acceptance decision is insufficient evidence and runtime defaults remain unchanged.

## Claims blocked

- Fixture target ranks are reviewed classification accuracy or empirical superiority.
- Raw scores are calibrated probabilities.
- Historical manifests are current pilot execution or human-review outcomes.
- Planned representative or targeted fixture work is completed source-bound review.
- Any candidate strategy, pool variant or fusion method is a selected production default.
- Any occurrence is release ready or release authorized.

Report fingerprint: `sha256:ade039c9914c6fc720773eee7fbfb2141ff087f3abf869d9ab56b5f54dfa5d09`.
Decision outcome: `insufficient_evidence`.
