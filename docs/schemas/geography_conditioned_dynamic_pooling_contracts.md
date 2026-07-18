# Geography-conditioned dynamic-pooling contracts

Status: implemented software and fixture schema catalog through Phase 15.

This document maps the accepted dynamic-pooling architecture to its canonical
durable artifacts, row grains, schema versions, validators, identities, and
scientific authority. The Python schema functions and validators named below
are normative for exact physical columns and enum domains; this catalog is the
stable operator and reviewer index.

No artifact in this catalog is taxonomic ground truth merely because it exists
or validates. Candidate, model, probability, human-review, statistical-support,
release, and publication maturity remain separate.

## Conventions

- Durable tabular artifacts use closed Polars schemas and Parquet. Small
  settings, manifests, decisions, receipts, and reports use JSON.
- Every table has a declared grain, canonical sort, schema version, and
  semantic row or artifact fingerprint. Empty tables retain the exact schema.
- Semantic fingerprints cover scientific/computational identity. Physical
  SHA-256 values cover exact serialized bytes. One does not replace the other.
- Paths, timestamps, workers, retries, multipart layout, and backend location
  do not change semantic identity unless a schema explicitly includes them.
- Unavailable values are null with a reason. They are not zero, false,
  biological absence, or failed performance.
- Manifests publish last. A consumer reads exact committed producer/consumer
  objects, never a dirty sibling worktree.

## Reference geography and immutable lookup

| Artifact | Schema version | Canonical grain | Producer/validator |
|---|---|---|---|
| `normalized_reference_geography.parquet` | `normalized-reference-geography-v1.0.0` | One normalized geography record per admitted reference media/observation identity | `references.normalized_geography` |
| `reference_geography_index.parquet` | `reference-geography-index-v1.0.0` | `(reference_media_id, route, visual_input_kind, embedding_fingerprint)` | `bioclip.reference_geography_index` |
| `global_reference_anchors.parquet` | `global-reference-anchors-v1.0.0` | One bounded global anchor membership per accepted taxon, route, observation/media and embedding | `bioclip.global_reference_anchors` |
| `geographic_reference_neighbours.parquet` | `geographic-reference-neighbours-v1.0.0` | `(reference_geography_row_fingerprint, lookup_scope, lookup_key, ...)` membership | `bioclip.geographic_reference_neighbours` |
| `reference_geography_index_manifest.json` | `reference-geography-index-manifest-v1.0.0` | One manifest for the complete index/anchor/neighbour snapshot | `bioclip.reference_geography_qa` |

The index references existing admitted embedding identities. It contains no
model weights or media bytes. Coordinate quality and precision constrain which
lookup scopes may be populated. Country-only, missing, invalid, and withheld
coordinates cannot fabricate local-cell evidence.

## Canonical Flickr scoring identity

| Artifact | Schema version | Canonical grain | Producer/validator |
|---|---|---|---|
| `flickr_scoring_units.parquet` | `flickr-scoring-unit-v1.0.0` | One routed organism/scoring unit; several units may share one photo embedding | `flickr_fetch.scoring_units` |
| `flickr_scoring_unit_associations.parquet` | `flickr-scoring-association-v1.0.0` | One retained discovery association per scoring unit | `flickr_fetch.scoring_units` |
| `flickr_scoring_unit_candidates.parquet` | `flickr-scoring-candidate-v1.0.0` | One complete candidate membership per scoring unit | `flickr_fetch.scoring_units` |
| `flickr_scoring_geography.parquet` | `flickr-scoring-geography-v1.0.0` | One geography evidence projection per scoring unit | `flickr_fetch.scoring_geography` |
| `flickr_geo_taxon_partitions.parquet` | `flickr-geo-taxon-partition-v1.0.0` | One scoring unit in one deterministic run/geo/taxon work partition | `flickr_fetch.scoring_partitions` |
| `flickr_partition_summary.parquet` | `flickr-partition-summary-v1.0.0` | One canonical partition summary | `flickr_fetch.scoring_partitions` |

`flickr_photo_id` is the source-photo grain; `organism_unit_id` is the
biological/routing grain; `photo_embedding_unit_id` is the reusable model-input
grain. These may be many-to-one and must not be collapsed accidentally. Query
hits are discovery provenance and never labels.

## Candidate union and scheduling

| Artifact | Schema version | Canonical grain | Producer/validator |
|---|---|---|---|
| `regional_candidate_species.parquet` | `regional-candidate-species-v1.0.0` | One accepted candidate taxon and complete inclusion provenance per regional set | `candidates.regional_union` |
| `family_geo_candidate_sets.parquet` | `family-geo-candidate-set-v1.0.0` | One candidate membership per Flickr scoring unit/complete union | `bioclip.family_geo_candidates` |
| `candidate_strategy_plans.parquet` | `candidate-strategy-plan-v1.0.0` | One ordered candidate row per candidate set and strategy | `candidates.strategy_ablation` |
| `candidate_strategy_metrics.parquet` | `candidate-strategy-metric-v1.0.0` | One strategy, cutoff and evaluation stratum | `evaluation.candidate_strategies` |
| `family_pruning_counterfactual.parquet` | `family-pruning-counterfactual-v1.0.0` | One eligible reviewed case/strategy counterfactual | `evaluation.candidate_strategies` |

The three strategy identities are `geography_first`, `family_first_safe`, and
`parallel_family_geography_union`. Their ordering may differ; their complete
membership and target-preservation contracts may not. Family and geography are
evidence and scheduling inputs, not identity or absence authority.

## Dynamic reference-pool plans

| Artifact | Schema version | Canonical grain | Producer/validator |
|---|---|---|---|
| `dynamic_reference_pool_plans.parquet` | `dynamic-reference-pool-plan-v1.0.0` | One query/candidate comparison plan under an immutable policy and index snapshot | `bioclip.dynamic_pool_contracts` |
| `dynamic_reference_pool_members.parquet` | `dynamic-reference-pool-member-v1.0.0` | One exact reference embedding membership in one global/local/safety pool | `bioclip.dynamic_pool_contracts` |
| `dynamic_reference_pool_summary.parquet` | `dynamic-reference-pool-summary-v1.0.0` | One pool-kind/candidate summary with actual counts, diversity and shortfalls | `bioclip.dynamic_pool_contracts` |
| `dynamic_pool_expansion_evidence.parquet` | `dynamic-pool-expansion-evidence-v1.0.0` | One raw expansion-signal set per plan/round | `bioclip.dynamic_pool_expansion` |
| `dynamic_pool_expansion_decisions.parquet` | `dynamic-pool-expansion-decision-v1.0.0` | One bounded expand/stop decision per plan/round | `bioclip.dynamic_pool_expansion` |
| `dynamic_pool_expansion_cache_reuse.parquet` | `dynamic-pool-expansion-cache-reuse-v1.0.0` | One reusable/required cache identity per expansion | `bioclip.dynamic_pool_expansion` |

Pool identity covers ordered members, selection reasons, quotas, effective
counts, shortfalls, geography state, policy, and parent index/candidate
fingerprints. Query geography and membership never enter the immutable image
embedding key. No-geography plans have a normal global pool and an explicit
local-unavailable reason.

## Full-frame embeddings, matrices, scores and fusion

The canonical target-aware model input is full-frame and uses
`target-aware-full-frame-embedding-set-v1.0.0`. One compatible source image,
model revision/weights, preprocessing and transform identity maps to one raw
embedding. Pool changes reuse it.

Matrix identities are:

- `family-prototype-matrix-signature-v1`;
- `candidate-prototype-matrix-signature-v1`;
- `dynamic-pool-reference-matrix-signature-v1`; and
- `cached-vector-matrix-v1`.

Vector work/result identities are `dynamic-vector-scoring-work-v1` and
`dynamic-vector-scoring-result-v1`. Batch metrics use
`pool-matrix-batch-metrics-v1` and report exact work items, matrix references,
unique matrices, rows, bytes, reuse, encoder invocations, and image
materializations.

| Artifact | Schema version | Canonical grain | Producer/validator |
|---|---|---|---|
| `dynamic_pool_candidate_scores.parquet` | `dynamic-pool-candidate-score-v2.0.0` | One Flickr scoring unit, candidate and provisional fusion method | `bioclip.dynamic_pool_scores` |
| `dynamic_pool_photo_summary.parquet` | `dynamic-pool-photo-summary-v2.0.0` | One Flickr scoring unit/method summary with top, alternatives and evidence maturity | `bioclip.dynamic_pool_scores` |

Raw component contracts preserve family, global and local prototype,
nearest-reference and top-k values plus coverage, disagreement and rank
movement. Fusion contracts preserve all four methods, complete rankings, ties,
and alternatives. None of these values is a probability.

## Review, statistical support and outcome lanes

| Artifact | Schema version | Canonical grain | Authority |
|---|---|---|---|
| `dynamic_pool_audit_frame.parquet` | `dynamic-pool-audit-frame-v1.0.0` | One connected duplicate/observation audit unit | Provisional model evidence only |
| `dynamic_pool_probability_audit_register.parquet` | `dynamic-pool-probability-register-v1.0.0` | One sampling-frame unit with inclusion probability and weight | Representative design |
| `dynamic_pool_probability_audit_sample.parquet` | `dynamic-pool-probability-sample-v1.0.0` | One selected representative unit | Representative design, not completed review |
| `dynamic_pool_failure_discovery_queue.parquet` | `dynamic-pool-failure-queue-v1.0.0` | One targeted diagnostic unit | Targeted only; no inclusion probability/weight |
| `dynamic_pool_occurrence_release_review_queue.parquet` | `dynamic-pool-release-review-queue-v1.0.0` | One final-release candidate requiring complete review gates | Review work only; no release authority |
| `reviewed_flickr_independence_components.parquet` | `reviewed-flickr-independence-component-v1.0.0` | One reviewed item assigned to a transitive independence component | Leakage control |
| `dynamic_pool_evaluation_splits.parquet` | `dynamic-pool-evaluation-split-v1.0.0` | One frozen independence component/split assignment | Calibration/validation/final-test control |
| Human-reviewed release lane | `human-reviewed-release-lane-v1.0.0` | One source item satisfying decisive review and all release prerequisites | Candidate for release, not publication receipt |
| Audited screening lane | `audited-screening-candidate-lane-v1.0.0` | One source item eligible only for screening | No occurrence release |
| Unresolved lane | `unresolved-candidate-queue-lane-v1.0.0` | One source item retained with an explicit unresolved reason | No occurrence release |

Representative and targeted purposes may overlap on a source item but may not
merge. Targeted work has null inclusion probabilities/weights and cannot
support unweighted population-quality claims. A selected work item is not a
completed source-bound human decision.

## Calibration, quality and remediation

`dynamic_pool_features.parquet` is the raw-evidence feature projection used by
grouped, leakage-safe calibration. Calibration, validation reliability,
screening-threshold selection, grouped quality estimates, and occurrence
outcome lanes use distinct fingerprints and splits. Provider assertions and
targeted-review outcomes cannot enter calibration/final-test labels as human
truth.

Quality reports retain overall, family, genus, species and five geographic
levels; estimate fields are null with an insufficient-sample reason when their
effective row/component minimum is not met. Remediation outputs are typed human
actions. Reference changes propagate through exact pool, matrix and scoring-
record dependency identities, with compatible Flickr embeddings remaining
reusable.

## Settings, plans and selection authority

`dynamic_pooling_settings.json` uses
`dynamic-pooling-settings-v1.0.0`. Candidate strategy and fusion method default
to null and require evidence fingerprints when selected. Pool policy remains a
separate immutable `dynamic-reference-pool-policy-v1.0.0` object.

The seven CLI operations emit `dynamic-pool-command-plan-v1.0.0` plans. A plan
can be structurally valid while selection requirements remain unmet. Plans
grant no calibration, human-verification, statistical-support, or release
authority. Live CLI adapters currently fail closed.

The Phase 15 `production_default_decision.json` uses
`dynamic-pool-production-default-decision-v1.0.0`. Its outcome is
`insufficient_evidence`, with zero eligible variants and no settings change.
The review projection is not a selected default.

## Downstream handoffs

TaxaLens is pinned to `e845dd98493979f37b04dbb6538e0d7b8758ca11` and consumes
`biominer-taxalens-dynamic-pool-handoff-v1.0.0` through a
`storage-handoff-inventory-v1.0.0` content-addressed archive. The handoff
contains six score/pool tables, a representative review frame, optional
quality sidecar, and explicit geographic-cell unavailability. TaxaLens owns
baseline-provider-union geographic impact and database identity.

ButterflyLens is pinned to `1cea643623f2f20a2bea72afc754c7b194db3278` and consumes
the ButterflyLens dynamic-pool handoff through exact committed JSON Schema,
Python, TypeScript, migration, pgTAP and vocabulary fixtures. BioMiner exports
pre-assignment review inputs, not reviewer assignments; ButterflyLens owns
database IDs, RLS, review events, maturity transitions, and release authority.

Both handoffs are create-only, validate all artifacts before a manifest-last
publish, keep semantic fingerprints separate from file SHA-256 values, and
grant no occurrence-release authority merely because import succeeds.

## Pilot evidence artifacts

The bounded pilot freezes its plan in
`config/pilot/geography_conditioned_dynamic_pool_pilot_v1.json`. Reports under
`reports/geo_dynamic_pooling/pilot/` bind structural candidate results, cached-
vector scores, review work, the 24-row selection table, the default decision,
and the integrated report to exact semantic fingerprints.

The pilot is fixture-backed. Historical real-execution manifests are listed
separately and do not count as current execution. There are zero source-bound
human labels, zero completed real reviews, and a remaining effective-review
shortfall of 86. No production default or occurrence release is authorized.

## Dependency and invalidation summary

```text
registry + admitted reference + cached embedding
  -> normalized geography -> index/anchors/neighbours
  -> complete candidate union -> strategy schedule
  -> pool plan -> exact pool members -> pool summary
  -> cached matrices + cached Flickr embedding -> raw components -> fusion rows
  -> representative frame / targeted queue / release-review work
  -> reviewed splits -> calibration -> quality -> outcome lanes
  -> exact reference-change impact -> selective rerun
  -> immutable TaxaLens / ButterflyLens handoffs
```

A changed reference membership invalidates affected pool/matrix/score
identities, not unrelated embeddings. A changed candidate strategy, fusion
method, calibration policy, review evidence, or release policy invalidates only
the layer and descendants whose fingerprint includes it. Unknown schema
versions, stale parents, mismatched fingerprints, incomplete manifests, or
unsafe maturity promotion fail closed.
