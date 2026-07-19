# ADR: geography-conditioned dynamic reference pooling

- Status: accepted architecture; software and fixture implementation complete
  through Phase 15; live empirical strategy selection remains pending
- Date: 2026-07-18
- Decision owner: Kris Kari
- Scope: BioMiner target-aware YOLOE-to-BioCLIP classification
- Baseline: `8d5112a2e715730010a27594202c53f9b36314d1`

## Context

BioMiner currently has reusable, content-bound reference embeddings and robust
species prototypes, but provisional ranking applies one route-compatible
support set with a fixed top-three mean. Regional candidate construction has
richer geographic evidence than the ranker: exact cell, buffer, country,
bioregion and global fallbacks become only a nullable exact-cluster diagnostic
at scoring time. The score artifact cannot identify distinct global and local
reference memberships.

The replacement must improve evidence selection without turning geography,
family, provider labels or raw similarity into taxonomic truth. It must also
avoid a costly and scientifically unnecessary encoding pass for every query,
candidate or geographic plan. The detailed baseline and gap register are in
`current_reference_pooling_audit.md`; downstream maturity and artifact
constraints are in `taxalens_butterflylens_pooling_handoff.md`.

## Implementation snapshot

The architecture below is implemented through the Phase 15 fixture and
production-decision boundary. The implementation includes:

- normalized reference geography, immutable index/anchor/neighbour artifacts,
  exact precision and QA manifests;
- complete family/geography candidate unions and all three target-preserving
  scheduling strategies;
- typed global/local pool plans, members, summaries, bounded uncertainty
  expansion, cached matrix identities and raw component scoring;
- four provisional fusion methods that preserve components, ties and
  alternatives;
- probability-audit registers, separate targeted failure discovery,
  fail-closed occurrence review, reviewed-evidence planning, leakage-safe
  splits, calibration, grouped quality reporting and remediation;
- exact reference-revision impact, selective reuse/rerun planning, stage graph,
  and typed settings. The seven plan-only CLI operations implemented in Phase
  13 were removed on 2026-07-19 because their disconnected bindings did not
  match the concrete artifact contracts;
- immutable TaxaLens and ButterflyLens handoffs validated against exact
  committed consumer objects; and
- a complete 24-variant fixture ablation and production-default decision.

The pilot decision is `insufficient_evidence`, not a selected strategy or a
rejection of measured production performance. It contains zero source-bound
human labels, leaves the full 86-effective-review shortfall, does not measure
MPS peak memory or comparable runtime, and leaves runtime settings unchanged.
The integrated report fingerprint is
`sha256:ade039c9914c6fc720773eee7fbfb2141ff087f3abf869d9ab56b5f54dfa5d09`.
Physical artifacts and canonical grains are catalogued in
`../schemas/geography_conditioned_dynamic_pooling_contracts.md`.

## Decision

BioMiner will insert three immutable layers between existing reference
embeddings/candidate sets and ranking:

1. a **reference index** over already cached embedding IDs and their admissible
   taxonomy, route, geography, source, observation and rights metadata;
2. a **comparison plan** that binds one Flickr scoring unit and its complete
   candidate safety union to explicit global and local pool memberships; and
3. a **component score** artifact that reports global, local, prototype,
   nearest-reference, top-k, coverage and disagreement evidence separately.

Pool membership is dynamic with respect to the query and the versioned
comparison policy. Embedding values are not. A change in geography, candidate
set, quota, distance band or expansion policy creates a new plan/pool identity
that points to the same compatible cached vectors.

Family evidence is a batching and retrieval accelerator, never a hard gate.
Geography is candidate and reference-selection evidence, never proof of
identity or absence. Every plan retains a geographically diverse global safety
pool. Missing or unusable query geography produces an explicit global-only
plan rather than a fabricated local pool.

## Logical data flow

```text
admitted reference media
  -> content/model/preprocessing-bound embeddings (computed once)
  -> immutable route-aware reference indexes

Flickr query + YOLOE route + query geography
  -> regional candidates + target + reviewed/visual safety candidates
  -> optional soft family-priority ordering
  -> fingerprinted comparison plan
       |-> global safety memberships
       |-> local memberships or local-unavailable reason
       `-> deterministic uncertainty expansion memberships when triggered
  -> batched similarity matrices over cached vectors
  -> separate global/local/prototype/coverage/disagreement components
  -> uncalibrated candidate ranking and review priority
  -> reviewed calibration/statistical/release layers (separate artifacts)
```

The candidate set is complete before any family-priority optimization. The
plan may schedule likely-family candidates first, but its final scored set must
still contain the configured target, regional candidates, reviewed
relationships, known mimics, historical false positives, visual neighbours,
and global safety candidates required by the active policy.

## First-class identities

The implemented physical schemas bind at least these semantic identities:

- query media, visual-input, transformation, route and embedding identity;
- candidate-set, accepted-taxon and registry identity;
- reference media, observation, source dataset/observer, route, visual-input,
  embedding and admission/review identity;
- reference-index snapshot and selection-policy identity;
- query geography value, precision, source and availability state;
- global pool, local pool, expansion and complete comparison-plan identity;
- model, weights, preprocessing and score-policy identity; and
- parent artifact fingerprints and producing repository commit.

Pool fingerprints cover ordered membership rows and selection reasons. A
global/local label without exact embedding IDs, ranks, quotas and shortfalls is
not a reconstructable pool.

## Candidate architecture

### Complete safety union

The regional candidate union remains authoritative for inclusion provenance.
Dynamic pooling may add candidates but may not remove any required candidate.
Candidate reasons remain multi-valued so a taxon can simultaneously be the
target, locally supported, a reviewed competitor and a visual neighbour.

The global safety contribution prevents local geography or an incorrect family
priority from hiding a plausible competitor. Its candidate policy is
versioned separately from reference-member quotas. The fixture ablation
compares geography-first, family-first-safe and parallel-union construction
without claiming one is superior before eligible held-out reviewed evidence
exists.

### Family as a soft accelerator

Family evidence may:

- choose an index shard and matrix batch order;
- allocate an initial retrieval budget among likely families;
- expand same-family local candidates efficiently; and
- expose a diagnostic family contribution or disagreement signal.

Family evidence must not:

- delete query-associated or target candidates;
- remove cross-family mimics, reviewed competitors, visual neighbours or
  historical errors;
- change a missing family prediction into a taxonomic assertion; or
- be presented as a calibrated species probability.

A low or unavailable family score causes ordinary safety-union processing,
not exclusion. Every output reports whether family priority changed scheduling,
membership or neither.

## Geographic reference architecture

### Local pool

When query geography is usable, the planner selects local evidence through
ordered, policy-bound scopes such as exact cell, buffered cells, bioregion,
country and bounded nearest geographic support. Selection records source and
query precision, distance availability, distance measure/version, scope,
rank, quota, effective count and shortfall. Coarse coordinates cannot create
finer geographic evidence.

Locality changes which admissible cached reference embeddings are compared; it
does not modify their vectors or admission status. Nearby references are
supporting evidence only. Missing a taxon locally is not negative evidence and
must not remove it from the complete candidate set.

### Global safety pool

Every plan has a global pool selected independently of query proximity. It is
geographically and source diverse where support permits, observation-aware,
route compatible and class balanced. Per-taxon opportunities are bounded so a
taxon with many media or transformed views does not receive an unreported
extreme-neighbour advantage.

The global pool is not merely "all remaining rows." It has an immutable pool
ID, quotas, effective counts, diversity/concentration fields, selection ranks,
shortfalls and policy fingerprint. It supplies the safety comparison when
local evidence is sparse, conflicting, unavailable or wrong.

### No-geography behavior

`no_geo` and `unassigned_geo` remain distinct query states. For either state:

- `local_pool_status` is unavailable with the exact reason;
- local counts, distances and local scores are null, not zero;
- the complete candidate safety union is retained;
- the global pool is selected normally; and
- downstream outputs state `global_only`, never `local_match=false`.

Coordinates withheld for privacy are also unavailable, not absent. A future
coarse region may be used only when its provenance and permitted precision are
explicit and the comparison policy supports that scope.

## Cached-embedding architecture

The immutable vector-cache key remains based on image content, visual-input
contract, model ID/revision/weights and preprocessing identity. Neither query
geography nor pool membership enters this key.

Reference indexes contain vector locations plus metadata and are invalidated by
embedding/admission/index-schema changes. Comparison plans contain selected
embedding IDs and are invalidated by candidate, geography, quota, selection-
policy or index-snapshot changes. Score artifacts are invalidated by plan,
query embedding, component-score or model changes. This separation permits:

- one BioCLIP load per worker rather than per species or pool;
- one raw embedding per compatible image/input identity;
- many reproducible plans over one frozen index;
- batched global/local matrix operations; and
- selective rebuilding from changed reference membership to affected plans and
  scores without touching unrelated vectors.

Conflicting vectors for one cache identity fail closed. A plan may never
materialize a new embedding artifact simply to express a geographic subset.

## Deterministic uncertainty expansion

The initial plan is deliberately bounded. It may expand only through a
versioned policy with maximum rounds and reference/candidate budgets. Triggers
may include:

- small uncalibrated rank margin;
- global/local rank or score disagreement;
- insufficient effective global or local support;
- high source, observer, geography or view concentration;
- prototype-versus-nearest-reference disagreement;
- family-priority disagreement with the complete safety union; and
- missing score components required by the active decision policy.

Trigger thresholds are configuration, not universal scientific constants.
They must be selected without final-test leakage and remain explicitly
uncalibrated until reviewed calibration evidence exists. Each expansion round
records the trigger values, added candidate/reference IDs, prior plan, policy,
budget use and stop reason. Expansion reuses cached query and reference
embeddings and cannot become an unbounded search.

Failure to resolve uncertainty is an output state and review-priority signal,
not permission to lower a release gate.

## Score and evidence semantics

The ranker will report at least:

- global and local prototype similarities;
- global and local nearest-reference similarities;
- global and local top-k means with configured and effective `k`;
- global/local support, independence, diversity and shortfall counts;
- global/local disagreement and rank movement;
- family-priority contribution and whether it changed membership;
- expansion rounds, triggers and stop reason; and
- an explicit probability-availability state.

Raw cosine similarities, weighted components, margins and fused ranking scores
are not probabilities. Any later calibrated probability must identify its
calibrator, reviewed training/calibration split, applicability, fingerprint and
reliability evidence. Component preservation is mandatory: a fused score may
not erase sparse local support or global/local disagreement.

## Evidence maturity

Dynamic pooling produces **candidate/model evidence**. Maturity advances only
through separate, fingerprint-bound layers:

1. model evidence available;
2. source-bound human review available;
3. quality estimate available for the relevant population/stratum;
4. expert review available when policy requires it;
5. all rights, taxonomy, duplicate, coordinate, date and release gates pass;
6. release-ready occurrence candidate; and
7. downstream publication under its own authority.

Unavailable/unrun evidence remains null with a reason. Zero reviewed rows is
not a zero error rate. Positive human review alone is not occurrence release,
and statistically supported population performance is not human verification
of each item.

## Failure and fallback policy

| Condition | Required behavior |
|---|---|
| Missing/withheld/unusable query geography | Global-only plan; local unavailable with reason. |
| No local support for a candidate | Retain candidate and global evidence; record local shortfall. |
| Global support shortfall | Use actual effective count; never duplicate rows to fill quota. |
| Family evidence missing or conflicting | Process complete safety union without family exclusion. |
| Stale index, plan or model fingerprint | Reject and rebuild only the affected layer. |
| Incompatible route/domain | Exclude that reference membership with reason; never coerce it. |
| Uncertainty budget exhausted | Emit unresolved state and review priority; do not fabricate confidence. |
| Rights/review/release dependency missing | Preserve internal evidence if permitted; block affected export. |

## Reproducibility, performance and operations

Selection order is deterministic after canonical normalization: explicit
priority fields followed by stable taxon, observation, media and embedding
identities. Distance and floating-point ties use declared stable tie-breakers.
All randomness is seeded and the seed is part of policy identity.

Indexes and pool tables are Parquet with manifests; small policies and reports
may be JSON. Production storage remains S3-compatible and work state remains
PostgreSQL; local filesystem/SQLite are explicit local modes. Workers claim
bounded partitions and coordinators perform canonical merge, sort, fingerprint
and manifest-last publication. Media, embeddings and model weights are never
embedded in a comparison-plan artifact.

Required efficiency metrics include model load count, cache hits/misses,
unique query/reference embeddings, plans and memberships, matrix batch sizes,
candidate and reference counts before/after expansion, wall/CPU/device time,
peak memory, bytes read/written and avoided recomputation. Uninstrumented
metrics remain unavailable rather than estimated.

## Downstream compatibility

TaxaLens at exact committed pin
`e845dd98493979f37b04dbb6538e0d7b8758ca11` receives immutable candidate,
review, quality and geographic-impact
artifacts without conflating reviewed with release-ready occurrence status.
ButterflyLens at exact committed pin
`1cea643623f2f20a2bea72afc754c7b194db3278` receives model-evidence and
classification-maturity artifacts
through its import adapter; BioMiner does not bypass RLS or write directly to
its tables. Both handoffs pin exact contracts and producer commits, preserve
rights and geographic precision, and publish manifests last.

Schema evolution is additive where possible. A breaking policy/schema change
creates a new version and migration/compatibility fixture; historical pool and
score artifacts are not reinterpreted in place.

## Alternatives rejected

### Keep fixed top-k over all route-compatible support

Rejected as the target architecture because it cannot reconstruct geographic
membership, exposes unequal retrieval opportunity, and cannot distinguish
local from global evidence. It remains a benchmark comparator until migration
and equivalence tests are complete.

### Local-only reference retrieval

Rejected because missing, sparse, imprecise or incorrect geography could hide
the correct species and amplify local sampling bias.

### Hard family-first pruning

Rejected because an incorrect coarse prediction would cause irreversible
species omission, especially for cross-family visual mimics and reviewed
historical errors.

### Re-encode references per geographic pool

Rejected because membership is metadata over immutable vectors. Re-encoding
would waste model work and make equivalent evidence identities diverge.

### Treat fused score as confidence or probability

Rejected because raw model evidence is not calibrated and component fusion can
hide disagreement and support shortfalls.

## Consequences, completed software gates and remaining evidence

The decision adds explicit index, plan, membership, component-score and
expansion artifacts. This increases schema surface area and storage metadata,
but makes every comparison reconstructable, cache-safe, selectively
invalidatable and auditable.

Deterministic software and fixture tests now prove:

1. all required candidates survive family and geography optimization;
2. every candidate has a global result and a local result or exact unavailable
   reason;
3. pool formation references existing embeddings and triggers no model call;
4. memberships are observation/source aware and report quota shortfalls;
5. no-geography plans contain no fabricated local distance or score;
6. expansion is deterministic, bounded and cache-reusing;
7. scores preserve global/local components and avoid probability language;
8. reference revisions selectively invalidate only affected downstream plans;
9. downstream fixtures preserve evidence maturity and release boundaries; and
10. strategy, quality and efficiency claims remain unavailable when eligible
    held-out or reviewed evidence does not exist.

These software gates do not complete a scientific run. Production selection
still requires source-bound human review, at least 86 effective reviewed
records under the frozen policy, at least 30 independent records in required
subgroups, a reviewed-precision lower bound of at least 0.95, comparable
instrumented computation, and an MPS peak-memory measurement within the
536,870,912-byte policy limit. No candidate strategy, pool variant, or fusion
method is a production default until those gates pass together.

## GitHits provenance

The required Task 0.2 and Subtask 0.2.1 GitHits searches were issued as one
bounded paired request and did not return within two minutes. The request was
terminated and recorded as unavailable in `provenance/githits.jsonl` under
`geo-pool-0.2` and `geo-pool-0.2.1`; no solution ID, external repository,
code, prose or claimed precedent was invented. This ADR is derived from the
committed BioMiner audits and pinned downstream contracts.
