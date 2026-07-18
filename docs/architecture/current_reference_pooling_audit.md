# Current reference-pooling audit

## Decision summary

BioMiner's current reference pipeline is **embedding-cache safe but not
pool-contract complete**. Reference images are content-bound and reusable, and
the provisional ranker accepts an already computed Flickr embedding. The
remaining gap is not a need to encode the same image once per candidate; it is
the absence of an explicit, balanced and fingerprinted comparison-plan artifact
that selects global and local embedding IDs before ranking.

The current pipeline should therefore be evolved by inserting candidate-plan,
reference-index and pool-membership contracts between the existing regional
candidate and ranking layers. Replacing the embedding layer or weakening the
complete-set/family-safety contracts would discard working safeguards.

This audit covers committed BioMiner code at
`bc821bfbfaee451f391c41806bc4a94aefbffcbc`. It is a source-level limitation
analysis, not a live accuracy, latency or memory benchmark.

## Data flow inspected

```text
regional occurrence + reviewed relationships + global visual neighbours
  -> regional candidate taxa (fingerprinted by target and geo cluster)
  -> target-aware complete-set plan (species + negatives + hierarchy diagnostics)

admitted support manifest + reference visual inputs
  -> cached reference embeddings (content/model/preprocessing identity)
  -> robust species/route/visual-input prototypes
  -> provisional ranker(query embedding, route, optional exact geo cluster)
  -> one uncalibrated ranking over route-compatible support
```

The two paths meet only conceptually. The current provisional ranker has no
`candidate_set_id` input, and the candidate artifact contains no selected
`reference_media_id`, `embedding_fingerprint` or pool role. Consequently, the
repository can prove candidate-set identity and embedding identity separately,
but cannot yet prove the exact support-image set used for each candidate in each
Flickr comparison.

## Fixed top-k behaviour

`provisional_reference_ranking()` defaults to `top_k=3` and
`prototype_method="trimmed_mean"`. It filters prototypes and reference support
by route, then loops over every taxon represented in the selected prototypes.
For each taxon it:

1. takes the maximum similarity over that taxon's selected prototype rows;
2. sorts every route-compatible support row for the taxon by descending cosine
   similarity with `reference_media_id` as the deterministic tie-break;
3. retains `min(3, available support rows)`;
4. averages the retained similarities using their actual count; and
5. orders candidates by
   `(prototype_similarity + top_k_reference_mean) / 2`.

Nearest-reference similarity is exposed, but it is not a separately weighted
term: the nearest row already participates in the top-k mean. The same `k`,
prototype method and formula apply to every candidate regardless of local
coverage, class dispersion, route/domain disagreement, source independence or
geographic distance. No uncertainty signal can expand or contract the selected
support.

This is deterministic and scientifically honest about being uncalibrated, but
it is fixed-support scoring rather than dynamic pooling.

## Prototype grouping

`provisional-prototypes-v1.0.0` groups observations by:

- accepted taxon key;
- species;
- route;
- visual-input kind.

It then builds normalized-mean, mean-centred-mean, trimmed-mean and medoid
variants. The default policy caps input at 64 independent observations per
species/route, uses one within-class prototype, trims 10%, requires two members
for a cluster and uses seed 42.

This existing cap is a meaningful class-balancing safeguard. It does not make
the prototype geographic: `geo_cluster_id`, country, bioregion and distance are
absent from the grouping key. It also does not constrain the raw reference rows
scanned by provisional ranking, because that function selects all
`support_train` rows on the requested route.

## Geography: candidate evidence versus ranking evidence

Geography is richer in candidate generation than in ranking, and those layers
must not be conflated.

The regional candidate union uses exact cell, buffered cell, country,
bioregion and global occurrence evidence. It fingerprints the geo-cluster-
scoped taxon union and records occurrence support and candidate reasons.

The provisional ranker reduces geography to one nullable value:

- if no `query_geo_cluster_id` is supplied, `geography_compatible` is null;
- otherwise it is true when any support row for the taxon has exactly that
  `geo_cluster_id`, and false when none does.

That Boolean does not affect the ranking score or reference selection. It does
not express distance, neighbouring cells, country/bioregion fallback,
continent, support density, nearest observation, or separate global and local
components. It can therefore record compatibility while the score still uses
the same complete route-compatible reference set.

## Missing global/local identities

No current production schema contains all of the following comparison-plan
fields:

- Flickr query/object embedding ID;
- candidate-set ID and candidate reason;
- candidate species and reference embedding ID;
- pool role (`global`, `local` or explicit safety expansion);
- geographic selector, distance and fallback reason;
- per-role quota, selected rank and availability shortfall;
- reference/source/observation independence identity;
- global-pool, local-pool and complete-plan fingerprints;
- selection policy and index fingerprints.

The reference embedding table has stable media, observation, content, model,
preprocessing, support and geographic identities. The regional candidate table
has stable taxon, geo-cluster, priority, reason and set identities. Neither
binds candidate taxa to selected embedding rows. A downstream score can name
its reference and support fingerprints, but it cannot reconstruct an explicit
global/local membership decision from the current ranking artifact alone.

## Candidate and reference class-size bias risks

These are risks to measure, not claims that a particular species is currently
misclassified.

| Risk | Current mechanism | Existing mitigation | Remaining limitation |
|---|---|---|---|
| Candidate-count variation | Regional sets grow with local same-family taxa, relationships, visual neighbours and fallbacks. | Target and complete candidate set are fingerprinted; priorities are deterministic. | No candidate quota by evidence axis, and no plan-level measure of how candidate-set size changes margins. |
| Global fallback expansion | `no_geo` and unassigned geography can include every same-family registry species plus explicit competitors. | Target, mimics, historical false positives and visual neighbours remain present. | Family size can dominate candidate count; source-axis contributions are not balanced. |
| More-reference opportunity | Ranking searches every route-compatible support row within a taxon before taking its best three. | Each taxon contributes one prototype/top-k score and deterministic ties. | A taxon with more observations or visual variants has more chances to produce extreme nearest neighbours. |
| Unequal effective k | A class with fewer than three rows uses all available rows and divides by that smaller count. | No missing rows are invented. | Scores have different support counts without an explicit coverage penalty or uncertainty expansion. |
| Prototype versus raw support | Prototype input is capped at 64 independent observations per species/route. | Deterministic observation-balanced subsampling and dispersion fields. | The raw top-k scan is not capped by the same selected observation set and is not stratified global/local. |
| Media/view multiplicity | Support can contain multiple visual-input variants bound to one media item. | Every variant has a distinct visual-input and transformation identity. | Fixed raw retrieval does not enforce one opportunity per observation or balance visual-input kinds. |

Required follow-up metrics are per-candidate support count, independent
observation count, dataset/observer count, visual-input-kind count, selected
global/local count, shortfall, source concentration, effective `k`, candidate
count by reason and sensitivity of rank/margin to balanced versus unbalanced
selection.

## Repeated-work audit

### Work already protected

Reference embedding generation groups pending inputs by a cache key containing:

- image-content hash;
- input-contract version;
- model ID and revision;
- preprocessing version;
- model-input fingerprint.

The cache is validated against model weights, preprocessing fingerprint and
attestation before reuse, and conflicting vectors for one identity fail closed.
Multiple support rows sharing one image/model identity reuse one vector. Resume
checkpoints also avoid recomputing completed support-row/visual-input pairs.

The provisional ranker receives `query_embedding` as data. It does not invoke
BioCLIP once per taxon or reference row. These properties must remain
non-negotiable.

### Work not yet protected or indexed

- Similarities are recomputed in Python for every query, candidate taxon and
  support row; there is no frozen global/local matrix index at this layer.
- Candidate construction and reference selection are disconnected, so there is
  no cache key for a complete comparison plan or selected pool.
- A future caller could incorrectly implement geography by rebuilding subset
  embedding artifacts. The current primitives permit safe reuse, but no pool
  contract requires selection-by-embedding-ID.
- A reference revision can reuse embeddings through existing affected-only
  contracts, but the current ranking artifact cannot identify which global or
  local pool memberships need rebuilding.

The architecture change should persist indexes and pool membership, not encode
new copies of images. Repeated matrix scoring may be optimized; repeated model
inference for the same identity must be rejected.

## Current no-geography behaviour

Flickr clustering reserves `no_geo` for records without usable geography and
`unassigned_geo` for geotagged records that do not resolve to a geographic
cluster. Both are explicit global-fallback cluster IDs, so missing location is
not silently treated as a real local cluster.

For either fallback, candidate generation preserves the target, reviewed
relationships, known mimics, historical false positives and visual neighbours.
It adds global-occurrence same-family taxa and, by default, every accepted
same-family registry species. `no_geo` records the reason
`global_no_geo_fallback`; unresolved geotags record
`global_unassigned_geo_fallback`.

This is a safe candidate fallback, not a global reference-pool implementation.
The provisional ranker still uses all route-compatible support. When passed
`query_geo_cluster_id="no_geo"`, it merely checks for exact `no_geo` support
metadata; when passed no cluster, geography is null. The next contract must
explicitly select a geographically diverse global pool and mark local evidence
unavailable, rather than treating missing local evidence as zero or as a false
local match.

## Current family-evidence role

BioMiner already enforces the most important hierarchy safety property in the
target-aware complete-set path:

- every regional species candidate is scored;
- the target must be present exactly once;
- family and genus classes are additional diagnostics;
- `hierarchy_pruning_applied` must be false; and
- `hierarchy_rankings_diagnostic_only` must be true.

Family also influences candidate construction. Local same-family taxa are
included, sparse local sets expand to country/bioregion same-family evidence,
and global fallback can add the whole registered family. This is an expansion
and batching opportunity, not an identity certificate: cross-family mimics,
visual neighbours, historical errors and other reviewed relationships remain
eligible.

The dynamic-pooling architecture may use family to select indexes, batch matrix
operations and add competitors. It must not let a weak family score delete the
query-associated species, configured target, geographic candidates, visual
neighbours, mimics, historical false positives or global safety candidates.

## Risk-ranked gap register

| Priority | Gap | Impact if uncorrected | Required contract response |
|---|---|---|---|
| Critical | No per-query global/local pool membership artifact | Exact evidence behind a score cannot be reconstructed or selectively invalidated. | Versioned comparison plan and pool-membership Parquet schemas with semantic fingerprints. |
| Critical | Candidate and support selection are not bound | A fingerprinted candidate set can be scored against an implicit or unintended support set. | Bind candidate-set, index, embedding, policy and pool fingerprints. |
| High | Geography does not affect provisional reference selection or score components | A geographic compatibility flag can overstate the geographic specificity of the evidence. | Separate global/local prototypes, nearest rows, top-k means, distance, coverage and disagreement. |
| High | Raw retrieval opportunity varies with support and view count | Extreme-neighbour evidence may favour larger or more duplicated classes. | Observation/source-aware quotas and reported availability/shortfall metrics. |
| High | Fixed k and formula ignore uncertainty | Sparse or conflicting evidence receives the same configured treatment. | Deterministic expansion triggers that reuse frozen embeddings. |
| Medium | No frozen matrix indexes at the ranking layer | Similarity traversal repeats and scaling behaviour is opaque. | Immutable global/geographic index manifests and vectorized scoring. |
| Medium | No-geo is candidate-safe but not pool-explicit | Local evidence can be ambiguous or simply absent without a dedicated state. | Global-only plan with `local_unavailable` reason and no fabricated distance. |
| Guardrail | Family expansion could become family pruning during optimization | Wrong coarse evidence could cause catastrophic species omission. | Tests requiring the full safety union regardless of family rank. |

## Acceptance boundary for the replacement design

The fixed path is considered superseded only when tests demonstrate that:

1. every YOLOE-eligible Flickr unit gets a deterministic comparison-plan ID;
2. every candidate gets a global membership result and a local result or an
   explicit local-unavailable reason;
3. selection references existing embedding IDs and performs no BioCLIP model
   call;
4. quotas are observation/source-aware and shortfalls are preserved;
5. family ranking cannot prune the complete safety union;
6. score output exposes global/local components and disagreement rather than
   hiding them in one number;
7. uncertainty expansion reuses the same Flickr and reference embeddings; and
8. affected-only invalidation can trace from a reference revision to pool,
   score and downstream evidence artifacts.

## GitHits provenance

Subtask `geo-pool-0.1.2` used GitHits solution
`36171a86-56f0-4b48-936a-6bc08ec5589d` to check audit framing against MIT and
Apache-2.0 examples. Adopted concepts were class-distribution comparison,
explicit unavailable geography, cache identity and soft family diagnostics.
The generated NumPy implementation and score bonus were rejected. No external
code or prose was copied; the authoritative findings above come from BioMiner's
committed source and tests.
