# Geography-conditioned dynamic-pooling baseline

## Verdict

BioMiner starts this goal with a strong adaptive GBIF evidence lifecycle and
reusable, content-bound BioCLIP reference embeddings, but not yet with a
first-class dynamic global/local reference pool. The current implementation
fingerprints reference embeddings, robust prototypes, regional candidate sets,
raw rankings, diagnostics and remediation dependencies. It does not yet
materialize balanced support-image membership per Flickr comparison plan or
score global and local evidence as separate components.

This is a code-and-artifact baseline, not a live biological-performance claim.
The full repository suite passed with **2,531 tests**. The checked-in Papilio
pilot is explicitly fixture-backed; zero Flickr labels have been human reviewed,
the 50-record audit queue remains pending, and no scientific release is
authorized.

The machine-readable companion is
`reports/geo_dynamic_pooling/current_baseline.json`.

## Immutable snapshot

| Evidence | Value |
|---|---|
| BioMiner branch | `main` |
| BioMiner starting/origin SHA | `c7eaa9bf3696a25a0c8229837819dccec4fb9d66` |
| Adaptive production-code completion | `477eaface3d1f5efa51255550f0ef8d6a7740f35` |
| Python | `3.14.5` |
| uv | `0.11.19` |
| Locked packages | 44; `uv lock --check` passed |
| `uv.lock` SHA-256 | `49e0c67867b37b25cc5de0522889c76c0943b3efd9c2de757e87484be9432a7d` |
| `pyproject.toml` SHA-256 | `6b0e611cd1b605b8f4b05a77e67f941594bbafcf6beccad4ce2bec86f6b49d1e` |
| Full test gate | `uv run pytest -q`: 2,531 passed, 0 failed, 111.98 s |
| Active BioMiner/YOLOE/BioCLIP job | none detected during the snapshot |

Known user-owned untracked files were excluded from the baseline. The only
goal-owned pre-report change was the required GitHits ledger entry.

## Downstream pin movement

The goal was written against TaxaLens
`1440596cf4403af61ba8d57481feacda7c4e3044` and ButterflyLens
`c8135a0cb0001245215cdc774d063ef49407fb26`. The first audit observed newer
commits (`16242d1e97b4b7cee6823ed604232ebcc4436daf` and
`054f37f97d9c1872831114643ae5b48e33aa4107`), and this baseline snapshot
observed another advance:

| Repository | Current committed SHA | Worktree | Consumed here |
|---|---|---|---|
| TaxaLens | `c5e87ead4fdb26d5c5624bbb8d8d67e46d8eddbc` | dirty | no |
| ButterflyLens | `9a190c4ba799cff094608516f1a2b087a606f040` | dirty | no |

The written pins are not silently moved. Subtask 0.1.3 must compare each
written pin with the latest committed object and approve a compatibility
position before BioMiner consumes any new contract. Uncommitted sibling files
are out of scope.

## Adaptive defaults and scientific boundary

The production default is `adaptive_gbif_fast_start`, using GBIF provisional
support and `provisional_reference_ranking`. `human_verified_strict` remains a
compatibility mode and `human_verified_flagged_only` is available for targeted
remediation. The default admission policy is
`adaptive-gbif-fast-start-v1` under schema
`reference-admission-policy-v1.0.0`, fingerprint
`sha256:cc9ff21db9679b3a21785f3b1c9a822cdda6436cdd244783c489ffb87802d23b`.

Unreviewed support is limited to GBIF adult-field references that pass taxon,
licence, decoded-size, subject-area, canonical-media, YOLOE-route and
observation-independence gates. Provider support is not called human verified.
It cannot enter calibration or final-test partitions, raw scores are not called
probabilities, and final Flickr inclusion requires human review.

## Current stage graph

The adaptive production order is:

1. Resolve taxon scope, build the registry, measure geographic spread, compile
   queries, enqueue/poll Flickr, cluster Flickr geography and build regional
   candidates.
2. Acquire reference metadata/media, deduplicate, quality-route and admit.
3. Build frozen reference embeddings and provisional prototypes.
4. Score Flickr provisionally, then require Flickr human verification and a
   statistical reference audit.
5. When a species is statistically flagged, target its reference review. When
   that review creates a bank revision, rebuild affected reference evidence and
   selectively rescore affected Flickr records.
6. Apply the final quality gate.

Admission blocks first scoring; reference human review does not block the first
adaptive score. Flickr human verification blocks final inclusion. Statistical
flags do not prove that a reference identity is wrong.

The code also retains the 12-stage default workflow and the 21-stage strict
reference-first workflow. Their complete ordered arrays, plus the adaptive
dependency edges and manual-stage inventory, are in the JSON companion.

## Artifact baseline

### Reference embeddings

`reference-embeddings-v3.0.0` is a 59-column, normalized `float32` Parquet
contract. Its identity binds the support row and reference media to the visual
input, raw and transformed content hashes, transformation, model name/revision/
weights, preprocessing and preprocessing attestation. It also carries the taxon,
geographic cluster, route, visual domain, support split, admission policy,
review state and artifact fingerprints.

This is the right base for dynamic pooling because cached embeddings can be
reused without rerunning BioCLIP. However, `geo_cluster_id` is support metadata;
the embedding table does not itself express membership in separate global and
local pools for a particular Flickr comparison.

### Provisional prototypes

`provisional-prototypes-v1.0.0` groups support by accepted taxon, species,
route and visual-input kind. It supports normalized mean, mean-centred mean,
trimmed mean and medoid prototypes; records independent observations, support
maturity, dispersion, outliers and fingerprints; and deterministically balances
the number of observations admitted per species/route before aggregation.

There is no geographic grouping key. The default creates one prototype per
group, caps a species/route at 64 observations, trims 10%, requires two members
for a cluster and uses seed 42.

### Provisional ranking

`provisional-reference-ranking-v1.0.0` defaults to `trimmed_mean` and fixed
`top_k=3`. For each candidate taxon it computes prototype similarity, nearest
reference evidence and the top-k reference mean, but its ordering score is only:

`(prototype_similarity + top_k_reference_mean) / 2`

Geography is a nullable Boolean indicating whether any support row exactly
matches the requested `geo_cluster_id`. There are no separate global/local pool
IDs, component scores, quotas, distances or uncertainty-driven expansion. The
contract correctly labels output as uncalibrated similarity/margin evidence,
sets probability unavailable and mandates human review before final inclusion.

### Regional and visual candidates

`regional-candidate-species-v1.0.0` with policy
`regional-candidate-union-v1.1.0` fingerprints a target-preserving union of
regional same-family taxa, congeners, known mimics, historical false positives,
visual neighbours, taxonomic neighbours and country/bioregion/global fallbacks.
The default seeks at least 20 local same-family candidates and preserves a
global registry fallback.

Geographic evidence uses exact/buffer/country/bioregion/global scopes, spatial
resolution 5 and `inverse-uncertainty-100km-v1.0.0` confidence weighting.
Visual neighbours come from frozen global prototypes. Candidate taxa and
selection reasons are first-class; balanced support-image embedding IDs are not.

## Quality audit and escalation

Reference diagnostics (`reference-quality-diagnostics-v1.0.0`) measure
within-class similarity, competitor margin, influence, route/domain mismatch and
outlier score. The default review threshold is 0.35, and the artifact explicitly
records `taxon_misidentification_conclusion=not_assessed`.

The bank-quality policy requires at least 30 reviewed records per group, 95%
confidence intervals, weights for targeted queues, calibrated probabilities for
calibration metrics and a statistical audit for approval. Escalation policy
`adaptive-reference-escalation-v1` persists every triggered threshold: precision
lower bound 0.8, false-positive ceiling 0.2, recall floor 0.75, competitor-
confusion ceiling 0.2, prototype-dispersion ceiling 0.35, high-influence-outlier
rate ceiling 0.2, route-imbalance ceiling 0.5 and minimum reference count 5.
Only flagged species/reference groups enter targeted review.

## Pilot and human-review truth

The checked-in Papilio demoleus path demonstrates contracts and selective reuse
with fixtures. It is not a completed live run:

- live execution: `not_executed_missing_local_artifacts`;
- fixture integration: passed;
- human-reviewed Flickr labels: 0;
- representative audit queue: 50 rows, pending;
- statistical metrics: insufficient sample;
- escalation: deferred pending human Flickr labels;
- live remediation: blocked pending that review;
- legitimately flagged species: 0;
- scientific release authorized: false.

Historical counts of 81 provider-supported references, 81 embeddings, 26
prototypes and 13,496 classified Flickr records belong to an earlier prototype
workflow and are not current adaptive-pilot evidence.

## Gap carried into design

The next architecture must preserve current embedding identity, route separation,
observation independence, evidence maturity, fingerprinting and human/statistical
gates while adding:

- an explicit hybrid candidate plan that cannot be catastrophically pruned by
  family;
- deterministic balanced global and geographic support-image membership;
- first-class pool and comparison-plan fingerprints;
- independently reported global/local prototype, nearest-reference and top-k
  components;
- distance, coverage, route/domain and candidate-reason evidence;
- uncertainty-driven expansion that selects cached embeddings rather than
  re-encoding images.

No production-readiness, quality improvement, live-source availability or
downstream compatibility claim is made by this baseline.
