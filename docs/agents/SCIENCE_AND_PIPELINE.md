# Scientific and pipeline rules

## Authority and identity

BioMiner's current registry is CoL XR-grounded and retains source evidence,
versions, accepted concepts, paths, name assertions, collisions, and query
eligibility.

Use versioned registry artifacts as identity authority:

```text
taxa.parquet
species_paths.parquet
taxon_relations.parquet
names.parquet
name_collision_ledger.parquet
name_evidence.parquet
source_snapshots.parquet
flickr_query_definitions.parquet
qa_findings.parquet
manifest.json
```

GBIF provides occurrence, geographic, media, and reconciliation evidence. A
provider assertion does not silently replace the accepted registry concept.

Taxonomic conflicts remain explicit. Do not join by display name when stable
keys exist.

## Name trust and query compilation

- Preserve T1–T5 source trust and review state.
- Generated translations remain low-trust candidates.
- One canonical normalized keyword may drive a physical request.
- Retain every source/name/taxon association.
- Tags and text are distinct query definitions.
- Collision, homonym, generic-token, and source-evidence QA may disable a term.
- Query eligibility never implies species identity.

## Flickr discovery

- Fetch metadata before media.
- Deduplicate canonical photo processing, not query-hit provenance.
- Preserve every logical query association and physical request fingerprint.
- A broad genus, family, order, or butterfly query does not support every
  descendant species.
- Metadata, title, tags, description, and comments create evidence or
  candidates, not truth.
- API accounting is application-wide and fail closed.
- Use the current planner and provider documentation for page sizes, windows,
  partitions, and quotas; do not duplicate constants in new modules.
- Resume pending work deterministically and requeue stale claims through the
  workstore contract.

## Geography

- Use accepted taxon keys and versioned geographic artifacts.
- Preserve coordinate uncertainty and source issues.
- Use compatible hierarchical spatial cells.
- Geography may:
  - build regional candidate sets;
  - select local references;
  - prioritize review;
  - provide structured features.
- Geography may not:
  - prove identity;
  - set the target to zero;
  - convert a query hit into an occurrence;
  - interpret no source evidence as biological absence.
- `no_geo`, unassigned, coarse fallback, and data-deficient states remain
  explicit.

## Candidate generation

For target-aware work:

- always preserve the target;
- include regional competitors, close congeners, known mimics, historical false
  positives, visual neighbours, and configured fallback candidates;
- persist every candidate reason and source version;
- do not delete species because a family/genus text rank was weak;
- higher-rank evidence is diagnostic or derived after species scoring.

The hierarchical cascade implementation and its generic metrics, review queue,
visual QA, charts, Xie adapter, row-compatibility branches, and mode aliases
have been removed. Production and diagnostics must not recreate them as a
fallback. Use target-verification and dynamic-pool evaluation contracts.

## Reference acquisition and admission

Accepted authority:

```text
docs/architecture/adaptive_gbif_reference_admission.md
src/biominer/references/admission.py
src/biominer/references/admission_eligibility.py
src/biominer/references/admission_compiler.py
src/biominer/references/readiness.py
```

Modes:

```text
adaptive_gbif_fast_start       current default
human_verified_strict          strict compatibility/high-stakes
human_verified_flagged_only    remediation
```

### Adaptive terminology

Use exactly:

```text
GBIF provider-asserted provisional support
```

Never call it:

```text
verified
human verified
ground truth
expert confirmed
```

The admission policy is immutable and fingerprinted. It owns provider,
reconciliation, licence, decode, dimension, subject-area, route, canonical
media, observation/observer diversity, and audit requirements.

Do not reimplement admission gates outside the policy evaluator.

A human rejection overrides provider assertion.

### Reference separation

Never mix incompatible:

```text
adult field
larval
pupal
egg
pinned specimen
artifact
ambiguous or unsuitable
```

One burst or one observation does not satisfy several independent support slots.

## YOLOE and visual inputs

YOLOE provides:

- subject presence;
- route;
- life-stage/domain evidence;
- detection and mask evidence;
- subject-area and quality flags.

It does not decide species.

Target-aware production uses the accepted canvas-preserving visual contract:

```text
raw_full_image
focused_full_frame
masked_full_frame
multi_object_full_frame
```

Do not silently spatially crop or manufacture detail. If the subject is too
small, lower evidence, abstain, or route to review.

Detector-crop modes have been removed. The stable detection schema retains
nullable historical crop columns only for consumer compatibility; current
producers must leave them null and `not_created`.

## BioCLIP

- Keep BioCLIP frozen.
- Load once per persistent worker.
- Embed each content/input/model/preprocessing fingerprint once.
- Reuse embeddings across candidate sets and reruns.
- Keep reference, Flickr, route, life-stage, domain, and split identity in the
  embedding artifact.
- Build balanced, observation-level prototypes.
- Support global, regional, metadata, and within-species multi-prototype groups
  only under versioned policies.
- Provider label count must not create class-size advantage.

### Provisional ranking

`provisional_reference_ranking` may use:

- prototype similarity;
- nearest-reference similarity;
- fixed top-k support mean;
- raw competitor margin;
- geography and domain compatibility.

It must expose:

```text
probability_available = false
mandatory human review before final inclusion
```

unless a valid independent calibrator is explicitly available.

## Geography-conditioned dynamic pooling

The accepted architecture and human governance records are:

```text
docs/architecture/geography_conditioned_dynamic_pooling.md
docs/architecture/statistical_support_and_human_verification.md
docs/governance/geography_conditioned_dynamic_pooling_policy.md
docs/schemas/geography_conditioned_dynamic_pooling_contracts.md
```

Software and fixture behavior is implemented through Phase 15, including
exact downstream handoffs and a complete 24-variant pilot decision. The pilot
outcome is `insufficient_evidence`: zero variants are eligible, no candidate
strategy, pool variant or fusion method is selected, and runtime settings are
unchanged.

The following remain mandatory:

- preserve the complete target-aware candidate union under every family and
  geography schedule;
- give every candidate global evidence and either local evidence or an exact
  local-unavailable reason;
- keep the immutable image embedding key independent of pool membership and
  reuse compatible embeddings across pool changes;
- preserve raw components, disagreement, coverage and alternatives rather than
  relabelling a fused score as probability;
- keep representative review, targeted discovery and occurrence-release work
  as separate contracts; and
- bind any later production selection to eligible live/review evidence and its
  exact fingerprints.

A fixture projection, CLI plan, successful handoff or consumer import does not
authorize a production default, human verification, statistical support,
occurrence release or publication maturity.

## Human review and release

Flickr scoring may precede review. Final inclusion may not.

A final Flickr row requires:

- decisive source-hash-bound human review;
- resolved duplicate group;
- supported identity;
- compatible life stage/domain;
- required geography/date evidence;
- independent second review or adjudication when required;
- exact upstream and reviewed-label fingerprints;
- final release permit.

These never qualify:

```text
unreviewed
pending
Skip
Can't view
uncertain
conflict
stale image hash
```

Reference review and Flickr review are separate evidence workflows.

## Statistical audit and remediation

Use independently human-reviewed Flickr labels to audit provisional banks.

- Keep representative audit and targeted failure discovery separate.
- Preserve inclusion probabilities and grouping.
- Use weighted estimates when required.
- Report insufficient samples as unavailable.
- Statistical findings prioritize human reference review; they do not prove an
  individual reference is wrong.
- Flag species/groups through a versioned escalation policy.
- Revise only affected support.
- Reuse unchanged embeddings.
- Rebuild affected prototypes/models only.
- Rescore affected Flickr records only.
- Compare paired before/after evidence.

## Comments

Comments may support, contradict, add alternatives, or identify life stage.
They must not:

- override hard visual negatives;
- create an unresolved taxon as accepted;
- release a record;
- hide candidate-set revisions.

When a comment adds a valid species candidate, reuse the existing full-frame
embedding and rescore the expanded candidate union.

## Evidence vocabulary

Keep these distinct:

```text
query provenance
provider assertion
automated QA
raw model evidence
calibrated probability
human review
consensus
quality estimate
release permit
scientific publication
```

Every public or report field must disclose the correct maturity.

## Removed legacy paths

The family-first hierarchy, genus shortcut, detector-crop path,
Gold/Silver/Bronze/Bin buckets, comment promotion, direct cloud scoring, and
rolling worker have been deleted. Preserve historical evidence in Git and
frozen reports; do not restore executable compatibility wrappers to satisfy a
stale test or artifact.

Migration requires explicit mode boundaries, artifact compatibility, replacement
tests, and updated docs.
