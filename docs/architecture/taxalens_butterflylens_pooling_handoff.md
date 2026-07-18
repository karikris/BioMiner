# TaxaLens and ButterflyLens pooling handoff audit

Status: Task 0.1.3 audit, updated by Task 1.2 compatibility review, 2026-07-18.
This document defines the
downstream constraints for BioMiner's geographic dynamic-reference-pooling
work. It is an interoperability audit, not evidence that a live model run or
human review occurred.

## Audit boundary and pin policy

The goal was written against TaxaLens
`1440596cf4403af61ba8d57481feacda7c4e3044` and ButterflyLens
`c8135a0cb0001245215cdc774d063ef49407fb26`. The most recent committed objects
audited were TaxaLens `c5e87ead4fdb26d5c5624bbb8d8d67e46d8eddbc` and
ButterflyLens `1cea643623f2f20a2bea72afc754c7b194db3278`. ButterflyLens's previous
audited pin was `fcee1a76886e37cb2f0d9badbe91b70a18a0e7c3`; Task 1.2 records the
additive, stricter-review compatibility decision separately. Both sibling
worktrees were dirty, so the audit used `git show`, `git grep`, and committed
trees only.

The executable compatibility fixture is
`tests/fixtures/downstream_pooling_contract_pins.json`. A newer sibling commit
must not silently replace either audited pin. Adoption requires a committed
fixture change, compatibility tests, provenance, and an explicit account of
semantic differences. BioMiner imports contracts and evidence artifacts, not
sibling implementation code or database state.

## TaxaLens contract findings

### Geographic review and quality

TaxaLens can stratify review by geographic cluster while retaining sampling
plan identity, representative versus targeted purpose, inclusion probability,
leakage groups, and reviewer independence. Geographic impact keeps separate
counts for candidates, reviewed positive and negative outcomes, uncertain and
pending items, media failures, skipped items, and release-ready candidates.
Summaries are projected at global, continent, country, and admin1 levels; a
source's coordinate precision must not be upgraded to a finer cell.

The quality sidecar contract is
`taxalens-verification-quality-snapshot:v1.1.0`. It carries review counts,
coverage, precision estimates, reviewer agreement, conflicts, reference
readiness, leakage results, milestone and data fingerprints, and a release
state of `not_evaluated`, `blocked`, or `release_ready`. Zero retained human
outcomes must stay an exact zero-review state with unavailable quality metrics;
it is not a zero error rate.

### Evidence maturity and release projections

Search hits and candidate rows are discovery evidence, not taxonomic labels.
Human review is a distinct evidence layer. The audited TaxaLens head tightens
the product language from a human-verified occurrence to a **release-ready
occurrence candidate** and requires a non-empty release-gate evidence ID.
Accordingly, BioMiner may export candidate and model evidence plus review and
quality dependencies, but must not infer occurrence release from a positive
review alone.

The geographic contracts preserve baseline availability, unavailable reasons,
distance availability, data-deficiency state, and separate reviewed versus
release-ready geographic additions. Missing baseline evidence means unknown,
not absence. The relevant stable schemas are:

- `taxalens-geographic-impact-cell:v1.0.0`
- `taxalens-geographic-impact-manifest:v1.0.0`
- `taxalens-geographic-impact-summary:v1.0.0`
- `taxalens-verification-quality-snapshot:v1.1.0`

### Artifact export expectations

TaxaLens's BioMiner label handoff is
`flickr_reviewed_labels_v2.parquet` with schema `reviewed-labels-v2`. It
requires a deterministic sort, unique photo identity, duplicate/owner split
integrity, campaign/question/sampling/decision hashes, effective append-only
review events, reviewer groups, blind-review state, and authority-appropriate
inverse-probability weights. Labels may only be derived from effective human
events, never from query text, provider labels, geography, detector output, or
BioCLIP scores.

Reference-review decisions use
`reference-review-decision-import-v1.0.0`, binding canonical request, media,
reviewer, round, timestamp, and outcome with deterministic validation.
Geographic manifests list every artifact's logical name, path, media type,
schema version, SHA-256, byte and row counts, snapshot identity, source
repository and commit, rights identity, availability, and unavailable reason.
BioMiner's new pool outputs must meet the same immutable-manifest standard.

## ButterflyLens contract findings

### Discovery, model evidence, and review layers

ButterflyLens separates discovery tables (`species`, name assertions, query
definitions and associations, API requests, Flickr photos) from durable media,
duplicate membership, pipeline stages, worker state, and `model_evidence`.
Model evidence kinds include YOLOE routes, BioCLIP embeddings, prototypes, and
candidate scores. Completed evidence requires model ID and revision, weights,
input and output hashes, and a unique evidence fingerprint. Raw scores remain
separate from calibrated probabilities, and a probability requires a named
calibrator.

Review is another layer: reviewer profiles, campaigns, assignments,
append-only events, consensus, reviewer reliability, and quality snapshots.
Provider assertions are not human review. BioMiner must preserve these layer
boundaries when a global/local pool comparison becomes a candidate-score
artifact.

The current pin further requires at least two repeated independent assignments
under `repeated-independent-v1`, blind model/peer evidence until decision, and
append-only authenticated review submission with correction lineage. BioMiner
may supply assignment inputs, but ButterflyLens assigns reviewers and database
identities and enforces those controls.

### Map impact, RLS, and import boundary

Geographic-impact rows model ALA, Flickr, YOLOE, BioCLIP, community, human,
release, distance, and date evidence with explicit `available`, `unavailable`,
`withheld`, or `not_applicable` states. Release candidates fail closed: human
support, consensus, an expert gate when required, coordinates, date, duplicate
independence, rights, quality, conflict freedom, and a complete evidence packet
must all pass.

Row-level security applies to every base table. Service-role ingestion is an
adapter responsibility; authenticated users receive no authority merely by
being authenticated. Review writes are identity-bound and append-only, and
approved/exported release transitions require curator or administrator
authority. BioMiner therefore publishes signed or hashed artifacts for a
downstream adapter and never writes directly to ButterflyLens tables or tries
to reproduce its RLS policy.

### Australian taxonomy and ALA geography

The Australian butterfly taxonomy pack contains 1,127 taxon records across six
families and uses AFD-backed stable `bltx:v1:*` keys. Its crosswalk records 553
complete, 560 partial, and 14 unresolved matches, with 741 open conflicts.
Those partial, unresolved, and conflict states must survive any BioMiner
candidate export; a family match cannot erase them or become a hard species
gate. The First Nations name review currently records zero assertions and zero
decisions, which must remain explicit.

The pinned ALA snapshot contains 236,897 normalized occurrence rows, 230,027
spatially eligible rows, and 23,744 aggregate cells at H3 resolutions 3, 5,
and 7 plus Australia, state, IBRA, and LGA summaries. These are provider
assertions, not human labels, and cannot support absence inference. Public
outputs must respect generalized precision. Rights review remains open for
16,753 records across three datasets, so the public release gate remains
blocked for affected products.

### Current evidence maturity

At the audited ButterflyLens commit, 2,906 media files had valid decodes but
zero were YOLOE-routed, zero had BioCLIP embeddings, and zero were human
verified. The committed states are `blocked_not_executed` for YOLOE and
`skipped_unfinished_by_goal_instruction` for BioCLIP. These values mean
unfinished or unavailable—not negative biological evidence.

The additive schema `butterflylens-classification-maturity:v1.0.0` exposes six
ordered states: butterfly detected, species candidate available, community
reviewed, quality estimate available, expert reviewed, and release ready. Each
state is either available with evidence fingerprints or unavailable with a
reason. `release_ready=true` requires all preceding states to be true, while
`scientific_claim_allowed` remains false in the interchange contract.
Fingerprint v1.1 (`butterflylens-evidence-fingerprint:v1.1.0`) uses a canonical
SHA-256 preimage with parent lineage. Content-addressed artifacts are immutable
and manifests publish last.

## BioMiner dynamic-pooling handoff requirements

The audited changes are additive and compatible, but the newer maturity
language is stricter and governs new output. Every global/local pool handoff
must therefore:

1. Pin BioMiner, TaxaLens, ButterflyLens, registry, source-snapshot, model,
   preprocessing, comparison-plan, and schema identities.
2. Keep cached embedding identity separate from pool membership so alternate
   comparison plans do not recompute image embeddings.
3. Export the global pool, local pool, union/candidate set, component scores,
   geography state and precision, family-routing contribution, effective `k`,
   support counts, exclusions, and fallback reason without calling a score a
   probability.
4. Preserve taxonomy conflicts, duplicate/owner groups, rights, coordinate
   precision, sampling purpose and inclusion probability.
5. Represent unrun, missing, withheld, and not-applicable evidence explicitly;
   never coerce them to `false`, zero, absence, or a failed biological claim.
6. Publish deterministic artifacts first and a manifest last, with SHA-256,
   bytes, rows, schemas, parent fingerprints, producer repository/commit, and
   unavailable reasons.
7. Treat database import and RLS as downstream adapter concerns. No BioMiner
   report or export is release authority by itself.
8. Require separate compatibility fixtures and migrations for future contract
   changes; never reinterpret retained historical evidence in place.

This audit establishes compatibility expectations only. It does not claim that
dynamic pools, live YOLOE/BioCLIP inference, statistical review, expert review,
or occurrence release have completed.
