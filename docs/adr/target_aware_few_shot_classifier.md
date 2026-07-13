# Target-aware Few-shot BioCLIP Classifier

Status: accepted

Date: 2026-07-13

## Context

BioMiner currently answers an open-world text-ranking question. Depending on
the selected mode, it either ranks a target-scoped species set or progressively
deletes taxonomy paths before ranking species. Neither design is a defensible
production answer to the pilot question:

> Does this image contain *Papilio demoleus* when compared with verified
> target images and the geographically plausible species most likely to be
> confused with it?

Flickr query matches are discovery evidence, not labels. GBIF and the unified
registry define taxonomic identity, while BioCLIP remains visual screening
evidence. The new classifier therefore needs a target-verification decision in
addition to a regional species ranking, and it must be able to abstain.

This ADR defines the intended architecture. It does not change the production
default; that requires the reviewed pilot and acceptance evidence specified by
the migration plan.

## Decision

Add a separate `target_aware_few_shot_classification` mode with two related
outputs:

1. A regional multiclass ranking over the target and known competitors.
2. A binary target-versus-rest verification result.

The target may be confirmed only when both outputs satisfy a versioned,
calibrated decision policy. Otherwise the result identifies a known non-target
class or abstains.

The mode has these invariants:

- Always score the target and the complete regional competitor union.
- Never remove species because of family, subfamily, tribe or genus scores.
- Use higher-rank scores only as diagnostics or structured features after
  species scoring.
- Keep BioCLIP 2.5 Huge frozen and load it once per persistent worker.
- Cache image embeddings by content, visual-input, model, revision and
  preprocessing fingerprints.
- Compare query image embeddings with manually verified support-image
  embeddings, prototypes and explicit negative banks.
- Keep adult field, larval, pupal, pinned-specimen and artifact reference banks
  separate.
- Retain the complete image canvas for every target-aware BioCLIP input.
  Detector boxes and masks may create full-frame attention variants but not
  spatial crops.
- Treat geography as candidate-selection and structured prior evidence. Missing
  or contradictory geography expands review or fallback behavior; it never
  proves absence or presence.
- Fit probability calibrators and abstention thresholds on data disjoint from
  classifier fitting. Persist split, model, data and threshold fingerprints.
- Block Flickr vision until the regional candidate and reviewed reference-bank
  readiness contract passes.
- Store durable data and model artifacts in S3 and durable work state in
  PostgreSQL. Local media remains a temporary cache.

The existing hierarchical and target-scope text modes remain explicit
diagnostic baselines during migration. Their historical outputs retain their
existing interpretation.

## Why Hierarchy Pruning Is Invalid

The classification-v3 cascade selects a fixed top-three beam at intermediate
ranks. The legacy seven-rank layout selects family top one, then genus top 20
and genus top three; a strictly greater than 0.90 shortcut can reduce the genus
set to one. Tests explicitly verify that species below excluded genera are
never scored.

This is catastrophic for target verification. A weak or wrong higher-rank text
prompt makes the target unobservable rather than merely lowering its evidence.
Once deleted, a species cannot recover even if its image embedding strongly
matches verified target support. Target verification requires the target and
known competitors to remain scoreable regardless of a family or genus rank.

## Why The Current Rerank Is Not Image-reference Reranking

Both current modes compute a species first pass from text prompts, retain up to
20 species and score those species again with a distinct text-prompt stage.
The current target-scope path also filters the global top 20 to family top one
when family metadata is present. No verified support image participates in
that second pass.

Changing or narrowing text prompts may alter scores, but it introduces no new
visual evidence. The new reranker must compare the already computed query
image embedding with verified reference embeddings, local/global prototypes
and explicit competitor or domain-negative evidence. Text remains an
independent supporting feature.

## Why Raw Scores Are Not Probabilities

`TaxonomyTextEmbeddingIndex.raw_similarities` returns normalized image-to-text
dot products. Candidate-relative softmax values are explicitly diagnostic.
The cascade records raw similarity margins, while the legacy genus guardrail
clamps the leading raw value into `[0, 1]`. The current evaluation report then
computes heuristic ECE from `species_top1_score`.

None of the following is a probability without a fitted calibrator:

- image-to-text cosine similarity;
- image-to-image cosine similarity;
- a difference between target and competitor similarities;
- a nearest-centroid or nearest-neighbour score;
- an SVM `decision_function` value.

Clamping, softmaxing a changing candidate set, or dividing by a temperature
chosen without held-out fitting does not create calibration. Probability-like
fields in the new mode must originate from a named calibrator fitted on
independent predictions, with method, split fingerprint, sample size and
training-data fingerprint persisted.

## Why Geography Cannot Certify A Label

The current range artifact aggregates GBIF country facets and can add same-
family or same-genus candidates. It does not describe reviewed image identity,
and GBIF occurrence coverage is incomplete, heterogeneous and affected by
preserved specimens, coordinate quality, historical records and introductions.
The Flickr stream is itself query-conditioned and cannot be treated as an
occurrence sample.

Geography may select plausible competitors, choose locally relevant support
images, contribute structured features and raise review priority. It cannot set
the target score to zero, convert a query match into truth, or turn absence from
an occurrence table into biological absence.

## Why Reference Images Require Manual Verification

GBIF taxon labels and iNaturalist Research Grade are useful preselection
evidence, not a BioMiner production review decision. Records may be
misidentified, mirrored across providers, licensed differently at occurrence
and media level, visually unsuitable, captive, preserved, obscured or assigned
to the wrong life-stage/domain bank.

Only rows with `verification_status = "verified"` may enter production support
embeddings. Verification must follow duplicate resolution and preserve source,
observation, creator, rights, licence, geographic, life-stage and visual-domain
provenance. Ambiguous, candidate and conflicting rows remain outside the
support split.

## Implementation-path Audit

| Path | Current responsibility and constraint | Target-aware migration boundary |
|---|---|---|
| `src/biominer/registry/range_discovery.py` | Resolves accepted species and writes country-facet counts to `range_countries.parquet`; no spatial cells, dataset-level occurrence provenance or resumable bulk acquisition. | Keep country evidence for compatibility. Add a separate multi-resolution geographic-spread compiler with occurrence suitability, dataset and snapshot provenance. |
| `src/biominer/registry/build.py` | Adds country range outputs only when a range seed is configured and includes them in local canonical promotion. Classification overlays are already removed in favor of unified `species_paths.parquet`. | Add geographic-spread/summary/QA stages and manifest counts without coupling taxonomy identity to occurrence absence. |
| `src/biominer/registry/publish.py` | Publishes the core unified-registry tables plus cell-level spread and species summaries after base and geographic fatal QA. It merges geographic provenance/QA, inventories checksums, and treats missing spread as unknown rather than negative. | Keep later regional candidate generation downstream of this evidence contract; publication must never convert geographic absence into a species exclusion. |
| `src/biominer/bioclip/candidate_sets.py` | Always inserts the target, limits registry candidates to target/same genus/same family, and adds geographic, query, metadata and comment candidates. It lacks regional reason lists, mimic/false-positive relationships and reference-derived neighbours. | Build a versioned regional union artifact. Preserve every inclusion reason and always retain the target; metadata and query evidence cannot become labels. |
| `src/biominer/bioclip/object_runner.py` | Default target-scope mode uses text scores, family-top-one filtering, a second text pass and raw-score bucket rules. It materializes crop, segmentation-crop or whole-image ablations and already exposes image embedding hooks. | Add a separate scorer that consumes cached full-frame embeddings, reference indexes, classifiers and calibrators. Do not silently change legacy row semantics. |
| `src/biominer/bioclip/path_cascade_classifier.py` | Prunes active paths at each rank; species under excluded genera are not scored. It reranks all retained top-20 species, but only with distinct text prompts. The legacy 0.90 genus shortcut clamps a raw score. | Retain as a diagnostic baseline. Target-aware scoring must bypass path pruning and the raw-score shortcut entirely. |
| `src/biominer/bioclip/path_taxonomy_store.py` | Validates exact schemas, checksums, taxonomy/prompt fingerprints and deterministic active paths. | Reuse the fail-closed fingerprint pattern for reference banks, prototypes, classifiers and calibrators; do not reuse active-path filtering for target eligibility. |
| `src/biominer/bioclip/taxonomy_embedding_cache.py` | Precomputes normalized text embeddings and validates model/taxonomy identity. Raw similarities are separated from diagnostic candidate-relative softmax values. | Keep as text evidence. Add a separate content-addressed image/reference embedding cache and prevent diagnostic values from being named probabilities. |
| `src/biominer/detection/policy.py` | Defines YOLO thresholds, crop defaults, Mac runtime settings and a narrow `butterfly_like` crop-eligibility label. | Replace the species-path gate with explicit visual-domain routes. Profile settings must feed every full-frame visual variant and worker path. |
| `src/biominer/detection/pipeline.py` | Runs bounded detection, emits stable detection rows, skips production crop metadata for noneligible detections and uses PIL/LANCZOS when available with nearest-neighbour fallback. | Keep YOLO as gate/router and retain all detection metadata. Target-aware processing must not materialize detector crops; subject-size/detail signals become abstention features. |
| `src/biominer/vision/gates.py` | Scores detected non-hard-negative rows as detector crops, falls back to whole image for no detection and excludes hard negatives. It has no life-stage or specimen routing. | Introduce adult, larval, pupal, specimen, insect, artifact, no-organism and ambiguous routes. Every target-aware scored route uses complete-canvas inputs. |
| `src/biominer/run/orchestrator.py` | Runs registry, Flickr, detection and BioCLIP stages in request order. It has no regional-candidate/reference readiness dependency before vision. | Add explicit reference-first stages and fail detection/scoring with actionable errors when readiness, fingerprints or verified support are missing. |
| `src/biominer/run/stages.py` | Defines the current linear discovery-to-comments stages. | Add geographic, clustering, candidate, reference, embedding, training, calibration, readiness and target-scoring stages. Manual review cannot auto-complete. |
| `src/biominer/run/paths.py` | Provides matching local and S3 URIs for current registry, detection, score, evidence and report artifacts. | Add mirrored durable paths for every geographic, reference, model, calibration and target-aware artifact; source media remains S3-only. |
| `src/biominer/evaluation/labels.py` | Reviewed-label v1 stores basic object/photo taxonomy and reviewer metadata. It lacks target presence, life stage/domain, duplicate/owner groups, dataset split and second review. | Add a v2 schema plus a v1 migration reader and leakage-relevant group fields. |
| `src/biominer/evaluation/metrics.py` | Evaluates hierarchical rows only with family/species top-k metrics and basic butterfly errors; target-scope rows are counted but excluded. | Add binary target-verification, selective prediction, OOD and stratified metrics over reviewed labels while retaining cascade diagnostics. |
| `src/biominer/evaluation/reports.py` | Writes hierarchical metrics/confusions and heuristic ECE using raw `species_top1_score`. | Consume `calibrated_target_probability`, calibration split metadata and grouped confidence intervals. Raw-score reliability plots are legacy diagnostics only. |

## Cloud-worker And Test Audit

`src/biominer/bioclip/cloud_work.py` and
`src/biominer/vision/cloud_work.py` already build deterministic work identities,
enqueue PostgreSQL work and commit S3 shards before work completion. Their
identities are currently crop/text-cascade oriented and have no reference-bank,
prototype, classifier or calibrator version. `src/biominer/vision/rolling_worker.py`
already overlaps bounded image, YOLO, scoring and commit stages and deletes
temporary inputs only after commit, but its default scoring path remains
detector-crop based.

These are extension points, not replacement targets. New work identities must
include reference, visual-input, model, preprocessing, classifier and
calibration fingerprints. Workers must load BioCLIP and immutable indexes once
per version and refresh only on version change.

Current tests strongly cover deterministic country facets, registry build,
taxonomy fingerprints, exact text-embedding caches, cascade pruning, crop
materialization, vision gates, bounded rolling workers and hierarchical
top-k reports. Several tests deliberately enforce behavior that is invalid for
the new mode, including deleting species below the genus beam and using crop
inputs. Those remain valid legacy-mode tests. New target-aware tests must prove
that the target survives wrong higher-rank scores, full-frame embeddings are
reused, life-stage banks remain separate, reference readiness gates vision,
calibration is leakage-safe and unsupported fingerprints fail closed.

## Artifact And Ownership Boundaries

- The unified registry owns accepted identity and taxonomy paths.
- Geographic artifacts own occurrence-derived candidate plausibility with
  provenance, never identity or truth.
- Reference acquisition owns candidate media and licences.
- Manual review owns support eligibility.
- Embedding/prototype artifacts own frozen visual representations.
- Classifier artifacts own raw decision functions.
- Calibrator artifacts own probability mappings and learned thresholds.
- Target-aware score artifacts own per-image evidence and abstention outcomes.
- Evaluation artifacts own claims about performance on frozen reviewed sets.

Each boundary is versioned and fail-closed. Downstream artifacts record all
upstream fingerprints needed to reject stale or mixed inputs.

## Alternatives Rejected

### Preserve The Taxonomic Cascade As The Production Verifier

Rejected because a higher-rank text error can prevent the target and its true
competitors from being scored.

### Treat A Second Prompt Pass As Few-shot Reranking

Rejected because it introduces no support-image evidence.

### Threshold Raw Cosines Or SVM Margins Directly

Rejected because the values are not calibrated probabilities and change with
model, prompt, support set and candidate composition.

### Use Geographic Absence As A Hard Negative

Rejected because occurrence coverage is not proof of biological absence and
Flickr coordinates may be missing or weak.

### Auto-verify GBIF Or Research-grade iNaturalist Images

Rejected because source labels are candidate evidence and do not resolve
duplicates, media suitability, licence, life stage or visual domain.

## Consequences

- Reference acquisition and manual review become prerequisites rather than
  optional enrichment.
- Inference becomes more expensive than a pruned cascade, but frozen query
  embeddings are computed once and reused across competitors and ablations.
- The system can return a known competitor, open-set non-match or abstention
  instead of forcing a target label.
- Missing geography or support coverage reduces acceptance coverage rather
  than silently increasing false certainty.
- Historical object-score tables remain readable; the new schema and mode make
  changed semantics explicit.
- The production default changes only after reviewed pilot evidence meets the
  configured acceptance policy. Until then rollback is selecting the existing
  diagnostic mode, not weakening target-aware thresholds.
