# Adaptive GBIF reference admission

- Status: accepted
- Date: 2026-07-17
- Decision ID: `adaptive-gbif-reference-admission-v1`

## Context

BioMiner's normal reference path currently requires every support image to
complete human review before it can enter the support manifest, reference
embeddings, or prototypes. That is defensible for a publication-critical bank,
but it makes manual review the critical path to the first useful BioCLIP
comparison. A large, carefully screened GBIF bank may therefore wait idle even
when metadata, licence, content, duplicate, route, and image-quality evidence
already make it suitable for provisional screening.

The repository's explicit Build Week prototype proved a narrower fact:
provider-supported references can be used for prototype-only retrieval while
retaining zero human-verified labels, no calibrated probability claim, and no
scientific-release authority. It did not create a general admission contract,
statistical escalation policy, or incremental remediation workflow.

Two extreme policies are unacceptable:

1. Requiring manual review of every GBIF reference before first inference is
   slow, spends scarce expert attention before model failure modes are known,
   and delays embeddings, prototypes, candidate prioritization, and review
   sampling.
2. Skipping controls and treating provider labels as truth is unsafe. GBIF
   records can contain taxon conflicts, unsuitable bases of record, absent or
   fossil occurrences, invalid media, duplicate bursts, licensing gaps,
   attribution gaps, route mismatches, small subjects, artifacts, and
   provider-label errors.

Adaptive escalation is the intended compromise: perform strict deterministic
screening first, use the qualifying bank only as **GBIF provider-asserted
provisional support**, evaluate the resulting system against independently
human-reviewed Flickr labels, and spend reference-review effort on species and
images where measured evidence says it is most valuable.

This ADR partially supersedes the “Why Reference Images Require Manual
Verification” prerequisite in
`docs/adr/target_aware_few_shot_classifier.md`. It removes universal
pre-inference review only for provisional support and provisional scoring.
It does not supersede that ADR's human-review requirements for strict support,
calibration labels, final-test labels, probability claims, or scientific
release.

## Decision

Add three explicit, versioned reference-admission modes:

1. `adaptive_gbif_fast_start`
   - new production default;
   - admits only qualifying GBIF provider-asserted images as provisional
     support;
   - permits frozen embeddings, route-separated prototypes, nearest-reference
     evidence, and provisional ranking without prior reference review;
   - requires a versioned downstream statistical audit;
   - escalates only flagged species/reference groups to human review.
2. `human_verified_strict`
   - preserves the existing fail-closed behavior;
   - admits only completed, human-verified support;
   - remains available for publication-critical and high-stakes work.
3. `human_verified_flagged_only`
   - starts from an existing provisional bank;
   - requires human review for statistically flagged species/reference groups;
   - is used for remediation and selective reruns.

The mode is never inferred from absent decisions. The default is represented
by the constant:

```python
DEFAULT_REFERENCE_ADMISSION_MODE = "adaptive_gbif_fast_start"
```

Every mode and policy is explicit in configuration and is included in the
semantic identity of support rows, readiness permits, embeddings, prototypes,
classifiers, calibrators, score outputs, reports, and revision-impact
artifacts.

## Evidence vocabulary

### GBIF provider-asserted provisional support

The only approved term for an automatically admitted, unreviewed GBIF
reference is:

> GBIF provider-asserted provisional support

It means:

- the provider supplied a taxon assertion;
- BioMiner reconciled that assertion to the requested accepted GBIF taxon;
- all configured deterministic admission gates passed;
- the image may support provisional visual comparison; and
- no independent human taxonomic verification is claimed.

An unreviewed GBIF image is never called verified, human verified, ground
truth, or expert confirmed.

### Strict human-verified support

Strict support is bound to a completed append-only review outcome,
`verification_status=verified`, `target_identity_verified=true`, decisive
life-stage/domain/view decisions, reviewer identity, source image hash, and all
existing support gates.

### Candidate Flickr evidence

An unreviewed Flickr image remains **candidate evidence**. It may be scored for
prioritization, triage, sampling, competitor identification, and review-queue
generation. It cannot enter a final occurrence dataset until a valid human
review outcome and every release gate pass.

Provider-asserted GBIF references and human-reviewed Flickr labels remain
different evidence sources and different review workflows.

## Admission policy

A typed, immutable `ReferenceAdmissionPolicy` owns admission behavior. Its
semantic fingerprint includes at least:

- schema and policy versions;
- mode;
- allowed provider sources;
- allowed unreviewed routes;
- accepted taxon reconciliation states;
- accepted licence policy states;
- decoded width and height minima;
- subject-area threshold;
- YOLOE route requirement;
- canonical-media requirement;
- images-per-observation limit;
- observer/photographer diversity rule;
- research-only licence permission;
- statistical-audit requirement; and
- audit policy version.

A mode or policy change invalidates stale readiness, embeddings, prototypes,
classifiers, calibrators, and scores through fingerprint mismatch. Old strict
artifacts remain readable as strict artifacts or require an explicit migration;
missing mode never becomes fast-start.

## Automated admission gates

An unreviewed image may enter provisional support only when all applicable
gates pass:

1. Source is GBIF.
2. Taxon reconciliation is exact or an accepted-name synonym resolving to the
   same accepted taxon key.
3. `uncertain_taxon_match=false`.
4. Occurrence is not absent.
5. Fossils are excluded.
6. Media is a supported still image.
7. Download completed.
8. Content type is valid.
9. Image decode succeeded.
10. Image SHA-256 exists.
11. Licence policy accepts the configured use.
12. Creator, source, and attribution are present.
13. Exact and perceptual duplicate processing completed.
14. Media is canonical.
15. Unresolved duplicate conflicts are excluded or sent to targeted review.
16. Provider-supplied identity matches the accepted candidate taxon.
17. Observation independence is enforced:
    - one image per observation by default;
    - one image per observer/photographer before additional images;
    - bursts and near-identical views do not fill independent quota slots.
18. YOLOE reference-quality routing completed.
19. YOLOE route is compatible with the requested bank route.
20. Artifact, logo, tattoo, and no-organism routes are excluded.
21. Ambiguous visual domains are excluded or sent to targeted review.
22. Subjects below the configured area threshold are excluded or flagged.
23. Full-frame visual-input generation succeeded.
24. Local/regional prototype use requires usable geography; a coordinate-less
    reference may still support a global prototype.

The default unreviewed route is `adult_field`. Any other unreviewed route must
be explicitly allowed. Adult field, larval, pupal, specimen, artifact, and
other incompatible domains never share one prototype.

YOLOE is a quality gate and router. It does not decide species identity.

Admission evaluation is pure: it returns admitted, excluded, or
review-required with complete reason codes and never mutates source records.
A human rejection always overrides a provider assertion.

## Provisional readiness

Add `ready_provisional` as a distinct readiness state. Its permit may authorize:

- reference embedding;
- route-separated prototype construction;
- nearest-reference and top-k support evidence;
- raw prototype/competitor margins;
- provisional Flickr ranking; and
- review-campaign generation.

It always requires:

- an explicit adaptive mode and policy fingerprint;
- qualifying provisional target support;
- route separation;
- complete provenance and automated QA;
- a versioned statistical audit plan;
- mandatory downstream Flickr review; and
- no effective human rejection for an admitted row.

By itself it never authorizes:

- calibrated probabilities;
- final-test or calibration labels;
- unreviewed Flickr output;
- population prevalence claims;
- scientific release; or
- public-display rights.

`ready` retains strict behavior. Readiness reports separate provisional counts,
human-verified counts, automated exclusions, human exclusions, and
review-required counts.

## Provisional scoring semantics

The initial scoring mode is `provisional_reference_ranking`. It may combine:

- frozen BioCLIP reference embeddings;
- robust species/route prototypes;
- nearest references and top-k means;
- raw target-versus-competitor margins;
- compatible geographic prototype scope; and
- domain/route compatibility.

Every output records:

- reference admission mode, policy version, and policy fingerprint;
- bank, embedding, prototype, model, preprocessing, and candidate fingerprints;
- provider-asserted and human-verified support counts;
- raw reference and competitor evidence;
- a provisional decision state;
- `probability_available=false` unless a valid independent calibrator exists;
- evidence maturity; and
- required human-review state.

Cosine similarity, nearest-reference similarity, prototype margin, and SVM
decision values are non-probabilistic. A probability-like output requires an
independent calibrator fitted only from appropriate human-reviewed labels.
Provider-asserted GBIF references cannot enter calibration or final-test labels.

Provisional scoring is allowed to abstain. Missing route-compatible support,
insufficient subject evidence, incompatible domains, stale fingerprints, or an
unsatisfied audit policy must produce explicit unavailable/abstain states, not
fallback confidence.

## Mandatory Flickr review and release

Machine scoring may precede Flickr review. Final occurrence inclusion may not.
A final Flickr row requires:

- a decisive human review;
- source-image-hash binding;
- resolved duplicates and conflicts;
- supported target identity;
- suitable life stage and visual domain;
- required geographic/date evidence;
- any required independent second review; and
- a release permit that includes the exact reviewed-label and upstream
  fingerprints.

`Skip`, `Can't view`, uncertain, pending, conflict, stale-hash, and unreviewed
outcomes never count as verified and never enter final occurrence exports.
No reference-admission mode weakens this boundary.

## Statistical species audit

The adaptive path must audit performance by target species, competitor species,
region, route, life stage, visual domain, source dataset, and admission basis.
It uses independently human-reviewed Flickr labels, not provider assertions.

Where sampling and sample size permit, report:

- precision and recall;
- false-positive and false-negative rates;
- PR-AUC;
- coverage and abstention rate;
- competitor confusion;
- grouped confidence intervals;
- calibrated metrics only when a valid calibrator exists; and
- raw-margin distributions otherwise.

Representative probability samples and targeted failure-discovery queues remain
distinct. Weighted estimates are required when sampling probabilities differ.
An insufficient sample produces an explicit unavailable result, not a guessed
metric.

Statistical evaluation may identify poor species performance, suspicious bank
dispersion, likely outlier references, and inadequate competitor coverage. It
does not prove that each unreviewed reference is correctly identified.

## Escalation and targeted reference review

A versioned `ReferenceBankQualityPolicy` maps measured evidence to persisted
flag reasons, including:

- precision/recall objectives not met;
- excessive false positives or false negatives;
- high target/competitor confusion;
- insufficient independent audit sample;
- excessive prototype dispersion;
- high-influence embedding outliers;
- route imbalance; and
- support shortfall.

Only flagged species/reference groups enter targeted reference review.
Individual references are prioritized by outlier score, competitor similarity,
prototype influence, route mismatch, provider concentration, repeated
involvement in errors, subject-area weakness, and duplicate ambiguity.

Outlier status means “review priority,” not “misidentified.” Review decisions
remain append-only and may verify, exclude, mark uncertain, correct route/view,
identify a possible alternative species, or require a second review.

## Bank revision

A review-driven revision:

1. excludes rejected references;
2. promotes verified references;
3. preserves unflagged provisional references;
4. increments the reference-bank version;
5. publishes an old-to-new change manifest;
6. fingerprints every changed semantic row; and
7. calculates downstream impact before execution.

The revision report distinguishes reviewed, verified, excluded, unchanged, and
still-provisional support. It does not rewrite provider assertions or historical
artifacts.

## Selective rerun

Content-addressed identity is the reuse boundary:

- unchanged image/model/preprocessing combinations reuse embeddings;
- unchanged YOLOE input/detector/policy combinations reuse route evidence;
- excluded references are filtered without recomputing their vectors;
- only new or content-changed references are embedded;
- only affected species/region/route prototypes and model rows are rebuilt;
- calibrators rebuild only when their human-reviewed training data changes; and
- Flickr rows rescore only when their target bank, relevant competitor bank,
  candidate union, decisive nearest reference, or configured impact band is
  affected.

Every selective run reports work performed and work avoided. A full rerun is a
deliberate fallback when impact cannot be bounded safely, never the silent
default.

## Artifact and ownership boundaries

| Artifact | Owns | Does not own |
|---|---|---|
| Registry | Accepted taxon identity and synonyms | Image correctness |
| GBIF acquisition | Provider assertion and source provenance | Human verification |
| Admission decision | Automated QA result and reason codes | Ground truth |
| Human reference review | Verified/excluded/uncertain reference decision | Flickr occurrence release |
| Embeddings/prototypes | Frozen visual representations | Probability |
| Provisional scores | Raw comparative evidence and abstention | Final occurrence label |
| Human Flickr review | Reviewed Flickr identity/domain outcome | Reference identity |
| Statistical audit | Measured class/region performance and escalation | Per-reference proof |
| Calibrator | Probability mapping from independent reviewed labels | Taxonomic authority |
| Release manifest | Exact permitted final rows and upstream identity | Unreviewed evidence |

## Release boundary matrix

| Evidence | Provisional support/prototype | Calibration/final-test label | Final occurrence export | Scientific release |
|---|---:|---:|---:|---:|
| Unreviewed GBIF provider assertion passing all gates | Yes | No | Not applicable | No |
| Human-verified GBIF reference | Yes | Only under explicit independent split policy | Not applicable | Subject to all release gates |
| Unreviewed Flickr candidate | Scoring/triage only | No | No | No |
| Decisively human-reviewed Flickr row | Evaluation when split-safe | Yes when policy/split permits | Yes when all release gates pass | Subject to release permit |
| Raw similarity or margin | Evidence only | Metric input only | Never as probability | No probability claim |
| Independently calibrated output | Scoring evidence | Yes | Only with reviewed row and permit | Subject to release policy |

## Why this compromise

Reviewing every GBIF image before first inference allocates equal expert effort
to references that may never influence an error. Removing all controls would
confuse provider assertions with truth and expose every downstream artifact to
unbounded contamination.

Adaptive escalation preserves deterministic safety gates, provenance, explicit
evidence maturity, abstention, strict mode, human-reviewed evaluation, and
release blocking. It changes the timing of reference review: expert effort is
spent after the system reveals which species, competitor relationships, routes,
and individual references plausibly drive measured failures.

Statistical auditing does not replace expert review. It makes the review budget
more informative.

## Alternatives rejected

### Keep universal pre-inference review as the only mode

Rejected as the default because it blocks time to first comparison and reviews
unaffected species before failure evidence exists. Retained as
`human_verified_strict`.

### Admit every GBIF image based on provider taxon key

Rejected because taxon key alone does not establish media validity, licence,
attribution, decode success, canonical identity, independence, route, domain,
subject quality, or absence of human rejection.

### Treat YOLOE as a species verifier

Rejected. YOLOE owns quality/domain routing, not taxonomic identity.

### Treat low similarity or an embedding outlier as misidentification

Rejected. Those values prioritize review and may reflect view, sex, life stage,
geography, image quality, or genuine intraspecific variation.

### Train and calibrate directly on provider assertions

Rejected by default. Provider assertions may support prototypes and explicit
provisional training policy, but cannot become independent calibration or
final-test truth.

### Rebuild and rescore everything after any review

Rejected because content and semantic fingerprints provide a safe, auditable
incremental boundary. Full rerun remains available when impact is uncertain.

## Consequences

- Time to first BioCLIP comparison no longer depends on universal reference
  review in the adaptive mode.
- The policy, schemas, readiness contract, artifact fingerprints, CLI,
  orchestration, reporting, tests, and migration guidance must change together.
- More evidence states exist, but their meaning becomes explicit instead of
  being hidden inside a prototype exception.
- Strict artifacts and strict mode remain supported.
- Provisional support increases the need for reliable statistical sampling,
  audit availability states, and targeted review operations.
- Human Flickr review remains mandatory for final occurrence data.
- The system can stop with abstention or blocked release when configured
  objectives are not met.

## Implementation order

1. Add versioned admission policy and schema fields.
2. Generalize support/readiness without weakening strict validation.
3. Compile deterministic GBIF automated admission decisions.
4. Bind embeddings, prototypes, models, scores, and reports to admission
   identity.
5. Add adaptive orchestration and make it the explicit default.
6. Prove the Flickr final-release gate independently.
7. Add statistically valid species audits and targeted review.
8. Add bank revision, impact analysis, cache reuse, and selective rescoring.
9. Run strict/adaptive integration, performance, and pilot evidence.
10. Migrate documentation and run the final acceptance audit.

Each step fails closed on missing mode, policy, provenance, or required
evidence.

## GitHits evidence

Task `gbif-fast-0.2` used GitHits solution
`6acd55ce-b7a3-4fef-ab3e-edeefd10b01f`, distilled from:

- `durandtibo/wildcat.pytorch` (MIT); and
- `zerozedsc/Raman-Spectroscopy-Analysis-Application@87b8f6a` (MIT).

Adopted concepts:

- never overwrite weak/provider labels with verified labels;
- evaluate by class on a separate verified set;
- identify underperforming classes before choosing review targets; and
- use uncertainty/error evidence to prioritize human review.

Rejected concepts:

- copying the generated PyTorch training implementation;
- pandas/CSV artifacts;
- calling softmax outputs calibrated confidence;
- using provider-confidence sampling when the provider does not supply a
  validated confidence measure; and
- generating a review queue from final-test rows in a way that contaminates the
  locked evaluation set.

BioMiner uses frozen BioCLIP embeddings, Polars/Parquet, append-only review
history, group-aware sampling, explicit availability states, and separate
calibration/final-test ownership. No external code or prose was copied.
