# Target-aware few-shot Phase 14 pilot verification and handoff

## Build Week prototype continuation

The original human-review gate described below is retained as the scientific
release history, but it no longer blocks the explicitly prototype-only Build
Week path. Tasks 14.2.2 through 14.2.5 now provide the trust-first layered
planner, biological and visual-domain negative candidates, bounded source
acquisition, and a reproducible acquisition plan. The current compact handoff
is
`examples/species/papilio_demoleus/pilot_prototype_acquisition_manifest.json`.

Task 14.3.1 freezes 82 biological and 11 visual-domain references in an
explicit 93-row prototype selection ledger. All 93 are R4 provider-supported
prototype evidence; none is represented as human verified. Every selection
has a distinct media and observation identifier. Enforcing independent
observations increased the honest shortfall to 553 across 34 scopes, including
30 target-adult and 19 target-caterpillar images. Planner scores remain ordinal
priorities rather than probabilities. No biological-negative source images
were downloaded during metadata acquisition.

Task 14.3.2 downloaded those 93 selected prototype media objects into the
configured S3 reference-media prefix and froze a 93-row Parquet inventory.
Every object passed bounded MIME, decode, dimension, byte-count, and SHA-256
verification; the inventory contains 92 JPEGs and one PNG totalling 61,081,834
bytes. Thirteen objects are policy-allowed and 80 remain research-only. A
separate resume run committed the same 93 inventory rows with zero HTTP
requests, zero retries, and 93 resumed objects. The compact, non-secret handoff
is
`examples/species/papilio_demoleus/pilot_prototype_download_manifest.json`.
These downloads remain prototype support evidence, not verified biological
labels. Task 14.3.3 must still resolve exact, near, observation, burst, owner,
and cross-provider duplicate families before any immutable bank freeze.

Task 14.3.3 then ran against ignored local storage after the S3 account reached
its download/transaction cap. Eighty-three locally valid objects resolved as
83 unique canonical media with no exact, perceptual-candidate,
same-observation, burst, provider-mirror, or GBIF/iNaturalist mirror
relationships. The identity ledger separately preserves 12 repeated owner
groups and 13 repeated photographer groups for later leakage-safe splitting;
those groups are not represented as visual duplicates. Ten Wikimedia Commons
downloads remain retryable operational failures (nine item deadlines and one
HTTP 429 exhaustion), were not evaluated for duplicate relationships, and are
not biological negatives. The compact handoff is
`examples/species/papilio_demoleus/pilot_prototype_duplicate_resolution_manifest.json`.

Task 14.3.4 applied deterministic automated QA to the same 93-row ledger using
ignored local storage. Eighty-three available images received intrinsic image
quality, metadata-disagreement, licence, attribution, life-stage, and visual-domain
checks; ten unavailable Wikimedia rows remained retryable operational failures.
One very-low-resolution biological candidate and one curated fruit-closeup with
no butterfly visual were excluded. The remaining 81 available images were routed
to review because full-bank subject-presence and subject-size detector evidence
is not authorized on this computer. No unmeasured evidence was guessed and no
row was represented as human taxonomically verified. The compact handoff is
`examples/species/papilio_demoleus/pilot_prototype_qa_manifest.json`.

Date: 2026-07-15. Confidence is high for the geographic workload,
GBIF-derived competitor evidence, metadata checkpoint integrity, local vision
execution limit, and the B0-B16 experiment contract. Confidence is unknown for
the classifier policy because no human-reviewed reference bank or off-machine
vision benchmark exists yet.

Phase 14 is not complete. The prototype-only path has progressed through Task
14.3.4 for 83 available objects, while the scientific release path still
requires attributable human review and an immutable verified reference-bank
freeze. Automated QA retains ten retryable operational failures and routes 81
available images for visual review; Task 14.3.5 prototype freeze is next. Tasks 14.4-14.6 remain
blocked from scientific release; Phase 15 is not authorized, and the
production classification default remains unchanged.

## Execution constraints

- All implementation commits are on `main`.
- Local BioCLIP or YOLOE build verification is capped at five images.
- Larger BioCLIP and YOLOE runs must execute on a different computer.
- The original Phase 14 preparation performed metadata-only GBIF requests
  locally. Task 14.3.2 later downloaded the explicitly selected prototype
  media to configured object storage; it invoked neither BioCLIP nor YOLOE and
  retained no local image cache.
- The off-machine contract is
  `config/pilot/papilio_demoleus_phase14_experiment_matrix.json`.

## Implementation commits

| Commit | Result |
|-|-|
| `d949c18` | real family-top1 species filtering, complete rerank of the family-constrained shortlist, removal of target injection from classification, and separately retained target-screening score/rank |
| `48cedf0` | regional competitor evidence, resumable metadata acquisition, source shortfalls, checkpoint duplicate reconciliation, scoped high-volume query handling, and the B0-B16 off-machine contract |
| `94efde4` | versioned family-first candidate provenance so old and new candidate semantics cannot be silently mixed |
| `82b6115` | trust-first layered regional prototype planning and evidence-level separation |
| `a4a006d` | registry-linked biological-negative source candidates |
| `a129311` | curated visual-domain prototype negatives |
| `aeb0e92` | completed prototype support-candidate acquisition |
| `d0760f5` | deterministic 93-row prototype reference selection ledger |

## Task 14.1: geographic Flickr workload

The source is the existing metadata-only Flickr candidate ledger. Search hits
remain discovery evidence rather than labels.

| Metric | Result |
|-|-:|
| Input query hits | 76,485 |
| Canonical photos | 13,501 |
| Geotagged photos | 13,501 |
| Located clusters | 76 |
| Fallback clusters | 1 |
| Unassigned geotagged records | 792 |
| Outlier records | 707 |
| Allocated target reference quota | 100 |

The quota uses `minimum-plus-sqrt-candidates-v1.1.0`, satisfies the configured
minimum for all 76 eligible clusters, allocates all 100 slots, and excludes
`no_geo` and `unassigned_geo` from reference support. The reviewed compact
manifest is
`examples/species/papilio_demoleus/pilot_geographic_workload_manifest.json`.

Key artifact hashes:

| Artifact | SHA-256 |
|-|-|
| Flickr clusters | `cba4651b967fae15f586e760859fb11ff608a603f792e338c7661d5563130b35` |
| Flickr assignments | `e12f6ef9582bf707c952c3974c91e9a8f226ca7ce8034cab8ea8c293b70b6f74` |
| Canonical geography | `a3c47a6f213191634f76655d859cdd8555a9b11a4e33a9041209eb23ba7c2bbf` |
| Query-hit evidence | `95448f3145d903f7f042fe41d74561475ef050f8df21b318ebacb252484e4f0b` |

## Task 14.2: target range and competitor source evidence

The target GBIF spread build completed with 19,201 occurrence records, 57,603
multi-resolution evidence rows, 16,678 spread rows, and 630 occurrences
eligible for range inference. It wrote 65 checkpoint parts. No invalid
coordinates or taxon-key mismatches were accepted.

| Artifact | Rows | SHA-256 |
|-|-:|-|
| `geographic_occurrence_evidence.parquet` | 57,603 | `1387f3c9967d322e537e9f2079b58641543707db863d2057c1caa4408d208a47` |
| `taxon_geographic_spread.parquet` | 16,678 | `46f38228cb5e7ba58f5f38f79a38dd87ae8fab80cf987e916f620d5cc4b9e83c` |

Twenty-one countries have at least five target range-inference occurrences.
Country-scoped GBIF species facets then produced 30 accepted *Papilio*
competitors. The selected top five for source acquisition are *Papilio
memnon*, *P. polytes*, *P. helenus*, *P. paris*, and *P. machaon*. The evidence
artifact hash is
`398990b3bef47e662e0b6afa70dba407412272906d08ab59e4a8ad54ab0c5ccc`.

The same compiler evaluated the five reviewed false-winning genera. These are
model-error candidates, not taxonomic truth labels.

| Reviewed genus | Accepted candidates | Evidence SHA-256 |
|-|-:|-|
| *Graphium* | 10 | `d5a75b9cec4497458e69fc92e4f812c90eeb83012b5a13a229fe6ece695e816d` |
| *Pachliopta* | 10 | `caef167cf587523e5b9ce037340b9771b9d149586db3a5ca6decab99fd300c85` |
| *Ornithoptera* | 10 | `74ed98e2a6772ee6f734576b08d672b6f856c8f382f15614347987675fb5bf85` |
| *Protographium* | 3 | `d08ca8ccf39acf7a4650d8800558cde5bd093ee787f1bd429371b1e5ff409232` |
| *Losaria* | 4 | `ee6798079df911617ff8e484cbbe6a01fcf75948450d691a342bf290a964491e` |

Each genus build made 21 successful country-facet requests with zero retries
and zero rate-limit events. The source-query plan contains 22 registry-matched
accepted taxa and explicit target, regional competitor, reviewed false-winner,
historical false-winner, broader Papilionidae, and larval quotas.

The global *Papilio machaon* still-image search returned 101,059 records, above
the 100,000 search ceiling. Its query is therefore constrained to the 21
target-supported countries, where the verified source count is 638. This is a
candidate-source scope decision, not a biological range assertion.

The positive/competitor metadata acquisition completed all 22 query checkpoints
across 314 pages and requests, with zero retries and zero rate-limit events.
No images were downloaded. The publication reconciled duplicate rows caused by
shifting source pages only when all semantic fields matched and only retrieval
timestamps differed; semantic conflicts would have failed publication.

| Metric | Result |
|-|-:|
| Raw checkpoint occurrence rows | 91,180 |
| Unique occurrence rows | 91,176 |
| Retrieval-only occurrence duplicates removed | 4 |
| Raw checkpoint media rows | 142,878 |
| Unique media candidates | 142,873 |
| Retrieval-only media duplicates removed | 5 |
| Eligible source media candidates | 838 |
| Human-verified source media | 0 |
| Located observations | 64,914 |
| Global-fallback observations | 26,262 |

The source-bank status remains
`awaiting_human_review_or_additional_sources`. The group-level gaps are:

| Group | Source candidates | Minimum | Candidate shortfall | Human-verified shortfall |
|-|-:|-:|-:|-:|
| Target adult | 289 | 50 | 0 | 50 |
| Five regional competitors | 300 | 100 | 8 per-species slots | 100 |
| Five reviewed false-winner genera | 90 | 100 | 60 per-species slots | 100 |
| Historical false winner | 0 | 20 | 20 | 20 |
| Broader Papilionidae | 60 | 100 | 40 | 100 |
| Target caterpillar, separate bank | 1 | 20 | 19 | 20 |
| Other-insect or moth negatives | 0 | 100 | 100 | 100 |
| Domain negatives | 0 | 0 | unresolved human selection | 0 |

The aggregate candidate shortfall is 247 and the human-verified shortfall is
490. Group totals are quota diagnostics and must not be mistaken for a count of
unique media across groups. The tracked compact handoff is
`examples/species/papilio_demoleus/pilot_reference_source_manifest.json`.

| Artifact | Bytes | SHA-256 |
|-|-:|-|
| `reference_observations.parquet` | 10,797,270 | `193d9d7451eebf75eaaf772fa761ad53cfeca9f04d84dae8c2caffe9fa64882f` |
| `reference_media_candidates.parquet` | 13,083,129 | `d9a4cd37eb8558f822807f753b2d027d37f33bcd9c32274b21516468d1827135` |
| `reference_metadata_report.json` | 1,055 | `f0a11c769b74a70169d9af2529212fa875b10469cc653fe5fd1051fd8b7ba2b1` |
| `reference_source_shortfalls.json` | 6,589 | `cadd38791d66e0f1875a35ee398b3203e82969957aa81fc0d69e83f1d08c0a99` |
| `reference_source_shortfalls.md` | 978 | `4badc1f6691f322c076c1aca2e99ee913a4d633bf7551cc4b2c0675df90ced84` |

## Task 14.3: human review and immutable freeze

Scientific-release status remains blocked awaiting human work. GBIF taxon
reconciliation and provider verification statuses do not verify an image for
BioMiner. The human-verified count is exactly zero.

The separate, explicitly non-scientific prototype path completed Tasks
14.3.1-14.3.5 using ignored local storage while S3 is capped. Of 93 selected
records, 81 provider-supported GBIF images entered the prototype bank, 2 were
excluded by QA, and 10 Wikimedia download failures remain retryable. The bank
is `prototype_ready_with_shortfalls`; it authorizes experimental screening but
not scientific release.

The deterministic transitive split produced 22 atomic leakage components and
26/30/13/12 rows in support/model-selection/calibration/final-test. All known
observation, owner, photographer, duplicate, exact-hash, perceptual-hash,
burst, and provider-mirror relationships stay within one split. Two selected
records lack owner evidence, so this is not a claim of complete leakage
protection. Exact 55/15/15/15 row proportions are impossible without splitting
an identity component.

Prototype readiness checks pass with 11 target adults, 3 regional competitor
species, 1 false-winner species, and 7 biological hard negatives in
support_train or the full support bank as applicable. The bank still lacks a
qualified visual-domain negative, a pinned specimen, a larval support-train
record, full-bank detector evidence, and independent human verification. The
tracked compact handoff is
`examples/species/papilio_demoleus/pilot_prototype_support_bank_manifest.json`.

The next operator must:

1. apply licence policy and retain complete attribution evidence;
2. download only selected media candidates;
3. resolve exact, perceptual, burst, owner, observation, and provider-mirror
   duplicates;
4. export the immutable review queue and record attributable decisions;
5. keep adult, larval, pinned-specimen, and visual-domain banks separate;
6. assign the 55/15/15/15 support/model-selection/calibration/final-test split
   by transitive leakage component;
7. publish readiness artifacts and pin their trusted SHA-256 outside the
   rewritable output directory.

No unresolved, uncertain, conflicting, attribution-incomplete, or
duplicate-ambiguous row may enter scientific support. Research-only licensing
is permitted only in the local prototype bank and remains explicitly labelled.

## Tasks 14.4-14.6: prototype benchmark, selection, and report

Status: Task 14.4.1's five-image local BioCLIP/YOLOE smoke and the explicitly
user-authorised 81-record local support-embedding run are authorized by the
prototype-only readiness artifact. Other larger experiments and scientific
model selection remain off-machine unless separately authorized. The
machine-readable matrix contains exactly B0-B16:

Task 14.4.1 passed on MPS with the exact BioCLIP 2.5 Huge revision
`191d741545e4c741cdef4b22c6eb69c945c1e592` and YOLOE checkpoint
`yoloe-26s-seg.pt`. BioCLIP returned a finite 5 x 1024 embedding matrix with
frozen preprocessing/model hashes, one persistent model load, and one cache
hit. YOLOE reused one persistent process across batches of three and two and
returned eight detections. Both Python 3.12 runtimes reported PyTorch 2.12.1,
MPS availability, MPS resolution, and enabled CPU fallback policy.

The detector's winning prompt routed four images to pupa/chrysalis exclusion
and one to ambiguous exclusion, including provider-supported adult records.
That is a recorded accuracy warning for subsequent experiments, not a reason
to reinterpret the source labels and not a runtime-smoke failure. The compact
handoff is
`examples/species/papilio_demoleus/pilot_prototype_vision_smoke_manifest.json`.

Task 14.4.2 now has an executable prototype-only frozen-embedding command and
local configuration. It validates the frozen support/readiness byte hashes and
semantic fingerprint without converting `provider_supported` evidence into a
human-review state. A failed batch is isolated to individual records; an
unreadable record or an explicit operator skip is retained in a retryable
failure Parquet while successful records continue. Completed embeddings are a
validated resume checkpoint, and prototype fitting consumes only
`support_train` after collapsing media by independent observation. Adult,
larval, and specimen routes remain separate, and visual-neighbour edges are
route-local.

The command first passed a bounded five-image MPS validation, then completed
the user-authorised full 81-record run using local storage only. It produced
81 finite, unit-normalized 1,024-dimensional embeddings, 26 global/regional
support-train prototypes, and 50 directed within-route visual-neighbour edges,
with zero failures or operator skips. BioCLIP loaded once in one persistent
worker and served seven requests with six model-cache hits. A second full
invocation reused all 81 embeddings without model recomputation.

The embedding artifact retains 80 adult-field and one larval row separately.
The larval row belongs to calibration rather than `support_train`, so emitting
no larval prototype is an explicit split consequence rather than route mixing.
No pinned specimen passed the prototype freeze. Every label remains
`provider_supported`; `human_verified_count` is zero. The compact handoff is
`examples/species/papilio_demoleus/pilot_prototype_embeddings_manifest.json`.

Task 14.4.3 completed the cumulative P1/P2/P3 target-aware classification run
against the frozen 13,501-record Flickr workload. Storage was local only; the
runner rejects S3 configuration, never instantiated an S3 client, and deleted
the temporary content-addressed image cache after each batch. The stages
completed at 100, 1,000, and 13,501 planned records. P3 classified 13,496
records and retained five Flickr download/decode failures as retryable
operational failures after three attempts. Those failures were skipped so the
remaining workload could progress and were never converted into biological
negatives.

Every classified record scored the fixed union of 34 species, two known
negative classes, and 11 visual-domain classes. The target appeared exactly
once per record, raw full images were retained, and neither hierarchy pruning
nor spatial cropping was applied. The durable local outputs contain 13,496
classification rows and 634,312 candidate-score rows. Polars and DuckDB QA
confirmed zero duplicate classifications, 34 species rows and 47 total
candidate rows per classified record, finite scores, route separation, and no
use of Flickr query hits as labels. All scores remain uncalibrated experimental
screening evidence rather than probabilities or taxonomic validation.

The full run used one persistent BioCLIP worker and one persistent YOLOE worker.
P3 sustained 2.274524 records per second with 1,765,261,312 bytes peak RSS. A
completed invocation returned from the SQLite checkpoint in 0.57 seconds with
no model work. The compact handoff is
`examples/species/papilio_demoleus/pilot_staged_flickr_manifest.json`.

Task 14.4.4 ran the local B0-B16 executable matrix over all 81 frozen prototype
records. The command rejects non-local paths and S3 authorization. It emitted
1,539 prediction rows across B0-B16 plus the three required B14 policies,
12,874 candidate-score rows, and a 19-row experiment summary. Candidate
taxonomy text was embedded once with BioCLIP on Apple MPS and then reused from
the local Parquet cache.

No record needed to be skipped, but the command supports an explicit
`skip_records` quarantine list so one operationally bad record cannot block the
remaining set. Such skips are written with `biological_negative=false`.
Provider-supported retrieval, target rank, raw margins, model agreement, and
abstention are reported. Classification accuracy and probability calibration
are not reported because all 81 human-verification flags are false.

B10 ran against the existing raw full-frame embeddings. B11 and B12 are
retained as an explicit local executable subset: they reuse raw full-frame
evidence and mark focused and masked embeddings unavailable rather than
inventing values. The compact handoff is
`examples/species/papilio_demoleus/pilot_prototype_b0_b16_manifest.json`.

- B0 current text-pruned and B1 zero-shot without pruning;
- B2 SimpleShot, B3 centered SimpleShot, B4 top-five references, and B5
  multi-prototype;
- B6 logistic regression, B7 embedding-only LinearSVC, B8 structured-feature
  LinearSVC, and B9 independently calibrated abstention;
- B10 raw full frame, B11 raw plus focused full frame, and B12 raw plus focused
  plus masked full frame;
- B13 global references and B14 cluster-conditioned references with global
  `no_geo` fallback;
- B15 taxonomy-text plus image fusion and B16 image-only evidence.

Selection uses model-selection only; calibration and thresholds use calibration
only; final-test cannot select either. Required evidence includes PR-AUC,
recall at high-precision operating points, calibration error, selective
coverage, cluster and `no_geo` slices, visual domain, life stage, source,
candidate species, leakage audits, failure rates, throughput, and peak memory.
Raw similarity or SVC margin is never labelled a probability.

## Validation

| Gate | Result |
|-|-|
| Family-first object pipeline, evaluation QA, and orchestration | 177 passed |
| Phase 14 prototype-freeze focused suite | 61 passed |
| Five-image runtime focused suite | 88 passed |
| Prototype embedding and compact-contract focused suite | 26 passed |
| Staged Flickr and species-generic boundary focused suite | 10 passed |
| GBIF checkpoint, metadata, workflow, and shortfall suite | 52 passed |
| Final full repository suite | 2,303 passed in 81.22 seconds |
| Changed-file Ruff | passed |
| `git diff --check` | passed |
| Completed smoke BioCLIP images | 5 |
| Completed smoke YOLOE images | 5 |
| Completed frozen support embeddings | 81 |
| Completed staged Flickr classifications | 13,496 |
| Retryable staged Flickr source failures | 5 |
| Local B0-B16 prototype prediction rows | 1,539 |
| Local B0-B16 prototype records skipped | 0 |

## Phase 15 decision

No default change is supported. Phase 15 requires a ready immutable reference
bank, complete B0-B16 off-machine results, a passed selection policy, and an
approved final Phase 14 report. None of those four gates may be inferred from
metadata availability or deterministic unit tests.
