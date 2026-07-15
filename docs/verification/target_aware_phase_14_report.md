# Target-aware few-shot Phase 14 pilot verification and handoff

Date: 2026-07-15. Confidence is high for the geographic workload,
GBIF-derived competitor evidence, metadata checkpoint integrity, local vision
execution limit, and the B0-B16 experiment contract. Confidence is unknown for
the classifier policy because no human-reviewed reference bank or off-machine
vision benchmark exists yet.

Phase 14 is not complete. Task 14.1 and the positive/competitor metadata portion
of 14.2 are implemented. Task 14.2 still lacks registry-linked moth and
other-insect negatives, has five source-candidate quota shortfalls, and leaves
domain-negative selection to human review. Task 14.3 is blocked on attributable
human review and immutable reference-bank freeze. Tasks 14.4-14.6 are
consequently blocked. Phase 15 is not authorized, and the production
classification default remains unchanged.

## Execution constraints

- All implementation commits are on `main`.
- Local BioCLIP or YOLOE build verification is capped at five images.
- Larger BioCLIP and YOLOE runs must execute on a different computer.
- This phase performed metadata-only GBIF requests locally. It downloaded no
  images and invoked neither BioCLIP nor YOLOE.
- The off-machine contract is
  `config/pilot/papilio_demoleus_phase14_experiment_matrix.json`.

## Implementation commits

| Commit | Result |
|-|-|
| `d949c18` | real family-top1 species filtering, complete rerank of the family-constrained shortlist, removal of target injection from classification, and separately retained target-screening score/rank |
| `48cedf0` | regional competitor evidence, resumable metadata acquisition, source shortfalls, checkpoint duplicate reconciliation, scoped high-volume query handling, and the B0-B16 off-machine contract |
| `94efde4` | versioned family-first candidate provenance so old and new candidate semantics cannot be silently mixed |

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

Status: blocked awaiting human work. GBIF taxon reconciliation and provider
verification statuses do not verify an image for BioMiner. At metadata-fetch
time the human-verified count is therefore exactly zero.

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

No unresolved, uncertain, conflicting, research-only, attribution-incomplete,
or duplicate-ambiguous row may enter support.

## Tasks 14.4-14.6: off-machine benchmark, selection, and report

Status: blocked by Task 14.3. The machine-readable matrix contains exactly
B0-B16:

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
| Phase 14 reference and matrix focused suite | 42 passed |
| GBIF checkpoint, metadata, workflow, and shortfall suite | 52 passed |
| Final full non-vision repository suite | 2,213 passed in 78.77 seconds |
| Changed-file Ruff | passed |
| `git diff --check` | passed |
| Local BioCLIP images | 0 |
| Local YOLOE images | 0 |

## Phase 15 decision

No default change is supported. Phase 15 requires a ready immutable reference
bank, complete B0-B16 off-machine results, a passed selection policy, and an
approved final Phase 14 report. None of those four gates may be inferred from
metadata availability or deterministic unit tests.
