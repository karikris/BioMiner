# Target-aware few-shot Phase 3 verification

Date: 2026-07-13. Confidence: high for deterministic regional candidate
construction, accepted-identity validation, relationship expansion, mandatory
target retention, and the inspected `no_geo` artifact. Confidence is unknown
for real regional occurrence overlap because no eligible external occurrence
artifact was available locally.

Phase 3 builds versioned occurrence and candidate-union contracts, compiles a
reviewed competitor relationship registry, and prevents hierarchy text ranks
from deleting candidates in target-scope reference comparison. The phase
started from `5233149b93f4d27d6d263a3f8f508a3d12b19059` on `main`.

## Task commits

| Task | Commit | Result |
|-|-|-|
| 3.1 regional occurrence index | `6f5097af2a14c13e7cec7c9b5b08f0e3fcdb5277` | accepted-key reconciliation, exact/buffer/country/bioregion/global scope precedence, eligible-record filtering, source support, dates, coordinate confidence, deterministic schema, and atomic Parquet output |
| 3.2 regional competitor union | `c6f690d2b37d2a6683d5e6ffd44af0fd78b2274b` | mandatory target, same-genus and same-family candidates, relationship and false-positive inputs, sparse geographic fallbacks, global `no_geo` expansion, reasons, flags, priorities, versions, and set fingerprint |
| 3.3 relationship registry | `4251ba76da5aa0ac74e34e2aa8ad47c0f951a54a` | generic six-type directed registry, accepted identity validation, review provenance, visual evidence fingerprints, active-version uniqueness, deterministic row fingerprints, and reviewed Papilio pilot seed |
| 3.4 target retention | `2287481344de0af658e5adf1c4680c4f5897dcb8` | versioned BioCLIP regional adapter, complete target-scope comparison, target force-retention, hierarchy priority without deletion, and per-species score provenance |

Each numbered task is a separate commit with its requested commit message.

## Test evidence

| Gate | Result |
|-|-|
| Task 3.1 focused tests | 8 passed |
| Task 3.2 focused and integration tests | 6 additional tests; full suite 1,059 passed |
| Task 3.3 candidate and relationship tests | 18 passed; full suite 1,071 passed |
| Task 3.4 candidate, object-scoring, cloud, rolling-worker, and orchestrator tests | 189 passed |
| Final full repository suite | 1,073 passed in 32.90 seconds |
| Compile validation | changed Python modules and tests passed `py_compile` |
| Whitespace validation | `git diff --check` passed |
| Configured formatter | Ruff was unavailable in the environment |

The occurrence aggregation was exercised with 50,000 deterministic synthetic
records in 0.243 seconds. The `no_geo` regional union built 892 accepted
Papilionidae candidates in under one second. These are local implementation
checks, not production throughput claims.

## Accepted identity and relationship seed

The accepted target is `gbif:1938069`, *Papilio demoleus*. The reviewed pilot
source records the current false-winning genera from the governing target
reference-bank policy:

| Genus | Accepted GBIF key | Accepted species expanded |
|-|-:|-:|
| *Graphium* | `gbif:1937188` | 108 |
| *Losaria* | `gbif:1939221` | 4 |
| *Ornithoptera* | `gbif:1937440` | 13 |
| *Pachliopta* | `gbif:1939152` | 18 |
| *Protographium* | `gbif:1939129` | 17 |

All five genus identities and the target were reconciled against accepted,
in-scope rows in `data/registry/butterflies-v2-20260712/taxa.parquet`. The
compiler contains no Papilio-specific names or keys. Each seed row is a
`historical_false_positive_genus` relationship with independent source-record
and semantic fingerprints. It influences comparison inclusion and priority;
it is not an image label or taxonomic validation.

## Candidate-set inspection

Phase 2 produced one Flickr cluster, `no_geo`, because all 18,041 available
candidate rows lacked valid Flickr coordinates. Phase 3 therefore exercised
the explicit global same-family fallback rather than inventing a location.

| Observation | Result |
|-|-|
| Geographic clusters inspected | one: `no_geo` |
| Candidate set ID | `regional:85e4a6d085db982b46ca013416d26a21` |
| Accepted candidate species | 892 |
| Unique candidate species | 892 |
| Target rows | exactly 1 |
| Historical false-positive rows | 160 |
| Candidate genera | 40 |
| Candidate rows with reasons and source versions | 892 of 892 |
| Candidate priorities | contiguous 0 through 891 |

The target remains present with zero occurrence support and no usable Flickr
geography. `no_geo` broadens to the accepted same-family registry. Geographic
evidence remains null rather than being guessed. Relationship reasons coexist
with global fallback reasons and do not replace them.

## Hierarchy-deletion regression

The target-scope BioCLIP adapter contract is
`object-bioclip-candidates-v2`; per-species scoring provenance is
`species-candidate-provenance-v1`.

This section records the historical Phase 3 behavior. Phase 14 supersedes new
target-scope diagnostic rows with `species-candidate-provenance-v2`: family
top one now filters the classification shortlist, target injection is removed,
and the fixed target score/rank remain separate screening evidence. Historical
v1 rows are not migrated or relabelled.

The required regression constructs twenty Papilionidae genera, assigns
*Papilio* the twentieth genus text rank, and verifies that *Papilio demoleus*
is still included in the species reference comparison and receives a rerank
score. A second regression places the target below a configured top-four
species shortlist and verifies mandatory force-retention. Family ranking now
prioritizes matching species but retains cross-family relationship candidates.

Every first-pass species persists accepted identity, scientific name,
candidate reasons, source versions, target flag, candidate priority,
first-pass rank and score, family-priority diagnostic, comparison membership,
and nullable rerank score. Open hierarchical classification remains a separate
mode and is not target-injected by this change.

## Generated artifact readback

The following generated artifacts are intentionally ignored by Git:

| Artifact | Rows x columns | Bytes | SHA-256 |
|-|-:|-:|-|
| `runs/target_aware_phase_3_papilio_demoleus_20260713/competitor_relationships.parquet` | 5 x 16 | 8,888 | `6bcff0b63d577cbd9fa1c10f250633b1f540c76386a712d406fbc380d9af14e7` |
| `runs/target_aware_phase_3_papilio_demoleus_20260713/regional_candidate_species.parquet` | 892 x 20 | 20,385 | `0275f006e3a1c4743aa635c2d71a7a3bf420f381d406170f1222aeba013e2d15` |

Independent Parquet readback revalidated physical schemas, row counts,
relationship and candidate fingerprints, target uniqueness, candidate
uniqueness, priority continuity, review state, and provenance completeness.

## Failure and determinism behavior

- Occurrence sources join only through accepted taxon keys; source names cannot
  redefine identity.
- Ineligible, absent, unreconciled, fossil, specimen, invalid-coordinate, and
  issue-bearing records cannot contribute range support.
- Exact scope evidence takes precedence over broader fallbacks for the same
  source and cannot be inflated by country or global duplicates.
- Missing coordinate uncertainty produces null confidence rather than an
  invented score.
- Mixed evidence versions, duplicate accepted identities, conflicting source
  records, invalid review state, and multiple enabled edge versions fail
  closed.
- Candidate generation is a union. Reasons accumulate, priority is ordering
  only, and every set must contain exactly one matching target.
- Missing geography broadens the union. Family and genus text ranks may reorder
  target-screening work but cannot delete comparison members.
- Input order does not alter relationship rows, candidate rows, set IDs, or
  semantic fingerprints.

## Tool and provenance notes

GitHits was queried fresh before every numbered task. The relationship design
used the versioned directed-edge and review-provenance pattern returned by
solution `c3753791-b5dd-48eb-aec0-79016ae59ef0`. Mandatory retention and
per-candidate decision provenance were checked against solution
`cbf01fbe-919e-4a54-9457-76b9eb4a3598`, distilled from permissively licensed
open-source references. Local contracts and tests, not generated examples,
remain authoritative.

Morph codebase search was attempted during discovery and returned HTTP 429.
Focused local call-site searches were used as the fallback. Valyu searches did
not produce sufficiently strong primary evidence for mimic identity, so no
external mimic claims were seeded. The five pilot genera come only from the
reviewed local migration policy.

## Repository hygiene

No generated Parquet, occurrence dump, image, model file, cache, secret, or
`.env` file is included in a Phase 3 commit. Pre-existing untracked files and
directories were not staged or modified.

## Unexecuted live work

No live GBIF, Flickr, iNaturalist, image-download, CUDA, YOLO, or BioCLIP model
operation was executed. No `regional_taxon_occurrence.parquet` was
materialized from real occurrence evidence. The inspected candidate union uses
the required `no_geo` fallback and cannot support claims about real regional
occurrence, overlap, abundance, or absence.
