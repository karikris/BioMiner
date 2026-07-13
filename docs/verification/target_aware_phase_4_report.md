# Target-aware few-shot Phase 4 verification

Date: 2026-07-14. Confidence is high for the reference metadata contracts,
source-specific normalization, retry and checkpoint behavior, cross-source
deduplication, balanced quota planning, recorded-fixture coverage, and the
inspected offline plan. Confidence is low for production support-bank
availability because the phase-completion plan replays six reduced target
records and does not fan out across the 892 candidate species.

Phase 4 adds reference observation and media contracts, GBIF and iNaturalist
metadata adapters, resumable source checkpoints, iNaturalist-through-GBIF
deduplication, and a deterministic geographically balanced support-bank
planner. It does not download or verify images. The phase started from
`c7bc16e5dca5c99a0e1d98c7d4b25be932857cf2` on `main`.

## Task commits

| Task | Commit | Result |
|-|-|-|
| 4.1 reference contracts | `04b30456ec24b22d6cad2b3608a01ba669da6374` | deterministic observation, media, plan, and source-page schemas; stable IDs; physical validation; atomic Parquet writers |
| 4.2 GBIF adapter | `85afe69feabd0c7b6d98d64f0d2d5cf65c73ddb6` | accepted-key occurrence search, four geographic scopes, occurrence/media licence separation, issue flags, retry counters, checkpoints, bulk-download handoff, and publisher-media preservation |
| 4.3 iNaturalist adapter | `e4226e5585a7739744d3ec87aa2324b4c29ca6c4` | Research Grade preselection, community identity, wild/photo/licence constraints, coordinate privacy, one-second pacing, cursor checkpoints, bulk alternatives, and GBIF mirror detection |
| 4.4 quota planner | `ea4de1419e64fa76f6b6375884defb3da3eb4ff5` | balanced per-class deficits, capped Hamilton cluster allocation, fallback priority, independent observations, diversity ordering, selection ledger, incremental top-up, reports, and cross-artifact validation |
| iNaturalist response correction | `f8e33cd5ff768543e01d6cd34250813dd3311d37` | exact reconciliation of the compact search shape through `community_taxon_id`, with conflicts still rejected |
| 4.5 recorded metadata tests | `12f2d4338008cfb3a4728bb52c8674fe7b4e2c40` | reduced API recordings, fixture checksums, exact normalization, licences, coordinates, fallback, retry, deduplication, shortfalls, target retention, and no automatic verification |

Each numbered task is a separate commit with its requested message. The
iNaturalist correction is separate because real response inspection exposed a
false assumption in Task 4.3 before the recorded-fixture tests were accepted.

## Test evidence

| Gate | Result |
|-|-|
| Task 4.4 reference planner and related suite | 76 passed |
| Full suite after Task 4.4 | 1,128 passed in 33.13 seconds |
| Compact iNaturalist response regression | 18 adapter tests passed |
| Full suite after response correction | 1,129 passed in 33.62 seconds |
| Task 4.5 recorded-fixture boundary | 4 passed |
| Final focused reference suite | 60 passed |
| Final full repository suite | 1,133 passed in 38.49 seconds |
| Changed Task 4.5 test lint | Ruff passed |
| Python bytecode compilation | `src` and `tests` passed `compileall` |
| Whitespace validation | `git diff --check` passed |

Repository-wide Ruff reports four pre-existing findings outside the Phase 4
reference changes: an `E712` comparison in
`src/biominer/candidates/regional_occurrence.py`, plus unused imports in
`src/biominer/flickr_fetch/metadata_poller.py`,
`src/biominer/registry/geographic_summary.py`, and
`src/biominer/registry/unified.py`. They were not mixed into this phase.

## Contract and identity behavior

Reference observations and media are separate physical artifacts. Observation
identity is source-scoped; media identity is source-, provider-media-, and
observation-scoped. Each normalized row retains the registry version, source
snapshot, query fingerprint, retrieval time, and source record identifier.
Occurrence licences never substitute for nested media licences.

The planner consumes the complete regional candidate union. For every species,
cluster, life-stage, and visual-domain stratum it records configured quota,
existing support, available independent observations, selected rows, and the
real shortfall. It allocates cluster demand with a configurable minimum and a
capped square-root Hamilton apportionment. Selection prioritizes lower fallback
levels, cluster distance, independent observations, and diversity of observer,
date, locality, background, and source. One observation cannot fill multiple
quota slots in the same plan.

No source status automatically verifies an image. GBIF identification labels
and iNaturalist Research Grade remain source metadata; every acquired media
candidate starts as `unreviewed`.

## iNaturalist compact-response correction

Default iNaturalist observation searches return a compact object containing
`community_taxon_id` and the expanded observation `taxon`, but normally omit
the expanded `community_taxon` object. The original adapter required the
expanded object and therefore excluded ordinary production search rows.

The corrected exact-match invariant is:

```text
community_taxon_id == taxon.id == requested source species ID
and taxon.rank == species
```

When the expanded community object exists, its ID and rank are cross-checked.
Missing IDs, conflicting identities, and non-species ranks remain unresolved or
conflicting. Research Grade alone is insufficient, and successful taxon
reconciliation still leaves image verification `unreviewed`.

## Recorded source fixtures

The default suite performs no network calls. Its checked-in fixtures are
reduced and redacted fragments recorded from the public source APIs on
2026-07-14:

| Fixture | Retained content | Bytes | SHA-256 |
|-|-|-:|-|
| `tests/fixtures/references/gbif_occurrence_search_v1.json` | two *Papilio demoleus* occurrences, one media item each, including an iNaturalist mirror and distinct parent/media licence values | 3,734 | `b79c67c1fc9961357e7c2cf80ae5bae4daf6c6269ba5c3b65acb93a782e95f4f` |
| `tests/fixtures/references/inaturalist_observation_search_v1.json` | four compact Research Grade search rows plus an empty local response, one media item each | 7,879 | `96ba9ffdf5ba245ee5c020088de1e6df5a4aa30ec83969345095cd898614c263` |

Observer names and fine-grained locality text are redacted. Pagination is reset
to each retained subset and the reductions are declared inside the fixtures.
`tests/fixtures/references/manifest.json` pins both checksums. The recordings
remain source-shaped JSON test inputs, not raw API dumps or production registry
data.

## Papilio demoleus metadata-only plan

The phase-completion plan ran the real planner against the Phase 3 candidate
artifact and the recorded source responses. It was an offline metadata replay:
no image URL was fetched, no authenticated request was made, and no row was
manually reviewed.

Input candidate artifact:

| Observation | Result |
|-|-|
| Target | `gbif:1938069`, *Papilio demoleus* |
| Candidate artifact | `runs/target_aware_phase_3_papilio_demoleus_20260713/regional_candidate_species.parquet` |
| Candidate species | 892 accepted Papilionidae species |
| Candidate clusters | one: `no_geo` |
| Target rows | exactly one |
| Candidate artifact SHA-256 | `0275f006e3a1c4743aa635c2d71a7a3bf420f381d406170f1222aeba013e2d15` |

The `no_geo` scope is inherited from Phase 2: the available Flickr candidates
contained no valid geotags. Reference metadata therefore used global fallback
level 3 rather than inventing a regional relationship.

### Source availability

| Source | Observations | Media candidates | Pending without exclusion | Excluded |
|-|-:|-:|-:|-:|
| GBIF | 2 | 2 | 0 | 2 |
| iNaturalist | 4 | 4 | 4 | 0 |
| Total | 6 | 6 | 4 | 2 |

All six observations reconcile exactly to the target identity, and all six
media rows remain `unreviewed`. The two GBIF records declare geospatial issues
and are excluded from planning. One is also the same observation/photo exposed
directly by iNaturalist; deduplication retains both provenance rows, keeps the
direct source, and marks the GBIF mirror excluded. Four distinct iNaturalist
observations are metadata-eligible before life-stage stratification.

### Licence availability

| Media licence as supplied | Count |
|-|-:|
| `cc-by-nc` | 4 |
| `http://creativecommons.org/licenses/by-nc/4.0/` | 1 |
| `http://creativecommons.org/licenses/by-nc-nd/4.0/` | 1 |

Occurrence licences remain separate: four `cc-by-nc` values from iNaturalist
and two `http://creativecommons.org/licenses/by-nc/4.0/legalcode` values from
GBIF. The plan preserves these source forms; Phase 5 must apply canonical
licence policy before downloading any media. A permissive parent occurrence
licence would not make a missing or denied nested media licence usable.

### Plan outcome

The default configuration requests 20 adult, unreviewed support observations
per class. Across 892 candidate species this is a balanced configured quota of
17,840. The recorded iNaturalist search rows expose no label-expanded
life-stage value, while the sole recorded adult GBIF row is excluded for
declared geospatial issues.

| Metric | All candidates | Target only |
|-|-:|-:|
| Requested | 17,840 | 20 |
| Available in adult/unreviewed stratum | 0 | 0 |
| Selected | 0 | 0 |
| Shortfall | 17,840 | 20 |

This is a metadata and coverage result, not a claim of biological absence or
source exhaustion. The replay contains no competitor metadata and only six
target records. The complete shortfall is the correct fail-closed outcome for
the inputs actually supplied. Reclassifying unknown-stage rows into an
`unknown/unreviewed` diagnostic stratum would produce selections, but it would
not satisfy the configured adult-support requirement and was not substituted
for it.

## Generated artifact readback

Generated Parquet and reports remain ignored by Git. Independent readback
revalidated all physical schemas and found 6 observations, 6 media rows, 892
plan rows, zero selections, and exactly one target plan row.

| Artifact | Rows or type | Bytes | SHA-256 |
|-|-:|-:|-|
| `runs/target_aware_phase_4_papilio_demoleus_20260714/reference_observations.parquet` | 6 | 18,834 | `675bedd51ea508b7892dc189acb16afd0c109ff51181123d8487e5dc720486e8` |
| `runs/target_aware_phase_4_papilio_demoleus_20260714/reference_media_candidates.parquet` | 6 | 13,916 | `ed7bee4b4e1fd0fc3845bc391eef7929fe66f9c8eff08ff67fdd782d8fc62936` |
| `runs/target_aware_phase_4_papilio_demoleus_20260714/reference_acquisition_plan.parquet` | 892 | 20,254 | `4c73c206135cfd9f94f9047b0ad6fde4adf78630fa1da6e401f1a27cf1793c5a` |
| `runs/target_aware_phase_4_papilio_demoleus_20260714/reference_acquisition_selections.parquet` | 0 | 3,217 | `9bcc076efa99066c52209940296dd6fda945a17a1e23a6cbf7cd2d8fa4501354` |
| `reports/target_aware_phase_4_papilio_demoleus_20260714/reference_metadata_plan.json` | compact run report | 5,389 | `97e0070df180c402af91cf9927cb969c89821d8d0c2aa513ebd9c4fc4e799fea` |
| `reports/target_aware_phase_4_papilio_demoleus_20260714/reference_metadata_plan.md` | compact summary | 1,073 | `63dc1924a588a32984a0bf96fa063a1b800dc3b74a2d376b70e7bc5024df1a4b` |

The generated planner JSON and Markdown are also present in the run directory.
The compact run report records command, run ID, PID, implementation Git SHA,
inputs, outputs, byte counts, checksums, source calls, retries, rate-limit
events, row counts, exclusions, licences, plan configuration, and unsupported
metrics.

## Failure, retry, and resume behavior

- GBIF and iNaturalist searches use bounded documented page sizes, identifying
  User-Agents, transient retry policies, `Retry-After`, and explicit counters.
- Permanent client errors are not retried. iNaturalist enforces at least one
  second between requests and sends no authentication by default.
- Source checkpoints bind query fingerprint, source version, and snapshot,
  persist normalized Parquet parts and compact state atomically, validate
  checksums and cursor continuity, and resume without refetching completed work.
- GBIF searches beyond the documented ceiling require an authenticated bulk
  occurrence download; iNaturalist search volumes beyond its API window require
  an export or the weekly GBIF dataset.
- Missing media licences, taxon conflicts, unsuitable occurrence bases,
  fossils, specimens, absence, geospatial issues, captive status, coordinate
  privacy, disagreement, and unsupported photo licences remain explicit
  exclusion reasons.
- Deduplication changes candidate eligibility, not provenance retention.
- Candidate classes, including the target, remain in the acquisition plan when
  availability is zero. Shortfalls are persisted rather than backfilled across
  species or hidden by duplicate photos.
- Input ordering does not change normalized IDs, candidate pools, plan IDs,
  selected observations, or report distributions.

## Tool and provenance notes

GitHits was queried fresh before every numbered task and before phase
completion. Recorded-response and retry tests used the verified HTTPX
`MockTransport` request/response boundary and injected-clock patterns. Useful
retry evidence was returned under solution
`58e867fc-1842-472d-bda6-3bcb239e58bd`. Examples that generated recordings at
test time, returned silent 404 fallbacks for unexpected routes, inherited
parent licences, or treated source labels as verified images were rejected.
The phase-completion report shape and deterministic fingerprint inventory were
cross-checked against fresh solution
`acac4feb-5fa5-4f2f-8f58-791a1eea37ac`.

The iNaturalist correction was verified against the official API source: the
default observations controller uses minimal association hydration, while
`details=all` expands the community taxon. The core Research Grade logic still
requires the community identity. Public API inspection confirmed the compact
shape before the reduced fixture was committed.

Morph codebase search was attempted during discovery/review and returned HTTP
429. Focused local call-site searches were used as the fallback. External tool
output was treated as untrusted development evidence and did not define
taxonomy, licence acceptance, verification, or a production selection.

## Repository hygiene

No raw API dump, downloaded image, model file, cache, generated run Parquet,
secret, token, or `.env` file is included in a Phase 4 commit. Reduced recorded
fixtures contain only the fields required by the tests and declare their
redactions. The pre-existing untracked files and directories were not staged or
modified.

## Unexecuted production work

No live metadata fan-out across the 892 candidate species was run. No image was
downloaded, decoded, deduplicated by content, manually reviewed, uploaded, or
embedded. No authenticated GBIF download, iNaturalist export, CUDA, YOLO, or
BioCLIP operation was performed. Source coverage, rate-limit duration, licence
yield, adult life-stage yield, and support-bank readiness therefore remain
unknown until the later production acquisition and review phases.

The Phase 3 candidate Parquet used by this replay is a generated, ignored
artifact and must be regenerated in a fresh clone. Phase 4 exposes adapters and
planner APIs but does not yet expose a reference-acquisition CLI or source
orchestrator. The inspected plan is therefore reproducible from the recorded
fixtures and generated candidate artifact, but it is not yet a production
command-line workflow.
