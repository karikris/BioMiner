# Target-aware few-shot Phase 2 verification

Date: 2026-07-13. Confidence: high for Flickr geography normalization,
deterministic clustering, query-provenance reporting, and the inspected local
candidate artifacts. Confidence is unknown for real target occurrence
geography because the available Flickr evidence contains no valid geotags.

Phase 2 normalizes Flickr candidate coordinates, clusters candidates without
claiming verified biological occurrences, preserves every query-discovery
link, and reports an implied geographic reference workload. The phase started
from `8d41eb4ffd7db5a205a5fc540c817acb6a950b6e` on `main`.

## Task commits

| Task | Commit | Result |
|-|-|-|
| 2.1 candidate geography | `fad782c271e2599dd21f3101799b9c7a8a844067` | Flickr coordinates, ordinal accuracy policy, precision-gated H3 cells, explicit administrative fields, warnings, and deterministic schema |
| 2.2 geographic clusters | `c76fd8c629dc4ff1938188117bcb1a29558e1f3d` | density-qualified connected components, stable IDs, adjacency and bounded fallbacks, distance/outlier evidence, `no_geo`, and atomic Parquet artifacts |
| 2.3 workload report | `141213646bb5083e3591a47890f2354f9a9f64f6` | candidate, cluster, country, query-tier, search-term, and minimum-plus-square-root reference-quota summaries |
| 2.4 acceptance tests | `d163471e21bb79432951ee9982af655f9fcd5841` | boundary, dateline, sparse-cell, accuracy, deterministic-refresh, stable-ID, and candidate-versus-occurrence coverage |
| Source-data correction | `5393ed2c5b15498c998f2ac1becbe89da6b4b2ff` | Flickr's exact `latitude=0`, `longitude=0`, `accuracy=0` missing-location sentinel is rejected; valid `(0,0)` coordinates with documented accuracy remain usable and warned |

Each numbered task is a separate commit with its requested commit message. The
sentinel correction is a separate follow-up commit because production-data
inspection exposed a hidden assumption not represented by the synthetic task
fixtures.

## Test evidence

| Gate | Result |
|-|-|
| Phase 2 focused geography, clustering, and report suite | 29 passed |
| Full repository suite after Task 2.4 | 1,044 passed |
| Full repository suite after sentinel correction | 1,045 passed in 31.55 seconds |
| Compile validation | `python -m compileall` passed for `src` and `tests` |
| Whitespace validation | `git diff --check` passed |
| Long-line scan | no lines above 119 characters in corrected source/test files |
| Configured lint/type checks | none found; Ruff was unavailable in the environment |

The clustering implementation was also exercised with 18,000 synthetic rows
in 0.46 seconds, with approximately 133,288 KiB maximum resident memory for the
process. This is a local implementation benchmark, not production throughput.

## Candidate input provenance

| Artifact | Rows/definitions | Bytes | SHA-256 |
|-|-:|-:|-|
| `runs/papilio_demoleus_ranked_slices_20260708T110046Z/evidence.parquet` | 18,041 canonical candidates | 4,231,211 | `77e46690132a49fea023d2f82f58990eb35f5a47627f83cd7ac2d13e92ff01cb` |
| `reports/papilio_demoleus_ranked_slices_20260708T110046Z_query_plan.json` | 247 definitions | 127,811 | `af8325020f801d92eee10da46b4eb9008b2e55f9357bda879c355c4f0ee3056e` |

The accepted target identity is `gbif:1938069`, *Papilio demoleus*. Exploding
the source `query_definition_ids` and joining them to the recorded query plan
produced 27,850 query-hit links. These links were retained independently of the
18,041 canonical candidate records, preserving the discovery-evidence
deduplication invariant.

## Production-candidate inspection

Every source row contained `latitude=0.0`, `longitude=0.0`, and `accuracy=0`.
Flickr documents accuracy levels from 1 through 16, so the exact zero tuple is
a missing-location placeholder, not evidence for a real coordinate at Null
Island. After normalization and independent Parquet readback:

| Observation | Result |
|-|-|
| Candidate records | 18,041 |
| Query-hit provenance links | 27,850 |
| `flickr_zero_geo_sentinel` rows | 18,041 |
| Valid geotagged candidates | 0 (0%) |
| Located geographic clusters | 0 |
| `no_geo` assignments | 18,041 (100%) |
| Outlier assignments | 0 |
| Cluster rows | one `no_geo` aggregate with 18,041 candidate members |
| Eligible geographic reference clusters | 0 |
| Implied quota allocated/unallocated | 0 / 50 |

This result does not show that *Papilio demoleus* is absent from any place. It
shows only that this fetched candidate dataset cannot support geographic
stratification. A future metadata enrichment or refetch that retrieves valid
Flickr location fields is required before Phase 7 can allocate references by
Flickr candidate cluster.

## Generated artifact readback

The following generated artifacts are intentionally ignored by Git:

| Artifact | Rows | Bytes | SHA-256 |
|-|-:|-:|-|
| `runs/target_aware_phase_2_papilio_demoleus_20260713/flickr_geography.parquet` | 18,041 | 744,615 | `2101093e015210e1c074e5d2d4e0eb21b8fe913c9c54a57d93e0d8bbc9a590f9` |
| `runs/target_aware_phase_2_papilio_demoleus_20260713/flickr_geo_clusters.parquet` | 1 | 9,593 | `db581a3ccc59f35f8b57e9100748fb4b6476f71a2a7a128baf1049692061cebb` |
| `runs/target_aware_phase_2_papilio_demoleus_20260713/flickr_geo_assignments.parquet` | 18,041 | 742,074 | `53c51c24fa6e45f6f75bc8138552a73bdf35cf2133ce17d84d4cf46431ab58cb` |
| `reports/target_aware_phase_2_papilio_demoleus_20260713/flickr_geographic_workload.json` | compact report | 8,518 | `fd741efb08580bc48ba7f55021491c26f35933dd823c9d56c65923dc125761a0` |
| `reports/target_aware_phase_2_papilio_demoleus_20260713/flickr_geographic_workload.md` | compact report | 2,915 | `9865c6c84bb40891f47f5b39c06f9f7107eca72a33ad6d4893c2293e204b2321` |

The readback revalidated row counts, the singleton `no_geo` cluster, assignment
identity, candidate-only flags, zero outliers, query-link instrumentation, and
the excluded `no_geo` reference quota. The production artifact records the
source commit `5393ed2c5b15498c998f2ac1becbe89da6b4b2ff`.

## Failure and determinism behavior

- Invalid, incomplete, out-of-range, and non-finite coordinate pairs remain
  auditable and cannot acquire H3 cells.
- Flickr accuracy is treated as an ordinal precision class, not a distance or
  decimal-place measurement. Unknown precision cannot silently claim cells.
- Cluster IDs derive from target identity, configuration, and sorted member
  cells; timestamps and input order cannot alter them.
- Dateline-aware bounding geometry and spherical distance calculations avoid
  planar longitude discontinuities.
- Sparse or unlocated candidates remain assigned and reported rather than
  disappearing from the workload.
- Administrative and bioregion fallbacks require explicit source evidence and
  a bounded distance to an existing cluster. No reverse-geocoded claim is
  invented.
- Reports fail closed on cross-artifact identity, target-key, configuration,
  assignment-count, or query-provenance disagreement.
- Reference quotas are preliminary workload estimates. They exclude `no_geo`
  and do not create accepted reference images or biological occurrence claims.

## Tool and provenance notes

GitHits was queried fresh before every numbered task and again before the
sentinel correction. Examples that treated missing coordinates as zero, used
unstable sequential cluster IDs, discarded sparse candidates, or conflated
candidate density with occurrence truth were rejected.

Morph codebase search was attempted at every discovery/review boundary and
returned HTTP 429. Focused local source and call-site searches were used as the
authoritative fallback. External behavior was checked against Flickr's primary
location/accuracy documentation. No external result defined taxonomy,
classification, or a bucket decision.

## Repository hygiene

No generated Parquet, query plan, candidate dump, live API response, image,
model file, cache, secret, or `.env` file is included in a Phase 2 commit. The
pre-existing untracked paths below were not staged or modified:

- `config/papilio_demoleus_flickr_estimator.sh2`;
- `config/papilio_demoleus_multilingual_keywords.json`;
- `docs/superpowers/`;
- `duplicate_query_terms_skipped`;
- `logs/`;
- `query_terms_added`.

## Unexecuted live work

No Flickr API, image download, GBIF call, CUDA, YOLO, BioCLIP, or reference-media
operation was executed in this phase. The 18,041 rows are search candidates,
not verified *Papilio demoleus* occurrences. Because no valid geotags were
available, geographic reference sampling remains blocked on source metadata,
while non-geographic reference construction can proceed in later phases.
