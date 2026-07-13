# Target-aware few-shot Phase 1 verification

Date: 2026-07-13. Confidence: high for the implemented geographic schemas,
checkpoint behavior, QA, and local publication contract; moderate for live GBIF
operation because this phase deliberately used deterministic clients and a
normalized Parquet fixture rather than authenticated production downloads.

Phase 1 adds multi-resolution cell geography, occurrence-level GBIF provenance,
resumable spread compilation, geographic summary and QA artifacts, and a
fail-closed registry publication boundary. Geographic absence remains unknown
evidence. It is never converted into a taxonomic mismatch, biological absence,
or classifier hard negative.

The phase started from `c11ab678d87c3108b7d37a1181202dd9e74b8a65` on
`main`. A pre-report fetch confirmed that `origin/main` remained at that commit
and the phase task head had no remote divergence.

## Task commits

| Task | Commit | Result |
|-|-|-|
| 1.1 geographic primitives | `613887d0315b977447196f62043fc2d1f366a593` | H3-backed cells, parents, neighbours, centres, distance, coordinate validation, uncertainty retention, optional `geo` dependency |
| 1.2 GBIF spread discovery | `0b8053f20e2303ab9b6b9c83c5fa7bc57ab48b4f` | paged-search ceiling, SIMPLE_PARQUET bulk handoff/reader, species-key validation, occurrence suitability, atomic resumable checkpoints, spread/evidence manifests |
| 1.3 summary and QA | `adb788bd3bdda27b300ae8effdf72d3061f1f678` | cell/resolution coverage, circular envelope, connectivity, density, source coverage, temporal evidence, outlier/data-deficiency policy, compact QA |
| 1.4 registry publication | `e0902830124a8593160cfe4e54bd374644df750e` | required geographic artifacts, source/QA merge, schema and fingerprint validation, checksummed final manifest, unknown-not-negative semantics |
| 1.5 acceptance tests | `e12d5d57ed6423fd9e3930c20f68fec54e836646` | real H3/Parquet dateline, corruption, completed-resume, QA, data-deficiency, and builder-to-publisher integration coverage |

Each numbered task is a separate commit with its requested commit message.

## Dependency evidence

| Check | Result |
|-|-|
| Python | `3.14.5` |
| H3 Python package | `4.5.0` |
| H3 constraint | optional `h3>=4.5,<5`; included in the test extra |
| `uv lock --check` | passed; 36 packages resolved |
| `pyproject.toml` SHA-256 | `0d99de91314d4232279a3d72e96e44345fa44485ff1ef6f9509822a1decd9a08` |
| `uv.lock` SHA-256 | `7321b73a0ad0282ba03c3f3014305c5c8a94fdfe733bf33bc10dfba94a73a012` |
| Configured lint/type checks | none found; Ruff was not installed, so compile, whitespace, physical-schema, focused, and full tests were used |

H3 remains lazily imported, so non-geographic BioMiner use does not require the
optional dependency. The lock contains CPython 3.14-compatible H3 wheels.

## Test evidence

| Gate | Result |
|-|-|
| Geographic primitive focused tests | 16 passed |
| Task 1.2 GBIF/range/geography suite | 46 passed |
| Task 1.3 geographic/registry suite | 51 passed |
| Task 1.4 publication, registry, and CLI suite | 98 passed |
| Task 1.5 geographic acceptance suite | 42 passed |
| Empty spread/evidence physical Parquet schema round-trip | exact schema equality passed |
| Compile validation | `python -m compileall` passed for changed modules/tests |
| Whitespace validation | `git diff --check` passed |
| Full repository suite | `1016 passed in 32.52s` |

The full suite increased from Phase 0's 974 tests to 1,016 without skipped live
tests, weakened assertions, or restored legacy behavior.

## Papilio demoleus fixture inspection

A deterministic fixture was built through the actual spread compiler, summary
builder, and registry publisher under
`/tmp/biominer-phase1-papilio-demoleus-15vds_7l`. The compact inspection record
is `phase1_inspection.json` in that directory. No generated Parquet artifact was
added to Git.

| Observation | Result |
|-|-|
| Accepted species | `gbif:1938069`, *Papilio demoleus* |
| Source snapshot | `gbif-download:phase1-fixture` |
| Occurrence inputs | 7 |
| Normalized occurrence-evidence rows | 17 |
| Spread rows | 14 |
| Spatial resolutions | 3, 5, 7 |
| Eligible occurrence count | 4 at each resolution |
| Preserved specimen count | 1 at each resolution; retained but ineligible for current-range inference |
| Invalid coordinate / taxon mismatch | 1 / 1, retained in evidence and QA |
| Occupied cells by resolution | 3: 3 cells; 5: 4 cells; 7: 4 cells |
| Regional connected components | 3 |
| Suspicious isolated cells | 1, retained and warned rather than deleted |
| Current / historical evidence | 4 / 1 |
| Data deficient | false under the recorded fixture policy |
| Geographic QA codes | invalid coordinate, taxon-key mismatch, extreme isolated outlier |
| Published runtime artifacts | 9, including spread, summary, merged QA/source snapshots, and manifest |
| Published artifact checksum verification | passed for every inventoried Parquet file |

The fixture also verified that a data-deficient species with an empty spread can
be published when its summary row explicitly records the deficiency. Structural
absence of a required summary row is rejected as a malformed register; that
check is distinct from biological absence.

## Failure and resume behavior

- GBIF occurrence search requests never exceed the documented 300-record page
  limit or the 100,000-record `offset + limit` ceiling.
- Taxa above that ceiling raise a structured bulk-download requirement using
  `SIMPLE_PARQUET`; normalized GBIF Parquet downloads resume inside a batch.
- Species identity prefers GBIF `speciesKey`, retaining accepted infraspecific
  records while rejecting mismatched or malformed species identifiers.
- Checkpoint identity covers registry, taxon, query, source snapshot, retrieval
  time, grid implementation/version, resolutions, and schema.
- Checkpoint parts are atomic and validated by schema, cursor continuity, row
  count, byte count, and SHA-256. A part written before state is safely adopted
  only when recomputed content matches. Corruption fails closed.
- A completed checkpoint performs zero source calls on replay.
- Preserved specimens, fossils, declared geospatial issues, invalid coordinates,
  and taxon mismatches remain auditable but cannot silently become current-range
  evidence.
- Registry publication verifies builder manifests, geographic schemas, one
  summary per accepted species, per-species spread fingerprints, fatal QA, and
  final staged checksums before promotion.

## Tool and provenance notes

GitHits was queried fresh before every task. The H3 package internals were
inspected before integration. Generated examples that omitted coordinate and
schema validation, used unsafe GBIF class-key matching, deep search paging,
per-request clients, pandas, destructive publication, synthetic H3 identifiers,
or JSON stand-ins for Parquet were rejected.

Morph codebase search was attempted for every discovery/review boundary and
returned HTTP 429. Focused local source and call-site searches were used as the
authoritative fallback. Headroom was invoked for the large geographic module;
its protected-code route retained the source without compression, so subsequent
review used focused line ranges.

External behavior was checked against primary GBIF occurrence search and
download documentation and the upstream H3 release/source material. No external
result was allowed to define taxonomy, query truth, or a bucket decision.

## Repository hygiene

The phase commits contain source, tests, lock metadata, and documentation only.
No secret, `.env`, live API response, downloaded image, model weight, cache,
DuckDB database, generated registry Parquet file, or temporary checkpoint was
committed. The pre-existing untracked paths below were not staged or modified:

- `config/papilio_demoleus_flickr_estimator.sh2`;
- `config/papilio_demoleus_multilingual_keywords.json`;
- `docs/superpowers/`;
- `duplicate_query_terms_skipped`;
- `logs/`;
- `query_terms_added`.

## Unexecuted live work

No authenticated GBIF bulk download or live occurrence search was executed.
No Flickr, iNaturalist, CUDA, YOLO, BioCLIP, or reference-media operation belongs
to this phase. The production bulk-download request contract and Parquet reader
are implemented, but live service credentials, queue latency, download size,
and real-world dataset citation diversity remain unmeasured until an explicitly
approved production run.
