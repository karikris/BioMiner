# GBIF media quality v4 implementation plan

Status: implemented; current local acceptance audit is 42/42 PASS

This plan governs the production-grade audit of the immutable rights-filtered
GBIF media database v3. It is subordinate to the repository scientific and
artifact policies and does not promote the legacy derivative into the current
ground-zero biological evidence lineage.

## Verified starting state

| Input or existing stage | Verified state | v4 treatment |
| --- | --- | --- |
| Raw DWCA archive | SHA-256 `3494944db9bca6917e0176e63852f8c233cdd7da95904c041c55ff8e8d31c6b9`; 75,352,491 occurrence rows, 18,680,565 multimedia rows, and 75,352,491 verbatim rows | Immutable source lineage |
| Joined occurrence/media Parquet | 18,680,565 rows; no unresolved multimedia foreign keys in the recorded audit | Recompute and reconcile, not assume |
| Rights-filtered v3 Parquet | SHA-256 `c96505f410723da57db4bd11bcffdc4e72be59ee59ecbaad8f4af8677229e57f`; 16,612,063 rows, 11,569,412 `gbifID` values, 114 columns | Immutable audited population |
| Intake audit v1 | Archive, members, Parquets, row groups, schemas, and join coverage recorded; producer worktree was dirty | Revalidate with committed v4 code |
| URL resolver | Source-bound identities, attempt/result sidecars, bounded probes, provider adapters, workstore, create-only publication | Reuse and harden as a v4 substage |
| Resolver pilot | Executed 823 fixed work rows: 764 network-eligible, 59 rights-blocked, 217 resolved, 547 eligible unresolved/non-image outcomes, and 2,068 stored attempts; all 217 resolutions passed structured-evidence review with zero wrong-occurrence substitutions | Preserve the checksum-bound execution and review audit; do not describe it as human visual inspection |
| Reference-only tail | 130,689 rows; 126,634 non-rights-blocked and 4,055 explicitly rights-blocked | Record rights status separately from URL/network eligibility |
| Temporal v1 | Exact candidate counts reproduced, but 2,236 rows were excluded as pre-1960 | Do not consume; v4 retains and flags ancient records |

The legacy v3 transformation history includes normalized values and row
filters. v4 must retain those 114 values exactly while also linking them to the
raw occurrence and multimedia assertions. A normalized v3 value is not
retroactively described as publisher-supplied source evidence.

## Target layout and ownership

Runtime publications are create-only and ignored by Git:

```text
data/derived/gbif_media_database/v4/
  source_lineage/
  occurrence_quality/
  media_assertion_quality/
  media_resource/
  derived_assertions/
  conflicts/
  quality_results/
  aggregates/
  checkpoints/
  manifest.json

reports/gbif_media_database/v4_final_20260729/
  source_funnel.md
  ... nineteen other required reports ...
  manifest.json
```

Source-controlled ownership is intentionally narrow:

| Path | Ownership |
| --- | --- |
| `src/biominer/gbif_quality/` | Check registry, field-applicability policy, schemas, local audit/enrichment stages, aggregation, publication, incremental identities, and report models |
| `src/biominer/gbif_media_resolution/` | Network-safe URL resolution, provider adapters, response evidence, probe cache, and pilot/full work queues |
| `src/biominer/gbif_temporal/` | Legacy temporal migration code; reusable parsing may move behind the v4 assertion contract, but its row-dropping publication is not a v4 input |
| `src/biominer/storage/` | Existing local/S3 artifact commit and Parquet conventions |
| `src/biominer/workstore/` | Existing PostgreSQL production and explicit SQLite local/test work state |
| `src/biominer/cli.py` and run orchestration | One registered audit command surface and production-stage integration; no standalone competing framework |
| `tests/fixtures/gbif_quality/` | Synthetic DWCA, golden results, mocked network fixtures, and deterministic review fixtures |

Untracked legacy scripts under `scripts/` remain user-owned during migration.
Equivalent logic will be consolidated only after a tested package replacement
exists; no dirty file will be overwritten or deleted.

## Phase contracts

1. **Repository and data audit.** Freeze scope, lineage, ownership, semantic
   boundaries, network policy, and versioning in this plan and the v4 ADR.
2. **Source reconciliation and baseline.** Inventory raw members and v3,
   produce a reason-coded 75M/18M/16M funnel, physical schema audit, field
   policy, and raw/applicable completeness using both row and occurrence
   denominators.
3. **Check registry and local validation.** Publish the versioned registry and
   one status per applicable check without requests; create occurrence and
   media assertion quality tables without repeating occurrence repairs.
4. **Deterministic enrichment.** Write sparse, evidence-bearing temporal,
   geographic, taxonomic, life-stage, and sex assertions; retain conflicts and
   prove idempotence.
5. **Resolver pilot.** Harden URL policy and adapters, regenerate the stable
   823-row sample, run it only with explicit live authorization, produce review
   material, and block expansion unless the precision gate passes.
6. **Targeted media remediation.** Resolve only the gated reference tail and
   scheduled missing-format/type resources with resumable host queues.
7. **Rights and attribution.** Normalize only explicit evidence, keep
   occurrence licence as context, and prioritize providers without blanket
   inference.
8. **Duplicates and AI readiness.** Separate assertion, URL, resource,
   content, and perceptual identities; publish leakage groups and independent
   readiness decisions/reasons.
9. **Incremental production and documentation.** Diff source/value hashes,
   resume committed parts, benchmark under 16 GB RSS, execute the complete
   metadata-only run, generate all required reports, and verify every global
   acceptance criterion.

Each independently verifiable task gets its own commit. A phase is pushed only
after its task tests, provenance checks, and `git diff --check` pass. Network
stages remain explicit and opt-in; an unavailable live gate is reported as
`NOT_TESTED`, never converted to success.

## Acceptance evidence map

Completion is proved from stored evidence rather than report prose:

- Source preservation: pre/post SHA-256 for archive, raw member Parquets, and
  v3, plus read-only source mounts in run configuration.
- Full traceability: source-lineage row counts and reason-coded funnel
  reconciliation for all 18,680,565 multimedia assertions; quality status for
  every v3 row.
- Scope separation: unique occurrence table keyed by `gbifID`, unique media
  assertion identity, and resource observations linked without URL/content
  identity conflation.
- Derived values: sparse assertion rows containing the complete required
  evidence envelope and categorical confidence.
- Unknown and semantic absence: registry-result counts for every status,
  with applicability denominators in column and gate reports.
- Restart/idempotence: interrupted fixture run, resumed full fixture run, and
  identical semantic fingerprints across clean reruns.
- Physical integrity: schemas, row-group reconciliation, part checksums,
  atomic commit receipts, and manifest-last validation.
- Resource bound: measured peak RSS, elapsed time, throughput, and bytes for
  the full metadata-only run.
- Network claims: probe/result denominators, cached attempt evidence, review
  decisions, uncertainty intervals, and an explicit record of whether a broad
  network run occurred.

The current executable audit records all 42 global acceptance criteria as
`PASS` and the final report manifest binds the 20 required reports (19
Markdown reports plus the manifest) to current-run evidence. The 126,634-row
resolver tail and any broad existing-URL run remain separate, explicit, future
operations.
