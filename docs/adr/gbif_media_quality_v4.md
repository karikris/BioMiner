# GBIF media quality v4 evidence architecture

Status: accepted

Date: 2026-07-22

## Context

The rights-filtered GBIF media database v3 contains 16,612,063 media rows for
11,569,412 occurrences. Its 95.43% basic tuple fill rate is a physical-null
measurement, not evidence that the same share is valid, reusable, reachable,
taxonomically reliable, spatially usable, or ready for machine learning.

The raw DWCA contains 75,352,491 occurrence rows and 18,680,565 multimedia
assertions. v3 is a legacy filtered and normalized derivative. Both populations
are needed: v3 defines the rows being quality-audited, while raw assertions
define source lineage and the reason-coded extraction funnel.

Existing exploratory publishers are insufficient as the authoritative v4.
The URL resolver's legacy `publish-v4` path can delete rights-blocked rows, and
the temporal v1 publisher deleted derived pre-1960 rows. The new contract
requires one traceable status per input row, retention of unresolved and
invalid records, and ancient dates retained as flags. Those outputs are not v4
inputs.

## Decision

### Scope and identity

The authoritative runtime publication is
`data/derived/gbif_media_database/v4/`. It retains all 16,612,063 v3 media rows.
Rights, temporal, spatial, taxonomic, and AI exclusions are statuses and sparse
assertions, not row deletions. A separate source-lineage/exclusion ledger
accounts for all 18,680,565 raw multimedia assertions and their transitions to
the v3 population.

Occurrence facts are evaluated once per `gbifID` and stored in an occurrence
quality table. Original multimedia assertions are evaluated in a media
assertion quality table. URL observations and downloaded-content observations
belong to a media resource table. URL equality is not content identity, and
content equality is not independent occurrence evidence.

Stable identities use the repository canonical semantic fingerprint:

- `source_row_id`: source snapshot identity, member/file identity, partition,
  and stable source row position;
- `media_assertion_id`: source row identity plus the multimedia assertion
  contract, never URL alone;
- resource observation identity: assertion, probe policy/version, target URL,
  and observation time or freshness window;
- derived assertion identity: source row, target field, rule version, and
  evidence fingerprint.

`gbifID` remains the current GBIF key, not the sole permanent identity.

### Source preservation and assertion layers

Raw DWCA members, their Parquets, and v3 are immutable inputs. v4 preserves all
114 v3 fields byte-semantically and never rewrites them. Raw publisher values,
GBIF-interpreted values, legacy normalized values, deterministic derived
values, provider assertions, model candidates, and human decisions remain
distinct layers.

The earlier normalized `identificationVerificationStatus` is explicitly a
legacy transformed value and is never presented as the publisher's Darwin Core
verification status. An accepted taxonomic name, `identifiedBy`, or model
prediction cannot synthesize identification verification.

### Semantic nulls and applicability

Physical absence is classified using a versioned per-column policy. Every
field records raw fill and, where defined, an applicable denominator. The
policy distinguishes missing, invalid-present, structurally absent,
not-applicable, withheld, generalized, conflicting, repairable-null, and
non-repairable-null states.

`NOT_APPLICABLE`, `WITHHELD`, and `GENERALIZED` are not completeness failures.
Examples include species on a genus-rank occurrence, infraspecific epithet at
species rank, a day absent from a deliberate month-precision date, and precise
locality suppressed by source policy.

### Check registry and results

One central versioned registry defines each check's namespaced ID, family,
scope, fields, applicability, result type, severity, determinism, repair
permission, evidence, method, output, and rule version. Applicable checks emit
exactly one of:

`PASS`, `FAIL`, `UNKNOWN`, `NOT_APPLICABLE`, `WITHHELD`, `GENERALIZED`,
`CONFLICT`, or `NOT_TESTED`.

No aggregate quality score replaces these dimensions. Aggregate reports group
explicit check results and gate reasons.

### Evidence model

Derived and repaired assertions are sparse. Each carries `source_row_id`,
`gbifID`, media assertion identity where applicable, target field, original
and derived values, evidence source, permitted source URL or record identifier,
retrieval time, source snapshot/version, method, rule version, categorical
confidence, validation status, conflict status, and reviewer status.

Permitted confidence classes are `DIRECT_SOURCE`,
`DETERMINISTIC_DERIVATION`, `PROVIDER_ASSERTION`,
`STRUCTURED_PAGE_METADATA`, `CONTROLLED_TEXT_EXTRACTION`, `MODEL_CANDIDATE`,
`MANUALLY_VERIFIED`, and `UNRESOLVED`. Numeric probability is prohibited unless
calibrated on labelled validation data.

### Network policy

Metadata-only validation is the default. Network work is explicit, bounded,
host-aware, resumable, cached, and opt-in. The existing resolver becomes a v4
substage rather than the owner of the v4 row set.

The 130,689 reference-only rows are first classified independently for rights
and URL safety. The currently reported 4,055-row difference is an explicit
rights block, not proof of URL ineligibility. Rights-blocked rows remain in v4
with an exclusion status.

The deterministic 823-row pilot must be regenerated from the pinned source,
executed, manually reviewed, and meet its precision and occurrence-identity
gates before a 126,634-row network run can start. No broad existing-URL probe
is implicit. Live failures remain evidence; `HEAD` failure alone does not mean
dead media.

Image hashes and decoder metadata are captured when another authorized stage
already downloads the image. v4 does not download millions of images merely
to fill audit fields.

### Versioning, publication, and refresh

`gbif-media-quality-v4` is the authoritative schema family. The existing
legacy URL `V4_SCHEMA_VERSION` is a migration implementation and must not
publish to the authoritative v4 directory; it will be adapted or retired only
after equivalent tested behavior exists.

Outputs are Zstandard Parquet parts plus small manifests/reports. Parts are
bounded, checkpointed, checksummed, and committed atomically. The manifest is
written last. Production uses S3-compatible storage and PostgreSQL work state;
filesystem and SQLite are explicit local/test modes.

Source and value hashes drive incremental refresh. Unchanged assertions reuse
prior local-check, probe, resource, and enrichment evidence when all source,
rule, adapter, taxonomy, boundary, and TTL fingerprints remain compatible.
Restarting a job does not reprocess committed compatible parts.

## Consequences

- The authoritative v4 media row count remains 16,612,063 even when a row is
  unusable, unresolved, rights-blocked, ancient, or conflicting.
- Source funnel reconciliation covers 18,680,565 multimedia assertions and
  75,352,491 occurrence rows without treating excluded rows as lost evidence.
- The occurrence table avoids repeating one repaired fact for every image.
- Existing v3, temporal v1, resolver receipts, and raw archives remain
  unchanged.
- Network-wide validity cannot be claimed from a pilot or sample.
- The v4 pipeline may report incomplete live phases as `NOT_TESTED`; it cannot
  weaken a gate or fabricate a completion result.

## Rejected alternatives

- One wide 114-plus-column rewrite: duplicates occurrence assertions and
  obscures evidence provenance.
- A single weighted quality score: collapses incomparable dimensions and hides
  unknown/not-applicable states.
- Filling from plausible neighbouring fields: creates synthetic completion.
- Using URL as media identity: conflates assertions, variants, and content.
- Deleting failed or restricted rows: violates traceability and biases reports.
- Running all URLs before a reviewed pilot: creates uncontrolled provider load
  and unsupported validity claims.
