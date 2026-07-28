# GBIF media quality v4 operations and data dictionary

Status: full local metadata pipeline accepted; bounded resolver pilot passed

The authoritative local population is the immutable 114-column v3 Parquet:
16,612,063 multimedia assertions linked to 11,569,412 GBIF occurrences. v4
adds evidence layers; it does not rewrite or delete source rows.

## Status model

Every applicable check uses `PASS`, `FAIL`, `UNKNOWN`, `NOT_APPLICABLE`,
`WITHHELD`, `GENERALIZED`, `CONFLICT`, or `NOT_TESTED`. `UNKNOWN` is not a
pass. `NOT_APPLICABLE` is not missing. A generalized or withheld coordinate is
not treated as an ordinary repairable null.

## Output tables

| Layer | Grain | Purpose |
| --- | --- | --- |
| `source_lineage/source_media_status.parquet` | raw multimedia assertion | Stable identity and reason-coded raw-to-v3 funnel status |
| `source_lineage/identity_v2/parts/**/*.parquet` | raw multimedia assertion | Download key, immutable source location, ingestion time, and cryptographic source-value hash |
| `occurrence_quality/occurrence_quality.parquet` | `gbifID` | Occurrence, temporal, spatial, taxonomic, and identification checks |
| `media_assertion_quality/media_assertion_quality.parquet` | media assertion | Request-free URL syntax, type/format, rights, and provenance checks |
| `derived_assertions/*` | sparse assertion/candidate | Original-preserving temporal, geographic, taxonomic, life-stage, and sex evidence |
| `rights_and_attribution/media_rights.parquet` | media assertion | Explicit media licence normalization and attribution evidence; occurrence licence remains separate |
| `duplicates/duplicate_membership.parquet` | media assertion | Row/URL groups, cross-label conflicts, and leakage identifiers |
| `media_resources/parts/**/*.parquet` | canonical URL resource | Canonical media identity and explicitly untested network/content observations |
| `ai_readiness/parts/*.parquet` | media assertion | Independent readiness gates, ingestion decision, and reason codes |
| `representativeness/*.parquet` | dimension, taxon, provider, or dataset | Raw and URL-adjusted support, bias flags, scorecards, and remediation evidence |
| `representativeness_concentration/concentration_metrics.parquet` | species, cohort, concentration dimension | Provider, creator, regional, and temporal HHI, maximum share, and effective count |
| `freshness/*.parquet` | provider/dataset or derived manifest | Timestamp conflicts and configurable stale/current classifications |
| `provider_enrichment/provider_enrichment_registry.parquet` | provider adapter | Versioned structured-metadata contracts for the seven remediation priorities; execution status is explicit |
| `provider_enrichment_v4/provider_archive_execution.parquet` | provider dataset snapshot | Checksum-bound execution status, archive coverage, exact item matches, and unresolved archive reasons |
| `provider_enrichment_v4/provider_item_evidence.parquet` | exact provider media item match | Item-scoped values bound by exact occurrence ID and direct media URL |
| `provider_enrichment_v4/provider_occurrence_context.parquet` | provider occurrence media ensemble | Current item values that share a stable occurrence ID but lack a safe item-level bridge; never automatically repaired |
| `provider_enrichment_v4/provider_derived_assertions.parquet` | sparse missing-field assertion | New provider values with source row, media assertion, archive row, method, and confidence provenance |
| `provider_enrichment_v4/provider_conflicts.parquet` | conflicting item field | Current provider item values that disagree with preserved v3 values |
| `provider_enrichment_v4/provider_media_outcomes.parquet` | prioritized-provider media assertion | One retained outcome for every targeted row, including unmatched, core-only, and unavailable archives |
| `provider_enrichment_v4/provider_field_summary.parquet` | provider dataset and media field | Before/after missingness, direct item evidence, conflicts, and unresolved counts |
| `quality_results/phase4_pilot_execution/v1/audit/*` | resolver pilot result, review, gate, provider, and URL pattern | Executed 823-row pilot evidence, including every terminal outcome and reviewed resolution |
| `quality_results/restart_validation_v3/*` | committed stage | Checksum-bound restart, orphan-staging, and unchanged-row validation |
| `quality_results/global_acceptance_v5/*` | global criterion | Terminal evidence for all 42 criteria, including dependency-scoped checksum and Parquet verification |
| `completeness_gates/*.parquet` | gate and reporting dimension | Seven cumulative-use gates with media, occurrence, URL, and status denominators |
| `quality_results/review_capsules/*.parquet` | deterministic review item | Sealed before/after/evidence capsules for rights, attribution, duplicates, and exclusions |
| `incremental_state/state/**/*.parquet` | media assertion | Binary domain hashes for future snapshot diffs |
| `incremental_validation/changed_row_queue.parquet` | changed assertion only | Sparse refresh queue; unchanged rows are absent |

The AI table uses the exact gate columns `MEDIA_ADDRESSABLE`,
`MEDIA_REACHABLE`, `MEDIA_DIRECT`, `MEDIA_DECODABLE`,
`MEDIA_TRANSCODE_REQUIRED`, `MEDIA_TECHNICALLY_VALID`, `RIGHTS_KNOWN`,
`RIGHTS_ALLOWED`, `OCCURRENCE_CORE_COMPLETE`, `TAXONOMICALLY_USABLE`,
`SPATIALLY_USABLE`, `IDENTIFICATION_PROVENANCE_PRESENT`,
`AI_DETECTION_READY`, `AI_CLASSIFICATION_READY`, `HUMAN_REVIEW_READY`,
`EXCLUDED`, and `UNRESOLVED`. Dimension thresholds 224, 512, and 768 pixels
are reporting gates. Without inspected bytes they remain `NOT_TESTED`.

## Local audit and deterministic enrichment

Run from the BioMiner repository with the pinned Python environment:

```bash
uv run biominer gbif-media-quality baseline
uv run biominer gbif-media-quality local-checks
uv run biominer gbif-media-quality enrich
uv run biominer gbif-media-quality source-lineage
uv run biominer gbif-media-quality rights --output-directory data/derived/gbif_media_database/v4-next/rights_and_attribution
uv run biominer gbif-media-quality duplicates --output-directory data/derived/gbif_media_database/v4-next/duplicates
uv run biominer gbif-media-quality ai-readiness --output-directory data/derived/gbif_media_database/v4-next/ai_readiness
uv run biominer gbif-media-quality representativeness --output-directory data/derived/gbif_media_database/v4-next/representativeness
uv run biominer gbif-media-quality concentration --output-directory data/derived/gbif_media_database/v4-next/representativeness_concentration
uv run biominer gbif-media-quality freshness --output-directory data/derived/gbif_media_database/v4-next/freshness
uv run biominer gbif-media-quality provider-registry --output-directory data/derived/gbif_media_database/v4-next/provider_enrichment
uv run biominer gbif-media-quality provider-archives --output-directory data/derived/gbif_media_database/v4-next/provider_enrichment_v4
uv run biominer gbif-media-quality media-resources --output-directory data/derived/gbif_media_database/v4-next/media_resources
uv run biominer gbif-media-quality gates --output-directory data/derived/gbif_media_database/v4-next/completeness_gates
uv run biominer gbif-media-quality review-capsules --output-directory data/derived/gbif_media_database/v4-next/quality_results/review_capsules
uv run biominer gbif-media-quality incremental --output-directory data/derived/gbif_media_database/v4-next/incremental_state
```

The CLI resolves the pinned v3 input and source snapshot from the v4 source
inventory unless they are explicitly overridden. Production manifests record
the exact input paths, configuration, code commit, row counts, part checksums,
and validation gates. Every publisher refuses to replace an existing data
directory; choose a new versioned destination for a new run.

## Final enriched Parquet and retained filtering lineage

`data/derived/gbif_media_final/current/` is the sole final source of truth only
when it contains exactly `gbif_media_final_enriched.parquet` and
`manifest.json`, includes the five terminal URL-resolution fields, and a
separate publication audit has independently passed. A staging Parquet is not
a publication. The currently running legacy wide builder is producing a base,
not the terminal final: after it seals `current`, rename that directory
create-only to `base-v1`, then publish the resolver-integrated output directly
to the newly absent `current`. Do not copy, clean inputs for, or advertise a
final dataset while its builder still owns `.current.staging/`.

The retained lineage is:

1. Join occurrence and multimedia on `gbifID`, retaining one row per media
   assertion and explicit unresolved join evidence.
2. Remove columns below the authorized 5% completeness threshold.
3. Normalize the user-authorized identification grouping so historical
   `identified` values become `accepted`; this grouping is not the verbatim
   Darwin Core status.
4. Retain rows with `identifiedBy` or an accepted verification grouping.
5. Exclude explicit Copyright and All Rights Reserved media.
6. Exclude the 2,236 media rows attached to pre-1960 occurrences, retaining
   their occurrence-level evidence in `temporal_derivations.parquet`.
7. Add validated derived year, month, and day fields without changing the
   original temporal fields.
8. Add occurrence, media, rights, duplicate, AI-readiness, registry, keyword,
   and Flickr-query evidence as nested columns.
9. Retain the original media identifier/reference fields and append
   `resolved_media_identifier`, `effective_media_identifier`,
   `media_identifier_resolution_status`,
   `media_identifier_resolution_id`, and
   `media_identifier_license_basis`. Matching unresolved and rights-blocked
   rows remain present with explicit terminal status.

The rights-filtered input has 16,612,063 rows. The post-1960 temporal
publication has 16,609,827 rows. Manifests, rather than these prose numbers,
remain authoritative for a rerun.

The bounded builder is the restartable implementation for future reruns. It
builds slim dimensions once, aligns them to stable source ordinals, writes
checksum-bound immutable parts, and assembles them sequentially. Its default
DuckDB memory limit is 8 GB and its telemetry records process peak RSS,
physical I/O, throughput, cache reuse, failures, and the terminal output
manifest. Reusing the same state directory validates and skips sealed work.

```bash
uv run python scripts/build_gbif_final_enriched_bounded.py \
  --temporal-parquet data/derived/gbif_media_temporal/v1/gbif_media_temporal.parquet \
  --pre-temporal-parquet data/reference/gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_completeness_ge_5pct_verification_normalized_identified_by_or_accepted_rights_filtered_parquet/occurrence_multimedia_year_ge_1960_completeness_ge_5pct_verification_normalized_identified_by_or_accepted_rights_filtered.parquet \
  --temporal-audit data/derived/gbif_media_temporal/v1/temporal_derivations.parquet \
  --registry-dir data/derived/gbif_flickr_keyword_registry/v2/registry \
  --source-assertions data/derived/gbif_flickr_keyword_registry/v2/enrichment/source_name_assertions.parquet \
  --quality-dir data/derived/gbif_media_database/v4 \
  --state-dir data/state/gbif-final-bounded-v1 \
  --output-dir data/derived/gbif_media_final/bounded-v1 \
  --producer-git-sha "<exact-builder-git-sha>" \
  --memory-limit 8GB \
  --threads 4
```

After the legacy base builder and every resolver work item complete, seal the
terminal resolver sidecar, move the base publication out of the canonical
name, and stream the resolver evidence into the sole `current` publication.
The move is reversible and must fail if `base-v1` already exists.

```bash
uv run biominer gbif-media-url-resolve finalize \
  --sqlite-workstore data/state/gbif-media-url-full-v1.sqlite \
  --output-root data/state/gbif-media-url-resolution/full-v1 \
  --output-directory data/state/gbif-media-url-resolution/full-v1/finalized-v1 \
  --run-id gbif-media-url-full-v1 \
  --expected-rows 130689

test -d data/derived/gbif_media_final/current
test ! -e data/derived/gbif_media_final/base-v1
mv data/derived/gbif_media_final/current \
  data/derived/gbif_media_final/base-v1

uv run python scripts/enrich_gbif_final_with_resolutions.py \
  --base-publication-directory data/derived/gbif_media_final/base-v1 \
  --resolution-directory data/state/gbif-media-url-resolution/full-v1/finalized-v1 \
  --output-directory data/derived/gbif_media_final/current \
  --repository-root . \
  --producer-git-sha "<exact-enrichment-producer-git-sha>" \
  --expected-resolution-rows 130689 \
  --batch-rows 50000 \
  --row-group-rows 100000

uv run python scripts/validate_gbif_final_resolution_enrichment.py \
  --output-directory data/derived/gbif_media_final/current \
  --base-publication-directory data/derived/gbif_media_final/base-v1 \
  --resolution-directory data/state/gbif-media-url-resolution/full-v1/finalized-v1 \
  --repository-root .
```

Independently audit the resolver-integrated publication before transfer or
cleanup. The expected producer SHA must be the exact value in the primary
manifest, not current `HEAD`.

```bash
uv run python scripts/audit_gbif_final_enriched.py \
  --publication-directory data/derived/gbif_media_final/current \
  --temporal-parquet data/derived/gbif_media_temporal/v1/gbif_media_temporal.parquet \
  --pre-temporal-parquet data/reference/gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_completeness_ge_5pct_verification_normalized_identified_by_or_accepted_rights_filtered_parquet/occurrence_multimedia_year_ge_1960_completeness_ge_5pct_verification_normalized_identified_by_or_accepted_rights_filtered.parquet \
  --registry-directory data/derived/gbif_flickr_keyword_registry/v2/registry \
  --source-assertions data/derived/gbif_flickr_keyword_registry/v2/enrichment/source_name_assertions.parquet \
  --quality-directory data/derived/gbif_media_database/v4 \
  --base-publication-directory data/derived/gbif_media_final/base-v1 \
  --resolution-directory data/state/gbif-media-url-resolution/full-v1/finalized-v1 \
  --output-directory data/derived/gbif_media_final/audit-v1 \
  --repository-root . \
  --expected-producer-git-sha "<producer-git-sha-from-primary-manifest>" \
  --memory-limit 8GB \
  --threads 4
```

Build the slim locator after the audit and before deleting any audited input.
It retains only stable IDs, direct/reference URLs, and species keys in DuckDB;
the full enriched rows remain solely in Parquet. URL, GBIF ID, species-key,
and registry-taxon-key indexes are reopened, benchmarked, and checksum-bound.

```bash
uv run python scripts/build_gbif_final_locator_index.py \
  --publication-directory data/derived/gbif_media_final/current \
  --publication-audit-directory data/derived/gbif_media_final/audit-v1 \
  --output-directory data/derived/gbif_media_final/locator-v1 \
  --repository-root . \
  --memory-limit 8GB \
  --threads 4
```

The superseded-artifact cleanup is dry-run by default. It permits only the 14
named targets encoded in `superseded_cleanup.py`: the terminally superseded
`base-v1`, the v1/v2 layers, and the pre-rights intermediate directories. It
explicitly protects v3, v4, the rights-filtered source, raw intake Parquet and
archive, unresolved-row audit, resolver state, the canonical `current`
publication, and its publication audit. The original obsolete set is about
38 GB; once `base-v1` is terminally superseded, the eligible total grows by
the base publication's physical size. Execution first persists checksummed
intent, rechecks each file immediately before unlinking, resumes after
interruption, rejects unexpected files, and writes its manifest last.

```bash
uv run python scripts/cleanup_superseded_gbif_artifacts.py \
  --repository-root . \
  --publication-audit-directory data/derived/gbif_media_final/audit-v1 \
  --state-directory data/state/gbif-final-superseded-cleanup-v1

uv run python scripts/cleanup_superseded_gbif_artifacts.py \
  --repository-root . \
  --publication-audit-directory data/derived/gbif_media_final/audit-v1 \
  --state-directory data/state/gbif-final-superseded-cleanup-v1 \
  --execute
```

## Resolver pilot and targeted URL resolution

Preparation is offline by default. Full-queue construction needs
`--allow-full-queue`; network work additionally needs `--execute-network`.
Never supply those flags merely to make an audit green.

The executed deterministic pilot has 823 assertions: 764 network-eligible and
59 explicitly rights-blocked. It made 2,068 bounded network attempts and
produced 217 resolved rows plus 547 eligible unresolved or non-image outcomes.
All 217 resolved rows passed an independent structured-evidence review, with
zero wrong-occurrence substitutions and a 95% Wilson precision interval of
0.982605 to 1.0. The review binds Flickr source photo identifiers to resolved
CDN identifiers or retains an unchanged direct provider URL, then requires
matching image MIME, signature, bounded decoder, sampled-byte, and provenance
evidence. It is explicitly an agent structured-evidence review, not a claim of
human visual inspection.

All ten pilot acceptance gates pass. The separately authorized, checkpointed
130,689-row reference-only run is active: 126,634 rows are network-eligible
and 4,055 are retained as rights-blocked. Its mutable queue is not terminal
evidence. Final resolved and unresolved counts may be reported only after the
create-only reducer publishes and independently verifies one result per input.
Every unresolved pilot row remains present with a terminal reason.

## Provider enrichment

Use explicit provider evidence only. Prefer bulk provider exports over
per-record requests. Never copy the occurrence licence into the media licence,
infer a creator from unrelated occurrence fields, or turn provider defaults
into direct source assertions. Publish candidates with evidence and review
status before promotion.

The provider registry exposes versioned adapters for the seven prioritized
providers. Registry publication is offline and does not mean an adapter has
executed: `execution_status=NOT_TESTED` remains authoritative until structured
provider evidence has been fetched, cached, and validated under explicit
network authorization.

`provider-archives` consumes a pinned, checksum-inventoried archive manifest.
It accepts a Darwin Core Multimedia row as direct item evidence only when both
its occurrence ID and identifier exactly match the preserved media assertion.
An occurrence-core licence is never promoted to a media licence, and
`recordedBy` is never promoted to creator. Multiple archive rows for one exact
item key fail closed as a conflict. Archives without a Multimedia extension,
unavailable archives, unmatched items, and item rows with no applicable new
fields remain explicit outcomes. The command performs no network requests and
refuses to overwrite an existing output directory.

When a provider replaces legacy media URLs, the archive may still resolve the
occurrence while lacking a safe one-to-one item bridge. Those current media
ensembles are retained separately as occurrence-scoped context, including raw
licence values, within-occurrence conflicts, and explicit Copyright or All
Rights Reserved item counts. They support change detection and review but
cannot repair an individual v3 media assertion.

## Duplicate analysis and AI readiness

URL identity is not content identity. Content SHA-256 and perceptual groups
remain `NOT_TESTED` until an authorized pipeline already has image bytes.
Use the occurrence, dataset-occurrence, creator, provider/dataset, location,
event, and source-platform group identifiers when building splits. Do not mix
the same group across train, validation, and test.

## Incremental refresh

Pass `--previous-state-glob` pointing at the prior `state/**/*.parquet`. The publisher
hashes URL, rights, spatial, temporal, identification, taxonomy, and provider
domains independently. Only new, deleted, or changed assertions enter
`changed_row_queue.parquet`. URL and provider TTL policy is stored separately;
taxonomy and boundary refresh depend on pinned version changes.

An unchanged full rerun is valid only when it queues zero rows and the current
and previous semantic fingerprints match. The recorded full-data validation
meets both conditions.

## Reports, rollback, and recovery

The current historical reports live in
`reports/gbif_media_database/v4_final_20260729/`; their manifest hashes all 19
Markdown reports, but their assertion that no broad run occurred is stale.
Historical executable acceptance directories remain immutable and are not
terminal evidence for the active run. The next terminal audit will publish
create-only under
`data/derived/gbif_media_database/v4/quality_results/global_acceptance_v5/`;
its report suite will publish create-only under
`reports/gbif_media_database/v4_terminal_20260729/`.
Runtime publications use staging directories and atomic rename. On
interruption, retain committed destinations, delete only the specific
incomplete staging directory after inspection, and restart into a new
destination. Rollback means selecting the previous manifest-bound output;
never mutate v3 or a committed v4 directory in place.

The bounded 823-row resolver pilot completed with 2,068 stored attempts. An
authorized 130,689-row reference-only run is active and checkpointed; it does
not probe the existing direct-URL population broadly. Until its terminal
manifest is published, only completed immutable shards are evidence and no
full-tail success rate is claimed. Reachability, redirect-final URLs, MIME
truth, decoding, image dimensions, content hashes, perceptual duplicates, and
model readiness remain explicitly `NOT_TESTED` outside their actual tested
denominators.
