# GBIF media quality v4 operations and data dictionary

Status: implemented local metadata pipeline; live resolver gate `NOT_TESTED`

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
| `occurrence_quality/occurrence_quality.parquet` | `gbifID` | Occurrence, temporal, spatial, taxonomic, and identification checks |
| `media_assertion_quality/media_assertion_quality.parquet` | media assertion | Request-free URL syntax, type/format, rights, and provenance checks |
| `derived_assertions/*` | sparse assertion/candidate | Original-preserving temporal, geographic, taxonomic, life-stage, and sex evidence |
| `rights_and_attribution/media_rights.parquet` | media assertion | Explicit media licence normalization and attribution evidence; occurrence licence remains separate |
| `duplicates/duplicate_membership.parquet` | media assertion | Row/URL groups, cross-label conflicts, and leakage identifiers |
| `ai_readiness/parts/*.parquet` | media assertion | Independent readiness gates, ingestion decision, and reason codes |
| `representativeness/*.parquet` | dimension, taxon, provider, or dataset | Raw and URL-adjusted support, bias flags, scorecards, and remediation evidence |
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
```

The rights, duplicate, AI-readiness, representativeness, incremental, and
report publishers are package APIs in `biominer.gbif_quality`. Their production
manifests record the exact input paths, configuration, code commit, row counts,
part checksums, and validation gates. A publisher refuses to replace an
existing data directory; choose a new versioned destination for a new run.

## Resolver pilot and targeted URL resolution

Preparation is offline by default. Full-queue construction needs
`--allow-full-queue`; network work additionally needs `--execute-network`.
Never supply those flags merely to make an audit green.

The current deterministic pilot has 823 assertions: 764 network-eligible and
59 explicitly rights-blocked. Live resolution, manual adjudication, Wilson
precision bounds, and wrong-occurrence review are still `NOT_TESTED`. Do not
start the 126,634-row eligible tail until every pilot acceptance gate passes.
Unresolved rows remain present with their failure evidence.

## Provider enrichment

Use explicit provider evidence only. Prefer bulk provider exports over
per-record requests. Never copy the occurrence licence into the media licence,
infer a creator from unrelated occurrence fields, or turn provider defaults
into direct source assertions. Publish candidates with evidence and review
status before promotion.

## Duplicate analysis and AI readiness

URL identity is not content identity. Content SHA-256 and perceptual groups
remain `NOT_TESTED` until an authorized pipeline already has image bytes.
Use the occurrence, dataset-occurrence, creator, provider/dataset, location,
event, and source-platform group identifiers when building splits. Do not mix
the same group across train, validation, and test.

## Incremental refresh

Point `previous_state_glob` at the prior `state/**/*.parquet`. The publisher
hashes URL, rights, spatial, temporal, identification, taxonomy, and provider
domains independently. Only new, deleted, or changed assertions enter
`changed_row_queue.parquet`. URL and provider TTL policy is stored separately;
taxonomy and boundary refresh depend on pinned version changes.

An unchanged full rerun is valid only when it queues zero rows and the current
and previous semantic fingerprints match. The recorded full-data validation
meets both conditions.

## Reports, rollback, and recovery

Reports live in `reports/gbif_media_database/v4/`; `manifest.json` is written
last and hashes every report. Runtime publications use staging directories and
atomic rename. On interruption, retain committed destinations, delete only the
specific incomplete staging directory after inspection, and restart into a
new destination. Rollback means selecting the previous manifest-bound output;
never mutate v3 or a committed v4 directory in place.

No broad network run was executed for the recorded local audit. Consequently,
reachability, redirect-final URLs, MIME truth, decoding, image dimensions,
content hashes, perceptual duplicates, and model readiness remain explicitly
`NOT_TESTED` where direct evidence is absent.
