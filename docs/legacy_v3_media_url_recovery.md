# Legacy v3 GBIF media URL recovery

This command recovers missing direct image URLs in the immutable legacy v3
GBIF media derivative. It does **not** promote v3 into the current GBIF
ground-zero production lineage. Every manifest labels that limitation.

The resolver preserves the original `media_references`, manually follows and
records redirects, validates public HTTP(S) targets, requires matching image
MIME/signature/decoder evidence, and records only bounded response prefixes.
Complete image download and `content_sha256` remain deferred.

## Run stages

Use configured PostgreSQL for production work state. Local development must
opt in to SQLite with `--sqlite-workstore`.

```bash
biominer gbif-media-url-resolve prepare \
  --source data/reference/.../source.parquet \
  --source-manifest data/reference/.../manifest.json \
  --output-root data/state/gbif-media-url-resolution/pilot \
  --run-id gbif-media-url-pilot-v1 \
  --mode pilot \
  --sqlite-workstore data/state/gbif-media-url-pilot.sqlite

biominer gbif-media-url-resolve work \
  --output-root data/state/gbif-media-url-resolution/pilot \
  --run-id gbif-media-url-pilot-v1 \
  --worker-id local-pilot \
  --max-batches 1 \
  --sqlite-workstore data/state/gbif-media-url-pilot.sqlite

biominer gbif-media-url-resolve finalize \
  --output-root data/state/gbif-media-url-resolution/pilot \
  --output-directory data/derived/gbif_media_url_resolution/pilot-v1 \
  --run-id gbif-media-url-pilot-v1 \
  --sqlite-workstore data/state/gbif-media-url-pilot.sqlite
```

Pilot preparation deterministically selects 100 rows for hosts with at least
1,000 affected records, 25 rows for hosts with 25–999 records, and all rows for
smaller hosts. Review 200 resolved results across adapters and hosts. Any
source/media identity mismatch disables that adapter before creating a new
`--mode full` run.

After the full run is finalized, publish v4:

```bash
biominer gbif-media-url-resolve publish-v4 \
  --source data/reference/.../source.parquet \
  --source-manifest data/reference/.../manifest.json \
  --resolution-directory data/derived/gbif_media_url_resolution/v1 \
  --output-directory data/derived/gbif_media_database/v4
```

The v4 publisher refuses pilot results, excludes explicit rights restrictions,
retains unresolved eligible records, preserves the source URL field, adds the
resolved/effective URL fields, and indexes raw URL, effective URL, and
`gbifID`. Both publication stages are create-only and write their manifest
last.

Unknown item-level media rights use the occurrence licence only as provisional
internal-research permission. Such records are labelled
`occurrence_license_fallback`; this does not establish media redistribution
rights.
