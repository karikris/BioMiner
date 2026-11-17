# Ranked Flickr searches by location status

The ranked-slice runner preserves the existing keyword and text/tag query pairs,
upload-time slicing, source-ID deduplication, query associations and shared rate
ledger. `--has-geo 1` remains the default. `--has-geo 0` searches for photos without
public geotags and uses ordinary 500-result pages. Geographic searches use 250.

Use `--utc-upload-bounds` to encode each requested date interval as inclusive UTC
Unix-second boundaries. The end date includes 23:59:59; an end date of today can
only return photos already public when Flickr serves the requests. These are
upload dates, not capture dates. The default end date is today's UTC date.

```sh
python scripts/run_papilio_demoleus_ranked_flickr_slices.py \
  --run-id example-nongeotagged-refresh \
  --output-root /private/flickr-runs \
  --report-dir /private/flickr-reports \
  --start-date YYYY-MM-DD --end-date YYYY-MM-DD \
  --has-geo 0 --utc-upload-bounds \
  --secrets-env /private/secrets.env \
  --shared-api-budget-db /private/flickr_refresh.sqlite \
  --plan-only
```

Replace both date placeholders. Inspect the plan, then remove `--plan-only` to
make API calls. The supplied shared ledger must have the existing `calls(at)`
schema. Never expose an API key in command arguments or output. A new location
policy requires a new state database; a mixed-policy resume fails closed.

Completed non-geotagged collections publish `nongeotagged_source_records.parquet`
and `.csv`; geographic collections retain `geotagged_source_records.parquet` and
`.csv`. Both preserve all search associations. Non-geotagged publication rejects
records containing valid public geotag coordinates instead of silently mixing
the two policies. Search results contain discovery metadata and supplied image
URLs; this command does not download image pixels or run model inference.

For an incremental refresh, overlap the earlier cutoff and compare canonical
provider/photo IDs with the old snapshot. Metadata updates and changed visibility
can also affect older photos, so a recent-upload search is not a full historical
reconciliation. Discovery results are candidate evidence, never species truth.
