# Geotagged Papilio demoleus rerun — 11 November 2026

The user requested a historical rerun of the 8 July, 18,041-image collection,
using its original runner and keywords, with previously productive keywords
queued first and all accepted images required to have public geotags. This is
an explicit historical reconciliation task, separate from the current GBIF
ground-zero production lineage.

Starting branch: `agent/yoloe-flickr-pilot`; starting SHA:
`ca99dfb02fd5a936223f9a417e433446ef2dc6e6`.
Existing YOLOE source and test edits were preserved and excluded from this task.

## Change

Global registry and fixed-upload-slice planners previously requested
`has_geo=0`, explicitly excluding geotagged photos. They now request
`has_geo=1` and use Flickr's 250-result geographic page size. Live search
responses are rejected before ingestion if coordinates are missing, nonfinite,
out of range, or lack an integer accuracy level between 1 and 16. Zero latitude
or longitude is valid with affirmative geotag accuracy evidence.

The original ranked-slice runner was recovered from commit `9dfa9a8`, at the
user's explicit request. Its 19 distinct term/search-field combinations and
annual/six-year partitions are retained. All 19 combinations have hits in the
previous snapshot. Positive combinations are assigned priority over unmatched
combinations without removing any keywords. The new run extends the final
upload interval through 11 November 2026.

The runner requires a separate geotagged state database, shares the rolling-hour
API-call ledger with the existing metadata recovery worker, and records raw
responses and keyword associations. Its final CSV and Parquet exports are
validated again. A result window cap marks a run incomplete rather than silently
claiming complete retrieval.

HTTPX returned HTTP 502 in the live environment; an equivalent rich search via
urllib returned HTTP 200 with 37 validated geotagged results. The restored
runner therefore defaults to urllib and the api.flickr.com endpoint. The
general poller retains its HTTPX default and supports the alternative transport.
Transport failures are redacted to avoid logging credential-bearing URLs.

## Validation and execution

- Full regression before final transport additions: **2,620 passed**.
- Final focused planner, validation, runner, poller, and report checks:
  **98 passed**.
- `git diff --check`: passed.
- Live run: `papilio_demoleus_geotagged_ranked_slices_20261111`.
- Initial queue: 247 date-sliced page-one requests; pagination adds further work.
- The first live audit confirmed 325 unique records with validated geotags.
- Live manifests, raw responses, state, logs, and code/input checksums stay under
  the ignored run directory; no credentials, source images, or runtime data are
  included in this commit.

GitHits was queried for a Flickr call example but returned no supported example.
No external code was copied. The controlling API contract is Flickr's official
[photos.search documentation](https://www.flickr.com/services/api/flickr.photos.search.html).
The local GitHits record is appended to `provenance/githits.jsonl`.

The run is discovery evidence, not taxonomic verification or an occurrence
release. The original dataset, reviews, and metadata recovery job are preserved.
