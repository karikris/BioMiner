# BioMiner

BioMiner is a lean Flickr Lepidoptera image-triage pipeline. It discovers Flickr photo metadata, filters obvious non-biodiversity records, classifies candidate images with BioCLIP 2.5, assigns evidence buckets, and uses targeted Flickr comment review to promote ambiguous Bronze records when comments provide matching species evidence.

BioCLIP output is screening evidence only. BioMiner does not claim taxonomic validation, does not publish verified Darwin Core occurrences, and does not keep a permanent Flickr image archive.

## Active Workflow

```text
Flickr photos.search metadata + image URLs
-> metadata/evidence parquet
-> anti-keyword and metadata filtering
-> temporary image download
-> BioCLIP 2.5 species + triage scoring
-> occurrence_bin, image_category, life_stage
-> delete downloaded image
-> targeted Bronze comment review
-> Gold/Silver/Bronze/Bin outputs + compact reports
```

The active package is `biominer` under `src/`. The CLI entry point is `biominer`.

## Repository Status

This repository is focused on Flickr Lepidoptera triage. The current active path is:

- bounded Flickr metadata polling;
- deterministic Flickr query planning with fixed upload-date slices;
- bounded broad-search coverage from 2004-02-10 through today;
- anti-keyword filtering of obvious non-biodiversity material;
- temporary image caching and cleanup after classification;
- BioCLIP 2.5 register-based batch classification helpers;
- rule-based occurrence bins and image category fields;
- targeted comment review for Bronze or ambiguous records;
- compact JSON/parquet reports and QA summaries.

BioCLIP/OpenCLIP/PyTorch runtime work may live in `karikris/BioCLIPMiner` or be imported through the BioCLIP runner. This repo keeps local unit tests free of network, Flickr credentials, CUDA, BioCLIP weights, and real downloaded images.

## Install

Python 3.14 or newer is required.

```bash
uv venv
uv pip install -e '.[test]'
```

Set Flickr credentials only for commands that call Flickr:

```bash
cp .env.example .env
# edit .env and set FLICKR_API_KEY
```

Generated data, local operator inputs, caches, parquet files, DuckDB files, virtual environments, model weights, raw API payloads, and image artifacts must not be committed.

## Main Commands

Show the command surface:

```bash
biominer --help
```

Build count-probe work items from a reviewed keyword JSON:

```bash
biominer build-papilio-demoleus-query-plan   --keywords-json config/papilio_demoleus_multilingual_keywords.json   --state-db data/state/flickr_poller.sqlite
```

Run one bounded metadata polling cycle:

```bash
biominer poll-once   --max-api-calls 3500   --workers 1   --state-db data/state/flickr_poller.sqlite   --raw-root data/raw   --evidence-output staging/evidence/poll_once_evidence.parquet
```

Apply evidence rules to a parquet evidence file:

```bash
biominer apply-rules   --evidence staging/evidence/poll_once_evidence.parquet   --output staging/evidence/classified.parquet
```

Drop obvious non-biodiversity records before downstream review:

```bash
biominer filter   --input staging/evidence/classified.parquet   --anti-keywords-json config/anti_keywords.json   --output staging/evidence/filtered.parquet   --dropped-output staging/evidence/dropped.parquet
```

Build and process the targeted comment-review queue:

```bash
biominer build-comment-review-queue   --input staging/evidence/classified.parquet   --state-db data/state/comment_review.sqlite

biominer review-comments-once   --state-db data/state/comment_review.sqlite   --max-api-calls 300

biominer apply-comment-review-decisions   --input staging/evidence/classified.parquet   --output staging/evidence/classified_with_comments.parquet   --state-db data/state/comment_review.sqlite
```

Inspect local API budget state or summarize an existing report:

```bash
biominer qa-rate-limit
biominer qa-summary --report reports/some_run_summary.json
```

Compact parquet shards or export bucket-specific parquet views:

```bash
biominer compact-parquet   --input-root staging/evidence/shards   --output staging/evidence/compacted.parquet

biominer export-bucket-views   --input staging/evidence/classified_with_comments.parquet   --output-dir reports/bucket_views
```

Clean a temporary image cache explicitly when needed:

```bash
biominer gc-cache --cache-root data/cache/images --delete
```

## Flickr Limits And BioMiner Policy

BioMiner separates three different concepts that should not be confused.

### 1. Flickr official result window

Flickr `flickr.photos.search` documents that it returns at most the first **4,000 results** for any given search query. This is a search-window constraint, not an hourly API-call quota.

### 2. BioMiner stable leaf-query threshold

BioMiner uses:

```text
STABLE_RESULT_THRESHOLD = 4000
```

This matches Flickr's documented 4,000-result accessible search window.

If a count probe is used for a narrow query and reports `total <= 4000`, BioMiner may enqueue normal page fetches for that exact query.

For broad butterfly discovery, BioMiner now avoids recursive count-probe expansion and seeds fixed upload-date slices directly.

### 3. BioMiner hourly API-call budget

BioMiner uses an operational API budget:

```text
SOFT_API_CALLS_PER_HOUR = 3500
HARD_API_CALLS_PER_HOUR = 3600
```

The hourly budget controls how many API calls the poller may make in a bounded run. It is separate from the 4,000 result threshold.

If the budget runs out, `poll-once` stops cleanly, leaves remaining work pending, and the next run resumes in deterministic database order.

## Flickr Discovery Rules

BioMiner uses Flickr `photos.search` for metadata discovery. The default request should use:

```text
media=photos
safe_search=1
content_types=0
extras=description,license,date_upload,date_taken,owner_name,last_update,geo,tags,machine_tags,o_dims,views,media,url_l,url_m
```

Image URL preference is:

```text
url_l -> url_m
```

`url_o` is not part of the default active path. Original images are diagnostic only.

Page-size policy:

```text
count probes: per_page=1
normal pages: per_page=500
bbox/geotagged pages: per_page=250
```

Flickr's `per_page` maximum for normal searches is 500, while geo/bbox queries return only 250 results per page.

## Stable Search-Space Coverage

BioMiner covers broad Flickr searches with deterministic upload-date slices, not blind deep paging.

Core invariant:

```text
Never page beyond Flickr's 4,000-result accessible window for any single query slice.
```

Planning logic:

```text
start at 2004-02-10 and advance to today
use 10-day upload-date slices through 2015-12-31
use 5-day upload-date slices from 2016-01-01 through today
enqueue pages 1..8 for each slice at per_page=500
```

If page 8 returns 500 records, report that upload-date slice as saturated at Flickr's result window. The slice is retained for downstream review, and future planning can split only those saturated windows more narrowly if needed.

Work-item ordering should be stable across reruns:

```text
split_depth
split_priority
date range start
date range end
slice index
bbox or region order
term
page number
query hash
```

All work items must have stable IDs derived from canonical query JSON. The database should use idempotent insertion, so reruns do not duplicate already planned work.

## Example: Querying `text=butterfly`

For broad `butterfly` discovery, the active runner seeds fixed upload-date slices instead of recursively probing the full query.

```text
method=flickr.photos.search
text=butterfly
media=photos
safe_search=1
content_types=0
min_upload_date=2004-02-10
max_upload_date=2004-02-19
per_page=500
page=1..8
```

The next slice is `2004-02-20..2004-02-29`, also pages 1..8. From 2016 onward, slices are 5 days wide. With the 3,500-call hourly budget, a bounded run stops after the current poll-once claim set and resumes remaining pending slices in deterministic SQLite order.

Recommended output layout:

```text
data/raw/flickr/photos_search/text/butterfly/
  normal_page-00001-<work_id>.json
  normal_page-00002-<work_id>.json
  ...
  normal_page-00007-<work_id>.json

staging/evidence/text_butterfly_metadata.parquet
reports/text_butterfly_fetch_manifest.json
reports/text_butterfly_fetch_profile.json
```

If a slice's eighth page is full, BioMiner reports that slice as saturated at Flickr's accessible window.

## Step 1: Metadata Fetch

Inputs:

```text
operator keyword JSON
Flickr API key
SQLite state database
```

Outputs:

```text
raw Flickr JSON
metadata/evidence parquet
API call ledger
work-item state
fetch reports
```

Rules:

- fetch metadata only;
- never download images in Step 1;
- reserve one API-call token before each request;
- stop at the hourly budget;
- requeue stale claimed work;
- resume pending work deterministically;
- dedupe source records by Flickr photo ID and image URL;
- write compact run metrics.

Required Step 1 metrics:

```text
api_calls_used
api_calls_remaining_soft
api_calls_remaining_hard
calls_per_hour
count_probes_completed
page_fetches_completed
split_probes_enqueued_by_reason
pending_count_probes
pending_page_fetches
records_fetched
records_per_call
duplicate_records_skipped
raw_response_bytes
parquet_rows
parquet_bytes
total_seconds
average_seconds_per_call
p50_seconds_per_call
p95_seconds_per_call
max_rss_kb
peak_traced_bytes
budget_limited_exit
```

Unsupported metrics should be written as `null` or `"not_instrumented"`, never guessed.

## Step 2: Metadata Filter

Inputs:

```text
Step 1 evidence parquet
operator anti-keyword JSON
```

Outputs:

```text
filtered candidates parquet
dropped records parquet
filter report
```

Drop obvious non-biodiversity material:

```text
artwork
tattoo
AI/generated
logo/brand
object/product
textile/pattern
museum/pinned specimen
other insect
not Lepidoptera
```

Keep butterfly life stages:

```text
adult
egg
caterpillar
larva
pupa
chrysalis
```

Step 2 must not make final species decisions. It only removes obvious non-biodiversity or hard-negative metadata records.

Required Step 2 metrics:

```text
input_rows
kept_rows
dropped_rows
drop_reasons
image_category_counts
life_stage_counts
null_image_url_count
null_date_count
null_geo_count
total_seconds
rows_per_second
max_rss_kb
peak_traced_bytes
```

## Step 3: BioCLIP 2.5 Classification

BioCLIP classification uses temporary image download and register-based processing.

Rules:

- use the register runner;
- keep one persistent model worker for the run;
- default `register_count=4`;
- default `register_size=20`;
- download images temporarily;
- classify;
- write prediction rows;
- delete staged image files;
- skip successful records on rerun using source/photo/image/model/checkpoint keys;
- use fake classifiers in tests.

Successful records are skipped on rerun for the same combination:

```text
source
flickr_photo_id
image_url
model_id
model_version
model_checkpoint
```

Required Step 3 metrics:

```text
records_seen
records_classified
records_skipped_existing
download_failures
bioclip_failures
images_downloaded
images_deleted_after_classification
cache_bytes_before
cache_bytes_after
max_staged_images
model_id
model_version
model_checkpoint
register_count
register_size
total_seconds
images_per_second
average_seconds_per_image
bucket_counts
score_distribution
max_rss_kb
peak_traced_bytes
gpu_memory_peak_mb
```

## Triage Rules

Occurrence bins:

```text
gold
silver
bronze
bin
in_review
```

Gold:

```text
adult butterfly
BioCLIP species score > 0.70
matching species evidence in Flickr title/tags/description/machine tags
image URL present
event date present
latitude and longitude present
image_category = adult_butterfly
no hard-negative category
```

Silver:

```text
BioCLIP species score 0.35 through 0.70
matching species evidence in Flickr metadata
image URL present
no hard-negative category
```

Also keep otherwise Gold-strength records in Silver if they are missing event date or geolocation.

Bronze:

```text
remaining butterfly records
adult butterflies without enough species agreement
egg/caterpillar/larva/pupa/chrysalis records
records requiring comment review or human review
```

Bin:

```text
records with no butterfly in any life stage
hard-negative visual/material categories
failed or irrelevant non-biodiversity records
```

Operational failures such as missing image URLs, missing BioCLIP output, failed downloads, and runtime failures should stay in review/error handling paths and remain retryable where appropriate.

## Category Model

BioMiner uses one shared image category column:

```text
image_category
```

Allowed initial values:

```text
adult_butterfly
life_stage_non_adult
museum_specimen
artwork
tattoo
ai_generated
logo_or_brand
object_or_product
textile_or_pattern
other_insect
not_lepidoptera
unknown
```

Allowed life stages:

```text
adult_butterfly
egg
caterpillar
larva
pupa
chrysalis
unknown
```

Default values are:

```text
image_category = adult_butterfly
life_stage = adult_butterfly
```

Non-adult life stages use:

```text
image_category = life_stage_non_adult
life_stage = egg | caterpillar | larva | pupa | chrysalis
```

## Step 4: Comment Review

Comments are a targeted review phase. BioMiner does not fetch comments for every record by default.

Default queue:

```text
Bronze records only
```

Records may also be queued when they are ambiguous:

```text
BioCLIP versus Flickr text mismatch
suspected species conflict
missing date
missing geo
unknown category
unknown life stage
low confidence
other incomplete evidence
```

Comment review may:

- confirm a BioCLIP species;
- reveal a conflicting species;
- provide date evidence;
- provide location clues;
- support promotion from Bronze to Gold or Silver.

Comment review must not:

- override hard-negative image categories;
- replace BioCLIP evidence;
- turn free-text place names into coordinates without safe structured resolution;
- force Gold while event date or geolocation is still missing.

Promotion rules:

```text
Bronze -> Gold:
  comments match BioCLIP species or accepted synonym
  Gold metadata/adult rules are also satisfied
  no hard-negative category

Bronze -> Silver:
  comments match BioCLIP species or accepted synonym
  species evidence is present
  Gold metadata/adult rules are incomplete

Remain Bronze:
  no comment match
  generic comments only
  species conflict
  non-adult life stage
  incomplete evidence
```

Required Step 4 metrics:

```text
queued_records
api_calls_used
comments_fetched
records_with_comments
species_matches
species_conflicts
gold_promotions
silver_promotions
retained_bronze
errors
total_seconds
average_seconds_per_call
max_rss_kb
peak_traced_bytes
```

## Reports

Reports should stay compact and machine-readable.

Expected report paths include:

```text
reports/query_term_totals.json
reports/flickr_split_progress.json
reports/api_budget_profile.json
reports/fetch_profile.json
reports/filter_profile.json
reports/bioclip_profile.json
reports/occurrence_bin_profile.json
reports/life_stage_profile.json
reports/no_geo_profile.json
reports/comment_review_profile.json
reports/cache_profile.json
reports/idempotency_profile.json
reports/code_cleanup_report.md
reports/agents_update_recommendations.json
```

Each run report should include:

```text
command
git_sha
run_id
pid if background
started_at
ended_at
status
environment summary without secrets
per-step timings
total_seconds
API budget profile
throughput profile
row counts
bucket/category/life-stage distributions
storage bytes by artifact class
memory RSS/peak
GPU memory if available
failure counts
```

Unsupported metrics should be written as `null` or `"not_instrumented"`, never guessed.

## Long Runs

For API fetches or BioCLIP processing that may run for minutes:

- start the run as a detached local process;
- redirect logs to `logs/`;
- write PID and manifest JSON to `reports/`;
- include command, git SHA, expected outputs, start time, and status in the manifest;
- end active agent work after the run starts;
- do not tail logs continuously;
- do not repeatedly poll progress unless explicitly asked.

## Tests

Run the local test suite:

```bash
pytest -q
```

Focused tests should cover:

```text
CLI surface
Flickr endpoint constraints
API-budget enforcement
fixed upload-date slicing
stable leaf-query threshold
deterministic resume order
metadata filter rules
category/life-stage rules
BioCLIP worker behavior with fakes
temporary image deletion
idempotency
comment-review queueing
comment-derived promotions
```

Tests must remain local and small. Do not add tests that require:

```text
network
Flickr credentials
CUDA
real BioCLIP weights
real downloaded images
large parquet artifacts
large DuckDB artifacts
model caches
```

## Out Of Scope

The current BioMiner scope deliberately excludes:

- Darwin Core occurrence publication as the active path;
- taxonomic validation claims;
- global comment fetching;
- permanent Flickr image archival;
- multi-key Flickr quota multiplication;
- network/CUDA/model-weight requirements in unit tests;
- blind deep paging of broad Flickr searches.
