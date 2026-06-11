# BioMiner

BioMiner is a lean Flickr Lepidoptera image-triage pipeline. It discovers Flickr photo metadata, stages image URLs for temporary download, records BioCLIP 2.5 screening output, assigns occurrence bins and image categories, and keeps enough evidence for later human review from the original Flickr URLs.

BioCLIP output is screening evidence only. BioMiner does not claim taxonomic validation, does not publish verified Darwin Core occurrences, and does not keep a permanent Flickr image archive.

## Active Workflow

```text
Flickr photos.search metadata + image URLs
-> staging/evidence parquet
-> temporary image download
-> BioCLIP 2.5 species + triage label scoring
-> occurrence_bin, image_category, life_stage
-> delete downloaded image
-> persist URL, stable hashes, model output, status, reports
```

The active package is `biominer` under `src/`. The CLI entry point is `biominer`.

## Repository Status

This repository is intentionally narrower than earlier BioMiner drafts. The active path is Flickr Lepidoptera triage, with current code focused on:

- bounded Flickr metadata polling;
- Papilio demoleus query-plan construction;
- temporary image caching and immediate cleanup after classification;
- BioCLIP 2.5 register-based batch classification helpers;
- rule-based occurrence bins and image category fields;
- targeted comment review for ambiguous or incomplete records;
- compact JSON/parquet reports and QA summaries.

BioCLIP/OpenCLIP/PyTorch runtime work has been moved to `karikris/BioCLIPMiner`. This repo keeps local unit tests free of network, Flickr credentials, CUDA, BioCLIP weights, and real downloaded images.

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

Generated data, local operator inputs, caches, parquet files, DuckDB files, virtual environments, and model/image artifacts are ignored by git.

## Main Commands

Show the command surface:

```bash
biominer --help
```

Build Papilio demoleus count-probe work items from a reviewed keyword JSON:

```bash
biominer build-papilio-demoleus-query-plan \
  --keywords-json config/papilio_demoleus_multilingual_keywords.json \
  --state-db data/state/flickr_poller.sqlite
```

Run one bounded metadata polling cycle:

```bash
biominer poll-once \
  --max-api-calls 3500 \
  --state-db data/state/flickr_poller.sqlite \
  --raw-root data/raw \
  --evidence-output staging/evidence/poll_once_evidence.parquet
```

Apply the compact evidence rules to a parquet evidence file:

```bash
biominer apply-rules \
  --evidence staging/evidence/poll_once_evidence.parquet \
  --output staging/evidence/classified.parquet
```

Build and process the targeted comment-review queue:

```bash
biominer build-comment-review-queue \
  --input staging/evidence/classified.parquet \
  --state-db data/state/comment_review.sqlite

biominer review-comments-once \
  --state-db data/state/comment_review.sqlite \
  --max-api-calls 300

biominer apply-comment-review-decisions \
  --input staging/evidence/classified.parquet \
  --output staging/evidence/classified_with_comments.parquet \
  --state-db data/state/comment_review.sqlite
```

Inspect local API budget state or summarize an existing report:

```bash
biominer qa-rate-limit
biominer qa-summary --report reports/some_run_summary.json
```

Clean a temporary image cache explicitly when needed:

```bash
biominer gc-cache --cache-root data/cache/images --delete
```

## Flickr Discovery Rules

BioMiner uses Flickr `photos.search` for metadata discovery. The default search request uses `media=photos`, `safe_search=1`, and metadata extras for description, license, upload/taken dates, owner name, geolocation, tags, machine tags, dimensions, and `url_l`/`url_m`.

Image URL preference is:

```text
url_l -> url_m
```

`url_o` is not part of the default active path. Original images are diagnostic only.

Metadata polling is designed as a bounded one-shot command:

```text
check API budget -> claim work items -> fetch pages -> write staging/evidence -> queue image triage -> write reports -> exit
```

The active limits are a 3,500-call soft hourly target and a 3,600-call hard hourly stop. Normal pages use `per_page=500`; geotagged/bbox pages use `per_page=250`; count probes use `per_page=1`. Oversized result sets are split instead of paging blindly past Flickr's accessible search window.

## Triage Rules

Occurrence bins:

```text
gold
silver
bronze
bin
in_review
```

Gold is an adult butterfly occurrence candidate with BioCLIP species score greater than `0.70`, matching species evidence in Flickr title/tags/description/machine tags, an image URL, event date, latitude, longitude, `image_category = adult_butterfly`, and no hard-negative category.

Silver is a species-supported butterfly candidate with BioCLIP species score from `0.35` through `0.70`, or an otherwise Gold-strength species match that is missing date or geo metadata.

Bronze is retained butterfly material that is not an adult occurrence candidate, including adult butterflies without enough species agreement and non-adult life stages.

Bin is material with no butterfly in any life stage, including museum specimens, artwork, tattoos, generated images, logos, products, textile/pattern imagery, other insects, and non-Lepidoptera records. Eggs, caterpillars, larvae, pupae, and chrysalides are useful Lepidoptera records and stay Bronze in this workflow.

Operational failures such as missing image URLs, missing BioCLIP output, failed downloads, and runtime failures stay in review/error handling paths and remain retryable where appropriate.

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

Default values are `image_category = adult_butterfly` and `life_stage = adult_butterfly`. Non-adult life stages use `image_category = life_stage_non_adult` plus the specific `life_stage`.

## Image Handling And Idempotency

Downloaded Flickr images are temporary. The register runner downloads images into bounded staging registers, computes hashes, classifies with a persistent BioCLIP worker, writes classification rows, and deletes staged image files after classification.

Successful records are skipped on rerun for the same source/photo/image/model combination:

```text
source
flickr_photo_id
image_url
model_id
model_version
model_checkpoint
```

Failures record a status and error string. Eligible download and BioCLIP failures can be retried; completed successful records are not reprocessed.

## Comment Review

Comments are a separate targeted review phase. BioMiner does not fetch comments for every record by default.

Records are queued for comment review when there is a BioCLIP versus Flickr text mismatch, suspected species conflict, missing date, missing geo, unknown category, unknown life stage, low confidence, or otherwise ambiguous evidence.

Comments may resolve species conflicts, recover structured date evidence, recover structured location clues, trigger missing-data requests, or promote an otherwise eligible record to Gold. Comments must not override hard-negative image categories, replace BioCLIP evidence, turn free-text place names into coordinates without safe structured resolution, or force Gold while date or geo is still missing.

## Reports

Reports should stay compact and machine-readable. Active and expected report paths include:

```text
reports/query_term_totals.json
reports/bbox_coverage_profile.json
reports/occurrence_bin_profile.json
reports/life_stage_profile.json
reports/no_geo_profile.json
reports/comment_review_profile.json
reports/api_budget_profile.json
reports/cache_profile.json
reports/idempotency_profile.json
reports/code_cleanup_report.md
reports/agents_update_recommendations.json
```

Unsupported metrics should be written as `null` or `"not_instrumented"`, never guessed.

## Tests

Run the local test suite:

```bash
pytest -q
```

Focused tests cover the CLI surface, Flickr endpoint constraints, API-budget enforcement, query splitting, evidence/category rules, BioCLIP worker behavior with fakes, temporary image deletion, idempotency, comment-review queueing, and comments-derived missing-data requests.

Tests must remain local and small. Do not add tests that require the network, Flickr credentials, CUDA, real BioCLIP weights, real downloaded images, large parquet/DuckDB artifacts, or model caches.

## Out Of Scope

The current BioMiner scope deliberately excludes:

- Darwin Core occurrence publication as the active path;
- taxonomic validation claims;
- global comment fetching;
- permanent Flickr image archival;
- multi-key Flickr quota multiplication;
- network/CUDA/model-weight requirements in unit tests.
