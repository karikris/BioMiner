# AGENTS.md

## Project

Create a Python 3.14 Flickr API pipeline that builds a Darwin Core-compatible biodiversity occurrence database from public Flickr photos, with BioCLIP 2 or the newest available BioCLIP-family model used for image classification and visual verification.

the project is python 3.14 - take advantage of parallel processing - set number of works as 16 or best number for my ultra 7 265k processor, optimise worker counts across various workflow steps. use polars dataframes, duckdb, and parquet for memory efficiency and speed. Add a constant (boolean: True/False) for outputting a csv file but make default as False - only output parquet.

commit and push all changes - never commit API keys or secrets.

follow a four-step process: Plan -> Execute -> Test -> Commit -> Plan etc..

Plan: Think through the approach before writing any code. Discuss the strategy and get alignment on what you're building.
Execute: write the code that matches plan.
Test: Run unit tests, check type safety, or perform manual QA. Validate that the implementation matches what was planned.
Commit: commit the code and start the cycle again for the next piece.

## Plan Mode

- Make the plan extremely concise. Sacrifice grammar for the sake of concision.
- At the end of each plan, give me a list of unresolved questions to answer, if any.

Initial test species:

```
Papilio demoleus
```

Target workflow:

```
Flickr API
  → collect public photo metadata
  → extract species and location clues
  → classify image with BioCLIP 2 / newest BioCLIP
  → compare image result with Flickr text, comments, geotag and range context
  → send uncertain records to review
  → export Darwin Core-compatible occurrence records
```

Important scientific principle:

```
The pipeline must expand biodiversity occurrence coverage.
Do not reject records merely because they fall outside known species ranges.
Treat unusual locations as discovery candidates, range-extension candidates, or analysis-stage outliers.
Outlier removal belongs in later analysis, not ingestion.
```

---

## Non-negotiable rules

1. Use the official Flickr API only.
2. Do not scrape Flickr pages.
3. Do not use browser automation.
4. Do not access private, restricted, or non-public photos.
5. Do not bypass Flickr API limits.
6. Treat `3600` as the absolute hourly safety cap for API calls.
7. Use a stricter soft cap of `3000` API calls/hour unless explicitly changed.
8. Also cap newly processed Flickr photo records to `3600` records/hour.
9. If API-call limit and record-limit conflict, use the stricter limit.
10. Store raw Flickr metadata unchanged before cleaning.
11. Store exact coordinates internally when available.
12. Do not generalise exact coordinates during ingestion.
13. Add publication-safety fields so exact coordinates can be modified later.
14. Flag any geolocation derived from an area larger than `100 km²`.
15. Do not reject occurrence candidates solely because the location is outside known range.
16. Never let BioCLIP 2 silently override Flickr metadata, comments, or human review.
17. Every accepted record must carry evidence, confidence, source, and review status.

---

## Required technology stack

Use:

```
Python >= 3.14
Polars
DuckDB
PyArrow
Parquet
httpx or aiohttp
Pydantic
pytest
BioCLIP 2 or newest BioCLIP-family model
OpenCLIP-compatible model wrapper where applicable
```

Use:

```
async I/O or ThreadPoolExecutor for Flickr API calls
ProcessPoolExecutor or GPU batch worker for image classification
Polars LazyFrames for transformations
DuckDB for local SQL analytics over Parquet
Parquet for durable pipeline outputs
```

Do not use pandas unless required at a dependency boundary.

---

## Repository structure

Create this structure:

```
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .env.example
├── config/
│   ├── pipeline.toml
│   ├── species_seed.csv
│   ├── regions.csv
│   ├── model_registry.toml
│   └── dwc_schema.yml
├── src/
│   └── flickr_bio_occurrence/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── flickr/
│       │   ├── client.py
│       │   ├── rate_limiter.py
│       │   ├── endpoints.py
│       │   └── work_items.py
│       ├── taxonomy/
│       │   ├── species_mapper.py
│       │   ├── name_cleaner.py
│       │   └── range_context.py
│       ├── geo/
│       │   ├── georeference.py
│       │   ├── area_flags.py
│       │   └── uncertainty.py
│       ├── vision/
│       │   ├── bioclip.py
│       │   ├── model_registry.py
│       │   ├── image_cache.py
│       │   ├── embeddings.py
│       │   ├── object_detection.py
│       │   └── future_classifier.py
│       ├── review/
│       │   ├── rules.py
│       │   └── queue.py
│       ├── dwc/
│       │   ├── mapper.py
│       │   └── exporter.py
│       ├── storage/
│       │   ├── parquet_io.py
│       │   └── duckdb_index.py
│       └── utils/
│           ├── hashing.py
│           ├── logging.py
│           └── time.py
├── tests/
│   ├── test_rate_limiter.py
│   ├── test_work_items.py
│   ├── test_species_mapper.py
│   ├── test_georeference.py
│   ├── test_vision_model_registry.py
│   ├── test_dwc_mapper.py
│   └── fixtures/
└── data/
    ├── raw/
    ├── bronze/
    ├── silver/
    ├── gold/
    ├── review/
    └── cache/
```

---

## Configuration defaults

Create `config/pipeline.toml`:

```toml
[flickr]
api_key_env = "FLICKR_API_KEY"
base_url = "https://www.flickr.com/services/rest/"
soft_api_calls_per_hour = 3000
hard_api_calls_per_hour = 3600
hard_photo_records_per_hour = 3600
default_per_page = 250
max_retries = 5
timeout_seconds = 30

[search]
test_species = "Papilio demoleus"
media = "photos"
content_types = "0"
safe_search = 1
has_geo = 1
extras = "description,license,date_upload,date_taken,geo,tags,machine_tags,owner_name,url_m,url_l,url_o,o_dims,last_update,media,views"

[partitioning]
split_by_species = true
split_by_region = true
split_by_year = true
split_by_month = true

[geo]
large_area_threshold_km2 = 100
store_exact_internal_coordinates = true
generalise_on_public_export = false

[vision]
preferred_model_family = "bioclip"
preferred_model_version = "bioclip2_or_newest"
fallback_model_version = "bioclip1"
top_k = 10
batch_size = 16
store_embeddings = true
auto_accept_requires_text_vision_agreement = false
text_vision_conflict_routes_to_review = true
outside_known_range_never_rejects = true

[storage]
format = "parquet"
duckdb_path = "data/flickr_bio_occurrence.duckdb"
```

Create `config/model_registry.toml`:

```toml
[models.bioclip2]
display_name = "BioCLIP 2"
role = "preferred"
status = "use_if_available"
task = "biology image-text classification and embedding"
notes = "Use BioCLIP 2 or the newest BioCLIP-family checkpoint available from the official project/model release source. Pin the exact checkpoint, package version, and model hash once installed."

[models.bioclip1]
display_name = "BioCLIP"
role = "fallback"
status = "fallback_only"
task = "biology image-text classification and embedding"
notes = "Use only if BioCLIP 2 or newer BioCLIP-family model is unavailable."

[models.future_butterfly_classifier]
display_name = "Fine-tuned butterfly classifier"
role = "future"
status = "not_initial"
task = "specialist butterfly classification after enough reviewed records exist"
```

---

## Work-item partitioning

Split all API harvesting into small work items.

Each work item must include:

```
species_name
species_query_terms
region_id
region_name
bbox
year
month
min_taken_date
max_taken_date
page
query_variant
status
attempt_count
last_error
```

Example test work item:

```json
{
  "species_name": "Papilio demoleus",
  "species_query_terms": [
    "Papilio demoleus",
    "lime butterfly",
    "chequered swallowtail",
    "citrus swallowtail"
  ],
  "region_id": "AU_QLD",
  "region_name": "Queensland",
  "bbox": "137.99,-29.18,153.55,-9.14",
  "year": 2024,
  "month": 1,
  "min_taken_date": "2024-01-01",
  "max_taken_date": "2024-01-31",
  "page": 1,
  "query_variant": "scientific_name"
}
```

Work-item identity must be deterministic:

```
sha256(species_name + region_id + year + month + page + query_variant)
```

Partitioning exists to:

```
avoid large query result caps
improve restartability
prevent runaway API use
make the pipeline auditable
allow region/year/month-level retries
```

---

## Flickr API collection rules

Use `flickr.photos.search` for initial discovery.

Required search arguments:

```
method=flickr.photos.search
text=<species/common-name query>
bbox=<region bbox>
min_taken_date=<month start>
max_taken_date=<month end>
has_geo=1
media=photos
content_types=0
safe_search=1
extras=<configured extras>
per_page<=250 for geo/bbox queries
format=json
nojsoncallback=1
```

Use follow-up endpoints only after deduplication:

```
flickr.photos.getInfo
flickr.photos.getExif
flickr.photos.geo.getLocation
flickr.photos.comments.getList
```

Do not call enrichment endpoints repeatedly for the same `flickr_photo_id`.

Maintain an endpoint-level cache table:

```
endpoint
photo_id
request_hash
response_hash
fetched_at
status
error_code
```

---

## Rate-limiting design

Implement a persistent global limiter.

The limiter must protect both:

```
API calls per hour
new Flickr photo records processed per hour
```

Required behaviour:

```
soft_api_calls_per_hour = 3000
hard_api_calls_per_hour = 3600
hard_photo_records_per_hour = 3600
```

Implementation requirements:

1. Use a persistent DuckDB or SQLite-backed call ledger.
2. Before every API call, check calls made in the previous rolling 3600 seconds.
3. If `soft_api_calls_per_hour` is reached, stop scheduling new work until tokens become available.
4. If `hard_api_calls_per_hour` would be exceeded, fail closed.
5. Do not retry failed requests without acquiring another token.
6. Count retries against the API-call budget.
7. Count `photos.search` returned new photo IDs against the hourly photo-record budget.
8. If a page would exceed the remaining hourly record budget, reduce `per_page` or defer the page.
9. Parallel workers must share the same limiter.
10. Unit tests must prove the limiter cannot exceed the hard cap under parallel execution.

Pseudo-interface:

```
class FlickrRateLimiter:
    def acquire_api_token(self, endpoint: str, work_item_id: str) -> None:
        ...

    def reserve_photo_record_slots(self, requested: int) -> int:
        ...

    def log_call(self, endpoint: str, work_item_id: str, status: str) -> None:
        ...

    def log_photo_records(self, photo_ids: list[str], work_item_id: str) -> None:
        ...
```

---

## Data layers

Use a bronze/silver/gold pattern.

### Raw layer

Store unmodified JSON responses.

```
data/raw/flickr/photos_search/
data/raw/flickr/get_info/
data/raw/flickr/get_exif/
data/raw/flickr/get_location/
data/raw/flickr/comments/
```

### Bronze layer

Flatten API responses into Parquet tables.

```
bronze_flickr_photo.parquet
bronze_flickr_exif.parquet
bronze_flickr_comment.parquet
bronze_flickr_location.parquet
```

### Silver layer

Create cleaned candidate biodiversity records.

```
silver_occurrence_candidate.parquet
silver_species_evidence.parquet
silver_location_evidence.parquet
silver_vision_prediction.parquet
silver_range_context.parquet
silver_review_queue.parquet
```

### Gold layer

Export Darwin Core-compatible tables.

```
dwc_occurrence.parquet
dwc_occurrence.csv
dwc_multimedia.parquet
dwc_identification_evidence.parquet
```

---

## Species mapping protocol

Follow the transferable structure of Chowdhury et al. 2024:

```
source selection
  → keyword/species-photo search
  → data extraction
  → georeferencing
  → quality control
```

For Flickr, adapt this as:

```
species dictionary
  → Flickr text/tag/comment search
  → BioCLIP 2 visual verification
  → geolocation extraction
  → range-context annotation
  → review status
```

Required species dictionary fields:

```
accepted_scientific_name
canonical_name
genus
specific_epithet
vernacular_names
synonyms
search_terms
taxon_rank
gbif_taxon_key
ala_taxon_id
inat_taxon_id
known_regions
sensitive_species_flag
```

For `Papilio demoleus`, include at minimum:

```
Papilio demoleus
lime butterfly
chequered swallowtail
citrus swallowtail
swallowtail
```

Species evidence must be stored separately from the final identification.

Species evidence fields:

```
flickr_photo_id
source_field
raw_text
matched_name
matched_name_type
resolved_scientific_name
confidence
evidence_rank
created_at
```

Valid `source_field` values:

```
title
description
tag
machine_tag
comment
bioclip2
bioclip_newest
manual_review
```

Do not assign `scientificName` directly from one weak signal.

---

## Species acceptance rules

Auto-accept species only when:

```
text evidence contains accepted scientific name or strong synonym
AND BioCLIP 2 / newest BioCLIP top-k supports the same species, genus, or visually related taxon group
AND image is not flagged as captive, pinned, museum, artwork, or non-wild
AND species is not sensitive or conservation-critical
AND there is no unresolved text/comment/vision conflict
```

Do not auto-reject because:

```
the location is outside known range
the record is geographically unusual
the record may represent a range extension
the species has few previous records in that region
```

Send to review when:

```
BioCLIP 2 disagrees with Flickr text
BioCLIP 2 confidence is low
only weak common-name evidence exists
comments contain a correction
location is novel or outside known range context
species is rare, threatened, sensitive, or conservation-critical
photo contains multiple organisms
life stage is caterpillar, pupa, egg, or unclear
geolocation was inferred from an area larger than 100 km²
```

Store review status as:

```
accepted
machine_suggested
needs_review
rejected
genus_only
family_only
range_extension_candidate
```

Important:

```
range_extension_candidate is not a rejection.
It is a flag for later biodiversity analysis.
```

---

## Range-context protocol

Do not use range context as a hard filter.

Purpose of range context:

```
describe whether the occurrence is already known, under-recorded, novel, or unusual
help reviewers prioritise records
support later biodiversity-distribution analysis
avoid prematurely discarding outliers
```

Range-context fields:

```
range_context_status
range_context_source
range_context_notes
range_extension_candidate
known_range_match
known_range_distance_km
analysis_outlier_candidate
```

Valid `range_context_status` values:

```
inside_known_range
near_known_range
outside_known_range
unknown_range
under_recorded_region
range_extension_candidate
not_evaluated
```

Rules:

```
Do not reject records solely because range_context_status = outside_known_range.
Do not remove outliers during ingestion.
Do not overwrite exact coordinates based on range context.
Do not generalise coordinates during ingestion because of range context.
Use range context only as an annotation and review-priority signal.
```

---

## Geolocation protocol

Location evidence priority:

```
1. Flickr explicit geotag
2. Flickr geo.getLocation
3. EXIF GPS, if shared
4. title / description locality
5. tags / machine tags
6. comments
7. region / group / query context
```

Store exact coordinates internally when available:

```
exact_decimalLatitude_internal
exact_decimalLongitude_internal
exact_coordinate_source
```

Also produce working Darwin Core coordinates:

```
decimalLatitude
decimalLongitude
coordinateUncertaintyInMeters
verbatimLocality
georeferenceSources
georeferenceRemarks
```

Do not generalise exact coordinates at ingestion time.

Add publication-preparation fields:

```
publish_decimalLatitude
publish_decimalLongitude
publish_coordinateUncertaintyInMeters
publication_generalisation_required
publication_generalisation_reason
```

Initially leave publication coordinates equal to null unless explicitly exporting a public dataset.

---

## Area larger than 100 km² rule

Any geolocation derived from an area larger than `100 km²` must be flagged.

Add fields:

```
georef_area_km2
georef_area_over_100km2
georef_precision_class
georef_review_required
```

Precision classes:

```
exact_gps
street_or_site
park_or_reserve_under_100km2
area_over_100km2
city_or_region
state_or_country
unknown
```

Rule:

```
if georef_area_km2 > 100:
    georef_area_over_100km2 = True
    georef_review_required = True
    georef_precision_class = "area_over_100km2"
```

Examples:

```
Exact Flickr GPS point → not flagged
Small park under 100 km² → not flagged
Large national park over 100 km² → flagged
City-level location → flagged
State-level location → flagged
Country-level location → flagged
```

---

## Image classification protocol

### Initial version: BioCLIP 2 / newest BioCLIP

Use BioCLIP 2, or the newest available BioCLIP-family model, as the preferred visual verification layer.

Implementation requirements:

```
Use a model registry.
Resolve the newest BioCLIP-family model explicitly.
Pin exact model name, checkpoint, package version, and hash.
Save the model metadata with every prediction.
Fail clearly if no model can be loaded.
Use BioCLIP 1 only as a fallback if BioCLIP 2/newer cannot be installed.
```

For each image:

1. Download the allowed preview image URL.
2. Hash image content.
3. Cache image locally for research processing.
4. Run BioCLIP 2 / newest BioCLIP against candidate labels.
5. Store top-k predictions.
6. Compare predictions with text-derived species evidence.
7. Do not let the model silently replace Flickr text or comments.
8. Do not reject a record because the location is outside known range.

Prompt labels should include:

```
a photo of Papilio demoleus
a photo of lime butterfly
a photo of chequered swallowtail
a photo of citrus swallowtail
a photo of a swallowtail butterfly
a photo of a butterfly
a photo of a moth
a photo of a caterpillar
a photo of a pupa or chrysalis
a photo of a pinned museum specimen
a photo of artwork or illustration
```

Vision prediction fields:

```
flickr_photo_id
model_family
model_name
model_version
model_checkpoint
model_hash
image_hash
image_url_used
top1_label
top1_score
topk_json
species_agreement_status
vision_review_required
created_at
```

Agreement statuses:

```
exact_species_agreement
same_genus_agreement
same_family_agreement
text_vision_conflict
vision_only
text_only
non_butterfly
uncertain
```

---

## Future vision upgrades

Design interfaces now, but do not implement all future models immediately.

### Future stage 1: object detection

Add an object detector to locate:

```
adult butterfly
caterpillar
pupa
egg
moth
pinned specimen
multiple organisms
non-organism
```

Pipeline:

```
full image
  → detector
  → crop organism
  → BioCLIP 2 / newest BioCLIP
  → store crop metadata
```

Detection fields:

```
bbox_x
bbox_y
bbox_width
bbox_height
detected_life_stage
detector_model_name
detector_model_version
detector_score
```

### Future stage 2: embedding retrieval

Build a reference image library from expert-verified biodiversity records.

For every Flickr image:

```
image embedding
  → nearest verified reference images
  → candidate taxa
  → reviewer-facing explanation
```

Store:

```
embedding_model_name
embedding_model_version
nearest_reference_ids
nearest_reference_taxa
nearest_reference_scores
```

### Future stage 3: fine-tuned butterfly classifier

Train only after enough reviewed records exist.

Training data sources may include:

```
expert-reviewed Flickr records
iNaturalist research-grade images where allowed
ALA / GBIF-linked media where allowed
museum or reference images where allowed
```

Fine-tuned classifier must support:

```
top-k predictions
genus fallback
family fallback
open-set rejection
rare-species review forcing
model-version tracking
confidence calibration
```

The fine-tuned classifier must not replace the evidence model. It adds another evidence source.

---

## Darwin Core mapping

Create Darwin Core-compatible occurrence records.

Minimum required output fields:

```
occurrenceID
basisOfRecord
eventDate
scientificName
verbatimIdentification
identificationVerificationStatus
decimalLatitude
decimalLongitude
coordinateUncertaintyInMeters
verbatimLocality
georeferenceSources
georeferenceRemarks
associatedMedia
associatedReferences
license
rightsHolder
dataGeneralizations
informationWithheld
occurrenceRemarks
dynamicProperties
```

Use:

```
basisOfRecord = "HumanObservation"
```

when the record is supported by Flickr user text, tags, comments, or other human-provided evidence.

Use:

```
basisOfRecord = "MachineObservation"
```

only when the record is based entirely on machine classification without human-provided species evidence.

`occurrenceID` must be deterministic:

```
sha256("flickr" + flickr_photo_id + resolved_scientific_name + eventDate + decimalLatitude + decimalLongitude)
```

`associatedReferences` should store the Flickr photo page URL.

`associatedMedia` should store the Flickr image URL only if licence and use rules allow it.

`dynamicProperties` should include model and pipeline evidence as JSON.

---

## Privacy and publication safety

Ingest exact coordinates when available, but do not assume they are safe to publish.

Flag records for publication review when:

```
species is sensitive
location is exact GPS
location appears to be private property
location is breeding/host-plant related
record is for threatened or range-restricted species
comments mention nest, colony, breeding site, host plant, or private garden
```

Fields:

```
sensitive_species_flag
private_property_possible
exact_location_publication_risk
publication_review_required
informationWithheld
dataGeneralizations
```

Do not remove exact coordinates from internal research tables unless explicitly instructed.

Do not generalise coordinates before internal analysis.

---

## Parallel processing rules

Use parallelism carefully.

Recommended split:

```
API fetching: async I/O or ThreadPoolExecutor
JSON flattening: Polars lazy transformations
BioCLIP 2 classification: GPU batch worker or ProcessPoolExecutor
DuckDB indexing: single writer where practical
```

Do not allow each worker to maintain its own independent rate limit.

All workers must share the persistent limiter.

Safe parallel pattern:

```
scheduler
  → creates work items
  → workers request permission from shared limiter
  → workers fetch API page
  → workers write raw response
  → workers enqueue enrichment only after deduplication
```

Avoid:

```
unbounded gather()
unbounded ThreadPoolExecutor
per-worker rate limiters
retry loops without limiter calls
multiple writers corrupting the same file
```

---

## Deduplication

Deduplicate early and often.

Primary key:

```
flickr_photo_id
```

Secondary duplicate checks:

```
image_hash
owner_id_hash + date_taken + coordinates + species_candidate
perceptual_hash in future vision stage
```

A single photo may create multiple candidate records only if:

```
multiple species are visible
or comments/text explicitly indicate multiple taxa
or object detection identifies multiple organisms
```

Otherwise, prefer one occurrence candidate per photo.

---

## DuckDB index

Create a DuckDB database over Parquet files.

Required views:

```sql
CREATE VIEW raw_photos AS
SELECT * FROM read_parquet('data/bronze/bronze_flickr_photo/**/*.parquet');

CREATE VIEW occurrence_candidates AS
SELECT * FROM read_parquet('data/silver/silver_occurrence_candidate/**/*.parquet');

CREATE VIEW dwc_occurrence AS
SELECT * FROM read_parquet('data/gold/dwc_occurrence/**/*.parquet');
```

Useful QA queries:

```sql
-- records by species
SELECT scientificName, count(*)
FROM dwc_occurrence
GROUP BY scientificName
ORDER BY count(*) DESC;

-- large-area georeferences
SELECT *
FROM occurrence_candidates
WHERE georef_area_over_100km2 = true;

-- records needing review
SELECT *
FROM occurrence_candidates
WHERE review_status = 'needs_review';

-- BioCLIP/text conflicts
SELECT *
FROM occurrence_candidates
WHERE species_agreement_status = 'text_vision_conflict';

-- possible range extensions, not rejected records
SELECT *
FROM occurrence_candidates
WHERE range_extension_candidate = true;
```

---

## CLI requirements

Create a CLI with these commands:

```bash
flickr-bio plan-work-items --species "Papilio demoleus"
flickr-bio fetch --species "Papilio demoleus" --region AU_QLD --year 2024 --month 1
flickr-bio enrich --species "Papilio demoleus"
flickr-bio classify --species "Papilio demoleus" --model bioclip2
flickr-bio classify --species "Papilio demoleus" --model newest-bioclip
flickr-bio build-candidates --species "Papilio demoleus"
flickr-bio export-dwc --species "Papilio demoleus"
flickr-bio qa-rate-limit
flickr-bio qa-summary
```

Add dry-run support:

```bash
flickr-bio fetch --species "Papilio demoleus" --dry-run
```

Dry-run must show:

```
planned API calls
planned maximum photo records
hourly limit status
work item count
output paths
selected BioCLIP model
```

---

## Testing requirements

Create tests before broad harvesting.

Required tests:

```
test_rate_limiter_never_exceeds_3600_calls_per_hour
test_rate_limiter_never_exceeds_3600_photo_records_per_hour
test_parallel_workers_share_global_limiter
test_work_item_ids_are_deterministic
test_partitioning_by_species_region_year_month
test_papilio_demoleus_seed_terms_present
test_geo_area_over_100km2_is_flagged
test_exact_flickr_coordinates_are_stored_internal
test_dwc_required_fields_present
test_bioclip2_or_newest_model_is_preferred
test_bioclip_conflict_routes_to_review
test_outside_known_range_never_auto_rejects
test_range_extension_candidate_is_annotation_not_rejection
test_no_private_or_scraping_endpoints_exist
```

Mock all Flickr API calls in tests.

Do not run live API tests by default.

---

## Acceptance criteria

The first acceptable version must:

1. Create work items split by species, region, year, and month.
2. Run a dry-run for `Papilio demoleus`.
3. Fetch public Flickr metadata through the API only.
4. Stay below 3000 API calls/hour by default.
5. Prove it cannot exceed 3600 API calls/hour.
6. Prove it cannot process more than 3600 new photo records/hour.
7. Save raw API responses.
8. Save bronze flattened Parquet.
9. Save silver occurrence candidates.
10. Run BioCLIP 2 or newest available BioCLIP-family top-k classification.
11. Compare BioCLIP output against Flickr title, description, tags, comments, geotag, and range context.
12. Send uncertain records to review.
13. Flag possible range-extension candidates without rejecting them.
14. Flag geolocations derived from areas larger than 100 km².
15. Store exact coordinates internally when available.
16. Export Darwin Core-compatible occurrence records.
17. Provide DuckDB QA views.
18. Include tests for rate limiting, partitioning, georeferencing, species mapping, BioCLIP model selection, range-extension annotation, and Darwin Core export.

---

## Development style for Codex

When modifying this repository:

1. Read this AGENTS.md before editing.
2. Prefer small, testable changes.
3. Keep pipeline stages separate.
4. Do not hide API calls in helper functions that bypass the rate limiter.
5. Do not add scraping dependencies.
6. Do not add browser automation.
7. Do not add private-data collection.
8. Do not hard-code API keys.
9. Do not commit downloaded images or raw secrets.
10. Use clear function names.
11. Add tests for every new stage.
12. Run formatting and tests before final output.

Required checks:

```bash
python -m pytest
python -m compileall src
```

If a dependency is missing, update `pyproject.toml` and document why.

---

## Final pipeline principle

This project must create biodiversity occurrence candidates, not pretend that every Flickr photo is a verified occurrence.

The database should preserve:

```
what Flickr said
what the image model predicted
where the location came from
how precise the location is
whether the location is novel or under-recorded
whether a human needs to review it
what can be safely exported later
```

The goal is not to delete outliers during ingestion.

The goal is to collect defensible, evidence-rich occurrence candidates so later biodiversity analysis can decide which records are reliable, novel, uncertain, or unsuitable for publication.
