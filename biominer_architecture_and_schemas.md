# BioMiner Future Architecture and Data Schemas

## 1. System architecture

```text
                        ┌──────────────────────────┐
                        │  Step 0 taxonomy/name    │
                        │  registry builder        │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ Flickr query compiler    │
                        │ atomic text/tag queries  │
                        └────────────┬─────────────┘
                                     │
                                     ▼
┌──────────────────────┐  ┌──────────────────────────┐
│ SQLite API ledger    │◄─┤ Flickr metadata poller    │
│ rate limits/resume   │  │ fixed date slices         │
└──────────────────────┘  └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │ Metadata evidence table  │
                           │ query-hit provenance     │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │ Anti-keyword filter      │
                           │ hard-negative screening  │
                           └────────────┬─────────────┘
                                        │
                                        ▼
┌──────────────────────┐  ┌──────────────────────────┐
│ GBIF occurrence feed │─►│ Geo species index         │
│ broad rounded cells  │  │ candidate priors         │
└──────────────────────┘  └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │ Candidate-set builder    │
                           │ metadata + geo + visual  │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │ Temporary image staging  │
                           │ content-addressed cache  │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │ BioCLIP 2.5 sidecar      │
                           │ persistent model worker  │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │ Funnel predictions       │
                           │ triage/family/genus/sp   │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │ Bucket rules             │
                           │ Gold/Silver/Bronze/Bin   │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │ Comment review queue     │
                           │ targeted only            │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │ Reports + QA + views     │
                           │ JSON/MD/Parquet/DuckDB   │
                           └──────────────────────────┘
```

## 2. Modules

### `biominer.registry`

Responsibilities:

- Build accepted Papilionoidea taxonomic spine.
- Compile family, genus, species, synonym, and vernacular name evidence.
- Manage multilingual terms and translation review status.
- Compile Flickr query definitions.
- Promote only QA-clean registry versions to `data/registry/current`.

### `biominer.flickr_fetch`

Responsibilities:

- Build deterministic query/date-slice work items.
- Poll Flickr metadata under API limits.
- Preserve query-hit provenance.
- Maintain SQLite state and rate-limit ledgers.

### `biominer.filter`

Responsibilities:

- Remove obvious non-biodiversity and hard-negative metadata records.
- Preserve plausible butterfly life-stage records.
- Write filtered and dropped evidence Parquet.

### `biominer.geo`

Responsibilities:

- Fetch or ingest GBIF occurrence reference records for accepted butterfly species.
- Round coordinates to deterministic global grid cells.
- Build geocell → species candidate indexes.
- Provide candidate fallback from fine to coarse grid levels.
- Track coordinate quality and occurrence support.

### `biominer.bioclip`

Responsibilities:

- Manage Python 3.12 sidecar runtime.
- Run one persistent BioCLIP 2.5 Huge worker per local run.
- Group records by candidate-set signature.
- Classify triage, family, genus, species top 20, and species top 5 rerun.
- Optionally emit normalized image embeddings.
- Delete temporary images after classification.

### `biominer.reports`

Responsibilities:

- Write JSON and Markdown reports.
- Export bucket views.
- Summarize QA, throughput, memory, and candidate coverage.

### `biominer.storage`

Responsibilities:

- Centralize Parquet write behavior.
- Maintain stable sorting and schema compatibility.
- Avoid CSV except at external boundaries.

## 3. Data zones

```text
data/registry/<version>/        durable taxonomic/name registry
data/geo/<version>/             durable GBIF-derived geo candidate register
data/embeddings/<run_id>/       optional model embeddings, no raw images
data/state/                     SQLite ledgers and work queues
data/cache/huggingface/         model cache, local only
data/cache/images/              temporary staged image cache, delete after classification
staging/evidence/               intermediate metadata/filter/classification outputs
reports/                        JSON, Markdown, bucket views, QA summaries
```

## 4. Core schemas

Types use logical names: `str`, `int`, `float`, `bool`, `date`, `datetime`, `json`, `list[str]`, `binary`, `fixed_size_list[float]`.

### 4.1 `taxa.parquet`

Primary taxonomic identity table.

| Column | Type | Notes |
|---|---:|---|
| `registry_version` | str | Registry version. |
| `taxon_key` | str | Stable source taxon key, usually GBIF accepted key. |
| `accepted_taxon_key` | str | Accepted key. |
| `parent_taxon_key` | str | Parent in accepted spine. |
| `scientific_name` | str | Accepted scientific name. |
| `canonical_name` | str | Canonical binomial/name. |
| `rank` | str | SUPERFAMILY/FAMILY/GENUS/SPECIES. |
| `taxonomic_status` | str | ACCEPTED/SYNONYM/etc. |
| `family` | str | Family name when applicable. |
| `genus` | str | Genus name when applicable. |
| `source` | str | gbif/col/inaturalist/etc. |
| `source_taxon_id` | str | Source-local identifier. |
| `lineage_json` | json | Full lineage snapshot. |
| `is_production_identity` | bool | True when accepted into production spine. |
| `created_at` | datetime | Build timestamp. |

### 4.2 `taxon_relations.parquet`

| Column | Type | Notes |
|---|---:|---|
| `registry_version` | str | Registry version. |
| `subject_taxon_key` | str | Child/source taxon. |
| `object_taxon_key` | str | Parent/accepted/related taxon. |
| `relation_type` | str | parent_of/synonym_of/member_of/etc. |
| `source` | str | Evidence source. |
| `confidence` | float | Optional normalized confidence. |
| `provenance_json` | json | Source payload summary. |

### 4.3 `names.parquet`

| Column | Type | Notes |
|---|---:|---|
| `registry_version` | str | Registry version. |
| `name_id` | str | Deterministic hash of taxon/name/source/language. |
| `taxon_key` | str | Accepted taxon key. |
| `name_string` | str | Scientific, synonym, common name, or translated term. |
| `normalized_name` | str | Casefolded, whitespace-normalized. |
| `language` | str | ISO code where known. |
| `script` | str | Optional script. |
| `name_type` | str | scientific/synonym/vernacular/translation/search_term. |
| `review_status` | str | accepted/reviewed/generated/unreviewed/rejected. |
| `source` | str | GBIF/CoL/iNat/manual/generated/etc. |
| `source_record_id` | str | Source identifier. |
| `is_query_eligible` | bool | True only for reviewed/safe terms. |

### 4.4 `name_evidence.parquet`

| Column | Type | Notes |
|---|---:|---|
| `registry_version` | str | Registry version. |
| `name_id` | str | Links to `names`. |
| `taxon_key` | str | Accepted taxon key. |
| `evidence_type` | str | source_match/manual_review/generated/corroborated. |
| `evidence_source` | str | Source label. |
| `evidence_url` | str | Optional URL. |
| `evidence_payload_json` | json | Compact source evidence. |
| `reviewer` | str | Optional reviewer. |
| `reviewed_at` | datetime | Optional review timestamp. |

### 4.5 `flickr_query_definitions.parquet`

| Column | Type | Notes |
|---|---:|---|
| `registry_version` | str | Registry version. |
| `query_id` | str | Deterministic hash. |
| `taxon_key` | str | Linked accepted taxon. |
| `name_id` | str | Source term. |
| `query_type` | str | tag/text/machine_tag. |
| `query_string` | str | Exact Flickr query term. |
| `language` | str | Term language. |
| `rank` | str | Taxon rank. |
| `family` | str | Family. |
| `genus` | str | Genus. |
| `review_status` | str | Term review status. |
| `is_enabled` | bool | Can be used for polling. |
| `disabled_reason` | str | If disabled. |

### 4.6 `flickr_photo_metadata.parquet`

| Column | Type | Notes |
|---|---:|---|
| `source` | str | `flickr`. |
| `flickr_photo_id` | str | Flickr ID. |
| `owner_nsid` | str | Flickr owner id. |
| `title` | str | Title. |
| `description` | str | Description text. |
| `tags` | list[str] | User tags. |
| `machine_tags` | list[str] | Machine tags. |
| `date_taken` | datetime | Flickr date taken if available. |
| `date_upload` | datetime | Upload timestamp. |
| `latitude` | float | Optional. |
| `longitude` | float | Optional. |
| `accuracy` | int | Flickr accuracy where provided. |
| `geo_source` | str | Flickr location source. |
| `image_url` | str | Temporary download candidate URL. |
| `license` | str | Flickr license id/text. |
| `media` | str | photo/video/etc. |
| `query_hit_count` | int | Number of query hits. |
| `query_hit_ids` | list[str] | Query definitions that found this record. |
| `query_hit_taxa_json` | json | Taxa represented in query hits. |
| `raw_metadata_path` | str | Optional local raw payload path if retained. |
| `created_at` | datetime | Poll timestamp. |

### 4.7 `metadata_filter_results.parquet`

| Column | Type | Notes |
|---|---:|---|
| `flickr_photo_id` | str | Flickr ID. |
| `filter_status` | str | kept/dropped/in_review. |
| `filter_reason` | str | Anti-keyword or rule reason. |
| `image_category_prior` | str | Metadata prior. |
| `life_stage_prior` | str | Metadata prior. |
| `matched_anti_keywords` | list[str] | Matched terms. |
| `retry_eligible` | bool | Operational retry flag. |

### 4.8 `gbif_occurrence_reference.parquet`

Reference occurrence records after GBIF ingestion and quality normalization.

| Column | Type | Notes |
|---|---:|---|
| `geo_version` | str | Geo register version. |
| `gbif_occurrence_key` | str | GBIF occurrence key. |
| `taxon_key` | str | GBIF taxon key. |
| `accepted_taxon_key` | str | Accepted species key. |
| `scientific_name` | str | Accepted scientific name. |
| `family` | str | Family. |
| `genus` | str | Genus. |
| `decimal_latitude` | float | Coordinate. |
| `decimal_longitude` | float | Coordinate. |
| `coordinate_uncertainty_m` | float | Optional. |
| `has_geospatial_issue` | bool | GBIF quality flag where available. |
| `basis_of_record` | str | OBSERVATION/PRESERVED_SPECIMEN/etc. |
| `country_code` | str | Optional. |
| `continent` | str | Optional. |
| `year` | int | Event year. |
| `dataset_key` | str | Source dataset. |
| `license` | str | GBIF/source license. |
| `issues` | list[str] | GBIF issue list. |
| `quality_weight` | float | BioMiner-derived weighting. |
| `included_in_geo_index` | bool | Whether used for candidate priors. |
| `excluded_reason` | str | If excluded. |

### 4.9 `geo_grid_cells.parquet`

| Column | Type | Notes |
|---|---:|---|
| `geo_version` | str | Geo version. |
| `grid_level` | str | G0_world/G1_realm/G2_20deg/etc. |
| `geocell_id` | str | Deterministic ID. |
| `lat_min` | float | Cell bound. |
| `lat_max` | float | Cell bound. |
| `lon_min` | float | Cell bound. |
| `lon_max` | float | Cell bound. |
| `cell_area_class` | str | coarse/medium/fine. |
| `parent_geocell_id` | str | Coarser cell. |
| `neighbour_geocell_ids` | list[str] | Adjacent cells. |

### 4.10 `geo_species_index.parquet`

| Column | Type | Notes |
|---|---:|---|
| `geo_version` | str | Geo version. |
| `grid_level` | str | Grid level. |
| `geocell_id` | str | Cell. |
| `accepted_taxon_key` | str | Accepted species key. |
| `scientific_name` | str | Species. |
| `family` | str | Family. |
| `genus` | str | Genus. |
| `occurrence_count` | int | Raw included occurrence count. |
| `record_count_weighted` | float | Quality-weighted count. |
| `dataset_count` | int | Distinct datasets. |
| `first_year` | int | Earliest record. |
| `last_year` | int | Latest record. |
| `basis_of_record_counts_json` | json | Count by basisOfRecord. |
| `coordinate_uncertainty_summary_json` | json | min/p50/p95/max. |
| `candidate_rank_prior` | float | Prior used for ordering. |
| `provenance_json` | json | Query/version/source summary. |

### 4.11 `geo_candidate_sets.parquet`

Per-photo candidate set generated from geo, metadata, and visual gates.

| Column | Type | Notes |
|---|---:|---|
| `run_id` | str | Candidate run id. |
| `flickr_photo_id` | str | Flickr ID. |
| `latitude_rounded` | float | Rounded input latitude. |
| `longitude_rounded` | float | Rounded input longitude. |
| `selected_geocell_id` | str | Main cell. |
| `selected_grid_level` | str | Main level. |
| `fallback_grid_level` | str | Fallback if used. |
| `neighbour_cells_used` | list[str] | Expanded cells. |
| `candidate_species_keys` | list[str] | Accepted species keys. |
| `candidate_species_count` | int | Count. |
| `candidate_sources_json` | json | geo/metadata/family/genus/query-hit. |
| `candidate_set_signature` | str | Stable hash of ordered candidate labels. |

### 4.12 `image_downloads.parquet`

| Column | Type | Notes |
|---|---:|---|
| `run_id` | str | Classification run. |
| `flickr_photo_id` | str | Flickr ID. |
| `image_url` | str | Used URL. |
| `image_hash` | str | SHA-256 hash. |
| `content_type` | str | Image content type. |
| `byte_size` | int | Image bytes. |
| `cache_path` | str | Temporary local path. |
| `download_status` | str | success/failed/skipped. |
| `download_error` | str | Error text. |
| `image_deleted_after_classification` | bool | Required true on success. |
| `retry_eligible` | bool | Operational retry. |

### 4.13 `classification_runs.parquet`

| Column | Type | Notes |
|---|---:|---|
| `run_id` | str | Run id. |
| `git_sha` | str | Source revision. |
| `model_name` | str | BioCLIP model name. |
| `model_checkpoint` | str | Checkpoint/revision. |
| `runtime_python` | str | Sidecar Python path. |
| `device_requested` | str | auto/mps/cuda/cpu. |
| `device_resolved` | str | Actual device. |
| `register_count` | int | Register count. |
| `register_size` | int | Batch size. |
| `download_workers` | int | Download workers. |
| `classification_mode` | str | triage/family/genus/species/hybrid/rescue. |
| `candidate_strategy` | str | all/metadata/geo/hierarchical/etc. |
| `started_at` | datetime | Start. |
| `ended_at` | datetime | End. |
| `status` | str | success/failed/partial. |
| `report_path` | str | JSON/MD report. |

### 4.14 `bioclip_predictions.parquet`

| Column | Type | Notes |
|---|---:|---|
| `run_id` | str | Run id. |
| `flickr_photo_id` | str | Flickr ID. |
| `image_hash` | str | Image hash. |
| `classification_status` | str | success/failed/skipped_existing. |
| `model_family` | str | bioclip. |
| `model_name` | str | Model. |
| `model_checkpoint` | str | Checkpoint. |
| `candidate_set_signature` | str | Candidate hash. |
| `triage_topk_json` | json | Triage results. |
| `triage_group_top` | str | Top group. |
| `family_topk_json` | json | Family top 3. |
| `genus_topk_json` | json | Genus top 8 per family summary. |
| `species_top20_json` | json | Species top 20. |
| `species_top5_rerun_json` | json | Strong rerun top 5. |
| `species_final_top1` | str | Final species. |
| `species_final_top1_score` | float | Final score. |
| `species_final_margin` | float | Top1/top2. |
| `species_final_entropy` | float | Entropy. |
| `geo_candidate_support` | bool | Whether final species present in geo candidates. |
| `metadata_name_support` | bool | Whether metadata supports final species/synonym/name. |
| `vision_review_required` | bool | Review flag. |
| `error` | str | Error text. |
| `retry_eligible` | bool | Retry flag. |
| `created_at` | datetime | Prediction timestamp. |

### 4.15 `image_embeddings.parquet`

| Column | Type | Notes |
|---|---:|---|
| `run_id` | str | Run id. |
| `image_hash` | str | Image hash. |
| `flickr_photo_id` | str | Flickr ID. |
| `model_name` | str | Model. |
| `model_checkpoint` | str | Checkpoint. |
| `preprocess_version` | str | Preprocessing identifier. |
| `embedding_dim` | int | Vector dimension. |
| `embedding` | fixed_size_list[float] | Normalized vector. |
| `created_at` | datetime | Timestamp. |

### 4.16 `bucket_assignments.parquet`

| Column | Type | Notes |
|---|---:|---|
| `run_id` | str | Run id. |
| `flickr_photo_id` | str | Flickr ID. |
| `occurrence_bin` | str | gold/silver/bronze/bin/in_review. |
| `triage_bin` | str | Compatibility bucket if needed. |
| `bin_reason` | str | Main reason. |
| `evidence_summary_json` | json | Compact model+metadata+geo evidence. |
| `hard_negative` | bool | Hard-negative flag. |
| `species_final_top1` | str | Final candidate species. |
| `species_score` | float | Final score. |
| `species_margin` | float | Margin. |
| `metadata_match_status` | str | none/species/synonym/common/conflict. |
| `geo_match_status` | str | supported/unsupported/no_geo/no_data. |
| `date_present` | bool | Date exists. |
| `geo_present` | bool | Coordinates exist. |
| `review_required` | bool | Needs review. |
| `retry_eligible` | bool | Operational retry. |

### 4.17 `comment_review_queue.sqlite`

Logical tables:

```sql
comment_review_queue(
  queue_id TEXT PRIMARY KEY,
  flickr_photo_id TEXT NOT NULL,
  run_id TEXT,
  queue_reason TEXT NOT NULL,
  priority INTEGER NOT NULL,
  status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

comment_observations(
  observation_id TEXT PRIMARY KEY,
  flickr_photo_id TEXT NOT NULL,
  comment_id TEXT,
  observed_term TEXT,
  normalized_term TEXT,
  matched_name_id TEXT,
  matched_taxon_key TEXT,
  evidence_type TEXT,
  created_at TEXT NOT NULL
);

comment_review_decisions(
  decision_id TEXT PRIMARY KEY,
  flickr_photo_id TEXT NOT NULL,
  run_id TEXT,
  decision TEXT NOT NULL,
  decision_reason TEXT NOT NULL,
  previous_bucket TEXT,
  new_bucket TEXT,
  created_at TEXT NOT NULL
);
```

## 5. Candidate-set signatures

Candidate-set signatures must be deterministic hashes over:

```text
classification_mode
candidate_strategy
model_checkpoint
ordered label set
prompt template version
geo_version
registry_version
```

Do not include row-order-specific or timestamp-specific values.

## 6. Bucket logic summary

```text
Gold:
  strong visual species support
  strong margin or low entropy
  species/synonym/reviewed-name metadata support
  geo support when geolocated
  date and image URL present
  no hard negative

Silver:
  moderate visual support or missing date/geo
  no hard negative
  no severe species conflict

Bronze:
  plausible butterfly
  uncertain species
  weak or conflicting support
  comment review useful

Bin:
  hard negative
  not butterfly / not Lepidoptera
  non-biodiversity material

InReview:
  retryable failures
  high uncertainty
  metadata/vision/geo conflicts
  insufficient candidate support
```

## 7. Benchmark schema

`reports/bioclip_benchmark_<run_id>.json` should include:

```json
{
  "run_id": "string",
  "git_sha": "string",
  "device_requested": "mps",
  "device_resolved": "mps",
  "mps_available": true,
  "parameter_matrix": [],
  "results": [
    {
      "register_count": 4,
      "register_size": 32,
      "download_workers": 4,
      "candidate_limit": 2000,
      "classification_mode": "hybrid",
      "candidate_strategy": "geo",
      "rows_in": 1000,
      "images_classified": 980,
      "images_per_second": null,
      "seconds_per_image": null,
      "rss_peak_memory_bytes": null,
      "mps_current_allocated_memory_bytes": null,
      "mps_driver_allocated_memory_bytes": null,
      "mps_recommended_max_memory_bytes": null,
      "download_failure_count": 0,
      "bioclip_failure_count": 0,
      "candidate_set_count": null,
      "bucket_counts": {},
      "notes": []
    }
  ]
}
```

Unsupported metrics must remain `null` or `not_instrumented`.
