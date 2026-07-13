# Target-aware few-shot artifact contracts

Status: normative design contract for the target-aware few-shot migration.

This document fixes artifact ownership, row grains, schemas, fingerprints, and
migration boundaries before runtime implementation. It does not add production
logic. Later tasks may add fields through a new schema version, but may not
silently change a field's meaning or reuse an existing version for an
incompatible physical schema.

The contracts are species-agnostic. A pilot configuration may select *Papilio
demoleus*, but no artifact identity or reusable schema is specific to that
species.

## 1. Normative conventions

### 1.1 Logical types

The tables below use these logical types:

| Notation | Durable representation |
|---|---|
| `str` | non-null UTF-8 string |
| `str?` | nullable UTF-8 string |
| `bool` / `bool?` | Boolean / nullable Boolean |
| `u8`, `u32`, `u64` | unsigned integer of the stated width |
| `i32` | signed 32-bit integer |
| `f32`, `f64` | IEEE floating point; NaN and infinity are forbidden |
| `date?` | nullable Arrow/Parquet date |
| `ts` / `ts?` | UTC Arrow/Parquet timestamp at microsecond precision |
| `list<T>` | non-null list; use an empty list for a known empty set |
| `struct<...>` | Arrow/Parquet struct with the listed typed fields |

Source-qualified accepted taxon keys are strings, for example `gbif:123`, even
when a source also exposes a numeric key. Geographic cell identifiers are
opaque strings outside the geography package. Enum columns are physically
`str` and are validated against the values in this document so additions
require an explicit schema-version decision.

Every Parquet artifact has an exact ordered physical schema implemented as a
Polars schema. Every row begins with a constant `schema_version: str`. Writers
must construct empty outputs with that exact schema and reject unknown columns,
wrong types, mixed versions, duplicate primary keys, invalid enum values, and
non-finite floats. Unknown scalar evidence is null, never zero, an empty string,
or a fabricated probability. Lists are sorted and deduplicated where their
meaning is set-like.

Dates, counts, distances, and confidence values retain their units in the field
name or schema description. A field named `*_probability` is permitted only for
an output from a persisted calibrator fitted on independent reviewed data. Raw
cosine similarities, text logits, SVC margins, geographic evidence scores, and
review confidence are not probabilities.

### 1.2 Determinism and publication

Each artifact definition below declares a primary key and deterministic sort
order. Workers may emit immutable shards, but the main process owns merge,
validation, sorting, deduplication, and publication. Parquet is the durable
tabular format. Compact JSON is limited to manifests, readiness, metrics, and
reports. NPZ is limited to non-executable numeric arrays.

S3 is the production durable store. Registry artifacts live under an immutable
`registry/version=<registry_version>/` root. Run artifacts live under the
existing `run_id=<run_id>/` root, grouped into `geography/`, `candidates/`,
`references/`, `models/`, `staging/`, and `reports/`. Local paths are build or
worker caches only. PostgreSQL stores leases, attempts, and resumable work
state; it is not the authoritative scientific artifact store.

Each artifact set has a `manifest.json`, published last after all referenced
objects validate. It contains:

- `schema_version = biominer-artifact-manifest-v1`;
- artifact-set name, version, status, run ID, registry version, and git SHA;
- start/end UTC timestamps and effective configuration;
- each file's URI, byte count, row count, physical schema version, full SHA-256
  checksum, semantic fingerprint, primary key, and sort order;
- dependency fingerprints and exact source snapshot versions;
- QA status and fatal/warning counts;
- supported metrics, with unavailable values recorded as null or
  `not_instrumented`.

An artifact set is readable only when its manifest exists, reports a complete
or explicitly allowed shortfall status, lists the object, and matches the
object checksum. Publication never overwrites a completed version. A failed or
partial prefix is not promoted and is not completion evidence.

### 1.3 Fingerprint algorithm

All hashes use the full lowercase SHA-256 digest prefixed with `sha256:`. An
eight-character or otherwise truncated digest is not an artifact identity.

Three hashes have distinct meanings:

1. `content_hash` hashes the exact object bytes. It detects corruption but may
   change when equivalent Parquet is written by another writer version.
2. `semantic_fingerprint` hashes canonical logical identity: schema version,
   ordered row values, effective semantic configuration, and dependency
   fingerprints. It excludes timestamps, byte sizes, URIs, attempts, workers,
   and other operational metadata.
3. `work_identity` hashes the semantic inputs required to decide whether a
   computation is already complete. Lease and retry metadata are excluded.

Canonical JSON uses UTF-8, sorted object keys, compact separators, explicit
nulls, and no NaN or infinity. Set-like arrays are sorted before hashing.
Integers are decimal. Floating values used in semantic fingerprints are hashed
as their declared IEEE byte representation, not locale-dependent formatted
text. Embedding and model arrays are C-contiguous little-endian arrays with a
declared dtype and shape; the manifest hashes their raw bytes and the complete
NPZ file.

Self-referential fields are excluded from their own preimage. A classifier
fingerprint, for example, covers its canonical manifest with
`classifier_fingerprint` omitted plus the checksum of `classifier_arrays.npz`.
The completed manifest then stores that fingerprint. Readers recompute it and
fail closed.

## 2. Geographic registry artifacts

Geographic evidence describes sourced known spread. Absence from these tables
is never a hard biological negative.

### 2.1 `taxon_geographic_spread.parquet`

Grain: one accepted species, source dataset, spatial cell, resolution, range
role, and source snapshot. Primary key and sort order:
`(accepted_taxon_key, source, source_dataset_key, spatial_resolution,
spatial_cell_id, known_range_role, source_snapshot_version)`.

Schema version: `taxon-geographic-spread-v1.0.0`.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | `str` | Constant above |
| `registry_version` | `str` | Accepted registry identity |
| `accepted_taxon_key` | `str` | Reconciled accepted species key |
| `gbif_species_key` | `u64` | Numeric GBIF accepted species key |
| `scientific_name` | `str` | Accepted scientific name |
| `source` | `str` | Evidence source, initially `GBIF` |
| `source_dataset_key` | `str` | Dataset identifier retained from source |
| `source_dataset_citation` | `str?` | Dataset citation or DOI citation |
| `source_query_hash` | `str` | Full hash of normalized query/download identity |
| `spatial_cell_id` | `str` | Opaque hierarchical cell identifier |
| `spatial_resolution` | `u8` | Grid resolution, interpreted by the geography config |
| `country_code` | `str?` | ISO 3166-1 alpha-2 when supported |
| `admin1` | `str?` | Source or reconciled first-order region |
| `bioregion` | `str?` | Versioned bioregion identifier |
| `centroid_latitude` | `f64` | Cell centre, degrees in `[-90, 90]` |
| `centroid_longitude` | `f64` | Cell centre, wrapped to `[-180, 180)` |
| `occurrence_count` | `u64` | All retained source records in this group |
| `georeferenced_occurrence_count` | `u64` | Records with usable coordinates |
| `range_inference_eligible_count` | `u64` | Records eligible for current-range evidence |
| `preserved_specimen_count` | `u64` | Retained but separately counted |
| `fossil_count` | `u64` | Retained but never current-range evidence |
| `geospatial_issue_count` | `u64` | Retained records with declared geo issues |
| `coordinate_uncertainty_summary` | `struct<count:u64,min_m:f64?,p50_m:f64?,p95_m:f64?,max_m:f64?>` | Known source uncertainty; no imputed precision |
| `earliest_occurrence_date` | `date?` | Earliest supported event date |
| `latest_occurrence_date` | `date?` | Latest supported event date |
| `basis_of_record_counts` | `list<struct<value:str,count:u64>>` | Sorted source categories |
| `establishment_means` | `list<str>` | Sorted distinct source values |
| `occurrence_status` | `str?` | Normalized source occurrence status |
| `known_range_role` | `str` | `native`, `introduced`, `vagrant`, `uncertain`, or `unknown` |
| `evidence_confidence` | `f32?` | Versioned evidence heuristic in `[0,1]`, not probability |
| `retrieved_at` | `ts` | Source retrieval time |
| `source_snapshot_version` | `str` | Immutable source snapshot/download identity |

### 2.2 `taxon_geographic_summary.parquet`

Grain: one accepted species and geographic evidence version. Primary key and
sort order: `(accepted_taxon_key, geographic_evidence_version)`.

Schema version: `taxon-geographic-summary-v1.0.0`.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | `str` | Constant above |
| `registry_version` | `str` | Accepted registry identity |
| `accepted_taxon_key` | `str` | Reconciled accepted species key |
| `scientific_name` | `str` | Accepted scientific name |
| `geographic_evidence_version` | `str` | Version of inputs and summary policy |
| `cell_counts_by_resolution` | `list<struct<resolution:u8,count:u64>>` | Sorted by resolution |
| `countries` | `list<str>` | Sorted ISO country codes |
| `admin_regions` | `list<str>` | Sorted scoped admin-region IDs |
| `occupied_envelope` | `struct<south:f64,north:f64,west:f64,east:f64,crosses_dateline:bool>` | Dateline-aware envelope |
| `disconnected_range_component_count` | `u32` | Components under versioned adjacency policy |
| `occurrence_density_summary` | `struct<min:f64?,p50:f64?,p95:f64?,max:f64?>` | Eligible records per occupied cell |
| `data_deficient` | `bool` | True when configured evidence minimum is unmet |
| `data_deficient_reasons` | `list<str>` | Auditable reason codes |
| `suspicious_outlier_cell_count` | `u64` | QA outliers, not silently discarded |
| `range_source_coverage` | `list<struct<source:str,dataset_count:u64,eligible_occurrence_count:u64>>` | Source coverage |
| `known_introduced_regions` | `list<str>` | Scoped cell/country/region identifiers |
| `current_evidence_count` | `u64` | Records classified as current evidence |
| `historical_evidence_count` | `u64` | Records distinguishable as historical |
| `spread_fingerprint` | `str` | Fingerprint of contributing spread rows |
| `created_at` | `ts` | Summary creation time |

Geographic source snapshots remain in the registry's existing source snapshot
artifact, extended by a later version if new fields are required. Geographic
QA findings use the registry QA artifact and must distinguish invalid source
records from biologically absent evidence.

## 3. Flickr geography artifacts

Flickr geography is candidate-distribution evidence. It is not verified range
or a target label.

### 3.1 `flickr_geography.parquet`

This supporting artifact is the canonical projection consumed by clustering.
Grain and primary key: one `(source, flickr_photo_id)`, sorted by that key.
Schema version: `flickr-geography-v1.0.0`.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | `str` | Constant above |
| `source` | `str` | `flickr` for current inputs |
| `flickr_photo_id` | `str` | Source photo identity |
| `source_record_hash` | `str` | Canonical metadata row hash |
| `latitude` | `f64?` | Original usable coordinate |
| `longitude` | `f64?` | Original usable coordinate |
| `coordinate_accuracy` | `f64?` | Original source accuracy, not invented precision |
| `coordinate_source` | `str?` | Source field or reconciliation method |
| `geotag_available` | `bool` | Usable coordinate pair present |
| `country_code` | `str?` | Source/reconciled ISO country |
| `admin1` | `str?` | Resolved first-order region |
| `coarse_cell_id` | `str?` | Opaque configured coarse cell |
| `regional_cell_id` | `str?` | Opaque configured regional cell |
| `local_cell_id` | `str?` | Opaque configured local cell |
| `coordinate_quality` | `str` | Versioned quality category |
| `geography_warning` | `str?` | Single primary warning code |
| `geography_warnings` | `list<str>` | Complete sorted warning codes |
| `geography_config_fingerprint` | `str` | Grid and reconciliation configuration |

`coordinate_quality` is one of `missing`, `invalid`, `unknown_precision`,
`flickr_world`, `flickr_country`, `flickr_region`, `flickr_city`, or
`flickr_street` under `flickr-accuracy-v1.0.0`. Flickr accuracy is retained as
the documented ordinal 1-16 level; it is never converted to metres. World and
country levels populate no H3 cells, region level populates only the coarse
cell, city level populates coarse and regional cells, and street level
populates all configured cells. A valid pair can therefore have
`geotag_available = true` while finer cell fields remain null. Unknown or
nonstandard accuracy populates no cells. Country and `admin1` values are
accepted only from explicit source fields; this projection performs no reverse
geocoding or location-name inference. Flickr's `latitude=0`, `longitude=0`,
`accuracy=0` response placeholder is normalized as missing geography with the
`flickr_zero_geo_sentinel` warning. A `(0,0)` pair carrying a documented
accuracy from 1 through 16 remains a usable but explicitly warned coordinate.

### 3.2 `flickr_geo_clusters.parquet`

Grain: one target scope and geographic cluster. Primary key and sort order:
`(target_accepted_taxon_key, geo_cluster_id)`.
Schema version: `flickr-geo-clusters-v1.0.0`.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | `str` | Constant above |
| `geo_cluster_id` | `str` | Stable hash of config and sorted member cells; `no_geo` is reserved |
| `target_accepted_taxon_key` | `str` | Target scope, not a label for members |
| `member_image_count` | `u64` | Candidate images assigned |
| `member_cell_count` | `u64` | Occupied cells represented |
| `member_cell_ids` | `list<str>` | Sorted cells used to derive identity |
| `centroid` | `struct<latitude:f64?,longitude:f64?>` | Great-circle centroid; null for `no_geo` |
| `medoid` | `struct<latitude:f64?,longitude:f64?>` | Member medoid; null for `no_geo` |
| `radius_quantiles_km` | `struct<p50:f64?,p90:f64?,p95:f64?,max:f64?>` | Great-circle radii |
| `bounding_geometry` | `struct<south:f64?,north:f64?,west:f64?,east:f64?,crosses_dateline:bool>` | Dateline-aware bounds |
| `countries` | `list<str>` | Sorted country codes |
| `admin_regions` | `list<str>` | Sorted scoped admin IDs |
| `source_resolution` | `u8?` | Cell resolution used; null for `no_geo` |
| `cluster_method` | `str` | Versioned deterministic method |
| `cluster_configuration_hash` | `str` | Full semantic configuration hash |
| `candidate_distribution_only` | `bool` | Must be true |
| `created_at` | `ts` | Build time, excluded from cluster identity |

### 3.3 `flickr_geo_assignments.parquet`

Grain and primary key: one `(source, flickr_photo_id)`, sorted by that key.
Schema version: `flickr-geo-assignments-v1.0.0`.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | `str` | Constant above |
| `source` | `str` | Source identity |
| `flickr_photo_id` | `str` | Photo identity |
| `source_record_hash` | `str` | Exact input record identity |
| `target_accepted_taxon_key` | `str` | Target scope only |
| `geo_cluster_id` | `str` | Cluster or `no_geo` |
| `distance_to_medoid_km` | `f64?` | Null for `no_geo` or coarse-only assignment |
| `assignment_method` | `str` | Local cell, adjacency, country, bioregion, or `no_geo` method |
| `coordinate_quality` | `str` | Carried from canonical projection |
| `fallback_scope` | `str?` | Country/bioregion/global scope used |
| `outlier` | `bool` | True when outside configured cluster support |
| `cluster_configuration_hash` | `str` | Must match cluster artifact |

No assignment may attach a photo to a remote cluster merely because that
cluster is mathematically nearest. `no_geo` and rejected remote assignments are
explicit and broaden candidate generation rather than deleting the target.

`h3-density-components-v1.0.0` selects cells meeting the configured minimum
image density, forms components using sorted H3 grid adjacency, and rejects
components below the configured image minimum. Occupied cells below the density
threshold may join only an adjacent component, and only when every candidate in
that cell is within the configured great-circle assignment distance after the
cluster medoid is recomputed. Final cluster IDs cover the target scope, complete
configuration hash, and sorted core-plus-adjacency member cells. The centroid
is the image-count-weighted spherical centroid of member cell centres; the
medoid is the member cell centre nearest that centroid by great-circle
distance. Low-precision coordinates may use a country or configured broad
bioregion only when that scope identifies exactly one cluster. Missing,
ambiguous, sparse, and remote candidates remain explicit `no_geo` assignments.
The target key scopes this candidate workload and never labels its images.

## 4. Regional candidate artifacts

### 4.1 `regional_taxon_occurrence.parquet`

Grain: one region, accepted species, source, and evidence version. Primary key
and sort order: `(regional_scope_id, accepted_taxon_key, source,
evidence_version)`.

Schema version: `regional-taxon-occurrence-v1.0.0`.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | `str` | Constant above |
| `regional_scope_id` | `str` | Geo cluster or spatial-cell scope |
| `regional_scope_type` | `str` | `geo_cluster`, `spatial_cell`, `country`, `bioregion`, or `global` |
| `accepted_taxon_key` | `str` | Reconciled accepted species |
| `scientific_name` | `str` | Accepted name |
| `family` | `str` | Accepted family |
| `subfamily` | `str?` | Reviewed accepted subfamily |
| `tribe` | `str?` | Reviewed accepted tribe |
| `genus` | `str` | Accepted genus |
| `occurrence_count` | `u64` | Eligible supporting records |
| `independent_dataset_count` | `u64` | Distinct source datasets |
| `earliest_occurrence_date` | `date?` | Earliest supported date |
| `latest_occurrence_date` | `date?` | Latest supported date |
| `coordinate_confidence` | `f32?` | Versioned evidence score, not probability |
| `overlap_type` | `str` | Exact/buffer/country/bioregion/global relationship |
| `source` | `str` | Evidence source |
| `source_dataset_keys` | `list<str>` | Sorted provenance |
| `evidence_version` | `str` | Source and reconciliation policy identity |
| `registry_version` | `str` | Taxonomic identity used for reconciliation |

Only records already marked range-inference eligible, present, taxon-key
matched, coordinate-valid, non-fossil, non-specimen, and free of declared
geospatial issues contribute. Their accepted keys are joined to accepted
species in `taxa.parquet`; names from occurrence sources are never authoritative
join keys. Reviewed subfamily and tribe values are added only from an enabled,
registry-consistent classification path.

Scope overlap precedence is `exact`, `buffer`, `country`, `bioregion`, then
`global`. For one scope, species, and source, only records from the strongest
available tier contribute, preventing broad fallbacks from inflating exact
support. Coordinate confidence is a versioned evidence score, not a
probability: version `inverse-uncertainty-100km-v1.0.0` averages
`1 / (1 + uncertainty_metres / 100000)` only when every selected record has a
reported uncertainty; otherwise it is null. The policy version is appended to
`evidence_version`.

### 4.2 `regional_candidate_species.parquet`

Grain: one candidate set and candidate species. Primary key and sort order:
`(candidate_set_id, candidate_priority, candidate_accepted_taxon_key)`.

Schema version: `regional-candidate-species-v1.0.0`.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | `str` | Constant above |
| `candidate_set_id` | `str` | Fingerprint-derived immutable set identity |
| `target_accepted_taxon_key` | `str` | Target that defines the verification task |
| `geo_cluster_id` | `str` | Includes `no_geo` |
| `candidate_accepted_taxon_key` | `str` | Species to score |
| `scientific_name` | `str` | Accepted candidate name |
| `family` | `str` | Candidate family |
| `genus` | `str` | Candidate genus |
| `candidate_reason` | `list<str>` | Sorted union of all inclusion reasons |
| `geographic_evidence_score` | `f32?` | Soft evidence, never a probability or hard gate |
| `occurrence_support` | `u64` | Eligible occurrence support count |
| `same_genus` | `bool` | Relation to target |
| `same_family` | `bool` | Relation to target |
| `known_mimic` | `bool` | Curated relationship present |
| `historical_false_positive` | `bool` | Reviewed historical relation present |
| `visually_nearest` | `bool` | False until versioned prototype evidence exists |
| `target_candidate` | `bool` | Exactly one true row per set |
| `candidate_priority` | `u32` | Ordering only; never pruning authorization |
| `source_versions` | `list<str>` | Registry, geographic, relation, and visual graph versions |
| `candidate_set_fingerprint` | `str` | Fingerprint of all sorted rows and dependencies |

The target row is mandatory even with no geography, zero occurrence support,
or poor text rank. Family and genus text ranks are not candidate deletion
inputs.

### 4.3 `competitor_relationships.parquet`

Grain: one directed subject/object relationship and evidence version. Primary
key and sort order: `(subject_accepted_taxon_key, relationship_type,
object_scope_type, object_scope_id, evidence_version)`.

Schema version: `competitor-relationships-v1.0.0`.

Required fields are `schema_version`, `subject_accepted_taxon_key`,
`object_scope_type` (`species` or `genus`), `object_scope_id`,
`relationship_type` (`known_mimic`, `close_congener`,
`historical_false_positive_species`, `historical_false_positive_genus`,
`taxonomic_neighbour`, or `visual_neighbour`), `source`, `source_record_id`,
`evidence_version`, `evidence_note`, `review_status`, `reviewed_by`,
`reviewed_at`, `enabled`, and `relationship_fingerprint`. Visual-neighbour
relationships additionally require prototype and model fingerprints. Curated
relationships may influence inclusion and priority, never establish an image
label.

## 5. Reference acquisition and review artifacts

Provider metadata is candidate evidence. Neither GBIF names nor iNaturalist
Research Grade status is manual verification for this support bank.

### 5.1 `reference_observations.parquet`

Grain and primary key: one `(source, source_observation_id)`, sorted by source
and ID. Schema version: `reference-observations-v1.0.0`.

Required columns are:

- identity: `schema_version: str`, `reference_observation_id: str`,
  `source: str`, `source_observation_id: str`, `source_taxon_id: str?`;
- taxonomy: `supplied_scientific_name: str?`, `accepted_taxon_key: str?`,
  `reconciled_scientific_name: str?`, `registry_version: str`,
  `taxon_reconciliation_status: str`;
- identification: `identification_quality: str?`,
  `community_taxon_status: str?`, `identification_disagreement: bool?`,
  `captive_or_cultivated: bool?`, `life_stage: str`, `sex: str?`;
- event/geography: `observed_at: ts?`, `latitude: f64?`, `longitude: f64?`,
  `coordinate_uncertainty: f64?` in metres, `coordinates_obscured: bool?`,
  `country: str?`, `country_code: str?`, `geo_cluster_id: str?`,
  `distance_to_cluster_medoid_km: f64?`;
- provenance: `source_dataset_key: str?`, `source_dataset_doi: str?`,
  `source_record_url: str?`, `source_record_hash: str`, `retrieved_at: ts`,
  `source_snapshot_version: str`;
- suitability flags: `geospatial_issue: bool`, `preserved_specimen: bool`,
  `fossil: bool`, `occurrence_absent: bool`,
  `uncertain_taxon_match: bool`, `basis_of_record_suitable: bool`.

`source_record_url` is an identifier only; scraped page content is not stored.
Observation and media licences are separate fields in separate artifacts.

### 5.2 `reference_media_candidates.parquet`

Grain and primary key: one `(source, provider_media_id,
reference_observation_id)`, sorted by that key. Schema version:
`reference-media-candidates-v1.0.0`.

Required columns are `schema_version`, `reference_media_id`,
`reference_observation_id`, `provider_media_id`, `source`, `media_identifier`,
`media_type`, `width`, `height`, `creator`, `rights_holder`, `licence`,
`licence_uri`, `attribution`, `occurrence_licence`, `original_provider`,
`media_position`, `source_checksum`, `source_checksum_algorithm`,
`download_status`, `verification_status`, `exclusion_reason`,
`licence_policy_status`, `retrieved_at`, and `source_snapshot_version`.
Dimensions are nullable `u32`; optional strings are nullable. Media and
occurrence licences must not be collapsed into one value.

### 5.3 `reference_acquisition_plan.parquet`

Grain: one target/competitor, geographic scope, life-stage, visual-domain,
source lane, and immutable plan version. Primary key and sort order:
`(acquisition_plan_id, candidate_accepted_taxon_key, geo_cluster_id,
life_stage, visual_domain, source, fallback_level)`.

Schema version: `reference-acquisition-plan-v1.0.0`. Required fields are
`schema_version`, `acquisition_plan_id`, `target_accepted_taxon_key`,
`candidate_set_id`, `candidate_accepted_taxon_key`, `scientific_name`,
`geo_cluster_id`, `life_stage`, `visual_domain`, `source`, `requested_count`,
`available_candidate_count`, `selected_candidate_count`, `shortfall_count`,
`fallback_level`, `selection_strategy`, `selection_seed`,
`max_distance_km`, `licence_policy_version`, `source_snapshot_version`,
`plan_configuration_fingerprint`, and `created_at`.

Quota planning preserves cluster, source, licence, observer, observation, date,
background, and locality diversity. A shortfall is a recorded outcome, not
synthetic support.

### 5.4 `reference_media_objects.parquet`

This supporting artifact records committed source-image objects without
mutating candidate metadata. Grain and primary key: one
`reference_media_id`. Schema version: `reference-media-objects-v1.0.0`.

Required fields are `schema_version`, `reference_media_id`, `source_object_uri`,
`content_type`, `source_byte_count`, `decoded_width`, `decoded_height`,
`sha256`, `perceptual_hash`, `duplicate_group_id`, `duplicate_type`,
`canonical_reference_media_id`, `provider_mirror_ids`, `downloaded_at`,
`download_attempt_count`, `licence_policy_status`, `decode_status`,
`quarantine_reason`, and `object_fingerprint`. Provider relationships and every
source row remain available after deduplication. Only a manifest-committed S3
object is `download_status=complete`.

### 5.5 Review state

`reference_review_queue.parquet` is a deterministic materialized queue; it is
not overwritten with decisions. Grain: one review request and media item.
Primary key: `review_request_id`. Schema version:
`reference-review-queue-v1.0.0`.

It includes `schema_version`, `review_request_id`, `reference_media_id`,
`reference_observation_id`, `accepted_taxon_key`, `scientific_name`,
`durable_preview_uri`, `duplicate_group_id`, provider identification evidence,
licence evidence, proposed `life_stage`, `visual_domain`, and `view`,
`review_reason`, `review_priority`, `required_review_count`, `review_status`,
`created_at`, `reference_bank_version`, and input fingerprints.

`reference_review_decisions.parquet` is the append-only scientific decision
record. Grain and primary key: one `review_decision_id`; sort by
`(reference_media_id, review_round, reviewed_at, review_decision_id)`. Schema
version: `reference-review-decisions-v1.0.0`.

It includes `schema_version`, `review_decision_id`, `review_request_id`,
`reference_media_id`, `review_round`, `reviewer_id`, `reviewed_at`,
`target_identity_verified: bool?`, `verification_status`, `life_stage`,
`visual_domain`, `view`, `review_confidence`, `review_notes`,
`exclusion_reason`, `second_review_required`, `conflicts_with_decision_id`, and
`decision_source_hash`.

Allowed values are:

- `life_stage`: `adult`, `larva`, `pupa`, `egg`, `unknown`;
- `visual_domain`: `live_field`, `pinned_specimen`, `artwork`, `logo`,
  `tattoo`, `partial_wing`, `dead_or_damaged_specimen`, `ambiguous`,
  `unsuitable`;
- `view`: `dorsal`, `ventral`, `lateral`, `frontal`, `oblique`, `unknown`;
- `verification_status`: `pending`, `verified`, `excluded`, `uncertain`,
  `conflict`, `second_review_required`.

`review_confidence` is reviewer metadata, not a calibrated model probability.
Only a resolved `verified` decision with `target_identity_verified=true`, an
accepted licence, a resolved duplicate group, and an allowed route may enter a
production support split.

### 5.6 Frozen support and readiness

`reference_support_manifest.parquet` is the immutable resolved projection used
by embedding and split construction. Its grain is one canonical verified media
item and route. It records all source observation/media IDs, content and
duplicate hashes, accepted taxonomy, resolved review decision IDs, licence and
attribution, geography, life stage, visual domain, view, support eligibility,
split, exclusion state, and `reference_bank_fingerprint`.

`reference_bank_summary.parquet` has one row per bank version, accepted species,
cluster scope, life stage, visual domain, and split. Schema version:
`reference-bank-summary-v1.0.0`. It records required, candidate, downloaded,
deduplicated, reviewed, verified, eligible, excluded, and shortfall counts;
source, licence, observer, observation, and geographic diversity counts; and
the reference-bank and support-manifest fingerprints.

`reference_bank_readiness.json` has schema version
`reference-bank-readiness-v1.0.0` and contains:

- `reference_bank_version`, target key, registry version, candidate-set
  fingerprints, support-manifest fingerprint, model/preprocessing identity,
  split fingerprint, creation time, and git SHA;
- `status`: `ready`, `ready_with_documented_shortfalls`,
  `awaiting_manual_review`, `blocked_licence`,
  `blocked_missing_target_support`, or `invalid`;
- every readiness check as an object with `check_id`, `status`, observed value,
  required value, affected species/clusters/routes, and artifact evidence;
- unresolved duplicate, licence, review, route-separation, attribution,
  leakage, target-minimum, competitor-minimum, and geographic-coverage counts;
- a sorted `documented_shortfalls` list and all dependent artifact checksums.

The JSON contains no fabricated pass. `ready_with_documented_shortfalls` is
allowed only by an explicit versioned policy, and missing target adult support
cannot be downgraded to a shortfall.

## 6. Embedding and prototype artifacts

Adult field, larval, and pinned-specimen routes are separate in every support,
prototype, classifier, calibrator, and threshold identity.

### 6.1 `reference_embeddings.parquet`

Grain and primary key: one canonical reference media item, visual input, and
model/preprocessing identity. Sort by accepted taxon, route, cluster, media ID,
and visual input. Schema version: `reference-embeddings-v1.0.0`.

Required fields are `schema_version`, `reference_media_id`,
`reference_observation_id`, `review_decision_id`, `duplicate_group_id`,
`accepted_taxon_key`, `scientific_name`, `geo_cluster_id`, `life_stage`,
`visual_domain`, `view`, `route`, `visual_input_kind`, `image_content_hash`,
`transformation_version`, `model_id`, `model_revision`, `model_checkpoint_hash`,
`preprocessing_version`, `model_fingerprint`, `embedding_dimension: u32`,
`embedding: list<f32>`, `embedding_norm: f32`, `support_split`,
`support_manifest_fingerprint`, `embedding_created_at`, and
`embedding_fingerprint`.

The vector length must equal `embedding_dimension`; all values must be finite;
the normalization policy is part of the model fingerprint. Only eligible rows
from the frozen support manifest are accepted.

### 6.2 `reference_prototypes.parquet`

Grain and primary key: one prototype ID. Sort by route, species, cluster scope,
life stage, visual domain, method, and ID. Schema version:
`reference-prototypes-v1.0.0`.

Required fields are `schema_version`, `prototype_id`, `accepted_taxon_key`,
`species` (the accepted scientific name), `cluster_scope_type`, `geo_cluster_id`, `life_stage`,
`visual_domain`, `view`, `route`, `prototype_method`, `prototype_group_id`,
`reference_count`, `independent_observation_count`,
`balanced_sampling_seed`, `mean_centered`, `embedding_dimension`, `embedding`,
`embedding_norm`, `model_fingerprint`, `reference_embedding_fingerprint`,
`support_manifest_fingerprint`, and `prototype_fingerprint`.

Embedding clusters may split a verified species/route group into prototypes;
they may never change its accepted taxon key.

### 6.3 `visual_neighbour_species.parquet`

Grain: one directed species-neighbour edge, route, and graph version. It records
subject and neighbour accepted keys, best prototype similarity, prototype IDs,
rank, route, model/prototype fingerprints, graph configuration, and graph
fingerprint. It adds candidate reasons but cannot remove geographically
plausible species.

### 6.4 `flickr_embeddings.parquet`

This supporting inference cache has one source photo, full-frame visual input,
and model/preprocessing identity per row. It records source/photo ID,
source-record hash, visual-input ID/kind/version, raw image content hash,
transformation and model fingerprints, finite embedding and norm, route and
quality metadata, creation time, and embedding fingerprint. No spatial crop
hash is its image identity. Raw full-image embeddings are reused across routes
and detections when their transformation identity is identical.

## 7. Training, classifier, and calibration artifacts

### 7.1 `few_shot_training_features.parquet`

Grain: one reviewed training item, visual input, route, target task, and feature
schema. Primary key is `training_example_id`; deterministic sort is split,
group ID, source item ID, and visual input. Schema version:
`few-shot-training-features-v1.0.0`.

The table includes:

- provenance and leakage control: source item/observation/owner IDs, duplicate
  and burst groups, provider-mirror group, geo cluster, split, reviewed-label
  ID, support/reference fingerprints, model/embedding fingerprints, route;
- labels: binary target-present label, accepted class key for multiclass use,
  visual-domain label, label certainty and suitability flags;
- raw frozen embedding: dimension, `list<f32>` values, norm, visual-input kind;
- reference evidence: target centroid, nearest, top-three, top-five, local and
  global prototype similarities; best regional, same-genus, historical false
  positive, family-negative, and domain-negative similarities; target-minus-
  competitor/domain margins; target-prototype and nearest-independent-support
  distances;
- text evidence: target ensemble similarity, best competitor similarity, and
  margin, all named as similarities rather than probabilities;
- geography: target/competitor overlap evidence, distances to target occurrence
  and support, candidate-source counts, and missing-geo indicator;
- detection/input quality: YOLOE route and score, subject-area ratio, mask
  coverage, visual-input kind, multiple-organism indicator, resolution, and
  quality flags;
- `feature_schema_fingerprint` and `training_data_fingerprint`.

Query source terms, Flickr discovery labels, or any other field that directly
leaks the reviewed answer are prohibited features.

### 7.2 `dataset_split_manifest.parquet`

Grain: one source item/group membership and split version. Primary key:
`(split_version, item_type, item_id)`. Schema version:
`dataset-split-manifest-v1.0.0`.

It records item, observation, exact-hash, perceptual-duplicate, observer,
photographer/Flickr-owner, burst, provider-mirror, and geo-cluster group IDs;
route, accepted class key, split (`support_train`, `model_selection`,
`calibration`, or `final_test`), grouping policy, deterministic seed, source
artifact fingerprint, and split fingerprint. A group may occur in exactly one
split. Loaders fail on leakage rather than repairing it silently.

### 7.3 `classifier_manifest.json` and `classifier_arrays.npz`

`classifier_manifest.json` schema version is
`few-shot-classifier-manifest-v1.0.0`. One immutable classifier directory
contains one manifest and one arrays file. The manifest requires:

- classifier version/fingerprint, task (`binary_target_verifier`,
  `regional_multiclass`, `visual_domain`, or `larval_target_verifier`), route,
  target key, ordered class labels, and estimator family;
- exact ordered feature names and dtypes, feature-schema fingerprint, scaling
  policy, and named array keys for means/scales where applicable;
- estimator configuration, deterministic seed, bounded search grid, selected
  parameters, class-weight policy, and fit-library versions;
- foundation model, preprocessing, reference bank, prototype, candidate set,
  training data, and split fingerprints;
- fit partition and group-aware validation policy; sample/class/group counts;
- `classifier_arrays.npz` URI, full file checksum, and every allowed array's
  name, dtype, shape, and raw-byte checksum;
- creation timestamp, git SHA, QA status, and non-calibrated validation metrics.

The NPZ contains numeric arrays only: coefficients, intercepts, class labels as
numeric indices, and feature scaling parameters. It contains no object arrays,
pickles, executable code, or embedded authoritative JSON. Readers use
`allow_pickle=False`, validate the exact key set/dtypes/shapes/checksums, and map
numeric class indices through the JSON manifest. Pickle and joblib artifacts are
not supported.

### 7.4 `calibration_artifacts.json`, `calibration_arrays.npz`, and report

`calibration_artifacts.json` schema version is
`few-shot-calibration-manifest-v1.0.0`. It requires calibration version and
fingerprint, classifier fingerprint, task and route, method (`sigmoid`,
`isotonic`, or `temperature`), ordered classes, independent-prediction artifact
fingerprint, split fingerprint, group-aware out-of-fold policy, calibration
sample/class/group counts, parameters represented directly or by named numeric
arrays, arrays checksum/schema, creation/git/library provenance, and calibration
metrics.

The same non-executable NPZ restrictions as classifier arrays apply. Isotonic
knots and values or other vector parameters live in NPZ; scalar Platt or
temperature parameters may live in JSON. The calibrator must not accept the
estimator's fitting predictions as calibration input.

The manifest also contains a versioned decision policy with learned threshold,
optimization metric, target precision objective, achieved calibration metrics,
calibration sample size, competitor-margin threshold, coverage/reference/OOD
requirements, abstention rules, and `decision_policy_fingerprint`. Unsupported
threshold provenance makes target confirmation invalid.

`calibration_report.parquet` records reliability-bin boundaries, weighted and
unweighted counts, mean prediction, observed frequency, route, split,
calibrator/classifier fingerprints, and Brier/log-loss/ECE contributions. Raw
similarity or SVC-margin bins cannot be labelled calibrated probability bins.

## 8. Target-aware inference artifacts

The production mode string is
`target_aware_few_shot_classification`. It does not overload
`target_scope_object_screening` or hierarchical cascade output.

### 8.1 `target_aware_object_scores.parquet`

Grain: one source photo, routed scoring unit, and target task after versioned
full-frame input aggregation. Primary key is `target_score_id`; sort by source,
photo ID, route, and scoring-unit ID. Schema version:
`target-aware-object-scores-v1.0.0`.

Identity fields:

- `schema_version`, `target_score_id`, `source`, `flickr_photo_id`,
  `source_record_hash`, `scoring_unit_id`, nullable `detection_id`, `route`,
  `geo_cluster_id`, `candidate_set_id`, `target_accepted_taxon_key`;
- `reference_bank_version`, `classifier_version`, `calibration_version`,
  `prompt_version`, `visual_input_version`, `classification_mode`;
- registry, geographic, candidate-set, reference-bank, support-manifest,
  model, preprocessing, prompt, classifier, calibrator, visual-input-fusion,
  and decision-policy fingerprints;
- `scored_at` and producer git SHA.

Target evidence fields:

- `target_raw_text_similarity`, `target_reference_centroid_similarity`,
  `target_nearest_reference_similarity`, `target_top_k_reference_similarity`,
  `target_local_prototype_similarity`, `target_global_prototype_similarity`;
- `target_classifier_margin`, `calibrated_target_probability`,
  `target_regional_rank`, and `target_global_rank`.

Competitor/non-match fields:

- `best_competitor_accepted_taxon_key`, `best_competitor_scientific_name`,
  `calibrated_best_competitor_probability`,
  `best_competitor_reference_similarity`, `target_competitor_margin`,
  `best_domain_negative_similarity`, `calibrated_non_target_probability`,
  `nonmatch_score`, `nonmatch_margin`, and `competitor_reason`.

Quality and structured-evidence fields:

- `yoloe_route`, nullable `detector_score`, nullable `subject_area_ratio`,
  nullable `mask_coverage`, `visual_input_disagreement`,
  `reference_coverage`, `geo_evidence`, nullable `ood_score`,
  `visual_detail_sufficient`, and sorted quality flags;
- `visual_input_evidence`, a list of structs preserving input ID/kind,
  transformation fingerprint, target/competitor raw scores, and aggregation
  weight for every full-frame variant;
- `regional_candidate_evidence`, a list of structs preserving candidate key,
  candidate reasons, raw reference/text/classifier scores, calibrated
  probability when supported, regional rank, and candidate-set fingerprint for
  every scored species.

Decision fields:

- `classification_decision`: `target_confirmed`,
  `target_probable_review`, `known_regional_competitor`,
  `known_nonregional_competitor`, `other_butterfly`,
  `non_butterfly_insect`, `pinned_specimen`, `visual_artifact`,
  `out_of_distribution`, `insufficient_visual_detail`,
  `insufficient_reference_coverage`, `no_geo_global_fallback`, or `abstain`;
- `abstained`, `abstention_reason`, `review_priority`,
  `model_decision_threshold`, and `threshold_provenance`.

Similarity and margin fields are nullable `f32`; calibrated probabilities are
nullable `f64` constrained to `[0,1]`; ranks are nullable `u32`; Boolean and
list fields are non-null. A target-confirmed row requires a compatible route,
calibrator and decision-policy fingerprints, supported threshold provenance,
complete candidate scoring, sufficient reference coverage, and no configured
abstention condition.

### 8.2 Normalized candidate evidence

`target_aware_candidate_scores.parquet` is a supporting normalized table for
large candidate unions. Grain and primary key: one `target_score_id` and
candidate accepted key. It stores all candidate reasons, raw evidence,
classifier margin/logit, calibrated probability only when the matching
calibrator supports it, ranks, support counts, geography, and every dependency
fingerprint. The nested candidate list in the object artifact is an exact
deterministic projection of these rows, not a top-k truncation.

### 8.3 `target_aware_photo_summary.parquet`

Grain and primary key: one `(source, flickr_photo_id, target key, decision-policy
fingerprint)`. It retains all route/scoring-unit IDs, chosen route, aggregate
target probability and competitor evidence, disagreement, final decision and
abstention, review priority, geo cluster, and upstream object-score
fingerprint. Larval, adult-field, and specimen scores are never averaged across
routes.

### 8.4 `target_aware_review_queue.parquet`

Grain: one immutable review request produced by a target-aware result. It
retains photo/scoring identity, durable media reference, route, decision,
abstention and priority reasons, target/competitor evidence, geo/candidate set,
quality flags, all model/data/policy fingerprints, and review status. It does
not convert a Flickr query term into a label.

## 9. Fingerprint dependency graph

The minimum identity chain is:

1. `registry_fingerprint` covers accepted taxonomy and source versions.
2. `geographic_spread_fingerprint` covers reconciled spread rows, registry,
   grid configuration, and source snapshots.
3. `flickr_cluster_fingerprint` covers exact candidate metadata hashes,
   geography projection, and cluster configuration.
4. `candidate_set_fingerprint` covers target, cluster, all candidate rows,
   relationship graph, registry, and geographic evidence.
5. `reference_bank_fingerprint` covers candidate metadata, committed content
   hashes, licence decisions, duplicate resolution, manual review decisions,
   route assignments, and exclusions.
6. `support_manifest_fingerprint` covers the frozen eligible rows and split.
7. `model_fingerprint` covers exact foundation-model ID/revision/checkpoint,
   preprocessing/transformation identity, embedding dimension/dtype, and
   normalization policy.
8. `reference_embedding_fingerprint` covers support and model fingerprints plus
   all sorted reference embedding rows.
9. `prototype_fingerprint` covers reference embeddings, prototype method,
   grouping, centering/normalization, seed, and vectors.
10. `training_data_fingerprint` covers exact labels, feature rows/schema,
    split, candidate/prototype/model inputs, and leakage policy.
11. `classifier_fingerprint` covers training data, split, feature order,
    estimator configuration, class mapping, and validated numeric arrays.
12. `calibrator_fingerprint` covers classifier, independent calibration
    predictions, split, method/classes, and validated parameters.
13. `decision_policy_fingerprint` covers calibrator, learned thresholds,
    optimization objective, route, coverage/OOD rules, and abstention policy.
14. Each target score work identity covers source image content, visual-input
    version, route, complete candidate set, reference/prototype/model,
    classifier, calibrator, prompt, and decision-policy fingerprints.

A mismatch at any required edge fails clearly. A new timestamp, URI, retry,
worker, or S3 multipart layout does not invalidate semantic computation.

## 10. Legacy object-score migration boundary

Current `object_bioclip_scores.parquet` rows and
`butterfly-cascade-output-v1.0.0` rows remain immutable historical artifacts.
The target-aware writer uses the new filenames and schema family above. It must
not append to, rewrite, cast, rename, or relabel a legacy dataset.

In particular:

- legacy `species_top1_score`, rerank scores, and rank scores remain raw
  similarities or historical classifier outputs; none is backfilled into
  `calibrated_target_probability`;
- legacy `crop_hash` remains historical identity and is not synthesized for
  whole-frame target-aware inputs;
- legacy family/genus pruning traces are retained for audit and never imported
  as target-aware candidate deletion evidence;
- completion of legacy work keys is not completion evidence for target-aware
  work;
- old triage buckets and thresholds keep their historical meaning.

An analytical version-aware union may expose a small common identity projection
only. It must add `score_schema_family`, `source_schema_version`,
`source_artifact_uri`, and `source_artifact_sha256`. Target-aware-only fields are
null for legacy rows. If an analyst requests a normalized legacy similarity, it
is named `legacy_species_top1_similarity`, never probability or confidence.
The union does not produce target-aware decisions for legacy rows.

Readers select a handler from the exact schema version. Unknown, mixed, or
unsupported versions fail closed. There is no row-level migration that invents
calibration, references, geography, full-frame evidence, or manual review.

Rollback retains the legacy registry/cache/output roots, work records, and the
previous capable release. Stopping new producers and pointing that release at
the old roots is rollback; deleting or rewriting either history is not.

## 11. Required artifact inventory

The migration is incomplete until these durable artifacts exist when their
phase applies:

| Domain | Required artifacts |
|---|---|
| Registry | `taxon_geographic_spread.parquet`, `taxon_geographic_summary.parquet` |
| Flickr geography | `flickr_geo_clusters.parquet`, `flickr_geo_assignments.parquet` |
| Candidate generation | `regional_taxon_occurrence.parquet`, `regional_candidate_species.parquet`, `competitor_relationships.parquet` |
| Reference acquisition | `reference_observations.parquet`, `reference_media_candidates.parquet`, `reference_acquisition_plan.parquet`, `reference_review_queue.parquet`, `reference_bank_summary.parquet`, `reference_bank_readiness.json` |
| Few-shot model | `reference_embeddings.parquet`, `reference_prototypes.parquet`, `visual_neighbour_species.parquet`, `few_shot_training_features.parquet`, `dataset_split_manifest.parquet`, `classifier_manifest.json`, `classifier_arrays.npz`, `calibration_artifacts.json`, `calibration_arrays.npz` |
| Inference | `target_aware_object_scores.parquet`, `target_aware_photo_summary.parquet`, `target_aware_review_queue.parquet` |
| Evaluation | `target_evaluation_metrics.json`, `target_evaluation_summary.md`, `target_calibration_bins.parquet`, `target_competitor_confusions.parquet`, `target_errors_for_review.parquet`, `target_metrics_by_geo_cluster.parquet`, `target_metrics_by_life_stage.parquet`, `target_metrics_by_visual_domain.parquet`, `target_ablation_results.parquet` |

Supporting normalized artifacts introduced in this contract are
`flickr_geography.parquet`, `reference_media_objects.parquet`,
`reference_review_decisions.parquet`, `reference_support_manifest.parquet`,
`flickr_embeddings.parquet`, `target_aware_candidate_scores.parquet`, and
`calibration_report.parquet`. They prevent mutable stage overloading and retain
the provenance required for later minimum artifacts.

Evaluation table schemas are finalized with reviewed-label schema v2 in Phase
13. Until then, every evaluation artifact must at least retain reviewed-label,
split, model, reference-bank, calibrator, and decision-policy fingerprints and
must reject Flickr query evidence as ground truth.

## 12. Implementation gates

Future schema implementations must prove, with deterministic tests, that:

- exact physical schemas and enum domains are enforced, including empty frames;
- sorting and semantic fingerprints are independent of worker completion order;
- invalid coordinates, dateline cells, `no_geo`, and remote-cluster rejection
  remain explicit;
- the target exists exactly once in every candidate set and no hierarchy text
  rank can remove it;
- only resolved verified references with accepted licences enter support;
- adult-field, larval, and specimen routes cannot share prototypes,
  classifiers, calibrators, or thresholds accidentally;
- exact/near duplicate and source-mirror groups cannot cross splits;
- model, reference, split, classifier, calibrator, and policy mismatches fail
  closed;
- NPZ loaders reject pickle/object arrays, unknown keys, and wrong shapes;
- no raw similarity or SVC margin is exposed as a probability;
- target-aware outputs are separate from immutable legacy object scores.

The reference-readiness manifest must pass before production Flickr detection,
embedding, or target-aware scoring begins. Geography may select and prioritize
candidates and support, but it never certifies an image label or forces the
target score to zero.
