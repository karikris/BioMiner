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
| `array<T,N>` | non-null fixed-size Arrow array with exactly `N` elements |
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

Canonical structured semantic preimages use
`biominer-canonical-semantic-v1`. The preimage begins with the fixed
`biominer-canonical-semantic\0v1\0` byte prefix. Every value then has a
one-byte type tag and an unsigned 64-bit big-endian payload length. Mapping
keys are UTF-8 strings sorted by their encoded key bytes. Lists and tuples
retain order and share the ordered-array tag. Integers use canonical ASCII decimal;
finite Python floats use little-endian IEEE-754 Float64 bytes, including the
sign bit of zero. NaN, infinity, naive datetimes, non-string mapping keys, and
unsupported types are rejected. Aware datetimes are normalized to UTC with
microsecond precision; dates retain their distinct type.

Set-like arrays are sorted by the owning contract before hashing. Embedding
and model arrays are C-contiguous little-endian arrays with a declared dtype
and shape; the manifest hashes their raw bytes and the complete NPZ file.
Canonical JSON remains the durable human-readable representation, not the
semantic hash preimage.

Moving Phase 6 identities from JSON-number text to this binary preimage is an
intentional cache break. Full-frame visual inputs, attention transforms,
evidence, unavailable variants, raw identity transforms, preprocessing and
quality policies use version 2. Target full-frame scoring units and embeddings
use version 3. OpenCLIP preprocessing attestations use version 2. Readers must
not reinterpret artifacts carrying the earlier versions under the new hash
algorithm.

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

Candidate generation is a union, not a hierarchy gate. It merges target,
same-genus range overlap, local same-family occurrence, reviewed relationship,
historical false-positive, and versioned visual-neighbour evidence. Country and
bioregion same-family rows are added only below the configured local-candidate
minimum. `no_geo` expands to the accepted same-family registry rather than
shrinking the comparison. A species can retain multiple sorted reasons.

`candidate_priority` is a deterministic ordinal derived after the union from
reason class, soft geographic evidence, occurrence support, taxonomy, and
accepted key. It authorizes ordering only. The set fingerprint covers the
complete sorted union, cluster, target, policy, and all registry, occurrence,
cluster, relationship, false-positive, and visual source versions.
Versioned visual-neighbour rows are filtered by the target as directed graph
edges and unioned into every geographic candidate set. Their source dependency
is recorded as `visual-neighbours:<graph_version>:<graph_fingerprint>`.
Supplying a graph never replaces target, geographic, taxonomic, mimic, or
historical-false-positive reasons; a species already present simply gains the
sorted `visually_nearest` reason and flag.
Geographic evidence uses fixed overlap weights `exact=1.0`, `buffer=0.8`,
`country=0.5`, `bioregion=0.35`, and `global=0.1`, multiplied by the
occurrence-count-weighted coordinate confidence. Missing confidence in any
selected source leaves the score null. The score remains soft evidence and is
stored as float32 before fingerprinting.

The BioCLIP candidate adapter contract is `object-bioclip-candidates-v2`.
It preserves the regional set ID, accepted key, target flag, ordinal, inclusion
reasons, and source versions on every candidate. In target-scope diagnostic
classification, family top one constrains the species-classification shortlist
when family metadata exists. The target is not injected into that shortlist.
Its raw all-candidate first-pass score and rank remain separate target-screening
fields. Every family-constrained shortlist member is reranked, while the
configured rerank width controls reporting rather than which shortlisted
species receive a rerank score. Target-scope object rows persist
`species-candidate-provenance-v2` for every species scored in the first pass,
including its regional provenance, first-pass rank and score, constrained
classification membership, family-match diagnostic, and rerank score when
compared. Historical v1 rows retain their target-injected/reordered membership
semantics and are not rewritten.

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
relationships additionally require nullable string fields `prototype_fingerprint`
and `model_fingerprint`, both populated with full lowercase SHA-256 fingerprints
for that relationship type and null otherwise. `reviewed_at` is a normalized UTC
ISO 8601 string. An enabled row must be reviewed with reviewer and timestamp
provenance; only one evidence version of a logical edge may be enabled. Genus
scope IDs are accepted genus scientific names, while species scope IDs are
accepted taxon keys. Curated relationships may influence inclusion and priority,
never establish an image label.

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

#### Prototype acquisition summaries

Build-week prototype planning additionally emits
`prototype_reference_source_summary.parquet` with schema version
`prototype-reference-source-summary-v1.0.0` and
`prototype_reference_shortfalls.parquet` with schema version
`prototype-reference-shortfalls-v1.0.0`. The source summary retains the
taxonomic or visual-domain scope, source, R1-R5 trust level, Layer A-E
geographic evidence, source/media/independent-observation counts, licence and
attribution qualification, and the count selected for download. The shortfall
artifact retains every requested scope, available and selected counts, route,
status, and reason. Visual-domain negatives use an explicit non-taxonomic
scope and are never placed in `candidate_accepted_taxon_key`.

Prototype source queries may set a positive `maximum_records` that is a
multiple of `page_size`. The bound is part of the query fingerprint, stops a
checkpoint cleanly at the configured count, and permits a time-bounded sample
when a provider's full result exceeds its interactive-search ceiling. The
report must disclose the bound and must not imply complete provider coverage.
Unbounded query fingerprints remain unchanged for checkpoint compatibility.

#### Prototype selection freeze

Task 14.3 freezes the metadata-qualified prototype choices in
`prototype_reference_selections.parquet` with schema version
`prototype-reference-selections-v1.0.0`. Each row carries an explicit
`candidate_scope_type`: biological rows use `accepted_taxon`, while curated
material and visual-domain negatives use `visual_domain`. Visual negatives
therefore never occupy `candidate_accepted_taxon_key`.

The freeze accepts R1-R4 evidence and rejects R5 unless a future attributable
manual promotion layer records the override. Every selected row must have an
acceptable licence, complete attribution, a compatible route, no known
identification conflict, and a unique `reference_observation_id`. Multiple
media from one observation cannot fill multiple support slots. Selection
scores remain ordinal priorities rather than probabilities, and documented
shortfalls remain preferable to duplicate-observation padding.

The same deterministic materialization emits
`prototype_reference_download_candidates.parquet` in the standard
`reference-media-candidates-v1.0.0` physical schema. It contains exactly one
row for every frozen prototype selection and no unselected media. Curated
visual-domain rows receive the same provider-derived observation and media ID
scheme as biological rows; their non-taxonomic scope remains in the prototype
selection ledger. The download boundary rejects stale media identity, URL,
licence, or attribution evidence before network access. The explicit prototype
licence policy keeps public-domain declarations distinct from CC0 while
allowing both when attribution evidence is complete.

### 5.4 `reference_media_objects.parquet`

This supporting artifact records committed source-image objects without
mutating candidate metadata. Grain and primary key: one
`reference_media_id`. Schema version: `reference-media-objects-v1.1.0`.

Required fields are `schema_version`, `reference_media_id`, `source_object_uri`,
`content_type`, `source_byte_count`, `decoded_width`, `decoded_height`,
`sha256`, `perceptual_hash`, `duplicate_group_id`, `duplicate_type`,
`canonical_reference_media_id`, `provider_mirror_ids`, `downloaded_at`,
`download_attempt_count`, `licence_policy_status`, `decode_status`,
`quarantine_reason`, and `object_fingerprint`. Provider relationships and every
source row remain available after deduplication.

Every valid decoded object has a lowercase `sha256:` content digest and a
versioned `dhash128-v1:` perceptual hash. The latter reads EXIF orientation,
resamples to a bounded 9x9 image with LANCZOS, applies the corresponding
orientation transform, composites transparency onto white, converts to
grayscale, and concatenates 64 horizontal with 64 vertical difference bits.
The preprocessing and hash version are part of the downloader and deduplication
policy fingerprints. Hashes from another version are rejected, not compared.

The downloader consumes acquisition selections only and never mutates media
candidates. It evaluates the configurable media-licence policy before network
access, distinguishes broadly reusable from research-only media, quarantines
missing or contradictory licence evidence, and retains attribution in the
committed checkpoint. Creative Commons codes and URIs must name a known suite
version and agree when both state a version.

Each provider policy names exact reviewed hosts, allowed schemes, URL rules,
and origin limits. iNaturalist downloads use sanctioned photo paths and the
configured image style. GBIF is an aggregator, so publisher media hosts are
configured explicitly; there is no arbitrary-public-host fallback. Default
ports and public DNS resolution are required where provider policy enables
network validation. Production connections disable environment proxies,
resolve on each new TCP connection, reject mixed public/private answers, and
dial only the validated numeric address while retaining the reviewed hostname
for the HTTP origin, Host header, TLS SNI, and certificate verification. The
injectable HTTP client is restricted to `MockTransport` test doubles; production
callers cannot replace this transport. Disabling public-address enforcement is
an explicit per-provider opt-out for reviewed private infrastructure.

`max_attempts` is one item-wide HTTP request budget and is never reset by a
redirect. `max_download_seconds` is an item-wide wall-clock deadline spanning
origin throttling, bounded DNS resolution, connect/TLS, response headers, body
streaming, and image validation. Production image decoding runs in a disposable
spawned process that is terminated when the remaining item budget expires;
per-operation `timeout_seconds` remains the stricter idle/connect bound where
applicable. A separate decode semaphore bounds concurrent decoder processes,
Linux children receive a configured address-space limit; on macOS, where that
limit cannot be lowered reliably, the parent polls the disposable child's RSS
and terminates it when the same configured ceiling is exceeded. Perceptual hash
alpha compositing and grayscale conversion operate on the 9x9 representation.
Palette and bilevel inputs use one temporary full-resolution RGB(A) or grayscale
buffer so Pillow can apply LANCZOS rather than its forced nearest-neighbour
resampler; pixel limits, the decode semaphore, and the platform-specific child
memory ceiling bound that buffer.

A source payload is eligible for commit only when its declared MIME type,
signature, and decoder format agree; it is a decodable single-frame raster
within byte and pixel limits; and any provider checksum matches. The downloader
records SHA-256, the perceptual hash, source bytes, and decoded dimensions. It
writes the
content-addressed object first, reads back its size and SHA-256, then writes the
per-media checkpoint. Resume revalidates the immutable input/policy binding and
the durable object's size and SHA-256 without another media request or object
overwrite. A valid row therefore means both object and checkpoint commits
succeeded; candidate lifecycle state remains a separate artifact.

Runs against one bank prefix may be incremental, but inventory compaction has a
single-writer invariant. Do not run concurrent download invocations against the
same prefix: each run reads, merges, validates, and atomically promotes the
cumulative inventory. Worker concurrency is internal to one invocation.

Intrinsic object state (`sha256`, `perceptual_hash`, decoded dimensions, and
object fingerprint) is committed by the downloader. Duplicate-group fields are
mutable derived inventory state. A checkpoint resume preserves those derived
fields only when the new committed row has the same intrinsic object
fingerprint; changed content clears them for recomputation.

Checkpoint v1 and object-inventory v1.0 rows are migrated explicitly. The
migrator revalidates the durable object's size and SHA-256, materializes that
object from storage, recomputes the perceptual hash in the isolated decoder,
and atomically promotes checkpoint v2 without another provider request.
Unselected legacy inventory rows use the same durable-object backfill during
the cumulative single-writer merge. Current source-byte and MIME policy is
checked before any legacy object storage access. Unknown, mixed, or malformed
legacy state fails closed.

Each invocation writes JSON and Markdown audit artifacts below its own
`reports/run_id=.../` prefix. The readable run component is bounded and paired
with a hash; generated run IDs are collision resistant. The Markdown summary is
promoted first and the JSON report last as the run-report commit marker. Fatal
validation, checkpoint, inventory, or storage errors use the same report schema
with `status=failed` and explicit `null` or `not_instrumented` metrics.

#### Duplicate relationship ledger

`reference_media_duplicate_relationships.parquet` stores direct duplicate and
provider-mirror evidence without deleting or coalescing candidate, observation,
or object provenance. Grain and primary key: one ordered media pair and
`duplicate_relationship_id`. Schema version:
`reference-media-duplicate-relationships-v1.0.0`.

Required fields are `schema_version`, `duplicate_relationship_id`,
`duplicate_group_id`, `canonical_reference_media_id`, ordered left/right media,
observation, source, and provider-media identifiers, `relationship_type`,
sorted `evidence_types`, `sha256_equal`, nullable
`perceptual_hash_distance`, `same_observation`, `provider_mirror`,
`resolution_status`, `policy_version`, and `policy_fingerprint`.

Exact SHA-256 equality is resolved duplicate evidence. Informative perceptual
matches within the stricter cross-observation threshold remain review-required;
same-observation matches within the configured threshold may be resolved as a
`resized_copy` or `near_identical_burst` when aspect ratios are compatible.
Low-information hashes, metadata conflicts, and visually incompatible provider
mirrors are never silently resolved. Licence identity includes an explicit
Creative Commons version, and component-level taxon/licence aggregation detects
conflicts even when the conflicting members are not adjacent in the sparse
ledger. The ledger stores a deterministic sparse spanning set of direct
threshold edges rather than a quadratic complete graph.
Dense Hamming neighborhoods beyond the configured bound fail closed. Connected
components provide leakage groups, while a linear, conservative varying-bit
distance upper bound prevents an A-B-C chain from being reported as a resolved
visual duplicate group.

Result validation binds every endpoint to the original normalized candidate and
observation inventories, requires candidate and observation sources to agree,
and requires each endpoint to belong to exactly one group. It recomputes exact,
perceptual, same-observation, and provider-mirror claims plus relationship type
and resolution from hashes, thresholds, informativeness, dimensions, metadata,
and provider provenance. Whole-result validation regenerates the deterministic
sparse ledger from intrinsic inputs, rejecting omitted required links, split
identical objects, and redundant noncanonical edges. Canonical selection is deterministic
and prefers allowed licences, accepted review and taxonomy evidence,
research-grade identity, configured source priority, larger decoded area,
larger source bytes, then media ID. Metadata-only
GBIF/iNaturalist aliases may participate in relationship evidence even when the
excluded mirror was never downloaded. All source IDs remain in the normalized
ledger and `provider_mirror_ids`; only the canonical ID identifies the preferred
physical object.

Deduplication publication uses an immutable, bounded run component. Annotated
objects and relationship Parquet are written below that run prefix, followed by
the Markdown summary; the run-scoped JSON report is written last as the commit
marker and records artifact URIs, bytes, and SHA-256 values. A failed
publication writes `status=failed` when report storage remains available and
never overwrites another run. The local writer stages a complete directory and
atomically renames it, so no path can expose a complete report with partial
Parquet state. Successful and failed cloud publications use the same structured
report shape and emit structured INFO start/completion/failure events. Their
Markdown includes command, run ID, PID, git SHA, timestamps, elapsed time,
input/output fingerprints, and artifact URIs.

### 5.5 Review state

`reference_review_queue.parquet` is a deterministic materialized queue; it is
not overwritten with decisions. Grain: one review request and media item;
primary key: `review_request_id`. Schema version:
`reference-review-queue-v1.0.0`. Its exact fields are `schema_version`,
`review_request_id`, `reference_media_id`, `reference_observation_id`,
`canonical_reference_media_id`, nullable `accepted_taxon_key`, nullable
`scientific_name`, `durable_preview_uri`, `media_object_fingerprint`,
`duplicate_group_id`, `source`, `provider_media_id`,
`provider_verification_status`, `creator`, `rights_holder`, `licence`,
`licence_uri`, `licence_policy_status`, `attribution`, `life_stage`,
`visual_domain`, `view`, `review_reason`, `review_priority`,
`required_review_count`, `review_status`, `created_at`,
`reference_bank_version`, and `input_fingerprint`.

`reference_review_queue_provenance.parquet` has one row per review request and
schema version `reference-review-queue-provenance-v1.0.0`. Its exact fields are
`schema_version`, `review_request_id`, `reference_media_id`,
`source_binding_fingerprint`, `source_leaf_fingerprints: list[str]`,
`queue_semantics_fingerprint`, `queue_row_fingerprint`, and
`input_fingerprint`. Source-leaf fingerprints are sorted, unique, full
lowercase SHA-256 values. The source-set fingerprint binds only evidence
reachable from that request, while the queue-row fingerprint covers every
immutable queue column and permits only the derived `review_status` projection
to change between imports.

Allowed queue `review_status` values are `pending`, `in_review`, `completed`,
`conflict`, `second_review_required`, and `cancelled`. The queue owns this
pending/conflict/second-review workflow projection; none of those states is a
human scientific disposition. Proposed `life_stage`, `visual_domain`, and
`view` values are nullable because the planner's provisional vocabulary is not
a human judgment. Any nonnull proposal uses the same closed vocabulary as a
decision; queue construction preserves null when upstream evidence cannot
supply a responsible mapping. In particular, it does not relabel planner
`unreviewed` or `field` values as reviewed `ambiguous` or `live_field` values.
The current workflow does not define an attributable cancellation record, so it
rejects a supplied `cancelled` state rather than trusting an unaudited queue
edit. The schema value is reserved for a later explicit cancellation command.

`reference_review_decisions.parquet` is the append-only scientific decision
record. Grain and primary key: one `review_decision_id`; sort by
`(reference_media_id, review_round, reviewed_at, review_decision_id)`. Schema
version: `reference-review-decisions-v1.0.0`.

Its exact fields are `schema_version`, `review_decision_id`,
`review_request_id`, `reference_media_id`, `review_round`, `verified_by`,
`reviewed_at`, `target_identity_verified: bool?`, `verification_status`,
`life_stage`, `visual_domain`, `view`, `review_confidence`, `review_notes`,
`exclusion_reason`, `second_review_required`,
`conflicts_with_decision_id`, and `decision_source_hash`. `verified_by` is the
single authoritative human actor for every disposition. A decision is never
updated in place; correction or disagreement produces a new decision row. The
base decisions writer uses atomic create-only publication and refuses an
existing target, including under concurrent first writers. Task 5.4 must merge
and validate retained history before publishing a new immutable ledger version
or append-only part.

`decision_source_hash` is the full lowercase SHA-256 of the canonical imported
source decision record, excluding transport paths and generated ledger IDs.
The importer recomputes it. It is deliberately excluded from the semantic
decision ID so delivery retries remain idempotent; merge validation must reject
an existing decision ID paired with a different source hash, and one source
hash cannot identify multiple semantic decisions.

Allowed values are:

- `life_stage`: `adult`, `larva`, `pupa`, `egg`, `unknown`;
- `visual_domain`: `live_field`, `pinned_specimen`, `artwork`, `logo`,
  `tattoo`, `partial_wing`, `dead_or_damaged_specimen`, `ambiguous`,
  `unsuitable`;
- `view`: `dorsal`, `ventral`, `lateral`, `frontal`, `oblique`, `unknown`;
- `verification_status`: `verified`, `excluded`, `uncertain`;
- `review_confidence`: `high`, `medium`, `low`, `unknown`.

Decision rows require a value from each `life_stage`, `visual_domain`, and
`view` vocabulary. In those fields, `unknown` means that the human inspected
the media but could not determine the value; null means no judgment and is not
valid in a human disposition. `review_confidence` is categorical review
metadata, not a calibrated probability or model score.

A `verified` disposition requires `target_identity_verified=true` and no
`exclusion_reason`. An `excluded` disposition requires a nonblank
`exclusion_reason`; its identity value is deliberately tri-state because an
image may depict the target and still be unusable. In particular,
`target_identity_verified=false` means that the target identity was
affirmatively disproved, not merely that the media was unsuitable. An
`uncertain` disposition requires `target_identity_verified=null`, nonblank
`review_notes`, and `second_review_required=true`.

The manual workflow has two commands:

```bash
biominer references export-review-queue \
  --acquisition-selections reference_acquisition_selections.parquet \
  --observations reference_observations.parquet \
  --media-candidates reference_media_candidates.parquet \
  --media-objects reference_media_objects.parquet \
  --duplicate-relationships reference_media_duplicate_relationships.parquet \
  --deduplication-report reference_media_deduplication_report.json \
  --reference-bank-version <version> \
  --output-dir <review-packet-directory> \
  --history-head <review-history-head>.json \
  [--include-research-only] \
  [--run-id <run-id>]
```

The export first validates the complete media-object, candidate, observation,
and relationship set against its committed deduplication report. It rejects a
truncated or mixed-run artifact set, including a canonical-only subset that
would otherwise resemble a legitimate singleton. It then collapses fully
resolved selected duplicate groups to their canonical media item. Selected
objects in an unresolved or conflicting group
remain separate review requests, so a provisional duplicate link cannot
transfer a human decision between media IDs. Each request retains
`source_object_uri` as `durable_preview_uri`; the export does not create an
untracked second image archive. The packet contains the validated queue, a
`reference_review_queue_provenance.parquet` companion, a typed decision
template, and an explicit typed empty `reference_review_decisions.parquet`
ledger. It is staged and published as one immutable,
create-only directory, so an existing packet is never silently replaced and a
failed export cannot expose a completed partial packet.
Each queue `input_fingerprint` combines the review-visible queue semantics with
a deterministic source-set binding. The companion records fingerprints for
every relevant selection, selected and canonical media record, source
observation, duplicate-group member, and duplicate relationship. Changing a
plan fingerprint, taxonomy, object identity, licence, source snapshot,
duplicate policy, proposal, or review quorum invalidates the request before any
retained decision is used; unrelated inventory rows do not. Operational changes
such as retrieval timestamps, transport URLs, queue timestamps, display
priority, or reason text change the packet-level queue-row fingerprint but do
not discard a still-applicable human decision.
Only `allowed` licence-policy rows are queued by default. The explicit
`--include-research-only` option permits scientific review of non-production
assets, but those rows remain blocked from production support selection.

Reviewers fill the exported Parquet decision template and import it with:

```bash
biominer references import-review-decisions \
  --review-queue <review-packet-directory>/reference_review_queue.parquet \
  --queue-provenance <review-packet-directory>/reference_review_queue_provenance.parquet \
  --decisions <completed-decisions>.parquet \
  --existing-decisions <review-packet-directory>/reference_review_decisions.parquet \
  --prior-review-report <authoritative-prior-report>.json \
  --history-head <review-history-head>.json \
  --output-dir <review-import-directory> \
  [--run-id <run-id>]
```

The decision input is Parquet with the exact physical schema returned by
`reference_review_decision_import_schema()`. CSV is not a supported review or
import format. The importer rejects unknown or missing fields, stale request or
media bindings, non-UTC timestamps, and invalid vocabulary values before it
publishes anything. It canonicalizes each source row, recomputes its source
hash and semantic decision ID, merges the required complete prior ledger, and
treats an identical re-delivery as idempotent. The importer proves that the
supplied queue is exactly the projection of that prior ledger before accepting
new rows; omitting conflict or uncertainty history is therefore rejected. The
initial export supplies the required typed empty ledger. Every continuation
must use the queue, provenance companion, and complete ledger from the latest
successful packet. The imported queue projection and complete
decision ledger are new immutable artifacts; neither the original queue nor a
prior decision row is updated in place.
Each import packet also stores the exact submitted
`reference_review_decision_import.parquet`. Its byte-bound frame fingerprint
and row count make the reported imported and idempotent-replay counts
recomputable; history-head advancement additionally binds the claimed existing
ledger count and fingerprint to the authoritative parent packet.

The history-head JSON is a mutable, lock-protected compare-and-swap pointer and
must live outside immutable packet directories. It pins the byte digest,
revision, path, and history ID of the only authoritative packet. Before import,
the CLI verifies the prior report against that head and verifies the queue,
provenance, and decision-ledger byte hashes recorded in the report. After the
new immutable packet commits, the head advances only if the parent digest is
still current. Reusing the root packet, a stale ancestor, a modified report, or
a recomputed sidecar is rejected; concurrent continuations can produce at most
one authoritative successor.
Root export requires a new, nonexistent head path and rejects a head inside the
output packet before writing any artifact. Revision zero uses
`reference_review_export_report.json`; revision one and later use the latest
`reference_review_import_report.json`.

The import packet also contains `reference_review_outcomes.parquet`,
`reference_review_conflicts.parquet`, `verified_reference_media.parquet`, and
`excluded_reference_media.parquet`. Outcomes are deterministic projections of
the queue and complete decision ledger. Conflict rows retain both open and
resolved conflict groups; correcting a dissent closes its prior group but does
not delete the audit history. Before publication, the writer recomputes every
projection, row count, and artifact fingerprint. The JSON report is written
last as the packet commit marker. Packet validation enforces every documented
artifact filename as well as its absolute URI, byte count, SHA-256 digest, and
Parquet projection.

Within each `(review_request_id, verified_by)` history, `review_round` is a
reviewer-local revision number. It starts at one and must be contiguous; a
correction is the next revision from that reviewer. Resolution retains every
revision for audit but uses only the latest revision from each reviewer as that
reviewer's effective judgment. Timestamps order evidence but cannot overwrite
a higher revision. Provider-supplied `verification_status` remains source
metadata exposed as `provider_verification_status`; importing a human decision
does not mutate it. Reviewer identifiers are canonical lowercase ASCII opaque
IDs; whitespace variants, Unicode confusables, and display names are rejected
instead of being counted as distinct reviewers. `verified_by` is an asserted
ledger identity, not a cryptographic signature: the production import
environment must authenticate the submitter and authorize that identity before
the decision Parquet reaches this command.

An effective `uncertain` judgment keeps the request unresolved and requests a
distinct second reviewer. A second review is another attributable decision,
not an in-place edit by the first reviewer. Effective judgments conflict when
their scientific disposition differs in verification status, target identity,
life stage, visual domain, or view. Differences in confidence or notes alone do
not create a conflict. Conflict is a derived queue state, not a human decision
status, and an explicit `conflicts_with_decision_id` remains provenance for the
disagreement. A majority never overwrites a dissenting effective scientific
judgment; the request stays `conflict` until a reviewer records a new revision
that removes the disagreement.

`conflicts_with_decision_id` is optional source provenance for the human row;
when populated it resolves to an earlier decision for the same request and
media from a distinct human. It does not turn `conflict` into a decision
disposition. The workflow derives normalized conflict groups and the queue's
conflict/second-review projection from the append-only decisions.
Verified and excluded files use schema
`reference-review-resolved-media-v1.0.0`. They preserve the immutable queue
proposal in `life_stage`, `visual_domain`, and `view`, and expose the human
determination separately as `resolved_life_stage`, `resolved_visual_domain`,
and `resolved_view`, with the resolved disposition, identity flag, effective
decision/reviewer IDs, and exclusion reasons. This keeps each request and input
fingerprint auditable without presenting the planner proposal as a human fact.
`select_verified_reference_media()` emits only media whose resolved human
disposition is `verified` with `target_identity_verified=true` and whose
per-item taxonomy, canonical identity, licence, attribution, duplicate state,
and visual-domain gates pass. Excluded, uncertain, incomplete, conflicting,
research-only, noncanonical, unresolved-duplicate, and prohibited-domain rows
are absent. Production consumers use its `resolved_life_stage`,
`resolved_visual_domain`, and `resolved_view` fields. Bank-level quota,
diversity, lifecycle separation, and split checks remain independent readiness
gates.

#### Curated visual-domain negative manifests

`reference_visual_domain_negative_manifest.parquet` is a local, manually
curated source ledger for explicit non-biological or unsuitable visual
evidence. Its schema version is
`reference-visual-domain-negative-manifest-v1.0.0`. The adapter accepts only a
strict JSON source document with schema version
`curated-visual-domain-negative-source-v1.0.0`; it performs no search, URL
discovery, HTTP request, or media download.

The closed negative categories are `artwork`, `logo`, `tattoo`,
`non_butterfly_insect_illustration`, `partial_wing`, and
`misleading_pattern`. Categories map deterministically to the existing visual
domains: non-butterfly insect illustrations remain `artwork`, and misleading
patterns remain `unsuitable`. `target_presence` is an independent manual label
with values `present`, `absent`, or `unknown`; a domain-negative label never
implies that the target is absent. This distinction is required for artwork,
tattoos, and partial wings that may depict or contain target morphology.

Every source row records a closed source kind, provider and record identity,
landing and media URIs, source snapshot, optional source SHA-256, curator
decision provenance, and explicit per-media rights metadata: creator, rights
holder, raw licence, licence URI, attribution, and a rights-evidence URI. The
compiler re-evaluates the raw licence with the central
`ReferenceLicencePolicy` and preserves its canonical licence, policy status,
reason, policy version, and policy fingerprint. Missing rights fields are
fatal. Research-only, quarantined, denied, pending, and excluded rows remain
auditable but cannot be enabled; an enabled row must be manually verified and
have policy status `allowed`. Pending rows use `review_confidence=unknown`;
verified and excluded rows require attributable review and a non-unknown
confidence; enabled rows require high or medium confidence.

The source and row schemas reject unknown fields, unknown categories, invalid
absolute URIs, conflicting review provenance, duplicate source identities,
duplicate media URIs, and duplicate supplied content hashes. Output is
deterministically sorted and binds every row to a stable source identity and a
complete row fingerprint. Publication is a create-only, atomic directory
operation that writes the Parquet ledger before
`reference_visual_domain_negative_manifest_report.json`. The report records
input and output fingerprints, licence/category/review counts, process and git
identity, and `network_requests = 0`. A failed validation or publication leaves
the output directory uncommitted and attempts to persist a sibling
`.failed.json` audit with the original error and timing metadata.

### 5.6 Frozen support and readiness

`reference_bank_split_assignments.parquet` has schema version
`reference-bank-split-assignments-v1.0.0`. It assigns each media item explicitly
to `support_train`, `model_selection`, `calibration`, or `final_test`, or records
an exclusion reason. There is no implicit support split. Every assignment binds
the split version, actor, timestamp, inclusion state, and semantic assignment
fingerprint. Those audit fields remain physical, but readiness projects split
semantics from media ID, split version, split, inclusion, and exclusion reason;
the actor, assignment timestamp, and transitive assignment fingerprint do not
change the frozen bank identity.

`reference_support_manifest.parquet` is the immutable resolved projection used
by embedding and split construction. Schema version
`reference-support-manifest-v2.0.0` uses an explicit semantic projection. It
excludes source-record, licence, and object locators; downloader object
fingerprints; queue request, decision, and reviewer IDs; and split-assignment
fingerprints from row and artifact semantic fingerprints. These fields remain
in every persisted row for retrieval and audit. Direct source snapshot,
source-record content, image-content, perceptual, taxonomy, review outcome,
licence, attribution, geography, route, split, and model-bank identities remain
semantic. Version 1 used locator-bearing audit provenance in its fingerprint
preimage and is not accepted as version 2. Its grain is one canonical verified
media item and route.

`reference_bank_summary.parquet` has one row per bank version, accepted species,
cluster scope, life stage, visual domain, and split. Schema version:
`reference-bank-summary-v1.0.0`. It records required, candidate, downloaded,
deduplicated, reviewed, verified, eligible, excluded, and shortfall counts;
source, licence, observer, observation, and geographic diversity counts; and
the reference-bank and support-manifest fingerprints.

`reference_bank_readiness.json` has schema version
`reference-bank-readiness-v2.0.0` and contains:

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
cannot be downgraded to a shortfall. The policy must contain at least one
cluster-scoped requirement for every geographic cluster in the candidate
union; an empty geographic requirement set is invalid rather than a vacuous
pass. Plans use the planner's composite candidate-union ID, while selections
remain bound to their regional source candidate-set IDs. Observation registry,
taxon/name, plan, selection, current media inventory, review queue, and review
source-leaf identities are cross-validated before prior manual decisions are
accepted.

Bank input fingerprints are computed from per-artifact allowlist projections.
Plan/selection/retrieval/download/queue/assignment timestamps, attempts, byte
counts, transport locators, review actors and queue-derived IDs/fingerprints
are excluded. Deduplication semantics are recomputed from the projected
observation, candidate, object, relationship, and versioned settings content
rather than trusting report hashes that may bind PID, git SHA, generation time,
or artifact URIs. Direct scientific content, source-snapshot, image, policy,
and model hashes remain semantic.

`reference_model_input_identity.json` uses schema version
`reference-model-input-identity-v2.0.0`. It persists the checkpoint URI for
retrieval and audit but excludes that relocatable URI from the semantic model
input fingerprint. The immutable model revision, checkpoint SHA-256, OpenCLIP
configuration, preprocessing contract, and input contract remain semantic.
Version 1 identities are rejected explicitly.

Compile and publish the three immutable readiness artifacts with:

```bash
biominer references validate-readiness \
  --candidate-species regional_candidate_species.parquet \
  --acquisition-plan reference_acquisition_plan.parquet \
  --acquisition-selections reference_acquisition_selections.parquet \
  --observations reference_observations.parquet \
  --media-candidates reference_media_candidates.parquet \
  --media-objects reference_media_objects.parquet \
  --duplicate-relationships reference_media_duplicate_relationships.parquet \
  --deduplication-report reference_media_deduplication_report.json \
  --review-queue reference_review_queue.parquet \
  --queue-provenance reference_review_queue_provenance.parquet \
  --review-decisions reference_review_decisions.parquet \
  --split-assignments reference_bank_split_assignments.parquet \
  --readiness-policy reference_bank_readiness_policy.json \
  --model-identity reference_model_input_identity.json \
  --registry-version <registry-version> \
  --reference-bank-version <bank-version> \
  --output-dir <immutable-readiness-directory>
```

The command is local and performs no network requests. It publishes blocked
business outcomes for audit and exits nonzero when vision is not permitted;
malformed or inconsistent inputs leave no committed output directory. A
non-dry production `run` that includes `detect_objects` or `score_bioclip`
requires both `--reference-bank-readiness <immutable-readiness-directory>` and
`--reference-bank-readiness-sha256 <trusted-sha256:-digest>`. The publication
command prints that digest; it must be pinned outside the rewritable artifact
directory. The orchestrator verifies the pin, artifact checksums, semantic
fingerprints, target, registry, and the runtime's complete independently
declared model-input identity before invoking a vision-stage handler or any
scoring call. A scorer missing model version, checkpoint SHA-256,
preprocessing version, input-contract version, or model-input fingerprint is
rejected. Dry runs and non-vision stage subsets do not require readiness
artifacts or a digest pin.

## 6. Embedding and prototype artifacts

Adult field, larval, and pinned-specimen routes are separate in every support,
prototype, classifier, calibrator, and threshold identity.

### 6.1 `reference_embeddings.parquet`

Grain and primary key: one frozen support row and full-frame visual input,
identified by `(support_row_fingerprint, visual_input_id)`. Sort by accepted
taxon, route, cluster, split, media ID, visual-input kind, content hash,
transformation version, transformation fingerprint, and visual-input ID.
Schema version: `reference-embeddings-v2.0.0`. Version 2 replaces the planned
version 1 physical schema before production publication; it preserves multiple
effective review decisions, uses fixed-width vectors, and makes model and
preprocessing attestations independently reconstructable.

The exact ordered fields are:

```text
schema_version, registry_version, reference_bank_version,
reference_media_id, reference_observation_id, source_snapshot_version,
review_decision_ids: list<str>, duplicate_group_id, readiness_sha256,
reference_bank_fingerprint, support_manifest_fingerprint,
model_input_fingerprint, input_contract_version, support_row_fingerprint,
accepted_taxon_key, scientific_name, geo_cluster_id, life_stage,
visual_domain, view, route, source_object_uri, source_image_sha256,
source_object_fingerprint, visual_input_id, visual_input_kind,
raw_image_content_hash, image_content_hash, transformation_version,
transformation_policy_fingerprint, transformation_fingerprint,
model_input_schema_version, model_name, model_version, model_id,
model_revision, model_checkpoint_uri, model_weights_sha256,
model_checkpoint_hash, model_fingerprint, preprocessing_version,
preprocessing_fingerprint, open_clip_version, open_clip_config_sha256,
preprocessing_attestation_version, preprocessing_config_json,
preprocessing_attestation_fingerprint, embedding_dimension: u32,
embedding: array<f32, embedding_dimension>, embedding_norm: f64,
support_split, embedding_created_at: ts, embedding_fingerprint
```

The vector length must equal `embedding_dimension`; all values must be finite;
the vector must be unit-normalized; and the stored norm must equal the norm of
the persisted Float32 vector. Embedding fingerprints encode `embedding_norm`
as little-endian Float64 followed by vector values as little-endian Float32.
They exclude embedding creation time, readiness bytes, source/model locator
URIs, queue review-decision IDs, and the downloader's locator-bearing object
fingerprint. The projected support-row identity, source snapshot and image
content hashes, transformation identity, model weights/configuration, and
preprocessing attestation remain semantic.

`model_fingerprint` is distinct from `model_input_fingerprint`: it hashes the
model-input fingerprint, embedding dimension, the declared `float32` dtype,
and `l2-unit-normalize-before-float32-persist-v1` normalization policy under
`reference-embedding-model-v1`. Only eligible rows from the frozen support
manifest are accepted.

Resumable local build state uses
`reference-embeddings-checkpoint-v2.0.0`. Its work identity excludes the
readiness object's byte checksum, relocatable source/model object URIs, and
regenerated review/object/split audit provenance. Every immutable checkpoint
part retains its own byte checksum and semantic fingerprint. On resume,
validated vectors are rebound to the current readiness and support-manifest
audit provenance before further model work or publication. Version 1
checkpoints are rejected explicitly.

### 6.2 `reference_prototypes.parquet`

Grain and primary key: one prototype ID. Sort by route, species, cluster scope,
life stage, visual domain, prototype kind, view, method, and ID. Schema version:
`reference-prototypes-v2.0.0`; version 1 artifacts are rejected rather than
silently interpreted without member or clustering provenance.

Required fields are `schema_version`, `prototype_id`, `accepted_taxon_key`,
`species` (the accepted scientific name), `cluster_scope_type`, `geo_cluster_id`, `life_stage`,
`visual_domain`, `view`, `route`, `visual_input_kind`, `prototype_method`, `prototype_group_id`,
`prototype_kind`, `metadata_group_id`, `embedding_cluster_id`,
`clustering_method`, `clustering_configuration_fingerprint`, the explicit
clustering threshold/minimum/maximum fields, `member_observation_ids`,
`member_observation_fingerprints`,
`reference_count`, `independent_observation_count`,
`balanced_sampling_seed`, `mean_centered`, `embedding_dimension`, `embedding`,
`embedding_norm`, `centering_fingerprint`, `model_fingerprint`, `reference_embedding_fingerprint`,
`support_manifest_fingerprint`, and `prototype_fingerprint`.

Prototype fitting consumes `support_train` embeddings only. Calibration,
model-selection, and final-test vectors do not enter a prototype, its centering
mean, or `reference_embedding_fingerprint`; that fingerprint identifies the
exact sorted `support_train` embedding subset consumed by the build. The full
frozen support-manifest fingerprint remains attached for split and bank
provenance.

Every independent observation contributes one effective vector per route and
visual-input kind. Multiple licensed views or media rows from the same
observation are averaged and normalized first, so a burst or provider mirror
cannot overweight the class centroid. `normalized_mean` then averages those
observation vectors and L2-normalizes the persisted Float32 prototype.

`simpleshot_mean_centered` uses one global centering context per route and
visual-input kind. The context takes an equal number of independent
observations from every species, selected deterministically with
`balanced_sampling_seed`; subtracts that unnormalized global mean from support
and query embeddings; normalizes the centered vectors; and then produces the
normalized class mean. The row records the exact `centering_fingerprint`.
Raw `normalized_mean` rows have null `balanced_sampling_seed` and
`centering_fingerprint`, because their identity must not change with a seed
they do not consume. A context with fewer than two species emits raw centroids
only. A zero-norm centered group is omitted and counted in the structured build
log rather than materializing an invalid vector.

Global rows use `cluster_scope_type = "global"`, `geo_cluster_id = "all"`, and
aggregate `prototype_kind = "aggregate"` rows use `view = "all"`. Regional
rows retain their actual cluster. Prototypes never mix routes or visual-input
kinds. Query centering reconstructs the same context
from the fingerprinted training embeddings and fails on a zero-norm result.
All stored prototypes are finite, unit-normalized Float32 arrays. Prototype
fingerprints encode semantic fields with the canonical binary identity
contract, followed by little-endian Float64 norm and little-endian Float32
vector bytes.

`prototype_kind = "metadata"` rows retain one concrete reviewed view and are
built at both global and regional scopes when their independent-observation
minimum is met. Aggregate fitting still collapses all licensed views from one
biological observation to one effective vector. Metadata fitting instead
collapses media to one vector per observation and reviewed view, so dorsal and
ventral evidence can form separate prototypes without a same-view burst gaining
extra weight.

`prototype_kind = "embedding_cluster"` rows are permitted only beneath one
persisted metadata parent with one accepted taxon key, life stage, visual
domain, route, view, visual-input kind, and geographic scope. Version
`deterministic_average_linkage_cosine_v1` sorts observations canonically,
performs bounded average-linkage clustering on cosine distance, folds
undersized clusters into the nearest valid cluster, and caps the final cluster
count. It emits clusters only when at least two minimum-size groups remain;
otherwise the metadata centroid is the explicit fallback. The configured
distance threshold, group/cluster minima, cluster maximum, observation bound,
configuration fingerprint, and exact sorted member IDs/fingerprints are
persisted. Sibling clusters must be disjoint and together cover the complete
metadata parent.

Embedding clustering is dependency-neutral in Phase 7 and is bounded by the
configuration rather than pulling the Phase 8 ML stack into registry-only
installations. It runs only inside a verified species metadata group. An
unsupervised cluster can split visual modes; it can never create, infer, or
change an accepted species label.

#### Nearest-reference evidence contract

`ReferenceEvidenceIndex` validates the frozen embedding and prototype artifacts
once and can then score repeated queries. A query carries its unit embedding,
route, full-frame visual-input kind, geographic cluster, and model fingerprint.
Every supplied accepted species is scored; family or genus evidence cannot
remove a candidate. Adult, larval, pupal, egg, and pinned-specimen observations
remain in separate route/input indexes.

Reference media are first collapsed to one unit vector per independent
biological observation. `support_count` and `local_support_count` therefore
count observations, not files or views. For each candidate, observations in
the query's real geographic cluster are deterministically selected first and
global observations fill the remainder. Selection is capped at the same
configured `balanced_reference_count` for every class and is ranked by the
seed, accepted taxon key, observation ID, route, and visual-input kind, never
by the embedding value. `no_geo` has no fabricated local pool and falls back
to global support.

The scorer returns nearest support similarity, fixed top-three and top-five
means, the normalized centroid of the selected balanced pool, persisted local
and global prototype similarities, and distance to the nearest independent
observation. The distance is cosine distance `1 - nearest_support_similarity`
and lies in `[0, 2]`. Top-three or top-five values are null when fewer than
three or five usable independent observations exist; the scorer never averages
a shorter list under those names. It records selected IDs, usable and selected
counts, local availability, and explicit insufficiency reasons.

Raw and SimpleShot mean-centered methods use the matching prototype method.
Mean-centered scoring reconstructs and attests the same seeded centering
context used by prototype fitting. Query/support zero directions are rejected
or reported as unusable rather than assigned a fabricated cosine. Optional
observation and duplicate-group exclusions prevent self-reference; affected
prototype means are recomputed from the retained observations. All returned
cosines remain similarities in `[-1, 1]`, including negative values. None are
probabilities or confidence estimates.

### 6.3 `visual_neighbour_species.parquet`

Grain: one directed species-neighbour edge, route, full-frame visual-input kind,
and graph version. Schema version: `visual-neighbour-species-v1.0.0`; algorithm
version: `global-aggregate-cosine-knn-v1`.

Each row records graph/configuration fingerprints and configured top-k/threshold;
edge ID; subject and neighbour accepted keys/names; route and visual-input kind;
prototype kind/method; best subject and neighbour prototype IDs; raw cosine
similarity and one-based neighbour rank; supporting prototype-pair count and
pair structs; embedding dimension; model, support-embedding, support-manifest,
and complete prototype-artifact fingerprints; and edge fingerprint. One graph
fingerprint is repeated on every row and binds the sorted edge fingerprints,
graph version, configuration, algorithm, and exact prototype artifact.

The initial graph deliberately compares exactly one `global` / `all` /
`aggregate` prototype per accepted species within each route and visual-input
contract. Metadata and embedding-cluster prototypes cannot give a class more
extreme-value opportunities. Same-species and cross-route pairs are prohibited.
Candidate neighbours are sorted by descending cosine similarity, then accepted
taxon key and prototype ID; equal scores therefore have stable ranks. A row is
emitted only above the configured similarity floor and within the configured
species-level top-k. Similarities remain `[-1, 1]` evidence, never probability.

The directed graph adds `visually_nearest` candidate evidence for the graph
subject. It cannot remove geographically plausible, taxonomic, curated mimic,
historical false-positive, or mandatory target candidates.

### 6.4 `flickr_embeddings.parquet`

This supporting inference cache has one source photo, full-frame visual input,
and model/preprocessing identity per row. It records source/photo ID,
source-record hash, visual-input ID/kind/version, raw image content hash,
transformation and model fingerprints, finite embedding and norm, route and
quality metadata, creation time, and embedding fingerprint. No spatial crop
hash is its image identity. Raw full-image embeddings are reused across routes
and detections when their transformation identity is identical.

### 6.5 Versioned taxonomic prompt ensembles

Target-aware text evidence uses prompt schema
`taxonomic-prompt-ensemble-v1.1.0` and prompt version
`bioclip-taxonomic-prompts-v1.0.0`. One ensemble is bound to an accepted species
key, accepted scientific name, route, life stage, root-to-species path, taxonomy
source/version/fingerprint, sorted evidence exclusions, an explicit geography
policy, and an ensemble fingerprint over that complete semantic record.

The deterministic built-in variants are the accepted scientific name, a literal
butterfly-species description, a route-compatible life-stage description, a
genus/family description, and the full accepted taxonomic path. The
`pinned_specimen` variant exists only for the specimen route. Each emitted prompt
retains its versioned template ID, accepted key, route, stage, evidence kind and
record ID, source, trust tier/language or reviewer state/identity, an explicit
`geography_bearing=false` marker, and its own semantic fingerprint.

Vernacular variants require a supported vernacular name class and trust tier
T1, T2, or T3, and always pair the vernacular with the accepted scientific name.
Generated translations, T4/T5 assertions, weak homonyms, rejected/disabled
names, and unsupported name classes are excluded. Free-form prompt aliases
require an accepted human review state and reviewer identity; route/stage
mismatches and specimen aliases outside the specimen route are excluded. Raw
Flickr query/search terms are not an input to this builder. Reviewed aliases
marked as geography-bearing are also excluded and must use the explicit
geographic ablation contract below. Exclusion reason and source-record identity
remain fingerprinted audit evidence rather than silently becoming a prompt.

Legacy classifiers may project an ensemble to ordered `PromptVariant` labels,
but prompt-score pooling is a separate, versioned operation. Prompt generation
does not average or otherwise collapse text evidence.

### 6.6 Prompt pooling and diagnostics

Prompt pooling results use schema `prompt-ensemble-pooling-result-v1.2.0` and
algorithm version `bioclip-prompt-pooling-v1.1.0`. Pooling consumes normalized
image and per-prompt text embeddings from one model fingerprint. The raw
per-prompt values are cosine similarities in `[-1,1]`; the label-set softmax
from the legacy BioCLIP worker is not a substitute and is not called raw
similarity.

Every run names exactly one strategy: normalized mean text embedding, maximum
prompt similarity, mean of the best two prompt similarities, or a normalized
text embedding built from learned prompt weights. Embedding pooling and score
pooling are distinct in the artifact. A normalized mean first normalizes every
text vector, averages the selected vectors, renormalizes the result, and only
then takes image cosine; it is not arithmetic mean similarity. Maximum and
best-two strategies operate on raw per-prompt cosines. Learned nonnegative
weights must name exactly every selected prompt variant, normalize to one, and
carry model, ensemble, subset, model-selection split, and artifact
fingerprints. Final-test-derived weights are invalid.

Selection is controlled by a fingerprinted route, life-stage, and visual-domain
subset policy. The default stage/domain builder includes accepted taxonomy
prompts but does not silently add vernacular or reviewed-alias prompts; those
families require explicit inclusion. Geography-bearing kinds are rejected by
generic subset selection unless a separate geography-ablation flag is true and
matches the ensemble's ablation fingerprint. Every result persists every
ensemble prompt's label, kind, template, variant fingerprint, raw similarity,
geography marker, subset and contribution flags, pooling weight, and selection
reason, including prompts outside the active subset. It also binds normalized
image/text embedding-set, model, ensemble, subset, optional weight-artifact,
geography-ablation, and result fingerprints. The result also carries the
accepted taxon key and scientific name from its ensemble, so downstream
evaluation cannot relabel a valid pooled result as a different candidate
species.

### 6.7 Structured geography and prompt ablations

The default prompt policy is `structured_evidence_only`. A normal taxonomic
ensemble contains no geographic evidence object, no geography-bearing prompt,
and no ablation fingerprint. Country, administrative region, bioregion,
locality, occurrence overlap, and distance remain structured candidate or
model features from the versioned geography artifacts in Sections 2 through 4.
They are not appended to visual text and do not modify visual similarity.

`structured-geographic-prompt-evidence-v1.0.0` is a narrow, fingerprinted
reference used only to define an experiment. It carries the accepted taxon key,
scope type and ID, display name and language, optional ISO country code, source
artifact and schema version, source record ID and fingerprint, and its own
semantic fingerprint. It neither creates a range assertion nor replaces the
source geography row. Accepted source contracts are the versioned taxon spread,
taxon summary, regional occurrence, and regional candidate artifacts; Flickr
query-hit or free-form metadata records are invalid sources.

A geography-bearing prompt can be derived only through
`bioclip-geographic-prompt-ablation-v1.0.0`. The builder requires a literal
explicit opt-in, a named ablation, a geography-free base ensemble, and structured
evidence for the same accepted taxon. It emits one marked variant and binds the
base ensemble fingerprint, complete evidence payload, prompt fingerprint, and
ablation fingerprint. Nested ablations, mismatched taxa, arbitrary location
strings, and geography-marked reviewed aliases in the normal builder fail
closed.

Default stage/domain pooling excludes the ablation even when its raw diagnostic
similarity is available. A subset that contains the geographic prompt kind must
separately enable the exact linked ablation. Pooling records whether that flag
was enabled, the ablation fingerprint, the selected geographic-prompt count,
and per-prompt `geography_bearing` state. This keeps country-conditioned text an
auditable validation experiment rather than a production morphology signal;
scenery or background correlation cannot enter the default score silently.

### 6.8 Prompt evaluation and validation-only selection

Prompt benchmarks use report schema
`taxonomic-prompt-evaluation-report-v1.0.0` and evaluation version
`taxonomic-prompt-evaluation-v1.0.0`. Input grain is one evaluation item,
dataset split, prompt configuration, and accepted candidate species. Each row
binds the split and candidate-set fingerprints, route, life stage, visual
domain, validated pooling result, and optional independent reference-image
cosine plus its evidence fingerprint. A pooling result's accepted key must
equal the candidate key.

Every configuration must evaluate the same item and candidate sets within a
split. Target identity, route/stage/domain, candidate order, and reference-image
evidence must be identical across configurations. Duplicate candidates,
missing targets, fewer than two species, incomplete configuration coverage, or
changed reference evidence fail closed. Prompt configurations explicitly state
whether common-name, taxonomic-path, or geographic-ablation prompts are
enabled; selected prompt diagnostics must agree with those flags.

Per-item metrics are defined as follows:

- `target_prompt_rank` is the one-based position of the highest-scoring active
  target prompt among every active candidate prompt. Ordering is descending raw
  cosine, then accepted candidate key and prompt-variant fingerprint.
- `target_species_rank` orders pooled candidate-species scores by descending
  cosine and then accepted key. Species recall@k is the fraction of items whose
  target species rank is at most k.
- `target_versus_competitor_text_margin` is the target pooled score minus the
  highest pooled score from a different accepted species. The winning
  competitor key and score remain in the item result.
- Prompt/reference correlation is Spearman rank correlation between pooled text
  scores and independent reference-image scores. Ties use average ranks. Fewer
  than two pairs or a constant rank vector yields null rather than a fabricated
  zero. Reports retain both pooled candidate-level correlation and the mean of
  defined per-item correlations.

Configuration summaries include overall metrics plus separate life-stage,
visual-domain, and life-stage-by-domain slices. Common-name and taxonomic-path
effects require matched baseline/treatment configurations differing only in
the named prompt family. Shared prompts must retain the same ensemble identity
and raw cosine, baseline prompts cannot disappear, and every added prompt must
belong to the named family. Paired item count and deltas for rank, recall@k,
margin, and reference correlation are fingerprinted. The default benchmark
requires both effect families to be measurable and one model fingerprint across
all configurations.

Prompt promotion uses selection schema `prompt-version-selection-v1.0.0` and
policy `validation-only-prompt-selection-v1.0.0`. Selection accepts only the
`model_selection` partition and excludes geography-bearing ablation
configurations. It maximizes configured species recall@k, then text margin,
then minimizes mean species rank and prompt rank, with configuration
fingerprint as the final deterministic tie break. The selection-input
fingerprint covers only eligible validation summaries; adding or changing
`final_test` results cannot alter either the chosen configuration or that input
fingerprint.

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

The builder accepts only typed, normalized evidence rather than arbitrary source
columns. `MODEL_FEATURE_COLUMNS` is an explicit allowlist disjoint from label
and provenance columns; query terms, query-definition IDs, discovery taxon
keys, and source labels do not exist in the physical schema. Labels remain in
the artifact solely as supervised outcomes.

Reference competitor features use balanced-pool centroid similarity. Each named
competitor category is reduced independently by maximum similarity.
`target_minus_best_competitor_margin` subtracts the maximum across regional,
same-genus, historical-false-positive, and family-negative categories from the
target centroid similarity; domain negatives retain their own margin. Target
prototype distance is `1 - target_global_prototype_similarity` and nearest-
support distance is `1 - target_nearest_reference_similarity`. Text margin
uses the analogous target-minus-best-competitor subtraction. Absent inputs
produce null outputs rather than zeroes or shortened fixed-k statistics.

Geographic distances are kilometres and must be non-negative. A `no_geo` row
has `missing_geo=true` and null regional overlap/distance values. Image width,
height, short side, long side, megapixels, and a short-side-below-224 indicator
are derived from the immutable visual input. The feature-schema fingerprint
binds the ordered model allowlist, dtypes, embedding dimension, derivation
versions, and prohibited-source list. The training-data fingerprint binds all
sorted row fingerprints. Leakage, observation, owner, duplicate, burst, and
provider-mirror groups may each occur in only one dataset split.

#### Nonparametric baseline runtime contract

`NonparametricBaselineIndex` consumes the validated `support_train` subset of
`reference_embeddings.parquet` and the matching complete
`reference_prototypes.parquet`. Construction validates and indexes immutable
reference evidence; it does not fit an sklearn estimator and exposes no
`fit()` operation. Every prediction records the exact model, support manifest,
support-embedding artifact, and prototype-artifact fingerprints.

The supported methods are `nearest_centroid`,
`mean_centered_nearest_centroid`, `top_k_nearest_neighbors`, and
`multi_prototype_nearest_class`. The first two use persisted global aggregate
prototypes. The mean-centered method reconstructs the seeded SimpleShot
centering context from the same support artifact, verifies every persisted
centering fingerprint, and applies that exact context to the query. A query or
duplicate-group exclusion causes the affected class centroid to be recomputed
from retained independent observations.

The nearest-neighbour method uses one vector per independent biological
observation and an exact positive `k`. It uses unweighted class votes; cosine
sum and nearest cosine are deterministic tie evidence only. It never shortens
`k`, treats negative cosine as a valid similarity rather than a negative vote,
or calls vote fraction a probability. Candidate key is the final stable tie
breaker.

Multi-prototype scoring uses the finest global persisted representation for
each reviewed metadata group: embedding-cluster children when present,
otherwise the metadata centroid, and the aggregate centroid only when no
metadata prototype exists. The class score is the maximum cosine over those
prototypes. Prototypes containing an excluded observation or duplicate group
are ineligible; clusters are never rebuilt during inference.

All methods receive the complete caller-supplied candidate union and isolate
support by YOLOE route and full-frame visual-input kind. A candidate with no
matching support/prototype, an undersized exact-k pool, a model mismatch, or an
unavailable SimpleShot context cannot be silently deleted. Recoverable
coverage failures return an abstained result with stable reason codes; invalid
artifact/query contracts fail closed. Outputs call raw values `raw_score` and
`raw_margin`, never confidence or probability.

#### Conventional classifier training runtime contract

`train_frozen_embedding_classifiers` validates the complete Task 8.2 feature
artifact, selects one task, target and route, and fits estimators only on
eligible `support_train` rows. `model_selection` rows are used only to compare
already refitted candidates; `calibration` and `final_test` labels are never
passed to an estimator, fold splitter, hyperparameter scorer, or model selector.
The run records separate consumed-partition and source-artifact fingerprints.

The default comparison is versioned as
`frozen-embedding-search-grid-v1`:

- L2-regularised logistic regression over the frozen embedding, with
  `C in {0.01, 0.1, 1, 10}`;
- `LinearSVC` over the frozen embedding, with the same `C` grid;
- `LinearSVC` over embedding plus structured evidence, with the same grid.

The optional bounded-pilot RBF SVC uses the embedding only,
`C in {0.1, 1, 10}`, and `gamma in {scale, 0.01, 0.1}`. It is absent unless
explicitly enabled and fails before fitting when the configured sample cap is
exceeded. SVC probability fitting is disabled. Logistic outputs and SVC
decision functions remain uncalibrated model outputs; every candidate records
`probability_calibrated=false`.

Structured input starts from the Task 8.2 model-feature allowlist. Frozen unit
embeddings pass through unchanged. Nullable continuous evidence is median
imputed and standardised inside each cross-validation pipeline. Boolean
indicators pass through as zero/one. Route, full-frame input kind and YOLOE
route use fixed complete enum one-hot columns. Open-ended quality flags use 32
fixed SHA-256 buckets under `visual-quality-flag-sha256-buckets-v1`, so their
vocabulary cannot leak from a held-out fold. Labels and provenance columns are
not materialised in the numeric matrix.

Hyperparameter selection uses balanced accuracy as the primary metric and
macro F1 as the secondary metric. `StratifiedGroupKFold` receives
`leakage_group_id` through `GridSearchCV`, with a deterministic seed and
bounded positive `n_jobs`. Before fitting, every class must have at least one
independent group per fold. Every generated fold is audited for complete class
coverage, disjoint train/validation groups, complete group coverage and exactly
one validation appearance per group. A group carrying conflicting labels is
fatal. Explicit class weights must name exactly the fitted classes.
Held-out and cross-validation metric ties prefer embedding-only `LinearSVC`,
then structured `LinearSVC`, then logistic regression, with pilot RBF last.

Binary and larval target-verifier labels are target key versus
`__non_target__`; regional multiclass uses reviewed accepted taxon keys; visual
domain uses the reviewed domain label. BioCLIP remains frozen. sklearn and
NumPy are imported lazily only when training begins, preserving registry-only
installations. Task 8.4 returns in-memory fitted pipelines and transparent
training metadata. `write_frozen_classifier` is the only supported Task 8.5
persistence boundary: it extracts a fitted logistic-regression or `LinearSVC`
candidate's numeric state and never serializes the pipeline or estimator
object. RBF pilot candidates remain in-memory comparison evidence and cannot
be persisted as linear artifacts.

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

The implemented physical schema is `dataset-split-manifest-v1.0.0` and adds
the general `stratification_label`, source, item fingerprint, configuration
fingerprint, assignment-policy version, four integer partition weights,
class-coverage policy, derived `leakage_component_id`, and component size.
`DatasetSplitItem` keeps source-observation, source-owner, observer,
photographer, Flickr-owner, generic duplicate, exact-hash, perceptual-duplicate,
burst, provider-mirror, and geographic-cluster identities separate for audit.
Person, owner, observation, and burst identities are source-scoped; global
duplicate, provider-mirror, and real geographic-cluster identities may connect
providers. `no_geo` means missing evidence and is never treated as one shared
group.

`transitive-multi-identity-leakage-groups-v1.0.0` computes connected components
across every populated identity using union-find. The assignment unit is the
complete connected component, including indirect chains through different
identity types. `deterministic-class-aware-component-allocation-v1.0.0` orders
components by class rarity, class breadth, size, and a seeded semantic hash,
then greedily minimises exact rational deviations from the configured item,
component, and class targets. Forward reservations prevent an early choice
from consuming the last component needed by an uncovered partition. Production
weights are 55/15/15/15 for support train, model selection, calibration, and
final test. Defaults require every class in all four partitions and fail before
assignment when a class has fewer than four independent components; all four
partitions must be non-empty even when that class-coverage check is explicitly
disabled.

The split fingerprint binds the configuration and every sorted item
fingerprint, derived component, and assignment. Readers validate the exact
physical schema and ordering, rebuild item and component fingerprints, rerun
the allocator, check every group namespace for cross-split reuse, and recompute
the split fingerprint. Publication is immutable by default and verifies the
Parquet round trip before returning.

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

The implementation writes little-endian Float64 coefficients, intercepts,
imputation statistics, scaler means/scales/variances, and little-endian Int64
class indices. Archive members are uncompressed NPY payloads with deterministic
names and timestamps. The canonical manifest is published last into a newly
created immutable directory. `load_frozen_classifier` rejects extra files,
symlinks, duplicate JSON keys, non-canonical JSON, oversized archives, duplicate
or compressed ZIP members, object arrays, unknown keys, non-finite values,
incorrect class order, and every checksum, dtype, shape, feature-layout, or
parent-fingerprint mismatch before returning a non-sklearn linear decision
function. Structured inputs apply only the persisted median and standard-scaler
statistics; embeddings and fixed indicator columns pass through unchanged.

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

The implemented calibration input is an explicit held-out prediction record:
prediction and source-item IDs, leakage-component ID, fold index, exact
`calibration` partition, true ordered class, estimator decision score vector,
and positive sample weight. Every fold audit binds sorted estimator-fit and
validation group sets. Validation requires the two sets to be disjoint within
each fold, requires every prediction group in exactly one validation fold, and
requires each fold's declared validation groups to equal its prediction groups.
The independent-prediction fingerprint covers the score rows and complete fold
group sets; the persisted fold summaries retain their counts and group-set
fingerprints without exposing person or owner identifiers.

`auto` fitting resolves to sigmoid calibration for binary tasks and temperature
scaling for multiclass tasks; `auto` is never persisted as an artifact method.
Sigmoid fitting uses Platt-smoothed targets and deterministic Newton steps with
line search. Isotonic fitting is rejected below 1,000 held-out predictions or
200 independent leakage components and stores its monotone knots and values as
little-endian Float64 arrays. Multiclass temperature fitting minimizes weighted
log loss over a bounded log inverse-temperature with a deterministic
golden-section search. Only the fit path imports scikit-learn for isotonic PAVA;
the strict runtime loader implements sigmoid, piecewise-linear interpolation,
and softmax scaling using numeric parameters alone.

Reliability rows use `probability_kind=calibrated_probability`, include every
ordered class and fixed equal-width bin (including empty bins), and never rename
the input `estimator_decision_score` as probability. The manifest is the commit
marker after deterministic NPZ and Parquet publication. Before policy fitting,
the writer carries a fingerprinted `not_fitted` decision-policy record with
`target_confirmation_enabled=false`; that pending record is not threshold
provenance.

A fitted selective policy uses schema version
`few-shot-decision-policy-v1.0.0`. Its calibration-only samples retain the
independent group ID, target label, calibrated target probability, competitor
margin, sample weight, and the route, reference-coverage, domain-negative, OOD,
visual-detail, and geo gates. Policy fitting searches the joint grid of observed
target-probability and competitor-margin thresholds. It maximizes weighted
target recall subject to an explicit weighted target-precision objective, with
deterministic ties resolved by precision, coverage, unweighted precision, and
then stricter thresholds. There is no default `0.90` threshold. If no joint
threshold satisfies the objective, the policy is persisted as `infeasible`
with null thresholds and target confirmation disabled.

The immutable policy record persists its status and version, both learned
thresholds, optimization metric and precision objective, achieved weighted and
unweighted calibration metrics, sample/group/class/grid counts, model,
classifier, calibrator, split, and sample fingerprints, all runtime requirements
and ordered abstention rules, and a fingerprint over that complete record.
Calibration loading recomputes the policy fingerprint and rejects identity,
threshold, metric, requirement, or rule tampering.

## 8. Target-aware inference artifacts

The production mode string is
`target_aware_few_shot_classification`. It does not overload
`target_scope_object_screening` or hierarchical cascade output.

### 8.0 `object_detections.parquet` routing prerequisite

The detector input to every target-aware scorer uses schema version
`object-detection-v2`. Version 2 retains the v1 image, bounding-box, score,
crop, model, status, and failure fields and adds immutable detector and routing
provenance:

- `detector_prompt: str?`, `detector_class_id: i32?`, and
  `detector_prompt_set_fingerprint: str?` preserve the normalized prompt,
  actual YOLOE result class ID, and order-sensitive prompt map used to decode
  that class ID;
- `mask_polygon_xyn: list<list<f64>>?` preserves the aligned instance-mask
  contour in normalized image coordinates without persisting a raw bitmap
  mask;
- `detection_route`, `routing_action`, nullable `bioclip_route`,
  `routing_priority`, and `routing_reason` preserve the route decision;
- `routing_policy_version` and `routing_policy_fingerprint` bind the complete
  possible-adult and ambiguous-review policy, including enable flags and
  inclusive score thresholds.

The closed route matrix is:

| Detection route | Routing action | BioCLIP comparison route | Contract |
|---|---|---|---|
| `adult_butterfly_field` | `score` | `adult_field` | Definite adults and possible adults meeting the configured recall threshold |
| `caterpillar_field` | `score` | `larval` | Larval evidence only; never an adult comparison |
| `pinned_specimen` | `score` | `pinned_specimen` | Specimen evidence only; never a live-field comparison |
| `ambiguous_visual_domain` | `review` or `exclude` | `adult_field` only for retained review | Configured low-priority recall review; review rows are not scored |
| `pupa_or_chrysalis` | `exclude` | null | Retained as a separate visual domain pending a compatible route |
| `possible_moth_or_other_insect` | `exclude` | null | Retained non-target insect evidence |
| `artwork_logo_tattoo_or_other_artifact` | `exclude` | null | Retained visual-artifact evidence |
| `no_relevant_organism` | `exclude` | null | No BioCLIP work |

`no_detection` means the detector produced no retained candidate and maps to
`no_relevant_organism`. Image decode/load failures and inference failures are
separate statuses and map fail-closed to `ambiguous_visual_domain`; they are
not rewritten as biological absence. Unknown raw prompts also fail closed even
when a legacy coarse label appears positive. Route-aware NMS suppresses only
overlapping detections in the same route so adult, larval, and specimen
evidence cannot erase one another.

The default BioCLIP gate is `routed_visual_domain`. A score requires a detected
row, valid routing-policy identity, `routing_action = score`, an exact
detection/comparison-route pair, and a comparison route explicitly supported
by the active scorer. `routing_action = review` is retained but never scored.
No-detection whole-image fallback is forbidden in this mode. The older
`butterfly_like_only` and `exclude_hard_negative` gates are explicit diagnostic
compatibility modes only. Detection and scoring work identities cover the
ordered prompt-set fingerprint, routing-policy fingerprint, gate mode, and
supported comparison routes; retry and lease metadata remain excluded.

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

Explicit non-match aggregation uses version
`regional-nonmatch-scoring-v1.0.0`. Candidate evidence contains exactly one
target taxon and at most one row per accepted candidate key. Every raw reference
value is named `cosine_similarity`, constrained to `[-1,1]`, and bound to one
shared score-contract fingerprint before maxima or margins are computed.
Calibrated values are separately constrained to `[0,1]` and require task,
classifier, and calibrator fingerprints. Target probabilities may come from the
matching binary/larval verifier or regional multiclass model; competitor
probabilities require the regional multiclass task; visual-domain probabilities
require the visual-domain task.

The scorer computes `best_target_reference_score`,
`best_known_competitor_score`, and `best_domain_negative_score` from comparable
raw reference evidence. `competitor_margin` is target reference similarity
minus best known-competitor reference similarity and is null unless both are
available. The required `best_non_target_score` is the maximum of the best raw
competitor score, best raw domain-negative score, and generic calibrated
non-target classifier evidence. Because that formula can cross raw-similarity
and probability scales, it is never labelled probability: the output records
the winning score kind, a heterogeneous-scale marker, evidence kind, and
evidence ID.

`nonmatch_margin` is computed only as calibrated target probability minus the
largest calibrated non-target probability across regional/nonregional
competitors, domain negatives, and generic target-verifier non-target evidence.
Raw and calibrated competitor winners retain separate taxon identities because
they need not be the same species. Generic non-target evidence yields
`abstain`, not a fabricated biological category. Maxima use stable taxon and
evidence-ID tie breaks, and the scoring fingerprint covers the complete sorted
input evidence plus every derived value. This layer exposes the closed decision
vocabulary; the calibrated selective policy applies the learned joint
target-probability and competitor-margin thresholds plus its fail-closed gates.

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
| Reference acquisition | `reference_observations.parquet`, `reference_media_candidates.parquet`, `reference_acquisition_plan.parquet`, `reference_media_objects.parquet`, `reference_media_duplicate_relationships.parquet`, `reference_review_queue.parquet`, `reference_bank_summary.parquet`, `reference_bank_readiness.json` |
| Few-shot model | `reference_embeddings.parquet`, `reference_prototypes.parquet`, `visual_neighbour_species.parquet`, `few_shot_training_features.parquet`, `dataset_split_manifest.parquet`, `classifier_manifest.json`, `classifier_arrays.npz`, `calibration_artifacts.json`, `calibration_arrays.npz` |
| Inference | `target_aware_object_scores.parquet`, `target_aware_photo_summary.parquet`, `target_aware_review_queue.parquet` |
| Evaluation | `target_evaluation_metrics.json`, `target_evaluation_summary.md`, `target_calibration_bins.parquet`, `target_competitor_confusions.parquet`, `target_errors_for_review.parquet`, `target_metrics_by_geo_cluster.parquet`, `target_metrics_by_life_stage.parquet`, `target_metrics_by_visual_domain.parquet`, `target_ablation_results.parquet` |

Supporting normalized artifacts introduced in this contract are
`flickr_geography.parquet`, `reference_media_objects.parquet`,
`reference_media_duplicate_relationships.parquet`,
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
