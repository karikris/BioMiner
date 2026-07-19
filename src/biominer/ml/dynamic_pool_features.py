"""Immutable model features over frozen dynamic-pool evidence and labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_review import QUERY_TIERS, RAW_SCORE_SEMANTICS
from biominer.evaluation.dynamic_pool_splits import (
    DYNAMIC_POOL_EVALUATION_SPLITS,
    validate_dynamic_pool_evaluation_splits,
)
from biominer.references.schemas import REFERENCE_ROUTES, REFERENCE_VISUAL_DOMAINS
from biominer.storage.parquet import write_parquet


DYNAMIC_POOL_FEATURE_SCHEMA_VERSION = "dynamic-pool-feature-table-v1.0.0"
DYNAMIC_POOL_FEATURE_FILE = "dynamic_pool_features.parquet"
DYNAMIC_POOL_FEATURE_DERIVATION_VERSION = "raw-evidence-explicit-missingness-one-hot-v1"
PROBABILITY_AVAILABLE = False

_ROUTES = tuple(sorted(REFERENCE_ROUTES))
_VISUAL_DOMAINS = tuple(sorted(REFERENCE_VISUAL_DOMAINS))
_QUERY_TIERS = tuple(sorted(QUERY_TIERS))
_BASE_FEATURE_NAMES = (
    "global_prototype_similarity",
    "global_nearest_reference_similarity",
    "global_top_k_mean_similarity",
    "raw_competitor_margin",
    "family_similarity",
    "family_similarity_available",
    "family_margin_to_next_raw",
    "family_margin_to_next_raw_available",
    "family_rank",
    "family_rank_available",
    "local_prototype_similarity",
    "local_prototype_similarity_available",
    "local_nearest_reference_similarity",
    "local_nearest_reference_similarity_available",
    "local_top_k_mean_similarity",
    "local_top_k_mean_similarity_available",
    "prototype_absolute_disagreement",
    "prototype_absolute_disagreement_available",
    "nearest_absolute_disagreement",
    "nearest_absolute_disagreement_available",
    "top_k_absolute_disagreement",
    "top_k_absolute_disagreement_available",
    "prototype_rank_movement",
    "prototype_rank_movement_available",
    "nearest_rank_movement",
    "nearest_rank_movement_available",
    "top_k_rank_movement",
    "top_k_rank_movement_available",
    "no_geo",
    "local_evidence_available",
    "route_compatible",
    "quality_flag_count",
    "global_support_coverage_fraction",
    "global_top_k_coverage_fraction",
    "global_observation_independence_fraction",
    "global_reference_count",
    "global_configured_reference_count",
    "global_independent_observation_count",
    "global_reference_shortfall_count",
    "local_support_coverage_fraction",
    "local_support_coverage_fraction_available",
    "local_top_k_coverage_fraction",
    "local_top_k_coverage_fraction_available",
    "local_observation_independence_fraction",
    "local_observation_independence_fraction_available",
    "local_reference_count",
    "local_configured_reference_count",
    "local_independent_observation_count",
    "local_reference_shortfall_count",
    "subject_area_ratio",
    "subject_area_ratio_available",
    "query_hit_count",
    "query_text_similarity",
    "query_text_similarity_available",
    "query_text_margin",
    "query_text_margin_available",
)
DYNAMIC_POOL_MODEL_FEATURE_NAMES = (
    *_BASE_FEATURE_NAMES,
    *(f"route={value}" for value in _ROUTES),
    *(f"visual_domain={value}" for value in _VISUAL_DOMAINS),
    *(f"query_tier={value}" for value in _QUERY_TIERS),
)


DYNAMIC_POOL_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "feature_derivation_version": pl.String,
    "feature_schema_fingerprint": pl.String,
    "feature_table_fingerprint": pl.String,
    "feature_row_fingerprint": pl.String,
    "feature_input_fingerprint": pl.String,
    "item_id": pl.String,
    "item_fingerprint": pl.String,
    "source_record_hash": pl.String,
    "source_artifact_fingerprint": pl.String,
    "review_decision_fingerprint": pl.String,
    "split_fingerprint": pl.String,
    "evaluation_split": pl.String,
    "independence_component_id": pl.String,
    "candidate_species_key": pl.String,
    "human_supported": pl.Boolean,
    "sampling_weight": pl.Float64,
    "score_component_fingerprint": pl.String,
    "model_fingerprint": pl.String,
    "reference_evidence_fingerprint": pl.String,
    "query_fingerprint": pl.String,
    "score_semantics": pl.String,
    "probability_available": pl.Boolean,
    "global_prototype_similarity": pl.Float64,
    "global_nearest_reference_similarity": pl.Float64,
    "global_top_k_mean_similarity": pl.Float64,
    "raw_competitor_margin": pl.Float64,
    "family_similarity": pl.Float64,
    "family_rank": pl.UInt32,
    "family_margin_to_next_raw": pl.Float64,
    "local_evidence_available": pl.Boolean,
    "local_evidence_unavailable_reason": pl.String,
    "local_prototype_similarity": pl.Float64,
    "local_nearest_reference_similarity": pl.Float64,
    "local_top_k_mean_similarity": pl.Float64,
    "prototype_absolute_disagreement": pl.Float64,
    "nearest_absolute_disagreement": pl.Float64,
    "top_k_absolute_disagreement": pl.Float64,
    "prototype_rank_movement": pl.Int32,
    "nearest_rank_movement": pl.Int32,
    "top_k_rank_movement": pl.Int32,
    "geographic_cluster_id": pl.String,
    "no_geo": pl.Boolean,
    "route": pl.String,
    "visual_domain": pl.String,
    "route_compatible": pl.Boolean,
    "quality_flag_count": pl.UInt32,
    "global_support_coverage_fraction": pl.Float64,
    "global_top_k_coverage_fraction": pl.Float64,
    "global_observation_independence_fraction": pl.Float64,
    "global_reference_count": pl.UInt32,
    "global_configured_reference_count": pl.UInt32,
    "global_independent_observation_count": pl.UInt32,
    "global_reference_shortfall_count": pl.UInt32,
    "local_support_coverage_fraction": pl.Float64,
    "local_top_k_coverage_fraction": pl.Float64,
    "local_observation_independence_fraction": pl.Float64,
    "local_reference_count": pl.UInt32,
    "local_configured_reference_count": pl.UInt32,
    "local_independent_observation_count": pl.UInt32,
    "local_reference_shortfall_count": pl.UInt32,
    "subject_area_ratio": pl.Float64,
    "primary_query_tier": pl.String,
    "query_hit_count": pl.UInt32,
    "query_text_similarity": pl.Float64,
    "query_text_margin": pl.Float64,
    "feature_count": pl.UInt32,
    "feature_names": pl.List(pl.String),
    "feature_vector": pl.List(pl.Float64),
}


@dataclass(frozen=True, slots=True)
class DynamicPoolFeatureInput:
    """Raw evidence for one reviewed candidate; no field is a probability."""

    item_id: str
    candidate_species_key: str
    score_component_fingerprint: str
    model_fingerprint: str
    reference_evidence_fingerprint: str
    query_fingerprint: str
    global_prototype_similarity: float
    global_nearest_reference_similarity: float
    global_top_k_mean_similarity: float
    raw_competitor_margin: float
    local_evidence_available: bool
    local_evidence_unavailable_reason: str | None
    geographic_cluster_id: str | None
    no_geo: bool
    route: str
    visual_domain: str
    route_compatible: bool
    quality_flag_count: int
    global_support_coverage_fraction: float
    global_top_k_coverage_fraction: float
    global_observation_independence_fraction: float
    global_reference_count: int
    global_configured_reference_count: int
    global_independent_observation_count: int
    global_reference_shortfall_count: int
    local_reference_count: int
    local_configured_reference_count: int
    local_independent_observation_count: int
    local_reference_shortfall_count: int
    primary_query_tier: str
    query_hit_count: int
    family_similarity: float | None = None
    family_rank: int | None = None
    family_margin_to_next_raw: float | None = None
    local_prototype_similarity: float | None = None
    local_nearest_reference_similarity: float | None = None
    local_top_k_mean_similarity: float | None = None
    prototype_absolute_disagreement: float | None = None
    nearest_absolute_disagreement: float | None = None
    top_k_absolute_disagreement: float | None = None
    prototype_rank_movement: int | None = None
    nearest_rank_movement: int | None = None
    top_k_rank_movement: int | None = None
    local_support_coverage_fraction: float | None = None
    local_top_k_coverage_fraction: float | None = None
    local_observation_independence_fraction: float | None = None
    subject_area_ratio: float | None = None
    query_text_similarity: float | None = None
    query_text_margin: float | None = None

    def __post_init__(self) -> None:
        for field in ("item_id", "candidate_species_key"):
            object.__setattr__(
                self, field, _required_text(getattr(self, field), field=field)
            )
        for field in (
            "score_component_fingerprint",
            "model_fingerprint",
            "reference_evidence_fingerprint",
            "query_fingerprint",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        for field in (
            "global_prototype_similarity",
            "global_nearest_reference_similarity",
            "global_top_k_mean_similarity",
            "family_similarity",
            "local_prototype_similarity",
            "local_nearest_reference_similarity",
            "local_top_k_mean_similarity",
            "query_text_similarity",
        ):
            object.__setattr__(
                self,
                field,
                _optional_bounded_float(
                    getattr(self, field), field=field, minimum=-1.0, maximum=1.0
                ),
            )
        for field in (
            "global_prototype_similarity",
            "global_nearest_reference_similarity",
            "global_top_k_mean_similarity",
        ):
            if getattr(self, field) is None:
                raise ValueError(f"{field} must be available")
        for field in (
            "raw_competitor_margin",
            "family_margin_to_next_raw",
            "query_text_margin",
        ):
            object.__setattr__(
                self,
                field,
                _optional_bounded_float(
                    getattr(self, field), field=field, minimum=-2.0, maximum=2.0
                ),
            )
        if self.raw_competitor_margin is None:
            raise ValueError("raw_competitor_margin must be available")
        for field in (
            "prototype_absolute_disagreement",
            "nearest_absolute_disagreement",
            "top_k_absolute_disagreement",
        ):
            object.__setattr__(
                self,
                field,
                _optional_bounded_float(
                    getattr(self, field), field=field, minimum=0.0, maximum=2.0
                ),
            )
        for field in (
            "global_support_coverage_fraction",
            "global_top_k_coverage_fraction",
            "global_observation_independence_fraction",
            "local_support_coverage_fraction",
            "local_top_k_coverage_fraction",
            "local_observation_independence_fraction",
            "subject_area_ratio",
        ):
            object.__setattr__(
                self,
                field,
                _optional_bounded_float(
                    getattr(self, field), field=field, minimum=0.0, maximum=1.0
                ),
            )
        for field in (
            "global_support_coverage_fraction",
            "global_top_k_coverage_fraction",
            "global_observation_independence_fraction",
        ):
            if getattr(self, field) is None:
                raise ValueError(f"{field} must be available")
        for field in (
            "quality_flag_count",
            "global_reference_count",
            "global_configured_reference_count",
            "global_independent_observation_count",
            "global_reference_shortfall_count",
            "local_reference_count",
            "local_configured_reference_count",
            "local_independent_observation_count",
            "local_reference_shortfall_count",
            "query_hit_count",
        ):
            object.__setattr__(self, field, _uint32(getattr(self, field), field=field))
        for field in (
            "family_rank",
            "prototype_rank_movement",
            "nearest_rank_movement",
            "top_k_rank_movement",
        ):
            value = getattr(self, field)
            if field == "family_rank":
                normalized = (
                    None if value is None else _positive_uint32(value, field=field)
                )
            else:
                normalized = None if value is None else _int32(value, field=field)
            object.__setattr__(self, field, normalized)
        for field in ("local_evidence_available", "no_geo", "route_compatible"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be boolean")
        cluster = _optional_text(
            self.geographic_cluster_id, field="geographic_cluster_id"
        )
        if self.no_geo and cluster is not None:
            raise ValueError("no_geo feature input cannot claim a geographic cluster")
        if not self.no_geo and cluster is None:
            raise ValueError(
                "georeferenced feature input requires geographic_cluster_id"
            )
        object.__setattr__(self, "geographic_cluster_id", cluster)
        reason = _optional_text(
            self.local_evidence_unavailable_reason,
            field="local_evidence_unavailable_reason",
        )
        local_optional = (
            self.local_prototype_similarity,
            self.local_nearest_reference_similarity,
            self.local_top_k_mean_similarity,
            self.local_support_coverage_fraction,
            self.local_top_k_coverage_fraction,
            self.local_observation_independence_fraction,
        )
        if self.local_evidence_available:
            if reason is not None or any(value is None for value in local_optional):
                raise ValueError(
                    "available local evidence requires every local value and no reason"
                )
        elif reason is None or any(value is not None for value in local_optional):
            raise ValueError(
                "unavailable local evidence requires a reason and null local values"
            )
        object.__setattr__(self, "local_evidence_unavailable_reason", reason)
        if (self.family_similarity is None) != (self.family_rank is None):
            raise ValueError("family similarity and rank must be available together")
        if (
            self.family_similarity is None
            and self.family_margin_to_next_raw is not None
        ):
            raise ValueError("family margin requires family evidence")
        local_derived = (
            self.prototype_absolute_disagreement,
            self.nearest_absolute_disagreement,
            self.top_k_absolute_disagreement,
            self.prototype_rank_movement,
            self.nearest_rank_movement,
            self.top_k_rank_movement,
        )
        if self.local_evidence_available and any(
            value is None for value in local_derived
        ):
            raise ValueError("available local evidence requires disagreement evidence")
        if not self.local_evidence_available and any(
            value is not None for value in local_derived
        ):
            raise ValueError("unavailable local evidence cannot claim disagreement")
        if self.no_geo and self.local_evidence_available:
            raise ValueError("no_geo feature input cannot claim local evidence")
        route = _choice(self.route, field="route", allowed=REFERENCE_ROUTES)
        domain = _choice(
            self.visual_domain,
            field="visual_domain",
            allowed=REFERENCE_VISUAL_DOMAINS,
        )
        query_tier = _choice(
            self.primary_query_tier,
            field="primary_query_tier",
            allowed=QUERY_TIERS,
        )
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "visual_domain", domain)
        object.__setattr__(self, "primary_query_tier", query_tier)
        if self.global_configured_reference_count == 0:
            raise ValueError("global_configured_reference_count must be positive")
        if self.global_reference_count > self.global_configured_reference_count:
            raise ValueError("global reference count exceeds configured count")
        if self.global_independent_observation_count > self.global_reference_count:
            raise ValueError("global independent count exceeds reference count")
        if (
            self.global_reference_shortfall_count
            != self.global_configured_reference_count - self.global_reference_count
        ):
            raise ValueError("global reference shortfall is inconsistent")
        if self.local_reference_count > self.local_configured_reference_count:
            raise ValueError("local reference count exceeds configured count")
        if self.local_evidence_available and self.local_configured_reference_count == 0:
            raise ValueError("available local evidence requires configured support")
        if self.local_independent_observation_count > self.local_reference_count:
            raise ValueError("local independent count exceeds reference count")
        if (
            self.local_reference_shortfall_count
            != self.local_configured_reference_count - self.local_reference_count
        ):
            raise ValueError("local reference shortfall is inconsistent")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": DYNAMIC_POOL_FEATURE_SCHEMA_VERSION,
                "score_semantics": RAW_SCORE_SEMANTICS,
                **asdict(self),
            }
        )


@dataclass(frozen=True, slots=True)
class DynamicPoolFeatureBuild:
    table: pl.DataFrame
    feature_schema_fingerprint: str
    feature_table_fingerprint: str
    row_count: int
    feature_count: int
    split_row_counts: tuple[tuple[str, int], ...]


def dynamic_pool_feature_schema_fingerprint() -> str:
    """Fingerprint the exact raw-to-vector feature contract."""

    return canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_FEATURE_SCHEMA_VERSION,
            "derivation_version": DYNAMIC_POOL_FEATURE_DERIVATION_VERSION,
            "score_semantics": RAW_SCORE_SEMANTICS,
            "model_feature_names": DYNAMIC_POOL_MODEL_FEATURE_NAMES,
            "route_categories": _ROUTES,
            "visual_domain_categories": _VISUAL_DOMAINS,
            "query_tier_categories": _QUERY_TIERS,
            "missing_value_semantics": (
                "raw_null_plus_availability_indicator_then_model_vector_zero"
            ),
            "label_columns_excluded_from_vector": ("human_supported",),
        }
    )


def build_dynamic_pool_feature_table(
    inputs: Sequence[DynamicPoolFeatureInput],
    split_manifest: pl.DataFrame,
) -> DynamicPoolFeatureBuild:
    """Join raw evidence to the complete frozen split without moving labels."""

    validate_dynamic_pool_evaluation_splits(split_manifest)
    if isinstance(inputs, str | bytes | bytearray):
        raise TypeError("inputs must be a sequence of DynamicPoolFeatureInput values")
    items = tuple(inputs)
    if not items:
        raise ValueError("dynamic-pool feature input must not be empty")
    if any(not isinstance(item, DynamicPoolFeatureInput) for item in items):
        raise TypeError("inputs must contain DynamicPoolFeatureInput values")
    by_id = {item.item_id: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("dynamic-pool feature item_id must be unique")
    manifest_ids = set(split_manifest["item_id"].to_list())
    if set(by_id) != manifest_ids:
        raise ValueError("feature inputs must exactly cover the frozen split manifest")
    schema_fingerprint = dynamic_pool_feature_schema_fingerprint()
    rows = []
    for split_row in split_manifest.iter_rows(named=True):
        item = by_id[str(split_row["item_id"])]
        if item.candidate_species_key != split_row["candidate_species_key"]:
            raise ValueError("feature candidate key differs from frozen split")
        vector = _feature_vector(item)
        base = {
            "feature_input_fingerprint": item.fingerprint,
            "item_fingerprint": split_row["item_fingerprint"],
            "split_fingerprint": split_row["split_fingerprint"],
            "evaluation_split": split_row["evaluation_split"],
            "independence_component_id": split_row["independence_component_id"],
            "review_decision_fingerprint": split_row["review_decision_fingerprint"],
            "human_supported": split_row["human_supported"],
            "sampling_weight": split_row["sampling_weight"],
            "feature_schema_fingerprint": schema_fingerprint,
            "feature_vector": vector,
        }
        rows.append(
            {
                "schema_version": DYNAMIC_POOL_FEATURE_SCHEMA_VERSION,
                "feature_derivation_version": (DYNAMIC_POOL_FEATURE_DERIVATION_VERSION),
                "feature_schema_fingerprint": schema_fingerprint,
                "feature_table_fingerprint": "",
                "feature_row_fingerprint": canonical_semantic_fingerprint(base),
                "feature_input_fingerprint": item.fingerprint,
                "item_id": item.item_id,
                "item_fingerprint": split_row["item_fingerprint"],
                "source_record_hash": split_row["source_record_hash"],
                "source_artifact_fingerprint": split_row["source_artifact_fingerprint"],
                "review_decision_fingerprint": split_row["review_decision_fingerprint"],
                "split_fingerprint": split_row["split_fingerprint"],
                "evaluation_split": split_row["evaluation_split"],
                "independence_component_id": split_row["independence_component_id"],
                "candidate_species_key": item.candidate_species_key,
                "human_supported": split_row["human_supported"],
                "sampling_weight": split_row["sampling_weight"],
                **asdict(item),
                "score_semantics": RAW_SCORE_SEMANTICS,
                "probability_available": PROBABILITY_AVAILABLE,
                "feature_count": len(DYNAMIC_POOL_MODEL_FEATURE_NAMES),
                "feature_names": list(DYNAMIC_POOL_MODEL_FEATURE_NAMES),
                "feature_vector": list(vector),
            }
        )
    rows.sort(key=lambda row: str(row["item_id"]))
    table_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_FEATURE_SCHEMA_VERSION,
            "feature_schema_fingerprint": schema_fingerprint,
            "split_fingerprint": split_manifest["split_fingerprint"].item(0),
            "row_fingerprints": [row["feature_row_fingerprint"] for row in rows],
        }
    )
    for row in rows:
        row["feature_table_fingerprint"] = table_fingerprint
    table = pl.DataFrame(rows, schema=DYNAMIC_POOL_FEATURE_SCHEMA, strict=True)
    validate_dynamic_pool_feature_table(table, split_manifest=split_manifest)
    return DynamicPoolFeatureBuild(
        table=table,
        feature_schema_fingerprint=schema_fingerprint,
        feature_table_fingerprint=table_fingerprint,
        row_count=table.height,
        feature_count=len(DYNAMIC_POOL_MODEL_FEATURE_NAMES),
        split_row_counts=tuple(
            (
                split,
                table.filter(pl.col("evaluation_split") == split).height,
            )
            for split in DYNAMIC_POOL_EVALUATION_SPLITS
        ),
    )


def validate_dynamic_pool_feature_table(
    table: pl.DataFrame,
    *,
    split_manifest: pl.DataFrame | None = None,
) -> None:
    """Recompute vectors and fingerprints; optionally verify the source split."""

    if not isinstance(table, pl.DataFrame):
        raise TypeError("dynamic-pool feature table must be a Polars DataFrame")
    if table.schema != DYNAMIC_POOL_FEATURE_SCHEMA:
        raise ValueError("dynamic-pool feature table schema does not match contract")
    if not table.height:
        raise ValueError("dynamic-pool feature table must not be empty")
    if not table.equals(table.sort("item_id")):
        raise ValueError("dynamic-pool feature table is not sorted")
    if table["item_id"].n_unique() != table.height:
        raise ValueError("dynamic-pool feature item_id must be unique")
    if set(table["schema_version"].to_list()) != {DYNAMIC_POOL_FEATURE_SCHEMA_VERSION}:
        raise ValueError("unsupported dynamic-pool feature schema version")
    if set(table["feature_derivation_version"].to_list()) != {
        DYNAMIC_POOL_FEATURE_DERIVATION_VERSION
    }:
        raise ValueError("unsupported dynamic-pool feature derivation")
    schema_fingerprint = dynamic_pool_feature_schema_fingerprint()
    if set(table["feature_schema_fingerprint"].to_list()) != {schema_fingerprint}:
        raise ValueError("dynamic-pool feature schema fingerprint mismatch")
    if set(table["score_semantics"].to_list()) != {RAW_SCORE_SEMANTICS}:
        raise ValueError("dynamic-pool raw score semantics changed")
    if any(table["probability_available"].to_list()):
        raise ValueError("raw dynamic-pool features cannot claim probabilities")
    if set(table["evaluation_split"].to_list()) != set(DYNAMIC_POOL_EVALUATION_SPLITS):
        raise ValueError("dynamic-pool feature table must retain every frozen split")
    if table["split_fingerprint"].n_unique() != 1:
        raise ValueError("dynamic-pool feature table has mixed split fingerprints")
    for component_id in table["independence_component_id"].unique().to_list():
        component = table.filter(pl.col("independence_component_id") == component_id)
        if component["evaluation_split"].n_unique() != 1:
            raise ValueError("dynamic-pool feature component crosses splits")
    row_fingerprints = []
    expected_names = list(DYNAMIC_POOL_MODEL_FEATURE_NAMES)
    for row in table.iter_rows(named=True):
        if row["feature_names"] != expected_names:
            raise ValueError("dynamic-pool feature names or order changed")
        if row["feature_count"] != len(expected_names):
            raise ValueError("dynamic-pool feature count mismatch")
        item = _input_from_row(row)
        if row["feature_input_fingerprint"] != item.fingerprint:
            raise ValueError("dynamic-pool feature input fingerprint mismatch")
        vector = _feature_vector(item)
        if tuple(row["feature_vector"]) != vector:
            raise ValueError("dynamic-pool feature vector mismatch")
        expected_row = canonical_semantic_fingerprint(
            {
                "feature_input_fingerprint": item.fingerprint,
                "item_fingerprint": row["item_fingerprint"],
                "split_fingerprint": row["split_fingerprint"],
                "evaluation_split": row["evaluation_split"],
                "independence_component_id": row["independence_component_id"],
                "review_decision_fingerprint": row["review_decision_fingerprint"],
                "human_supported": row["human_supported"],
                "sampling_weight": row["sampling_weight"],
                "feature_schema_fingerprint": schema_fingerprint,
                "feature_vector": vector,
            }
        )
        if row["feature_row_fingerprint"] != expected_row:
            raise ValueError("dynamic-pool feature row fingerprint mismatch")
        row_fingerprints.append(expected_row)
    expected_table = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_FEATURE_SCHEMA_VERSION,
            "feature_schema_fingerprint": schema_fingerprint,
            "split_fingerprint": table["split_fingerprint"].item(0),
            "row_fingerprints": row_fingerprints,
        }
    )
    if set(table["feature_table_fingerprint"].to_list()) != {expected_table}:
        raise ValueError("dynamic-pool feature table fingerprint mismatch")
    if split_manifest is not None:
        _validate_split_binding(table, split_manifest)


def write_dynamic_pool_feature_table(
    table: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write and round-trip validate a dynamic-pool feature artifact."""

    validate_dynamic_pool_feature_table(table)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= DYNAMIC_POOL_FEATURE_FILE
    written = write_parquet(table, destination, overwrite=overwrite)
    loaded = pl.read_parquet(written)
    validate_dynamic_pool_feature_table(loaded)
    if not table.equals(loaded):
        raise ValueError("dynamic-pool feature Parquet round-trip mismatch")
    return written


def load_dynamic_pool_feature_table(path: str | Path) -> pl.DataFrame:
    """Load and validate a dynamic-pool feature artifact."""

    source = Path(path)
    if source.is_dir():
        source /= DYNAMIC_POOL_FEATURE_FILE
    table = pl.read_parquet(source)
    validate_dynamic_pool_feature_table(table)
    return table


def _feature_vector(item: DynamicPoolFeatureInput) -> tuple[float, ...]:
    values: dict[str, float] = {
        "global_prototype_similarity": item.global_prototype_similarity,
        "global_nearest_reference_similarity": (
            item.global_nearest_reference_similarity
        ),
        "global_top_k_mean_similarity": item.global_top_k_mean_similarity,
        "raw_competitor_margin": item.raw_competitor_margin,
        "no_geo": float(item.no_geo),
        "local_evidence_available": float(item.local_evidence_available),
        "route_compatible": float(item.route_compatible),
        "quality_flag_count": float(item.quality_flag_count),
        "global_support_coverage_fraction": (item.global_support_coverage_fraction),
        "global_top_k_coverage_fraction": item.global_top_k_coverage_fraction,
        "global_observation_independence_fraction": (
            item.global_observation_independence_fraction
        ),
        "global_reference_count": float(item.global_reference_count),
        "global_configured_reference_count": float(
            item.global_configured_reference_count
        ),
        "global_independent_observation_count": float(
            item.global_independent_observation_count
        ),
        "global_reference_shortfall_count": float(
            item.global_reference_shortfall_count
        ),
        "local_reference_count": float(item.local_reference_count),
        "local_configured_reference_count": float(
            item.local_configured_reference_count
        ),
        "local_independent_observation_count": float(
            item.local_independent_observation_count
        ),
        "local_reference_shortfall_count": float(item.local_reference_shortfall_count),
        "query_hit_count": float(item.query_hit_count),
    }
    optional_fields = (
        "family_similarity",
        "family_margin_to_next_raw",
        "family_rank",
        "local_prototype_similarity",
        "local_nearest_reference_similarity",
        "local_top_k_mean_similarity",
        "prototype_absolute_disagreement",
        "nearest_absolute_disagreement",
        "top_k_absolute_disagreement",
        "prototype_rank_movement",
        "nearest_rank_movement",
        "top_k_rank_movement",
        "local_support_coverage_fraction",
        "local_top_k_coverage_fraction",
        "local_observation_independence_fraction",
        "subject_area_ratio",
        "query_text_similarity",
        "query_text_margin",
    )
    for field in optional_fields:
        value = getattr(item, field)
        values[field] = 0.0 if value is None else float(value)
        values[f"{field}_available"] = float(value is not None)
    for route in _ROUTES:
        values[f"route={route}"] = float(item.route == route)
    for domain in _VISUAL_DOMAINS:
        values[f"visual_domain={domain}"] = float(item.visual_domain == domain)
    for query_tier in _QUERY_TIERS:
        values[f"query_tier={query_tier}"] = float(
            item.primary_query_tier == query_tier
        )
    if set(values) != set(DYNAMIC_POOL_MODEL_FEATURE_NAMES):
        missing = sorted(set(DYNAMIC_POOL_MODEL_FEATURE_NAMES) - set(values))
        extra = sorted(set(values) - set(DYNAMIC_POOL_MODEL_FEATURE_NAMES))
        raise AssertionError(
            f"feature derivation mismatch: missing={missing}, extra={extra}"
        )
    return tuple(values[name] for name in DYNAMIC_POOL_MODEL_FEATURE_NAMES)


def _input_from_row(row: Mapping[str, object]) -> DynamicPoolFeatureInput:
    return DynamicPoolFeatureInput(
        **{field: row[field] for field in DynamicPoolFeatureInput.__dataclass_fields__}
    )


def _validate_split_binding(
    table: pl.DataFrame,
    split_manifest: pl.DataFrame,
) -> None:
    validate_dynamic_pool_evaluation_splits(split_manifest)
    if set(table["item_id"].to_list()) != set(split_manifest["item_id"].to_list()):
        raise ValueError("feature table does not exactly cover split manifest")
    split_by_id = {
        str(row["item_id"]): row for row in split_manifest.iter_rows(named=True)
    }
    fields = (
        "item_fingerprint",
        "source_record_hash",
        "source_artifact_fingerprint",
        "review_decision_fingerprint",
        "split_fingerprint",
        "evaluation_split",
        "independence_component_id",
        "candidate_species_key",
        "human_supported",
        "sampling_weight",
    )
    for row in table.iter_rows(named=True):
        source = split_by_id[str(row["item_id"])]
        if any(row[field] != source[field] for field in fields):
            raise ValueError("feature row differs from frozen split manifest")


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field).casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


def _choice(value: object, *, field: str, allowed: frozenset[str]) -> str:
    text = _required_text(value, field=field)
    if text not in allowed:
        raise ValueError(f"unsupported {field}: {text}")
    return text


def _optional_bounded_float(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{field} must be finite and in [{minimum}, {maximum}]")
    return result


def _uint32(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if not 0 <= value < 2**32:
        raise ValueError(f"{field} must fit UInt32")
    return value


def _positive_uint32(value: object, *, field: str) -> int:
    result = _uint32(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _int32(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if not -(2**31) <= value < 2**31:
        raise ValueError(f"{field} must fit Int32")
    return value


__all__ = [
    "DYNAMIC_POOL_FEATURE_DERIVATION_VERSION",
    "DYNAMIC_POOL_FEATURE_FILE",
    "DYNAMIC_POOL_FEATURE_SCHEMA",
    "DYNAMIC_POOL_FEATURE_SCHEMA_VERSION",
    "DYNAMIC_POOL_MODEL_FEATURE_NAMES",
    "DynamicPoolFeatureBuild",
    "DynamicPoolFeatureInput",
    "build_dynamic_pool_feature_table",
    "dynamic_pool_feature_schema_fingerprint",
    "load_dynamic_pool_feature_table",
    "validate_dynamic_pool_feature_table",
    "write_dynamic_pool_feature_table",
]
