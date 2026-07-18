"""Immutable plan, membership and summary contracts for dynamic reference pools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.schemas import REFERENCE_ROUTES
from biominer.storage.parquet import write_parquet
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


DYNAMIC_POOL_PLAN_SCHEMA_VERSION = "dynamic-reference-pool-plan-v1.0.0"
DYNAMIC_POOL_MEMBER_SCHEMA_VERSION = "dynamic-reference-pool-member-v1.0.0"
DYNAMIC_POOL_SUMMARY_SCHEMA_VERSION = "dynamic-reference-pool-summary-v1.0.0"
DYNAMIC_POOL_PLANS_FILE = "dynamic_reference_pool_plans.parquet"
DYNAMIC_POOL_MEMBERS_FILE = "dynamic_reference_pool_members.parquet"
DYNAMIC_POOL_SUMMARY_FILE = "dynamic_reference_pool_summary.parquet"

DYNAMIC_POOL_SCOPES = frozenset({"global", "local", "safety_expansion"})
DYNAMIC_POOL_ROLES = frozenset(
    {
        "global_core",
        "nearest_local",
        "neighbouring_region",
        "regional_same_genus",
        "regional_same_family",
        "query_associated",
        "visual_neighbour",
        "historical_confusion",
    }
)
DYNAMIC_POOL_GEOGRAPHIC_SCOPES = frozenset(
    {
        "global",
        "exact_local_cell",
        "neighbouring_local_cell",
        "regional_cell",
        "coarse_cell",
        "bioregion",
        "admin1",
        "country",
        "nearest_geodesic",
        "not_applicable",
    }
)
DYNAMIC_POOL_SCORING_STAGES = frozenset(
    {"initial", "uncertainty_expansion", "selective_rescore"}
)
DYNAMIC_POOL_DISTANCE_STATUSES = frozenset(
    {"available", "unavailable", "not_applicable"}
)
DYNAMIC_LOCAL_POOL_STATUSES = frozenset({"available", "unavailable"})

_ROLE_SCOPE = {
    "global_core": "global",
    "nearest_local": "local",
    "neighbouring_region": "local",
    "regional_same_genus": "local",
    "regional_same_family": "local",
    "query_associated": "safety_expansion",
    "visual_neighbour": "safety_expansion",
    "historical_confusion": "safety_expansion",
}
_NO_LOCAL_QUERY_QUALITIES = frozenset(
    {"no_geo", "unassigned_geo", "withheld", "invalid"}
)
_VISUAL_INPUT_KINDS = frozenset(
    {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
        MULTI_OBJECT_FULL_FRAME_KIND,
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PLAN_ID_PATTERN = re.compile(r"dynamic-pool-plan:[0-9a-f]{64}\Z")
_POOL_ID_PATTERN = re.compile(r"dynamic-reference-pool:[0-9a-f]{64}\Z")
_REFERENCE_MEDIA_ID_PATTERN = re.compile(r"reference-media:[0-9a-f]{64}\Z")
_REFERENCE_OBSERVATION_ID_PATTERN = re.compile(r"reference-observation:[0-9a-f]{64}\Z")

_PLAN_CONTEXT_FIELDS = (
    "run_id",
    "flickr_query_id",
    "flickr_photo_id",
    "organism_unit_id",
    "visual_input_id",
    "query_embedding_fingerprint",
    "scoring_stage",
    "query_route",
    "registry_version",
    "reference_bank_version",
    "reference_geography_index_fingerprint",
    "candidate_set_id",
    "candidate_set_fingerprint",
    "query_geo_cluster_id",
    "query_coordinate_quality",
    "local_pool_status",
    "local_pool_unavailable_reason",
    "selection_policy_version",
    "selection_policy_fingerprint",
    "model_id",
    "model_revision",
    "model_weights_sha256",
    "model_fingerprint",
    "preprocessing_fingerprint",
    "configured_global_per_candidate",
    "configured_local_per_candidate",
    "configured_safety_per_candidate",
    "maximum_expansion_rounds",
)
_PLAN_INPUT_FIELDS = ("plan_id", *_PLAN_CONTEXT_FIELDS)
_PLAN_SORT = (
    "run_id",
    "flickr_photo_id",
    "organism_unit_id",
    "visual_input_id",
    "scoring_stage",
    "plan_id",
)

_MEMBER_INPUT_FIELDS = (
    "plan_id",
    "run_id",
    "flickr_query_id",
    "flickr_photo_id",
    "organism_unit_id",
    "visual_input_id",
    "query_embedding_fingerprint",
    "scoring_stage",
    "query_route",
    "candidate_set_id",
    "candidate_set_fingerprint",
    "candidate_accepted_taxon_key",
    "candidate_scientific_name",
    "reference_media_id",
    "reference_observation_id",
    "reference_embedding_fingerprint",
    "reference_route",
    "reference_visual_input_kind",
    "pool_scope",
    "pool_role",
    "geographic_scope",
    "geographic_distance_km",
    "geographic_distance_status",
    "geographic_distance_reason",
    "fallback_level",
    "selection_rank",
    "independent_observation_group",
    "observer_id_hash",
    "reference_country_code",
    "inclusion_reason",
    "selection_policy_fingerprint",
    "source",
    "source_dataset_key",
    "registry_version",
    "reference_bank_version",
    "reference_geography_index_fingerprint",
    "model_id",
    "model_revision",
    "model_weights_sha256",
    "model_fingerprint",
    "preprocessing_fingerprint",
    "expansion_round",
)
_MEMBER_SORT = (
    "plan_id",
    "pool_scope",
    "pool_role",
    "candidate_accepted_taxon_key",
    "expansion_round",
    "selection_rank",
    "reference_observation_id",
    "reference_media_id",
)
_POOL_GROUP = (
    "plan_id",
    "pool_scope",
    "pool_role",
    "candidate_accepted_taxon_key",
    "expansion_round",
)


def dynamic_reference_pool_plan_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "plan_id": pl.String,
        "run_id": pl.String,
        "flickr_query_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "visual_input_id": pl.String,
        "query_embedding_fingerprint": pl.String,
        "scoring_stage": pl.String,
        "query_route": pl.String,
        "registry_version": pl.String,
        "reference_bank_version": pl.String,
        "reference_geography_index_fingerprint": pl.String,
        "candidate_set_id": pl.String,
        "candidate_set_fingerprint": pl.String,
        "query_geo_cluster_id": pl.String,
        "query_coordinate_quality": pl.String,
        "local_pool_status": pl.String,
        "local_pool_unavailable_reason": pl.String,
        "global_pool_ids": pl.List(pl.String),
        "local_pool_ids": pl.List(pl.String),
        "safety_pool_ids": pl.List(pl.String),
        "selection_policy_version": pl.String,
        "selection_policy_fingerprint": pl.String,
        "model_id": pl.String,
        "model_revision": pl.String,
        "model_weights_sha256": pl.String,
        "model_fingerprint": pl.String,
        "preprocessing_fingerprint": pl.String,
        "configured_global_per_candidate": pl.UInt32,
        "configured_local_per_candidate": pl.UInt32,
        "configured_safety_per_candidate": pl.UInt32,
        "maximum_expansion_rounds": pl.UInt16,
        "plan_fingerprint": pl.String,
    }


def dynamic_reference_pool_member_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "plan_id": pl.String,
        "pool_id": pl.String,
        "run_id": pl.String,
        "flickr_query_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "visual_input_id": pl.String,
        "query_embedding_fingerprint": pl.String,
        "scoring_stage": pl.String,
        "query_route": pl.String,
        "candidate_set_id": pl.String,
        "candidate_set_fingerprint": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "candidate_scientific_name": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "reference_embedding_fingerprint": pl.String,
        "reference_route": pl.String,
        "reference_visual_input_kind": pl.String,
        "pool_scope": pl.String,
        "pool_role": pl.String,
        "geographic_scope": pl.String,
        "geographic_distance_km": pl.Float64,
        "geographic_distance_status": pl.String,
        "geographic_distance_reason": pl.String,
        "fallback_level": pl.UInt8,
        "selection_rank": pl.UInt32,
        "independent_observation_group": pl.String,
        "observer_id_hash": pl.String,
        "reference_country_code": pl.String,
        "inclusion_reason": pl.String,
        "selection_policy_fingerprint": pl.String,
        "source": pl.String,
        "source_dataset_key": pl.String,
        "registry_version": pl.String,
        "reference_bank_version": pl.String,
        "reference_geography_index_fingerprint": pl.String,
        "model_id": pl.String,
        "model_revision": pl.String,
        "model_weights_sha256": pl.String,
        "model_fingerprint": pl.String,
        "preprocessing_fingerprint": pl.String,
        "expansion_round": pl.UInt16,
        "member_fingerprint": pl.String,
    }


def dynamic_reference_pool_summary_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "plan_id": pl.String,
        "pool_id": pl.String,
        "run_id": pl.String,
        "flickr_query_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "scoring_stage": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "candidate_scientific_name": pl.String,
        "pool_scope": pl.String,
        "pool_role": pl.String,
        "configured_reference_count": pl.UInt32,
        "effective_reference_count": pl.UInt32,
        "shortfall_count": pl.UInt32,
        "independent_observation_count": pl.UInt32,
        "observer_identity_available_count": pl.UInt32,
        "independent_observer_count": pl.UInt32,
        "source_dataset_count": pl.UInt32,
        "country_count": pl.UInt32,
        "distance_available_count": pl.UInt32,
        "minimum_distance_km": pl.Float64,
        "maximum_distance_km": pl.Float64,
        "maximum_fallback_level": pl.UInt8,
        "expansion_round": pl.UInt16,
        "selection_policy_fingerprint": pl.String,
        "reference_geography_index_fingerprint": pl.String,
        "model_fingerprint": pl.String,
        "pool_membership_fingerprint": pl.String,
        "summary_fingerprint": pl.String,
    }


def dynamic_reference_pool_plan_id(values: Mapping[str, object]) -> str:
    """Create the pre-membership plan identity from its complete context."""

    if not isinstance(values, Mapping):
        raise TypeError("dynamic pool plan identity values must be a mapping")
    _require_exact_fields(values, set(_PLAN_CONTEXT_FIELDS), label="plan identity")
    context = _normalized_plan_context(values)
    digest = canonical_semantic_fingerprint(
        {"schema_version": "dynamic-reference-pool-plan-id-v1", **context}
    ).removeprefix("sha256:")
    return f"dynamic-pool-plan:{digest}"


def build_dynamic_reference_pool_members(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    """Build canonical member rows and derive immutable pool identities."""

    _require_mapping_sequence(rows, label="dynamic pool member rows")
    normalized: list[dict[str, object]] = []
    for row in rows:
        _require_exact_fields(row, set(_MEMBER_INPUT_FIELDS), label="pool member")
        normalized.append(_normalized_member(row))
    if not normalized:
        return pl.DataFrame(schema=dynamic_reference_pool_member_schema())

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in normalized:
        grouped.setdefault(tuple(row[field] for field in _POOL_GROUP), []).append(row)

    output: list[dict[str, object]] = []
    for group_key in sorted(grouped, key=lambda key: tuple(str(item) for item in key)):
        group = sorted(
            grouped[group_key],
            key=lambda row: (
                int(row["selection_rank"]),
                str(row["reference_observation_id"]),
                str(row["reference_media_id"]),
            ),
        )
        ranks = [int(row["selection_rank"]) for row in group]
        if len(ranks) != len(set(ranks)):
            raise ValueError("dynamic pool selection ranks repeat within a pool")
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("dynamic pool selection ranks must be contiguous from one")
        observation_groups = [
            str(row["independent_observation_group"]) for row in group
        ]
        if len(observation_groups) != len(set(observation_groups)):
            raise ValueError(
                "dynamic pool cannot fill multiple slots from one observation group"
            )
        pool_id = _pool_id(group)
        for item in group:
            complete = {
                "schema_version": DYNAMIC_POOL_MEMBER_SCHEMA_VERSION,
                "pool_id": pool_id,
                **item,
            }
            complete["member_fingerprint"] = canonical_semantic_fingerprint(complete)
            output.append(complete)

    frame = pl.DataFrame(
        output,
        schema=dynamic_reference_pool_member_schema(),
        orient="row",
        strict=True,
    ).sort(*_MEMBER_SORT)
    validate_dynamic_reference_pool_members(frame)
    return frame


def build_dynamic_reference_pool_plans(
    rows: Sequence[Mapping[str, object]],
    members: pl.DataFrame,
) -> pl.DataFrame:
    """Build plan rows and bind them to exact derived member pools."""

    validate_dynamic_reference_pool_members(members)
    _require_mapping_sequence(rows, label="dynamic pool plan rows")
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in rows:
        _require_exact_fields(source, set(_PLAN_INPUT_FIELDS), label="pool plan")
        plan_id = _required_text(source["plan_id"], field="plan_id")
        context = _normalized_plan_context(source)
        if plan_id != dynamic_reference_pool_plan_id(context):
            raise ValueError("dynamic pool plan_id does not match plan context")
        if plan_id in seen:
            raise ValueError("dynamic pool plan_id is duplicated")
        seen.add(plan_id)
        selected = members.filter(pl.col("plan_id") == plan_id)
        if selected.is_empty():
            raise ValueError("dynamic pool plan has no member rows")
        _validate_plan_member_context(context, selected)
        pools = {
            scope: sorted(
                selected.filter(pl.col("pool_scope") == scope)["pool_id"]
                .unique()
                .to_list()
            )
            for scope in DYNAMIC_POOL_SCOPES
        }
        if not pools["global"]:
            raise ValueError("dynamic pool plan requires a global pool")
        _validate_local_plan_state(context, local_pool_ids=pools["local"])
        complete = {
            "schema_version": DYNAMIC_POOL_PLAN_SCHEMA_VERSION,
            "plan_id": plan_id,
            **context,
            "global_pool_ids": pools["global"],
            "local_pool_ids": pools["local"],
            "safety_pool_ids": pools["safety_expansion"],
        }
        complete["plan_fingerprint"] = canonical_semantic_fingerprint(complete)
        output.append(complete)
    if set(members["plan_id"].to_list()) != seen:
        raise ValueError("dynamic pool plan/member identity sets differ")

    frame = (
        pl.DataFrame(
            output,
            schema=dynamic_reference_pool_plan_schema(),
            orient="row",
            strict=True,
        ).sort(*_PLAN_SORT)
        if output
        else pl.DataFrame(schema=dynamic_reference_pool_plan_schema())
    )
    validate_dynamic_reference_pool_plans(frame)
    return frame


def _build_summaries_core(
    plans: pl.DataFrame,
    members: pl.DataFrame,
) -> pl.DataFrame:
    """Derive summaries without recursively cross-validating artifacts."""

    validate_dynamic_reference_pool_plans(plans)
    validate_dynamic_reference_pool_members(members)
    plan_lookup = {str(row["plan_id"]): row for row in plans.iter_rows(named=True)}
    output: list[dict[str, object]] = []
    for pool_id in sorted(members["pool_id"].unique().to_list()):
        group = members.filter(pl.col("pool_id") == pool_id).sort(*_MEMBER_SORT)
        first = group.row(0, named=True)
        plan = plan_lookup.get(str(first["plan_id"]))
        if plan is None:
            raise ValueError("dynamic pool members reference an unknown plan")
        scope = str(first["pool_scope"])
        configured_field = {
            "global": "configured_global_per_candidate",
            "local": "configured_local_per_candidate",
            "safety_expansion": "configured_safety_per_candidate",
        }[scope]
        configured = int(plan[configured_field])
        effective = group.height
        if effective > configured:
            raise ValueError("dynamic pool effective count exceeds configured quota")
        observers = [value for value in group["observer_id_hash"].to_list() if value]
        countries = [
            value for value in group["reference_country_code"].to_list() if value
        ]
        distances = [
            float(value)
            for value in group["geographic_distance_km"].to_list()
            if value is not None
        ]
        membership_fingerprint = canonical_semantic_fingerprint(
            {
                "schema_version": "dynamic-reference-pool-membership-set-v1",
                "pool_id": pool_id,
                "member_fingerprints": group["member_fingerprint"].to_list(),
            }
        )
        summary: dict[str, object] = {
            "schema_version": DYNAMIC_POOL_SUMMARY_SCHEMA_VERSION,
            "plan_id": first["plan_id"],
            "pool_id": pool_id,
            "run_id": first["run_id"],
            "flickr_query_id": first["flickr_query_id"],
            "flickr_photo_id": first["flickr_photo_id"],
            "organism_unit_id": first["organism_unit_id"],
            "scoring_stage": first["scoring_stage"],
            "candidate_accepted_taxon_key": first["candidate_accepted_taxon_key"],
            "candidate_scientific_name": first["candidate_scientific_name"],
            "pool_scope": scope,
            "pool_role": first["pool_role"],
            "configured_reference_count": configured,
            "effective_reference_count": effective,
            "shortfall_count": configured - effective,
            "independent_observation_count": group[
                "independent_observation_group"
            ].n_unique(),
            "observer_identity_available_count": len(observers),
            "independent_observer_count": len(set(observers)),
            "source_dataset_count": group["source_dataset_key"].n_unique(),
            "country_count": len(set(countries)),
            "distance_available_count": len(distances),
            "minimum_distance_km": min(distances) if distances else None,
            "maximum_distance_km": max(distances) if distances else None,
            "maximum_fallback_level": max(group["fallback_level"].to_list()),
            "expansion_round": first["expansion_round"],
            "selection_policy_fingerprint": first["selection_policy_fingerprint"],
            "reference_geography_index_fingerprint": first[
                "reference_geography_index_fingerprint"
            ],
            "model_fingerprint": first["model_fingerprint"],
            "pool_membership_fingerprint": membership_fingerprint,
        }
        summary["summary_fingerprint"] = canonical_semantic_fingerprint(summary)
        output.append(summary)
    frame = (
        pl.DataFrame(
            output,
            schema=dynamic_reference_pool_summary_schema(),
            orient="row",
            strict=True,
        ).sort("plan_id", "pool_scope", "pool_role", "candidate_accepted_taxon_key")
        if output
        else pl.DataFrame(schema=dynamic_reference_pool_summary_schema())
    )
    validate_dynamic_reference_pool_summaries(frame)
    return frame


def build_dynamic_reference_pool_summaries(
    plans: pl.DataFrame,
    members: pl.DataFrame,
) -> pl.DataFrame:
    """Derive and cross-check count, diversity, distance and shortfall evidence."""

    frame = _build_summaries_core(plans, members)
    validate_dynamic_reference_pool_artifacts(plans, members, frame)
    return frame


def validate_dynamic_reference_pool_plans(frame: pl.DataFrame) -> None:
    _require_frame_schema(frame, dynamic_reference_pool_plan_schema(), label="plans")
    if frame.is_empty():
        return
    if frame["plan_id"].n_unique() != frame.height:
        raise ValueError("dynamic pool plan IDs are not unique")
    if not frame.equals(frame.sort(*_PLAN_SORT)):
        raise ValueError("dynamic pool plans are not canonically sorted")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != DYNAMIC_POOL_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported dynamic pool plan schema version")
        context = {field: row[field] for field in _PLAN_CONTEXT_FIELDS}
        normalized = _normalized_plan_context(context)
        if any(normalized[field] != row[field] for field in _PLAN_CONTEXT_FIELDS):
            raise ValueError("dynamic pool plan fields are not canonical")
        if row["plan_id"] != dynamic_reference_pool_plan_id(context):
            raise ValueError("dynamic pool plan_id does not match plan context")
        for field in ("global_pool_ids", "local_pool_ids", "safety_pool_ids"):
            values = row[field]
            if not isinstance(values, list) or values != sorted(set(values)):
                raise ValueError(f"{field} must be a sorted unique pool ID list")
            if any(not _POOL_ID_PATTERN.fullmatch(str(value)) for value in values):
                raise ValueError(f"{field} contains an invalid pool ID")
        if not row["global_pool_ids"]:
            raise ValueError("dynamic pool plan requires a global pool")
        pool_sets = [
            set(row["global_pool_ids"]),
            set(row["local_pool_ids"]),
            set(row["safety_pool_ids"]),
        ]
        if any(
            left & right
            for index, left in enumerate(pool_sets)
            for right in pool_sets[index + 1 :]
        ):
            raise ValueError("dynamic pool plan scope lists must be disjoint")
        _validate_local_plan_state(context, local_pool_ids=row["local_pool_ids"])
        _validate_fingerprint(row, field="plan_fingerprint")


def validate_dynamic_reference_pool_members(frame: pl.DataFrame) -> None:
    _require_frame_schema(
        frame, dynamic_reference_pool_member_schema(), label="members"
    )
    if frame.is_empty():
        return
    if frame["member_fingerprint"].n_unique() != frame.height:
        raise ValueError("dynamic pool member fingerprints are not unique")
    if not frame.equals(frame.sort(*_MEMBER_SORT)):
        raise ValueError("dynamic pool members are not canonically sorted")
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != DYNAMIC_POOL_MEMBER_SCHEMA_VERSION:
            raise ValueError("unsupported dynamic pool member schema version")
        normalized = _normalized_member(row)
        if any(normalized[field] != row[field] for field in _MEMBER_INPUT_FIELDS):
            raise ValueError("dynamic pool member fields are not canonical")
        if not _POOL_ID_PATTERN.fullmatch(str(row["pool_id"])):
            raise ValueError("dynamic pool member pool_id is invalid")
        _validate_fingerprint(row, field="member_fingerprint")
        grouped.setdefault(tuple(row[field] for field in _POOL_GROUP), []).append(row)
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["selection_rank"]),
                str(row["reference_observation_id"]),
                str(row["reference_media_id"]),
            ),
        )
        if [int(row["selection_rank"]) for row in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise ValueError("dynamic pool selection ranks must be contiguous from one")
        observation_groups = [
            str(row["independent_observation_group"]) for row in ordered
        ]
        if len(observation_groups) != len(set(observation_groups)):
            raise ValueError(
                "dynamic pool cannot fill multiple slots from one observation group"
            )
        if len({str(row["candidate_scientific_name"]) for row in ordered}) != 1:
            raise ValueError("dynamic pool candidate name conflicts within a pool")
        expected = _pool_id(
            [{field: row[field] for field in _MEMBER_INPUT_FIELDS} for row in ordered]
        )
        if {str(row["pool_id"]) for row in ordered} != {expected}:
            raise ValueError("dynamic reference pool identity mismatch")


def validate_dynamic_reference_pool_summaries(frame: pl.DataFrame) -> None:
    _require_frame_schema(
        frame, dynamic_reference_pool_summary_schema(), label="summaries"
    )
    if frame.is_empty():
        return
    if frame["pool_id"].n_unique() != frame.height:
        raise ValueError("dynamic pool summaries repeat pool IDs")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != DYNAMIC_POOL_SUMMARY_SCHEMA_VERSION:
            raise ValueError("unsupported dynamic pool summary schema version")
        if not _PLAN_ID_PATTERN.fullmatch(str(row["plan_id"])):
            raise ValueError("dynamic pool summary plan_id is invalid")
        if not _POOL_ID_PATTERN.fullmatch(str(row["pool_id"])):
            raise ValueError("dynamic pool summary pool_id is invalid")
        for field in (
            "run_id",
            "flickr_query_id",
            "flickr_photo_id",
            "organism_unit_id",
            "scoring_stage",
            "candidate_accepted_taxon_key",
            "candidate_scientific_name",
            "pool_scope",
            "pool_role",
        ):
            if _required_text(row[field], field=field) != row[field]:
                raise ValueError(f"dynamic pool summary {field} is not canonical")
        if row["scoring_stage"] not in DYNAMIC_POOL_SCORING_STAGES:
            raise ValueError("unsupported dynamic pool summary scoring_stage")
        if row["pool_scope"] not in DYNAMIC_POOL_SCOPES:
            raise ValueError("unsupported dynamic pool summary scope")
        if (
            row["pool_role"] not in DYNAMIC_POOL_ROLES
            or _ROLE_SCOPE[str(row["pool_role"])] != row["pool_scope"]
        ):
            raise ValueError("dynamic pool summary role conflicts with scope")
        for field in (
            "selection_policy_fingerprint",
            "reference_geography_index_fingerprint",
            "model_fingerprint",
            "pool_membership_fingerprint",
        ):
            _sha256(row[field], field=field)
        configured = int(row["configured_reference_count"])
        effective = int(row["effective_reference_count"])
        shortfall = int(row["shortfall_count"])
        if effective > configured or shortfall != configured - effective:
            raise ValueError("dynamic pool summary shortfall is inconsistent")
        bounded_counts = (
            "independent_observation_count",
            "observer_identity_available_count",
            "source_dataset_count",
            "country_count",
            "distance_available_count",
        )
        if any(int(row[field]) > effective for field in bounded_counts):
            raise ValueError("dynamic pool summary diversity count exceeds members")
        if int(row["independent_observer_count"]) > int(
            row["observer_identity_available_count"]
        ):
            raise ValueError("independent observer count exceeds available identities")
        distance_count = int(row["distance_available_count"])
        minimum = row["minimum_distance_km"]
        maximum = row["maximum_distance_km"]
        if distance_count == 0 and (minimum is not None or maximum is not None):
            raise ValueError("distance-free summary cannot contain distance bounds")
        if distance_count > 0 and (
            minimum is None or maximum is None or float(minimum) > float(maximum)
        ):
            raise ValueError("dynamic pool summary distance bounds are invalid")
        _validate_fingerprint(row, field="summary_fingerprint")


def validate_dynamic_reference_pool_artifacts(
    plans: pl.DataFrame,
    members: pl.DataFrame,
    summaries: pl.DataFrame,
) -> None:
    """Cross-check plan lists and derived summaries against exact member rows."""

    validate_dynamic_reference_pool_plans(plans)
    validate_dynamic_reference_pool_members(members)
    validate_dynamic_reference_pool_summaries(summaries)
    plan_lookup = {str(row["plan_id"]): row for row in plans.iter_rows(named=True)}
    if set(members["plan_id"].to_list()) != set(plan_lookup):
        raise ValueError("dynamic pool plan/member identity sets differ")
    if set(summaries["plan_id"].to_list()) != set(plan_lookup):
        raise ValueError("dynamic pool plan/summary identity sets differ")
    if set(summaries["pool_id"].to_list()) != set(members["pool_id"].to_list()):
        raise ValueError("dynamic pool summary/member pool identity sets differ")
    for plan_id, plan in plan_lookup.items():
        selected = members.filter(pl.col("plan_id") == plan_id)
        if selected["expansion_round"].max() > plan["maximum_expansion_rounds"]:
            raise ValueError("dynamic pool expansion round exceeds plan maximum")
        for scope, field in (
            ("global", "global_pool_ids"),
            ("local", "local_pool_ids"),
            ("safety_expansion", "safety_pool_ids"),
        ):
            actual = sorted(
                selected.filter(pl.col("pool_scope") == scope)["pool_id"]
                .unique()
                .to_list()
            )
            if actual != plan[field]:
                raise ValueError("dynamic pool plan membership list mismatch")
    rebuilt = _build_summaries_core(plans, members)
    if not summaries.equals(rebuilt):
        raise ValueError("dynamic pool summaries do not match member evidence")


def write_dynamic_reference_pool_artifacts(
    plans: pl.DataFrame,
    members: pl.DataFrame,
    summaries: pl.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_dynamic_reference_pool_artifacts(plans, members, summaries)
    destination = Path(output_dir)
    return {
        "plans": write_parquet(plans, destination / DYNAMIC_POOL_PLANS_FILE),
        "members": write_parquet(members, destination / DYNAMIC_POOL_MEMBERS_FILE),
        "summary": write_parquet(summaries, destination / DYNAMIC_POOL_SUMMARY_FILE),
    }


def _normalized_plan_context(values: Mapping[str, object]) -> dict[str, object]:
    required_text = (
        "run_id",
        "flickr_query_id",
        "flickr_photo_id",
        "organism_unit_id",
        "visual_input_id",
        "query_embedding_fingerprint",
        "scoring_stage",
        "query_route",
        "registry_version",
        "reference_bank_version",
        "reference_geography_index_fingerprint",
        "candidate_set_id",
        "candidate_set_fingerprint",
        "query_coordinate_quality",
        "selection_policy_version",
        "selection_policy_fingerprint",
        "model_id",
        "model_revision",
        "model_weights_sha256",
        "model_fingerprint",
        "preprocessing_fingerprint",
        "local_pool_status",
    )
    context = {
        field: _required_text(values[field], field=field) for field in required_text
    }
    context.update(
        {
            "query_geo_cluster_id": _optional_text(
                values["query_geo_cluster_id"], field="query_geo_cluster_id"
            ),
            "local_pool_unavailable_reason": _optional_text(
                values["local_pool_unavailable_reason"],
                field="local_pool_unavailable_reason",
            ),
            "configured_global_per_candidate": _nonnegative_int(
                values["configured_global_per_candidate"],
                field="configured_global_per_candidate",
                maximum=2**32 - 1,
            ),
            "configured_local_per_candidate": _nonnegative_int(
                values["configured_local_per_candidate"],
                field="configured_local_per_candidate",
                maximum=2**32 - 1,
            ),
            "configured_safety_per_candidate": _nonnegative_int(
                values["configured_safety_per_candidate"],
                field="configured_safety_per_candidate",
                maximum=2**32 - 1,
            ),
            "maximum_expansion_rounds": _nonnegative_int(
                values["maximum_expansion_rounds"],
                field="maximum_expansion_rounds",
                maximum=2**16 - 1,
            ),
        }
    )
    if context["scoring_stage"] not in DYNAMIC_POOL_SCORING_STAGES:
        raise ValueError("unsupported dynamic pool scoring_stage")
    if context["query_route"] not in REFERENCE_ROUTES:
        raise ValueError("unsupported dynamic pool query_route")
    if context["local_pool_status"] not in DYNAMIC_LOCAL_POOL_STATUSES:
        raise ValueError("unsupported local_pool_status")
    for field in (
        "visual_input_id",
        "query_embedding_fingerprint",
        "reference_geography_index_fingerprint",
        "candidate_set_fingerprint",
        "selection_policy_fingerprint",
        "model_weights_sha256",
        "model_fingerprint",
        "preprocessing_fingerprint",
    ):
        _sha256(context[field], field=field)
    return context


def _normalized_member(values: Mapping[str, object]) -> dict[str, object]:
    required_text = (
        "plan_id",
        "run_id",
        "flickr_query_id",
        "flickr_photo_id",
        "organism_unit_id",
        "visual_input_id",
        "query_embedding_fingerprint",
        "scoring_stage",
        "query_route",
        "candidate_set_id",
        "candidate_set_fingerprint",
        "candidate_accepted_taxon_key",
        "candidate_scientific_name",
        "reference_media_id",
        "reference_observation_id",
        "reference_embedding_fingerprint",
        "reference_route",
        "reference_visual_input_kind",
        "pool_scope",
        "pool_role",
        "geographic_scope",
        "geographic_distance_status",
        "independent_observation_group",
        "inclusion_reason",
        "selection_policy_fingerprint",
        "source",
        "source_dataset_key",
        "registry_version",
        "reference_bank_version",
        "reference_geography_index_fingerprint",
        "model_id",
        "model_revision",
        "model_weights_sha256",
        "model_fingerprint",
        "preprocessing_fingerprint",
    )
    row = {field: _required_text(values[field], field=field) for field in required_text}
    row.update(
        {
            "geographic_distance_km": _optional_nonnegative_float(
                values["geographic_distance_km"], field="geographic_distance_km"
            ),
            "geographic_distance_reason": _optional_text(
                values["geographic_distance_reason"],
                field="geographic_distance_reason",
            ),
            "fallback_level": _nonnegative_int(
                values["fallback_level"], field="fallback_level", maximum=255
            ),
            "selection_rank": _positive_int(
                values["selection_rank"], field="selection_rank", maximum=2**32 - 1
            ),
            "observer_id_hash": _optional_text(
                values["observer_id_hash"], field="observer_id_hash"
            ),
            "reference_country_code": _optional_text(
                values["reference_country_code"], field="reference_country_code"
            ),
            "expansion_round": _nonnegative_int(
                values["expansion_round"], field="expansion_round", maximum=2**16 - 1
            ),
        }
    )
    if row["reference_country_code"] is not None:
        row["reference_country_code"] = str(row["reference_country_code"]).upper()
    if not _PLAN_ID_PATTERN.fullmatch(str(row["plan_id"])):
        raise ValueError("dynamic pool member plan_id is invalid")
    if not _REFERENCE_MEDIA_ID_PATTERN.fullmatch(str(row["reference_media_id"])):
        raise ValueError("dynamic pool reference_media_id is invalid")
    if not _REFERENCE_OBSERVATION_ID_PATTERN.fullmatch(
        str(row["reference_observation_id"])
    ):
        raise ValueError("dynamic pool reference_observation_id is invalid")
    if row["scoring_stage"] not in DYNAMIC_POOL_SCORING_STAGES:
        raise ValueError("unsupported dynamic pool scoring_stage")
    if row["query_route"] not in REFERENCE_ROUTES:
        raise ValueError("unsupported dynamic pool query_route")
    if row["reference_route"] not in REFERENCE_ROUTES:
        raise ValueError("unsupported dynamic pool reference_route")
    if row["reference_route"] != row["query_route"]:
        raise ValueError("dynamic pool query and reference routes conflict")
    if row["reference_visual_input_kind"] not in _VISUAL_INPUT_KINDS:
        raise ValueError("unsupported reference visual-input kind")
    if row["pool_scope"] not in DYNAMIC_POOL_SCOPES:
        raise ValueError("unsupported dynamic pool scope")
    if row["pool_role"] not in DYNAMIC_POOL_ROLES:
        raise ValueError("unsupported dynamic pool role")
    if _ROLE_SCOPE[str(row["pool_role"])] != row["pool_scope"]:
        raise ValueError("dynamic pool role conflicts with pool scope")
    if row["geographic_scope"] not in DYNAMIC_POOL_GEOGRAPHIC_SCOPES:
        raise ValueError("unsupported dynamic pool geographic scope")
    if row["geographic_distance_status"] not in DYNAMIC_POOL_DISTANCE_STATUSES:
        raise ValueError("unsupported geographic distance status")
    _validate_distance(row)
    country = row["reference_country_code"]
    if country is not None and (
        len(str(country)) != 2
        or not str(country).isalpha()
        or not str(country).isupper()
    ):
        raise ValueError("reference_country_code must be uppercase ISO alpha-2")
    for field in (
        "visual_input_id",
        "query_embedding_fingerprint",
        "candidate_set_fingerprint",
        "reference_embedding_fingerprint",
        "selection_policy_fingerprint",
        "reference_geography_index_fingerprint",
        "model_weights_sha256",
        "model_fingerprint",
        "preprocessing_fingerprint",
    ):
        _sha256(row[field], field=field)
    observer = row["observer_id_hash"]
    if observer is not None:
        _sha256(observer, field="observer_id_hash")
    return row


def _validate_distance(row: Mapping[str, object]) -> None:
    status = row["geographic_distance_status"]
    distance = row["geographic_distance_km"]
    reason = row["geographic_distance_reason"]
    if status == "available":
        if distance is None or reason is not None:
            raise ValueError(
                "available geographic distance requires value and no reason"
            )
    elif distance is not None or reason is None:
        raise ValueError(
            "unavailable geographic distance requires null value and reason"
        )
    scope = row["pool_scope"]
    geographic_scope = row["geographic_scope"]
    if scope == "global" and geographic_scope != "global":
        raise ValueError("global pool members require global geographic_scope")
    if scope == "local" and geographic_scope in {"global", "not_applicable"}:
        raise ValueError("local pool members require a local geographic scope")
    if scope == "safety_expansion" and geographic_scope != "not_applicable":
        raise ValueError("safety expansion members require not_applicable geography")


def _pool_id(group: Sequence[Mapping[str, object]]) -> str:
    digest = canonical_semantic_fingerprint(
        {
            "schema_version": "dynamic-reference-pool-id-v1",
            "members": [
                {field: row[field] for field in _MEMBER_INPUT_FIELDS} for row in group
            ],
        }
    ).removeprefix("sha256:")
    return f"dynamic-reference-pool:{digest}"


def _validate_plan_member_context(
    context: Mapping[str, object], members: pl.DataFrame
) -> None:
    pairs = {
        "run_id": "run_id",
        "flickr_query_id": "flickr_query_id",
        "flickr_photo_id": "flickr_photo_id",
        "organism_unit_id": "organism_unit_id",
        "visual_input_id": "visual_input_id",
        "query_embedding_fingerprint": "query_embedding_fingerprint",
        "scoring_stage": "scoring_stage",
        "query_route": "query_route",
        "candidate_set_id": "candidate_set_id",
        "candidate_set_fingerprint": "candidate_set_fingerprint",
        "registry_version": "registry_version",
        "reference_bank_version": "reference_bank_version",
        "reference_geography_index_fingerprint": (
            "reference_geography_index_fingerprint"
        ),
        "selection_policy_fingerprint": "selection_policy_fingerprint",
        "model_id": "model_id",
        "model_revision": "model_revision",
        "model_weights_sha256": "model_weights_sha256",
        "model_fingerprint": "model_fingerprint",
        "preprocessing_fingerprint": "preprocessing_fingerprint",
    }
    for plan_field, member_field in pairs.items():
        if (
            members[member_field].n_unique() != 1
            or members[member_field].item(0) != (context[plan_field])
        ):
            raise ValueError(f"dynamic pool member context conflicts on {member_field}")


def _validate_local_plan_state(
    context: Mapping[str, object], *, local_pool_ids: Sequence[object]
) -> None:
    status = context["local_pool_status"]
    reason = context["local_pool_unavailable_reason"]
    quality = context["query_coordinate_quality"]
    if status == "available":
        if not local_pool_ids or reason is not None:
            raise ValueError(
                "available local pool requires IDs and no unavailable reason"
            )
        if quality in _NO_LOCAL_QUERY_QUALITIES:
            raise ValueError("query coordinate quality cannot support a local pool")
    elif local_pool_ids or reason is None:
        raise ValueError("unavailable local pool requires no IDs and an exact reason")
    if quality in _NO_LOCAL_QUERY_QUALITIES and status != "unavailable":
        raise ValueError("no-geo query state must make the local pool unavailable")


def _validate_fingerprint(row: Mapping[str, object], *, field: str) -> None:
    payload = dict(row)
    fingerprint = payload.pop(field)
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError(f"dynamic pool {field} mismatch")


def _require_frame_schema(
    frame: pl.DataFrame, schema: dict[str, pl.DataType], *, label: str
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"dynamic pool {label} must be a Polars DataFrame")
    if frame.schema != schema:
        raise ValueError(f"dynamic pool {label} schema mismatch")


def _require_mapping_sequence(
    rows: Sequence[Mapping[str, object]], *, label: str
) -> None:
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise TypeError(f"{label} must be a sequence")
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError(f"{label} must contain mappings")


def _require_exact_fields(
    values: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    missing = expected - set(values)
    unexpected = set(values) - expected
    if missing or unexpected:
        raise ValueError(
            f"dynamic {label} fields mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    result = _nonnegative_int(value, field=field, maximum=maximum)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{field} must be an integer in [0, {maximum}]")
    return value


def _optional_nonnegative_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric or null")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical SHA-256 fingerprint")
    return text


__all__ = [
    "DYNAMIC_LOCAL_POOL_STATUSES",
    "DYNAMIC_POOL_DISTANCE_STATUSES",
    "DYNAMIC_POOL_GEOGRAPHIC_SCOPES",
    "DYNAMIC_POOL_MEMBERS_FILE",
    "DYNAMIC_POOL_MEMBER_SCHEMA_VERSION",
    "DYNAMIC_POOL_PLANS_FILE",
    "DYNAMIC_POOL_PLAN_SCHEMA_VERSION",
    "DYNAMIC_POOL_ROLES",
    "DYNAMIC_POOL_SCOPES",
    "DYNAMIC_POOL_SCORING_STAGES",
    "DYNAMIC_POOL_SUMMARY_FILE",
    "DYNAMIC_POOL_SUMMARY_SCHEMA_VERSION",
    "build_dynamic_reference_pool_members",
    "build_dynamic_reference_pool_plans",
    "build_dynamic_reference_pool_summaries",
    "dynamic_reference_pool_member_schema",
    "dynamic_reference_pool_plan_id",
    "dynamic_reference_pool_plan_schema",
    "dynamic_reference_pool_summary_schema",
    "validate_dynamic_reference_pool_artifacts",
    "validate_dynamic_reference_pool_members",
    "validate_dynamic_reference_pool_plans",
    "validate_dynamic_reference_pool_summaries",
    "write_dynamic_reference_pool_artifacts",
]
