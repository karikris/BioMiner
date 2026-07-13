from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from uuid import uuid4

import polars as pl

from biominer.storage.parquet import write_parquet


REFERENCE_OBSERVATIONS_SCHEMA_VERSION = "reference-observations-v1.2.0"
REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION = "reference-media-candidates-v1.0.0"
REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION = "reference-acquisition-plan-v1.1.0"
REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION = (
    "reference-acquisition-selections-v1.0.0"
)
REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION = "reference-media-objects-v1.1.0"
REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_SCHEMA_VERSION = (
    "reference-media-duplicate-relationships-v1.0.0"
)
REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION = "reference-review-queue-v1.0.0"
REFERENCE_REVIEW_DECISIONS_SCHEMA_VERSION = "reference-review-decisions-v1.0.0"

REFERENCE_OBSERVATIONS_FILE = "reference_observations.parquet"
REFERENCE_MEDIA_CANDIDATES_FILE = "reference_media_candidates.parquet"
REFERENCE_ACQUISITION_PLAN_FILE = "reference_acquisition_plan.parquet"
REFERENCE_ACQUISITION_SELECTIONS_FILE = "reference_acquisition_selections.parquet"
REFERENCE_MEDIA_OBJECTS_FILE = "reference_media_objects.parquet"
REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_FILE = (
    "reference_media_duplicate_relationships.parquet"
)
REFERENCE_REVIEW_QUEUE_FILE = "reference_review_queue.parquet"
REFERENCE_REVIEW_DECISIONS_FILE = "reference_review_decisions.parquet"

TAXON_RECONCILIATION_STATUSES = frozenset(
    {"accepted_key_exact", "accepted_name_synonym", "unresolved", "conflict"}
)
DOWNLOAD_STATUSES = frozenset(
    {"pending", "complete", "failed", "quarantined", "excluded"}
)
VERIFICATION_STATUSES = frozenset(
    {"unreviewed", "accepted", "rejected", "needs_review"}
)
LICENCE_POLICY_STATUSES = frozenset(
    {"unreviewed", "allowed", "research_only", "quarantined", "denied"}
)
DECODE_STATUSES = frozenset(
    {
        "not_attempted",
        "valid",
        "invalid_content_type",
        "decode_failed",
        "download_failed",
    }
)
REFERENCE_MEDIA_RASTER_CONTENT_TYPES = frozenset(
    {"image/gif", "image/jpeg", "image/png", "image/tiff", "image/webp"}
)
DUPLICATE_TYPES = frozenset(
    {
        "unique",
        "exact",
        "provider_mirror",
        "resized_copy",
        "near_identical_burst",
        "mixed",
        "unresolved_perceptual_candidate",
    }
)
DUPLICATE_RELATIONSHIP_TYPES = frozenset(
    {
        "exact",
        "provider_mirror",
        "resized_copy",
        "near_identical_burst",
        "perceptual_candidate",
    }
)
DUPLICATE_RESOLUTION_STATUSES = frozenset({"resolved", "review_required", "conflict"})
DUPLICATE_EVIDENCE_TYPES = frozenset(
    {
        "exact_sha256",
        "perceptual_hash",
        "provider_identifier",
        "same_observation",
        "metadata_conflict",
        "component_metadata_conflict",
    }
)
REFERENCE_LIFE_STAGES = frozenset({"adult", "larva", "pupa", "egg", "unknown"})
REFERENCE_VISUAL_DOMAINS = frozenset(
    {
        "live_field",
        "pinned_specimen",
        "artwork",
        "logo",
        "tattoo",
        "partial_wing",
        "dead_or_damaged_specimen",
        "ambiguous",
        "unsuitable",
    }
)
REFERENCE_VIEWS = frozenset(
    {"dorsal", "ventral", "lateral", "frontal", "oblique", "unknown"}
)
REFERENCE_REVIEW_CONFIDENCE_VALUES = frozenset({"high", "medium", "low", "unknown"})
REFERENCE_REVIEW_QUEUE_STATUSES = frozenset(
    {
        "pending",
        "in_review",
        "completed",
        "conflict",
        "second_review_required",
        "cancelled",
    }
)
REFERENCE_REVIEW_DECISION_STATUSES = frozenset({"verified", "excluded", "uncertain"})

_OBSERVATION_SORT = ["source", "source_observation_id"]
_MEDIA_SORT = ["source", "provider_media_id", "reference_observation_id"]
_PLAN_SORT = [
    "acquisition_plan_id",
    "candidate_accepted_taxon_key",
    "geo_cluster_id",
    "life_stage",
    "visual_domain",
    "source",
    "fallback_level",
]
_PLAN_PRIMARY_KEY = [
    "acquisition_plan_id",
    "candidate_accepted_taxon_key",
    "geo_cluster_id",
    "life_stage",
    "visual_domain",
]
_SELECTION_SORT = [
    "acquisition_plan_id",
    "candidate_accepted_taxon_key",
    "geo_cluster_id",
    "life_stage",
    "visual_domain",
    "selection_rank",
    "reference_media_id",
]
_MEDIA_OBJECT_SORT = ["reference_media_id"]
_MEDIA_DUPLICATE_RELATIONSHIP_SORT = [
    "duplicate_group_id",
    "left_reference_media_id",
    "right_reference_media_id",
]
_REFERENCE_REVIEW_QUEUE_SORT = [
    "review_priority",
    "reference_media_id",
    "review_request_id",
]
_REFERENCE_REVIEW_DECISION_SORT = [
    "reference_media_id",
    "review_round",
    "reviewed_at",
    "review_decision_id",
]
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PERCEPTUAL_HASH_PATTERN = re.compile(r"dhash128-v1:[0-9a-f]{32}\Z")
_DUPLICATE_GROUP_ID_PATTERN = re.compile(r"reference-duplicate-group:[0-9a-f]{32}\Z")
_DUPLICATE_RELATIONSHIP_ID_PATTERN = re.compile(
    r"reference-duplicate-relationship:[0-9a-f]{32}\Z"
)
_REVIEW_REQUEST_ID_PATTERN = re.compile(r"reference-review-request:[0-9a-f]{64}\Z")
_REVIEW_DECISION_ID_PATTERN = re.compile(r"reference-review-decision:[0-9a-f]{64}\Z")
_REFERENCE_MEDIA_ID_PATTERN = re.compile(r"reference-media:[0-9a-f]{64}\Z")
_REFERENCE_OBSERVATION_ID_PATTERN = re.compile(r"reference-observation:[0-9a-f]{64}\Z")


def reference_observation_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_observation_id": pl.String,
        "source": pl.String,
        "source_observation_id": pl.String,
        "source_taxon_id": pl.String,
        "supplied_scientific_name": pl.String,
        "accepted_taxon_key": pl.String,
        "reconciled_scientific_name": pl.String,
        "registry_version": pl.String,
        "taxon_reconciliation_status": pl.String,
        "identification_quality": pl.String,
        "community_taxon_status": pl.String,
        "identification_disagreement": pl.Boolean,
        "captive_or_cultivated": pl.Boolean,
        "observer_id": pl.String,
        "locality": pl.String,
        "life_stage": pl.String,
        "sex": pl.String,
        "observed_at": pl.Datetime("us", "UTC"),
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "coordinate_uncertainty": pl.Float64,
        "coordinates_obscured": pl.Boolean,
        "country": pl.String,
        "country_code": pl.String,
        "geo_cluster_id": pl.String,
        "distance_to_cluster_medoid_km": pl.Float64,
        "source_dataset_key": pl.String,
        "source_dataset_doi": pl.String,
        "source_record_url": pl.String,
        "source_record_hash": pl.String,
        "retrieved_at": pl.Datetime("us", "UTC"),
        "source_snapshot_version": pl.String,
        "source_query_fingerprint": pl.String,
        "fallback_level": pl.UInt8,
        "geospatial_issue": pl.Boolean,
        "preserved_specimen": pl.Boolean,
        "fossil": pl.Boolean,
        "occurrence_absent": pl.Boolean,
        "uncertain_taxon_match": pl.Boolean,
        "basis_of_record_suitable": pl.Boolean,
    }


def reference_media_candidate_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "provider_media_id": pl.String,
        "source": pl.String,
        "media_identifier": pl.String,
        "media_type": pl.String,
        "width": pl.UInt32,
        "height": pl.UInt32,
        "creator": pl.String,
        "rights_holder": pl.String,
        "licence": pl.String,
        "licence_uri": pl.String,
        "attribution": pl.String,
        "occurrence_licence": pl.String,
        "original_provider": pl.String,
        "media_position": pl.UInt32,
        "source_checksum": pl.String,
        "source_checksum_algorithm": pl.String,
        "download_status": pl.String,
        "verification_status": pl.String,
        "exclusion_reason": pl.String,
        "licence_policy_status": pl.String,
        "retrieved_at": pl.Datetime("us", "UTC"),
        "source_snapshot_version": pl.String,
    }


def reference_acquisition_plan_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "acquisition_plan_id": pl.String,
        "target_accepted_taxon_key": pl.String,
        "candidate_set_id": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "geo_cluster_id": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "source": pl.String,
        "requested_count": pl.UInt32,
        "existing_support_count": pl.UInt32,
        "available_candidate_count": pl.UInt32,
        "selected_candidate_count": pl.UInt32,
        "shortfall_count": pl.UInt32,
        "fallback_level": pl.UInt8,
        "selection_strategy": pl.String,
        "selection_seed": pl.UInt64,
        "max_distance_km": pl.Float64,
        "licence_policy_version": pl.String,
        "source_snapshot_version": pl.String,
        "plan_configuration_fingerprint": pl.String,
        "created_at": pl.Datetime("us", "UTC"),
    }


def reference_acquisition_selection_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_selection_id": pl.String,
        "acquisition_plan_id": pl.String,
        "target_accepted_taxon_key": pl.String,
        "candidate_set_id": pl.String,
        "source_candidate_set_id": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "geo_cluster_id": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "source": pl.String,
        "fallback_level": pl.UInt8,
        "selection_rank": pl.UInt32,
        "selection_round": pl.String,
        "distance_to_cluster_medoid_km": pl.Float64,
        "observer_id": pl.String,
        "observed_date": pl.Date,
        "locality": pl.String,
        "background_group_id": pl.String,
        "licence": pl.String,
        "source_snapshot_version": pl.String,
        "selection_strategy": pl.String,
        "selection_seed": pl.UInt64,
        "plan_configuration_fingerprint": pl.String,
        "selected_at": pl.Datetime("us", "UTC"),
    }


def reference_media_object_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "source_object_uri": pl.String,
        "content_type": pl.String,
        "source_byte_count": pl.UInt64,
        "decoded_width": pl.UInt32,
        "decoded_height": pl.UInt32,
        "sha256": pl.String,
        "perceptual_hash": pl.String,
        "duplicate_group_id": pl.String,
        "duplicate_type": pl.String,
        "canonical_reference_media_id": pl.String,
        "provider_mirror_ids": pl.List(pl.String),
        "downloaded_at": pl.Datetime("us", "UTC"),
        "download_attempt_count": pl.UInt32,
        "licence_policy_status": pl.String,
        "decode_status": pl.String,
        "quarantine_reason": pl.String,
        "object_fingerprint": pl.String,
    }


def reference_media_duplicate_relationship_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "duplicate_relationship_id": pl.String,
        "duplicate_group_id": pl.String,
        "canonical_reference_media_id": pl.String,
        "left_reference_media_id": pl.String,
        "right_reference_media_id": pl.String,
        "left_reference_observation_id": pl.String,
        "right_reference_observation_id": pl.String,
        "left_source": pl.String,
        "right_source": pl.String,
        "left_provider_media_id": pl.String,
        "right_provider_media_id": pl.String,
        "relationship_type": pl.String,
        "evidence_types": pl.List(pl.String),
        "sha256_equal": pl.Boolean,
        "perceptual_hash_distance": pl.UInt16,
        "same_observation": pl.Boolean,
        "provider_mirror": pl.Boolean,
        "resolution_status": pl.String,
        "policy_version": pl.String,
        "policy_fingerprint": pl.String,
    }


def reference_review_queue_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "canonical_reference_media_id": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "durable_preview_uri": pl.String,
        "media_object_fingerprint": pl.String,
        "duplicate_group_id": pl.String,
        "source": pl.String,
        "provider_media_id": pl.String,
        "provider_verification_status": pl.String,
        "creator": pl.String,
        "rights_holder": pl.String,
        "licence": pl.String,
        "licence_uri": pl.String,
        "licence_policy_status": pl.String,
        "attribution": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "review_reason": pl.String,
        "review_priority": pl.UInt32,
        "required_review_count": pl.UInt8,
        "review_status": pl.String,
        "created_at": pl.Datetime("us", "UTC"),
        "reference_bank_version": pl.String,
        "input_fingerprint": pl.String,
    }


def reference_review_decision_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "review_decision_id": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "review_round": pl.UInt16,
        "verified_by": pl.String,
        "reviewed_at": pl.Datetime("us", "UTC"),
        "target_identity_verified": pl.Boolean,
        "verification_status": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "review_confidence": pl.String,
        "review_notes": pl.String,
        "exclusion_reason": pl.String,
        "second_review_required": pl.Boolean,
        "conflicts_with_decision_id": pl.String,
        "decision_source_hash": pl.String,
    }


def make_reference_observation_id(source: str, source_observation_id: str) -> str:
    return "reference-observation:" + _semantic_digest(
        {
            "source": _required_text(source, field="source").casefold(),
            "source_observation_id": _required_text(
                source_observation_id,
                field="source_observation_id",
            ),
        }
    ).removeprefix("sha256:")


def make_reference_media_id(
    source: str,
    provider_media_id: str,
    reference_observation_id: str,
) -> str:
    return "reference-media:" + _semantic_digest(
        {
            "source": _required_text(source, field="source").casefold(),
            "provider_media_id": _required_text(
                provider_media_id,
                field="provider_media_id",
            ),
            "reference_observation_id": _required_text(
                reference_observation_id,
                field="reference_observation_id",
            ),
        }
    ).removeprefix("sha256:")


def make_acquisition_plan_id(
    *,
    target_accepted_taxon_key: str,
    candidate_set_id: str,
    plan_configuration_fingerprint: str,
) -> str:
    fingerprint = _full_sha256(
        plan_configuration_fingerprint,
        field="plan_configuration_fingerprint",
    )
    return "reference-plan:" + _semantic_digest(
        {
            "target_accepted_taxon_key": _required_text(
                target_accepted_taxon_key,
                field="target_accepted_taxon_key",
            ),
            "candidate_set_id": _required_text(
                candidate_set_id,
                field="candidate_set_id",
            ),
            "plan_configuration_fingerprint": fingerprint,
        }
    ).removeprefix("sha256:")


def make_reference_selection_id(
    *,
    acquisition_plan_id: str,
    reference_media_id: str,
    candidate_accepted_taxon_key: str,
    geo_cluster_id: str,
    life_stage: str,
    visual_domain: str,
) -> str:
    return "reference-selection:" + _semantic_digest(
        {
            "acquisition_plan_id": _required_text(
                acquisition_plan_id,
                field="acquisition_plan_id",
            ),
            "reference_media_id": _required_text(
                reference_media_id,
                field="reference_media_id",
            ),
            "candidate_accepted_taxon_key": _required_text(
                candidate_accepted_taxon_key,
                field="candidate_accepted_taxon_key",
            ),
            "geo_cluster_id": _required_text(geo_cluster_id, field="geo_cluster_id"),
            "life_stage": _required_text(life_stage, field="life_stage"),
            "visual_domain": _required_text(visual_domain, field="visual_domain"),
        }
    ).removeprefix("sha256:")


def make_reference_review_request_id(
    *,
    reference_media_id: str,
    media_object_fingerprint: str,
    reference_bank_version: str,
    input_fingerprint: str,
) -> str:
    return "reference-review-request:" + _semantic_digest(
        {
            "schema_version": REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION,
            "reference_media_id": _canonical_required_text(
                reference_media_id,
                field="reference_media_id",
            ),
            "media_object_fingerprint": _canonical_full_sha256(
                media_object_fingerprint,
                field="media_object_fingerprint",
            ),
            "reference_bank_version": _canonical_required_text(
                reference_bank_version,
                field="reference_bank_version",
            ),
            "input_fingerprint": _canonical_full_sha256(
                input_fingerprint,
                field="input_fingerprint",
            ),
        }
    ).removeprefix("sha256:")


def make_reference_review_decision_id(
    *,
    review_request_id: str,
    reference_media_id: str,
    review_round: int,
    verified_by: str,
    reviewed_at: datetime,
    target_identity_verified: bool | None,
    verification_status: str,
    life_stage: str,
    visual_domain: str,
    view: str,
    review_confidence: str,
    review_notes: str | None,
    exclusion_reason: str | None,
    second_review_required: bool,
    conflicts_with_decision_id: str | None,
) -> str:
    if target_identity_verified is not None and not isinstance(
        target_identity_verified,
        bool,
    ):
        raise ValueError("target_identity_verified must be Boolean or null")
    if not isinstance(second_review_required, bool):
        raise ValueError("second_review_required must be Boolean")
    return "reference-review-decision:" + _semantic_digest(
        {
            "schema_version": REFERENCE_REVIEW_DECISIONS_SCHEMA_VERSION,
            "review_request_id": _canonical_required_text(
                review_request_id,
                field="review_request_id",
            ),
            "reference_media_id": _canonical_required_text(
                reference_media_id,
                field="reference_media_id",
            ),
            "review_round": _positive_int(review_round, field="review_round"),
            "verified_by": _canonical_required_text(
                verified_by,
                field="verified_by",
            ),
            "reviewed_at": _utc_datetime_text(reviewed_at, field="reviewed_at"),
            "target_identity_verified": target_identity_verified,
            "verification_status": _canonical_choice(
                verification_status,
                field="verification_status",
                choices=REFERENCE_REVIEW_DECISION_STATUSES,
            ),
            "life_stage": _canonical_choice(
                life_stage,
                field="life_stage",
                choices=REFERENCE_LIFE_STAGES,
            ),
            "visual_domain": _canonical_choice(
                visual_domain,
                field="visual_domain",
                choices=REFERENCE_VISUAL_DOMAINS,
            ),
            "view": _canonical_choice(view, field="view", choices=REFERENCE_VIEWS),
            "review_confidence": _canonical_choice(
                review_confidence,
                field="review_confidence",
                choices=REFERENCE_REVIEW_CONFIDENCE_VALUES,
            ),
            "review_notes": _nullable_nonblank_text(
                review_notes,
                field="review_notes",
            ),
            "exclusion_reason": _nullable_nonblank_text(
                exclusion_reason,
                field="exclusion_reason",
            ),
            "second_review_required": second_review_required,
            "conflicts_with_decision_id": _nullable_nonblank_text(
                conflicts_with_decision_id,
                field="conflicts_with_decision_id",
            ),
        }
    ).removeprefix("sha256:")


def reference_observations_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    frame = _frame(
        rows, schema=reference_observation_schema(), sort_by=_OBSERVATION_SORT
    )
    validate_reference_observations(frame)
    return frame


def reference_media_candidates_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    frame = _frame(rows, schema=reference_media_candidate_schema(), sort_by=_MEDIA_SORT)
    validate_reference_media_candidates(frame)
    return frame


def reference_acquisition_plan_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    frame = _frame(rows, schema=reference_acquisition_plan_schema(), sort_by=_PLAN_SORT)
    validate_reference_acquisition_plan(frame)
    return frame


def reference_acquisition_selections_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    frame = _frame(
        rows,
        schema=reference_acquisition_selection_schema(),
        sort_by=_SELECTION_SORT,
    )
    validate_reference_acquisition_selections(frame)
    return frame


def reference_media_objects_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    frame = _frame(
        rows,
        schema=reference_media_object_schema(),
        sort_by=_MEDIA_OBJECT_SORT,
    )
    validate_reference_media_objects(frame)
    return frame


def reference_media_duplicate_relationships_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    frame = _frame(
        rows,
        schema=reference_media_duplicate_relationship_schema(),
        sort_by=_MEDIA_DUPLICATE_RELATIONSHIP_SORT,
    )
    validate_reference_media_duplicate_relationships(frame)
    return frame


def reference_review_queue_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    frame = _frame(
        rows,
        schema=reference_review_queue_schema(),
        sort_by=_REFERENCE_REVIEW_QUEUE_SORT,
    )
    validate_reference_review_queue(frame)
    return frame


def reference_review_decisions_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    frame = _frame(
        rows,
        schema=reference_review_decision_schema(),
        sort_by=_REFERENCE_REVIEW_DECISION_SORT,
    )
    validate_reference_review_decisions(frame)
    return frame


def validate_reference_observations(frame: pl.DataFrame) -> None:
    _validate_physical_frame(
        frame,
        schema=reference_observation_schema(),
        schema_version=REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        sort_by=_OBSERVATION_SORT,
        primary_key=_OBSERVATION_SORT,
        artifact="reference observations",
    )
    if frame["reference_observation_id"].n_unique() != frame.height:
        raise ValueError("reference observation IDs must be unique")
    for row in frame.iter_rows(named=True):
        source = _required_text(row["source"], field="source")
        source_id = _required_text(
            row["source_observation_id"],
            field="source_observation_id",
        )
        expected_id = make_reference_observation_id(source, source_id)
        if row["reference_observation_id"] != expected_id:
            raise ValueError("reference observation ID mismatch")
        _required_text(row["registry_version"], field="registry_version")
        _required_text(row["life_stage"], field="life_stage")
        _required_text(
            row["source_snapshot_version"],
            field="source_snapshot_version",
        )
        status = _choice(
            row["taxon_reconciliation_status"],
            field="taxon_reconciliation_status",
            choices=TAXON_RECONCILIATION_STATUSES,
        )
        accepted_key = _optional_text(row["accepted_taxon_key"])
        reconciled_name = _optional_text(row["reconciled_scientific_name"])
        if status in {"accepted_key_exact", "accepted_name_synonym"}:
            if accepted_key is None or reconciled_name is None:
                raise ValueError("accepted reconciliation requires accepted identity")
        if status in {"unresolved", "conflict"} and not row["uncertain_taxon_match"]:
            raise ValueError("unresolved reconciliation must be marked uncertain")
        _full_sha256(row["source_record_hash"], field="source_record_hash")
        _full_sha256(
            row["source_query_fingerprint"],
            field="source_query_fingerprint",
        )
        _fallback_level(row["fallback_level"])
        if row["retrieved_at"] is None:
            raise ValueError("retrieved_at is required")
        observed_at = row["observed_at"]
        if observed_at is not None and observed_at > row["retrieved_at"]:
            raise ValueError("observed_at cannot be after retrieved_at")
        _coordinate_pair(row["latitude"], row["longitude"])
        _optional_nonnegative_finite(
            row["coordinate_uncertainty"],
            field="coordinate_uncertainty",
        )
        distance = _optional_nonnegative_finite(
            row["distance_to_cluster_medoid_km"],
            field="distance_to_cluster_medoid_km",
        )
        if distance is not None and (
            row["latitude"] is None
            or row["longitude"] is None
            or _optional_text(row["geo_cluster_id"]) is None
        ):
            raise ValueError("cluster distance requires coordinates and geo_cluster_id")
        country_code = _optional_text(row["country_code"])
        if country_code is not None and (
            len(country_code) != 2 or country_code != country_code.upper()
        ):
            raise ValueError("country_code must be uppercase ISO alpha-2")
        for field in ("uncertain_taxon_match", "basis_of_record_suitable"):
            if row[field] is None:
                raise ValueError(f"{field} must be boolean")


def validate_reference_media_candidates(frame: pl.DataFrame) -> None:
    _validate_physical_frame(
        frame,
        schema=reference_media_candidate_schema(),
        schema_version=REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
        sort_by=_MEDIA_SORT,
        primary_key=_MEDIA_SORT,
        artifact="reference media candidates",
    )
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("reference media IDs must be unique")
    for row in frame.iter_rows(named=True):
        source = _required_text(row["source"], field="source")
        provider_media_id = _required_text(
            row["provider_media_id"],
            field="provider_media_id",
        )
        observation_id = _required_text(
            row["reference_observation_id"],
            field="reference_observation_id",
        )
        expected_id = make_reference_media_id(
            source,
            provider_media_id,
            observation_id,
        )
        if row["reference_media_id"] != expected_id:
            raise ValueError("reference media ID mismatch")
        _required_text(row["media_identifier"], field="media_identifier")
        _required_text(row["media_type"], field="media_type")
        _required_text(
            row["source_snapshot_version"],
            field="source_snapshot_version",
        )
        _choice(
            row["download_status"],
            field="download_status",
            choices=DOWNLOAD_STATUSES,
        )
        _choice(
            row["verification_status"],
            field="verification_status",
            choices=VERIFICATION_STATUSES,
        )
        _choice(
            row["licence_policy_status"],
            field="licence_policy_status",
            choices=LICENCE_POLICY_STATUSES,
        )
        checksum = _optional_text(row["source_checksum"])
        checksum_algorithm = _optional_text(row["source_checksum_algorithm"])
        if (checksum is None) != (checksum_algorithm is None):
            raise ValueError("source checksum and algorithm must be populated together")
        if row["retrieved_at"] is None:
            raise ValueError("retrieved_at is required")


def validate_reference_acquisition_plan(frame: pl.DataFrame) -> None:
    _validate_physical_frame(
        frame,
        schema=reference_acquisition_plan_schema(),
        schema_version=REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
        sort_by=_PLAN_SORT,
        primary_key=_PLAN_PRIMARY_KEY,
        artifact="reference acquisition plan",
    )
    for row in frame.iter_rows(named=True):
        target_key = _required_text(
            row["target_accepted_taxon_key"],
            field="target_accepted_taxon_key",
        )
        candidate_set_id = _required_text(
            row["candidate_set_id"],
            field="candidate_set_id",
        )
        fingerprint = _full_sha256(
            row["plan_configuration_fingerprint"],
            field="plan_configuration_fingerprint",
        )
        expected_id = make_acquisition_plan_id(
            target_accepted_taxon_key=target_key,
            candidate_set_id=candidate_set_id,
            plan_configuration_fingerprint=fingerprint,
        )
        if row["acquisition_plan_id"] != expected_id:
            raise ValueError("reference acquisition plan ID mismatch")
        for field in (
            "candidate_accepted_taxon_key",
            "scientific_name",
            "geo_cluster_id",
            "life_stage",
            "visual_domain",
            "source",
            "selection_strategy",
            "licence_policy_version",
            "source_snapshot_version",
        ):
            _required_text(row[field], field=field)
        requested = int(row["requested_count"])
        existing = int(row["existing_support_count"])
        available = int(row["available_candidate_count"])
        selected = int(row["selected_candidate_count"])
        shortfall = int(row["shortfall_count"])
        if selected > requested or selected > available:
            raise ValueError("selected reference count exceeds requested or available")
        if shortfall != requested - selected:
            raise ValueError("reference shortfall must equal requested minus selected")
        if existing < 0:
            raise ValueError("existing support count must be nonnegative")
        _fallback_level(row["fallback_level"])
        _optional_nonnegative_finite(row["max_distance_km"], field="max_distance_km")
        if row["created_at"] is None:
            raise ValueError("created_at is required")


def validate_reference_acquisition_selections(frame: pl.DataFrame) -> None:
    _validate_physical_frame(
        frame,
        schema=reference_acquisition_selection_schema(),
        schema_version=REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION,
        sort_by=_SELECTION_SORT,
        primary_key=["reference_selection_id"],
        artifact="reference acquisition selections",
    )
    if frame["reference_selection_id"].n_unique() != frame.height:
        raise ValueError("reference selection IDs must be unique")
    selected_media_count = frame.select(
        pl.struct(["acquisition_plan_id", "reference_media_id"]).n_unique()
    ).item()
    if selected_media_count != frame.height:
        raise ValueError(
            "reference media may be selected only once per acquisition plan"
        )
    selected_observation_count = frame.select(
        pl.struct(["acquisition_plan_id", "reference_observation_id"]).n_unique()
    ).item()
    if selected_observation_count != frame.height:
        raise ValueError(
            "reference observations may fill only one quota slot per acquisition plan"
        )
    for row in frame.iter_rows(named=True):
        for field in (
            "acquisition_plan_id",
            "target_accepted_taxon_key",
            "candidate_set_id",
            "source_candidate_set_id",
            "candidate_accepted_taxon_key",
            "scientific_name",
            "geo_cluster_id",
            "life_stage",
            "visual_domain",
            "reference_media_id",
            "reference_observation_id",
            "source",
            "selection_strategy",
            "source_snapshot_version",
        ):
            _required_text(row[field], field=field)
        expected_id = make_reference_selection_id(
            acquisition_plan_id=str(row["acquisition_plan_id"]),
            reference_media_id=str(row["reference_media_id"]),
            candidate_accepted_taxon_key=str(row["candidate_accepted_taxon_key"]),
            geo_cluster_id=str(row["geo_cluster_id"]),
            life_stage=str(row["life_stage"]),
            visual_domain=str(row["visual_domain"]),
        )
        if row["reference_selection_id"] != expected_id:
            raise ValueError("reference selection ID mismatch")
        if row["selection_round"] not in {
            "independent_observation",
            "same_observation_fallback",
        }:
            raise ValueError("unsupported reference selection round")
        _fallback_level(row["fallback_level"])
        _optional_nonnegative_finite(
            row["distance_to_cluster_medoid_km"],
            field="distance_to_cluster_medoid_km",
        )
        _full_sha256(
            row["plan_configuration_fingerprint"],
            field="plan_configuration_fingerprint",
        )
        if row["selected_at"] is None:
            raise ValueError("selected_at is required")


def validate_reference_media_objects(frame: pl.DataFrame) -> None:
    _validate_physical_frame(
        frame,
        schema=reference_media_object_schema(),
        schema_version=REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
        sort_by=_MEDIA_OBJECT_SORT,
        primary_key=_MEDIA_OBJECT_SORT,
        artifact="reference media objects",
    )
    for row in frame.iter_rows(named=True):
        _required_text(row["reference_media_id"], field="reference_media_id")
        policy_status = _choice(
            row["licence_policy_status"],
            field="licence_policy_status",
            choices=LICENCE_POLICY_STATUSES,
        )
        decode_status = _choice(
            row["decode_status"],
            field="decode_status",
            choices=DECODE_STATUSES,
        )
        _full_sha256(row["object_fingerprint"], field="object_fingerprint")
        attempt_count = _nonnegative_int(
            row["download_attempt_count"],
            field="download_attempt_count",
        )
        mirrors = row["provider_mirror_ids"]
        if not isinstance(mirrors, list):
            raise ValueError("provider_mirror_ids must be a non-null list")
        normalized_mirrors = [
            _required_text(value, field="provider_mirror_ids") for value in mirrors
        ]
        if normalized_mirrors != sorted(set(normalized_mirrors)):
            raise ValueError("provider_mirror_ids must be sorted and unique")
        perceptual_hash = row["perceptual_hash"]
        if perceptual_hash is not None:
            _perceptual_hash(perceptual_hash, field="perceptual_hash")
        group_fields = (
            row["duplicate_group_id"],
            row["duplicate_type"],
            row["canonical_reference_media_id"],
        )
        populated_group_fields = sum(value is not None for value in group_fields)
        if populated_group_fields not in {0, len(group_fields)}:
            raise ValueError("duplicate group fields must be populated together")
        if populated_group_fields:
            duplicate_group_id = _required_text(
                row["duplicate_group_id"], field="duplicate_group_id"
            )
            if _DUPLICATE_GROUP_ID_PATTERN.fullmatch(duplicate_group_id) is None:
                raise ValueError("duplicate_group_id has an unsupported namespace")
            _choice(
                row["duplicate_type"],
                field="duplicate_type",
                choices=DUPLICATE_TYPES,
            )
            _required_text(
                row["canonical_reference_media_id"],
                field="canonical_reference_media_id",
            )
        elif normalized_mirrors:
            raise ValueError("provider mirror IDs require a resolved duplicate group")
        if row["reference_media_id"] in normalized_mirrors:
            raise ValueError("provider_mirror_ids cannot contain the row itself")
        if any(
            not value.startswith("reference-media:") for value in normalized_mirrors
        ):
            raise ValueError("provider_mirror_ids has an unsupported namespace")

        if decode_status == "valid":
            if policy_status not in {"allowed", "research_only"}:
                raise ValueError(
                    "valid reference media requires an allowed or research-only licence"
                )
            source_object_uri = _required_text(
                row["source_object_uri"],
                field="source_object_uri",
            )
            _choice(
                row["content_type"],
                field="content_type",
                choices=REFERENCE_MEDIA_RASTER_CONTENT_TYPES,
            )
            source_byte_count = _positive_int(
                row["source_byte_count"],
                field="source_byte_count",
            )
            decoded_width = _positive_int(
                row["decoded_width"],
                field="decoded_width",
            )
            decoded_height = _positive_int(
                row["decoded_height"],
                field="decoded_height",
            )
            if (
                min(source_byte_count, decoded_width, decoded_height, attempt_count)
                <= 0
            ):
                raise ValueError(
                    "valid reference media requires positive object metrics"
                )
            sha256 = _full_sha256(row["sha256"], field="sha256")
            _perceptual_hash(row["perceptual_hash"], field="perceptual_hash")
            if sha256.removeprefix("sha256:") not in source_object_uri:
                raise ValueError("source object URI must contain the SHA-256 digest")
            if row["downloaded_at"] is None:
                raise ValueError("valid reference media requires downloaded_at")
            if row["quarantine_reason"] is not None:
                raise ValueError(
                    "valid reference media cannot have a quarantine reason"
                )
            continue

        for field in (
            "source_object_uri",
            "source_byte_count",
            "decoded_width",
            "decoded_height",
            "sha256",
            "downloaded_at",
        ):
            if row[field] is not None:
                raise ValueError(f"non-valid reference media cannot populate {field}")
        if row["content_type"] is not None:
            _required_text(row["content_type"], field="content_type")
        _required_text(row["quarantine_reason"], field="quarantine_reason")
        if perceptual_hash is not None or populated_group_fields or normalized_mirrors:
            raise ValueError(
                "non-valid reference media cannot populate deduplication state"
            )

    grouped = {
        str(group_id[0] if isinstance(group_id, tuple) else group_id): group
        for group_id, group in frame.filter(pl.col("duplicate_group_id").is_not_null())
        .partition_by("duplicate_group_id", as_dict=True)
        .items()
    }
    for group_id, group in grouped.items():
        canonical_ids = group["canonical_reference_media_id"].unique().to_list()
        if len(canonical_ids) != 1:
            raise ValueError(
                f"duplicate group {group_id} has inconsistent canonical IDs"
            )
        canonical_id = str(canonical_ids[0])
        if canonical_id not in set(group["reference_media_id"].to_list()):
            raise ValueError(f"duplicate group {group_id} canonical row is missing")
        duplicate_types = group["duplicate_type"].unique().to_list()
        if len(duplicate_types) != 1:
            raise ValueError(
                f"duplicate group {group_id} has inconsistent duplicate types"
            )
        if duplicate_types == ["unique"] and group.height != 1:
            raise ValueError(
                "unique duplicate groups must contain exactly one media row"
            )


def validate_reference_media_duplicate_relationships(frame: pl.DataFrame) -> None:
    _validate_physical_frame(
        frame,
        schema=reference_media_duplicate_relationship_schema(),
        schema_version=REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_SCHEMA_VERSION,
        sort_by=_MEDIA_DUPLICATE_RELATIONSHIP_SORT,
        primary_key=["duplicate_relationship_id"],
        artifact="reference media duplicate relationships",
    )
    if (
        frame.select("left_reference_media_id", "right_reference_media_id")
        .unique()
        .height
        != frame.height
    ):
        raise ValueError("duplicate relationship media pairs must be unique")
    for row in frame.iter_rows(named=True):
        relationship_id = _required_text(
            row["duplicate_relationship_id"],
            field="duplicate_relationship_id",
        )
        if _DUPLICATE_RELATIONSHIP_ID_PATTERN.fullmatch(relationship_id) is None:
            raise ValueError("duplicate relationship ID has an unsupported namespace")
        for field in (
            "canonical_reference_media_id",
            "left_reference_media_id",
            "right_reference_media_id",
            "left_reference_observation_id",
            "right_reference_observation_id",
            "left_source",
            "right_source",
            "left_provider_media_id",
            "right_provider_media_id",
            "policy_version",
        ):
            _required_text(row[field], field=field)
        duplicate_group_id = _required_text(
            row["duplicate_group_id"], field="duplicate_group_id"
        )
        if _DUPLICATE_GROUP_ID_PATTERN.fullmatch(duplicate_group_id) is None:
            raise ValueError("duplicate_group_id has an unsupported namespace")
        left_id = str(row["left_reference_media_id"])
        right_id = str(row["right_reference_media_id"])
        if left_id >= right_id:
            raise ValueError(
                "duplicate relationship media IDs must be ordered and distinct"
            )
        expected_left_id = make_reference_media_id(
            str(row["left_source"]),
            str(row["left_provider_media_id"]),
            str(row["left_reference_observation_id"]),
        )
        expected_right_id = make_reference_media_id(
            str(row["right_source"]),
            str(row["right_provider_media_id"]),
            str(row["right_reference_observation_id"]),
        )
        if left_id != expected_left_id or right_id != expected_right_id:
            raise ValueError(
                "duplicate relationship endpoint provenance conflicts with its media ID"
            )
        relationship_type = _choice(
            row["relationship_type"],
            field="relationship_type",
            choices=DUPLICATE_RELATIONSHIP_TYPES,
        )
        evidence = row["evidence_types"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("duplicate relationship evidence_types must be non-empty")
        normalized_evidence = [
            _choice(value, field="evidence_types", choices=DUPLICATE_EVIDENCE_TYPES)
            for value in evidence
        ]
        if normalized_evidence != sorted(set(normalized_evidence)):
            raise ValueError(
                "duplicate relationship evidence_types must be sorted and unique"
            )
        expected_relationship_id = (
            "reference-duplicate-relationship:"
            + _semantic_digest(
                {
                    "left_reference_media_id": left_id,
                    "right_reference_media_id": right_id,
                    "evidence_types": normalized_evidence,
                }
            ).removeprefix("sha256:")[:32]
        )
        if relationship_id != expected_relationship_id:
            raise ValueError("duplicate relationship ID does not match its evidence")
        distance = row["perceptual_hash_distance"]
        if distance is not None:
            _nonnegative_int(distance, field="perceptual_hash_distance")
            if distance > 128:
                raise ValueError("perceptual_hash_distance cannot exceed 128")
        if (distance is not None) != ("perceptual_hash" in normalized_evidence):
            raise ValueError("perceptual distance must match perceptual hash evidence")
        for field in ("sha256_equal", "same_observation", "provider_mirror"):
            if row[field] is None:
                raise ValueError(f"{field} must be a non-null Boolean")
        sha256_equal = bool(row["sha256_equal"])
        same_observation = bool(row["same_observation"])
        provider_mirror = bool(row["provider_mirror"])
        if sha256_equal != ("exact_sha256" in normalized_evidence):
            raise ValueError("SHA-256 equality must match exact hash evidence")
        if same_observation != ("same_observation" in normalized_evidence):
            raise ValueError("same-observation flag must match observation evidence")
        if provider_mirror != ("provider_identifier" in normalized_evidence):
            raise ValueError(
                "provider-mirror flag must match provider identifier evidence"
            )
        if sha256_equal != (relationship_type == "exact"):
            raise ValueError("exact relationship type must match SHA-256 equality")
        if relationship_type == "provider_mirror" and not provider_mirror:
            raise ValueError(
                "provider-mirror relationships require provider identifier evidence"
            )
        if relationship_type in {"resized_copy", "near_identical_burst"} and (
            not same_observation or distance is None
        ):
            raise ValueError(
                "resized and burst relationships require perceptual "
                "same-observation evidence"
            )
        resolution_status = _choice(
            row["resolution_status"],
            field="resolution_status",
            choices=DUPLICATE_RESOLUTION_STATUSES,
        )
        if relationship_type == "perceptual_candidate" and resolution_status == (
            "resolved"
        ):
            raise ValueError("perceptual candidates cannot be resolved automatically")
        if {"metadata_conflict", "component_metadata_conflict"} & set(
            normalized_evidence
        ) and resolution_status != "conflict":
            raise ValueError("metadata conflict evidence requires conflict resolution")
        _full_sha256(row["policy_fingerprint"], field="policy_fingerprint")


def validate_reference_review_queue(frame: pl.DataFrame) -> None:
    _validate_physical_frame(
        frame,
        schema=reference_review_queue_schema(),
        schema_version=REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION,
        sort_by=_REFERENCE_REVIEW_QUEUE_SORT,
        primary_key=["review_request_id"],
        artifact="reference review queue",
    )
    for row in frame.iter_rows(named=True):
        request_id = _canonical_required_text(
            row["review_request_id"],
            field="review_request_id",
        )
        if _REVIEW_REQUEST_ID_PATTERN.fullmatch(request_id) is None:
            raise ValueError("review_request_id has an unsupported namespace")
        reference_media_id = _canonical_required_text(
            row["reference_media_id"],
            field="reference_media_id",
        )
        if _REFERENCE_MEDIA_ID_PATTERN.fullmatch(reference_media_id) is None:
            raise ValueError("reference_media_id has an unsupported namespace")
        for field in (
            "reference_observation_id",
            "canonical_reference_media_id",
            "durable_preview_uri",
            "source",
            "provider_media_id",
            "licence",
            "review_reason",
            "reference_bank_version",
        ):
            _canonical_required_text(row[field], field=field)
        canonical_media_id = str(row["canonical_reference_media_id"])
        if _REFERENCE_MEDIA_ID_PATTERN.fullmatch(canonical_media_id) is None:
            raise ValueError(
                "canonical_reference_media_id has an unsupported namespace"
            )
        observation_id = str(row["reference_observation_id"])
        if _REFERENCE_OBSERVATION_ID_PATTERN.fullmatch(observation_id) is None:
            raise ValueError("reference_observation_id has an unsupported namespace")
        expected_media_id = make_reference_media_id(
            str(row["source"]),
            str(row["provider_media_id"]),
            observation_id,
        )
        if reference_media_id != expected_media_id:
            raise ValueError(
                "reference_media_id does not match its provider provenance"
            )
        duplicate_group_id = _canonical_required_text(
            row["duplicate_group_id"],
            field="duplicate_group_id",
        )
        if _DUPLICATE_GROUP_ID_PATTERN.fullmatch(duplicate_group_id) is None:
            raise ValueError("duplicate_group_id has an unsupported namespace")
        media_object_fingerprint = _canonical_full_sha256(
            row["media_object_fingerprint"],
            field="media_object_fingerprint",
        )
        input_fingerprint = _canonical_full_sha256(
            row["input_fingerprint"],
            field="input_fingerprint",
        )
        reference_bank_version = str(row["reference_bank_version"])
        expected_id = make_reference_review_request_id(
            reference_media_id=reference_media_id,
            media_object_fingerprint=media_object_fingerprint,
            reference_bank_version=reference_bank_version,
            input_fingerprint=input_fingerprint,
        )
        if request_id != expected_id:
            raise ValueError("review request ID does not match its immutable inputs")
        accepted_key = _nullable_nonblank_text(
            row["accepted_taxon_key"],
            field="accepted_taxon_key",
        )
        scientific_name = _nullable_nonblank_text(
            row["scientific_name"],
            field="scientific_name",
        )
        if (accepted_key is None) != (scientific_name is None):
            raise ValueError(
                "accepted_taxon_key and scientific_name must be populated together"
            )
        _canonical_choice(
            row["provider_verification_status"],
            field="provider_verification_status",
            choices=VERIFICATION_STATUSES,
        )
        _canonical_choice(
            row["licence_policy_status"],
            field="licence_policy_status",
            choices=LICENCE_POLICY_STATUSES,
        )
        proposed_values = (
            ("life_stage", REFERENCE_LIFE_STAGES),
            ("visual_domain", REFERENCE_VISUAL_DOMAINS),
            ("view", REFERENCE_VIEWS),
        )
        for field, choices in proposed_values:
            if row[field] is not None:
                _canonical_choice(row[field], field=field, choices=choices)
        _canonical_choice(
            row["review_status"],
            field="review_status",
            choices=REFERENCE_REVIEW_QUEUE_STATUSES,
        )
        _nonnegative_int(row["review_priority"], field="review_priority")
        _positive_int(
            row["required_review_count"],
            field="required_review_count",
        )
        _utc_datetime_text(row["created_at"], field="created_at")
        for field in (
            "creator",
            "rights_holder",
            "licence_uri",
            "attribution",
        ):
            _nullable_nonblank_text(row[field], field=field)
    inconsistent_groups = (
        frame.group_by("duplicate_group_id")
        .agg(pl.col("canonical_reference_media_id").n_unique().alias("canonical_count"))
        .filter(pl.col("canonical_count") > 1)
    )
    if not inconsistent_groups.is_empty():
        raise ValueError("a duplicate group cannot name multiple canonical media items")
    inconsistent_canonicals = (
        frame.group_by("canonical_reference_media_id")
        .agg(pl.col("duplicate_group_id").n_unique().alias("group_count"))
        .filter(pl.col("group_count") > 1)
    )
    if not inconsistent_canonicals.is_empty():
        raise ValueError("a canonical media item cannot belong to multiple groups")


def validate_reference_review_decisions(frame: pl.DataFrame) -> None:
    _validate_physical_frame(
        frame,
        schema=reference_review_decision_schema(),
        schema_version=REFERENCE_REVIEW_DECISIONS_SCHEMA_VERSION,
        sort_by=_REFERENCE_REVIEW_DECISION_SORT,
        primary_key=["review_decision_id"],
        artifact="reference review decisions",
    )
    decision_rows = list(frame.iter_rows(named=True))
    decisions_by_id = {str(row["review_decision_id"]): row for row in decision_rows}
    duplicate_votes = (
        frame.group_by("review_request_id", "review_round", "verified_by")
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate_votes.is_empty():
        raise ValueError("a reviewer may record only one decision per request round")
    if frame["decision_source_hash"].n_unique() != frame.height:
        raise ValueError("each source decision record may produce only one decision")
    request_media_conflicts = (
        frame.group_by("review_request_id")
        .agg(pl.col("reference_media_id").n_unique().alias("media_count"))
        .filter(pl.col("media_count") > 1)
    )
    if not request_media_conflicts.is_empty():
        raise ValueError("a review request cannot refer to multiple media items")
    for row in decision_rows:
        decision_id = _canonical_required_text(
            row["review_decision_id"],
            field="review_decision_id",
        )
        if _REVIEW_DECISION_ID_PATTERN.fullmatch(decision_id) is None:
            raise ValueError("review_decision_id has an unsupported namespace")
        review_request_id = _canonical_required_text(
            row["review_request_id"],
            field="review_request_id",
        )
        if _REVIEW_REQUEST_ID_PATTERN.fullmatch(review_request_id) is None:
            raise ValueError("review_request_id has an unsupported namespace")
        reference_media_id = _canonical_required_text(
            row["reference_media_id"],
            field="reference_media_id",
        )
        if _REFERENCE_MEDIA_ID_PATTERN.fullmatch(reference_media_id) is None:
            raise ValueError("reference_media_id has an unsupported namespace")
        review_round = _positive_int(row["review_round"], field="review_round")
        verified_by = _canonical_required_text(
            row["verified_by"],
            field="verified_by",
        )
        reviewed_at = row["reviewed_at"]
        _utc_datetime_text(reviewed_at, field="reviewed_at")
        target_identity_verified = row["target_identity_verified"]
        if target_identity_verified is not None and not isinstance(
            target_identity_verified,
            bool,
        ):
            raise ValueError("target_identity_verified must be Boolean or null")
        verification_status = _canonical_choice(
            row["verification_status"],
            field="verification_status",
            choices=REFERENCE_REVIEW_DECISION_STATUSES,
        )
        life_stage = _canonical_choice(
            row["life_stage"],
            field="life_stage",
            choices=REFERENCE_LIFE_STAGES,
        )
        visual_domain = _canonical_choice(
            row["visual_domain"],
            field="visual_domain",
            choices=REFERENCE_VISUAL_DOMAINS,
        )
        view = _canonical_choice(row["view"], field="view", choices=REFERENCE_VIEWS)
        review_confidence = _canonical_choice(
            row["review_confidence"],
            field="review_confidence",
            choices=REFERENCE_REVIEW_CONFIDENCE_VALUES,
        )
        review_notes = _nullable_nonblank_text(
            row["review_notes"],
            field="review_notes",
        )
        exclusion_reason = _nullable_nonblank_text(
            row["exclusion_reason"],
            field="exclusion_reason",
        )
        second_review_required = row["second_review_required"]
        if not isinstance(second_review_required, bool):
            raise ValueError("second_review_required must be Boolean")
        conflicts_with_decision_id = _nullable_nonblank_text(
            row["conflicts_with_decision_id"],
            field="conflicts_with_decision_id",
        )
        if conflicts_with_decision_id is not None:
            if (
                _REVIEW_DECISION_ID_PATTERN.fullmatch(conflicts_with_decision_id)
                is None
            ):
                raise ValueError(
                    "conflicts_with_decision_id has an unsupported namespace"
                )
            if conflicts_with_decision_id == decision_id:
                raise ValueError("a review decision cannot conflict with itself")
            conflicting_row = decisions_by_id.get(conflicts_with_decision_id)
            if conflicting_row is None:
                raise ValueError(
                    "conflicts_with_decision_id must resolve in the decision ledger"
                )
            if (
                conflicting_row["review_request_id"] != review_request_id
                or conflicting_row["reference_media_id"] != reference_media_id
            ):
                raise ValueError(
                    "conflicting decisions must concern the same request and media"
                )
            if conflicting_row["verified_by"] == verified_by:
                raise ValueError("a reviewer cannot conflict with their own decision")
            conflicting_reviewed_at = conflicting_row["reviewed_at"]
            _utc_datetime_text(
                conflicting_reviewed_at,
                field="conflicting reviewed_at",
            )
            if conflicting_reviewed_at >= reviewed_at:
                raise ValueError("a conflict pointer must identify an earlier decision")
        _canonical_full_sha256(
            row["decision_source_hash"],
            field="decision_source_hash",
        )

        if verification_status == "verified":
            if target_identity_verified is not True:
                raise ValueError("verified decisions require confirmed target identity")
            if exclusion_reason is not None:
                raise ValueError("verified decisions cannot have an exclusion reason")
            if second_review_required or conflicts_with_decision_id is not None:
                raise ValueError(
                    "verified decisions cannot require or identify a conflicting review"
                )
        elif verification_status == "excluded":
            if exclusion_reason is None:
                raise ValueError("excluded decisions require an exclusion reason")
            if second_review_required or conflicts_with_decision_id is not None:
                raise ValueError(
                    "excluded decisions cannot require or identify a conflicting review"
                )
        else:
            if target_identity_verified is not None:
                raise ValueError("uncertain decisions cannot assert target identity")
            if review_notes is None:
                raise ValueError("uncertain decisions require review notes")
            if exclusion_reason is not None:
                raise ValueError("uncertain decisions cannot have an exclusion reason")
            if not second_review_required:
                raise ValueError("uncertain decisions require a second review")

        expected_id = make_reference_review_decision_id(
            review_request_id=review_request_id,
            reference_media_id=reference_media_id,
            review_round=review_round,
            verified_by=verified_by,
            reviewed_at=reviewed_at,
            target_identity_verified=target_identity_verified,
            verification_status=verification_status,
            life_stage=life_stage,
            visual_domain=visual_domain,
            view=view,
            review_confidence=review_confidence,
            review_notes=review_notes,
            exclusion_reason=exclusion_reason,
            second_review_required=second_review_required,
            conflicts_with_decision_id=conflicts_with_decision_id,
        )
        if decision_id != expected_id:
            raise ValueError("review decision ID does not match its semantic content")


def write_reference_observations(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    validate_reference_observations(frame)
    return write_parquet(
        frame,
        _artifact_path(output, REFERENCE_OBSERVATIONS_FILE),
        overwrite=overwrite,
    )


def write_reference_media_candidates(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    validate_reference_media_candidates(frame)
    return write_parquet(
        frame,
        _artifact_path(output, REFERENCE_MEDIA_CANDIDATES_FILE),
        overwrite=overwrite,
    )


def write_reference_acquisition_plan(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    validate_reference_acquisition_plan(frame)
    return write_parquet(
        frame,
        _artifact_path(output, REFERENCE_ACQUISITION_PLAN_FILE),
        overwrite=overwrite,
    )


def write_reference_acquisition_selections(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    validate_reference_acquisition_selections(frame)
    return write_parquet(
        frame,
        _artifact_path(output, REFERENCE_ACQUISITION_SELECTIONS_FILE),
        overwrite=overwrite,
    )


def write_reference_media_objects(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    validate_reference_media_objects(frame)
    return write_parquet(
        frame,
        _artifact_path(output, REFERENCE_MEDIA_OBJECTS_FILE),
        overwrite=overwrite,
    )


def write_reference_media_duplicate_relationships(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    validate_reference_media_duplicate_relationships(frame)
    return write_parquet(
        frame,
        _artifact_path(output, REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_FILE),
        overwrite=overwrite,
    )


def write_reference_review_queue(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    validate_reference_review_queue(frame)
    return write_parquet(
        frame,
        _artifact_path(output, REFERENCE_REVIEW_QUEUE_FILE),
        overwrite=overwrite,
    )


def write_reference_review_decisions(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    validate_reference_review_decisions(frame)
    return _write_parquet_create_only(
        frame,
        _artifact_path(output, REFERENCE_REVIEW_DECISIONS_FILE),
    )


def _frame(
    rows: Sequence[Mapping[str, object]],
    *,
    schema: dict[str, pl.DataType],
    sort_by: list[str],
) -> pl.DataFrame:
    materialized = list(rows)
    if not materialized:
        return pl.DataFrame(schema=schema)
    unknown = sorted(set().union(*(set(row) for row in materialized)) - set(schema))
    if unknown:
        raise ValueError(f"reference rows have unknown fields: {unknown}")
    missing = sorted(
        set(schema) - set.intersection(*(set(row) for row in materialized))
    )
    if missing:
        raise ValueError(f"reference rows are missing fields: {missing}")
    return pl.DataFrame(materialized, schema=schema, strict=True).sort(sort_by)


def _validate_physical_frame(
    frame: pl.DataFrame,
    *,
    schema: dict[str, pl.DataType],
    schema_version: str,
    sort_by: list[str],
    primary_key: list[str],
    artifact: str,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    if frame.schema != schema:
        raise ValueError(f"{artifact} frame does not match the physical schema")
    if not frame.equals(frame.sort(sort_by)):
        raise ValueError(f"{artifact} frame is not in deterministic sort order")
    if frame.height and frame["schema_version"].unique().to_list() != [schema_version]:
        raise ValueError(f"{artifact} schema version mismatch")
    duplicates = frame.group_by(primary_key).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError(f"{artifact} contains duplicate primary keys")


def _coordinate_pair(latitude: object, longitude: object) -> None:
    if (latitude is None) != (longitude is None):
        raise ValueError("latitude and longitude must be populated together")
    if latitude is None:
        return
    lat = float(latitude)
    lon = float(longitude)
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("coordinates must be finite")
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError("coordinates are outside WGS84 bounds")


def _optional_nonnegative_finite(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return parsed


def _nonnegative_int(value: object, *, field: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} must be a nonnegative integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return parsed


def _positive_int(value: object, *, field: str) -> int:
    parsed = _nonnegative_int(value, field=field)
    if parsed == 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _fallback_level(value: object) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 3:
        raise ValueError("fallback_level must be between 0 and 3")
    return parsed


def _choice(value: object, *, field: str, choices: frozenset[str]) -> str:
    text = _required_text(value, field=field)
    if text not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return text


def _canonical_choice(
    value: object,
    *,
    field: str,
    choices: frozenset[str],
) -> str:
    text = _canonical_required_text(value, field=field)
    if text not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return text


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _canonical_required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    text = _required_text(value, field=field)
    if value != text:
        raise ValueError(f"{field} must not have surrounding whitespace")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _nullable_nonblank_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _canonical_required_text(value, field=field)


def _utc_datetime_text(value: object, *, field: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field} must be a timezone-aware UTC datetime")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


def _full_sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 digest")
    return text


def _canonical_full_sha256(value: object, *, field: str) -> str:
    text = _canonical_required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 digest")
    return text


def _perceptual_hash(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _PERCEPTUAL_HASH_PATTERN.fullmatch(text) is None:
        raise ValueError(
            f"{field} must be a lowercase versioned 128-bit difference hash"
        )
    return text


def _semantic_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _artifact_path(output: str | Path, filename: str) -> Path:
    path = Path(output)
    return path if path.suffix.casefold() == ".parquet" else path / filename


def _write_parquet_create_only(frame: pl.DataFrame, output: Path) -> Path:
    staged = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        write_parquet(frame, staged, overwrite=False)
        os.link(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    return output


__all__ = [
    "DECODE_STATUSES",
    "DOWNLOAD_STATUSES",
    "DUPLICATE_EVIDENCE_TYPES",
    "DUPLICATE_RELATIONSHIP_TYPES",
    "DUPLICATE_RESOLUTION_STATUSES",
    "DUPLICATE_TYPES",
    "LICENCE_POLICY_STATUSES",
    "REFERENCE_ACQUISITION_PLAN_FILE",
    "REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION",
    "REFERENCE_ACQUISITION_SELECTIONS_FILE",
    "REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION",
    "REFERENCE_MEDIA_CANDIDATES_FILE",
    "REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION",
    "REFERENCE_MEDIA_OBJECTS_FILE",
    "REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION",
    "REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_FILE",
    "REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_SCHEMA_VERSION",
    "REFERENCE_MEDIA_RASTER_CONTENT_TYPES",
    "REFERENCE_OBSERVATIONS_FILE",
    "REFERENCE_OBSERVATIONS_SCHEMA_VERSION",
    "REFERENCE_LIFE_STAGES",
    "REFERENCE_REVIEW_CONFIDENCE_VALUES",
    "REFERENCE_REVIEW_DECISIONS_FILE",
    "REFERENCE_REVIEW_DECISIONS_SCHEMA_VERSION",
    "REFERENCE_REVIEW_DECISION_STATUSES",
    "REFERENCE_REVIEW_QUEUE_FILE",
    "REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION",
    "REFERENCE_REVIEW_QUEUE_STATUSES",
    "REFERENCE_VIEWS",
    "REFERENCE_VISUAL_DOMAINS",
    "TAXON_RECONCILIATION_STATUSES",
    "VERIFICATION_STATUSES",
    "make_acquisition_plan_id",
    "make_reference_review_decision_id",
    "make_reference_review_request_id",
    "make_reference_selection_id",
    "reference_acquisition_selection_schema",
    "reference_acquisition_selections_frame",
    "make_reference_media_id",
    "make_reference_observation_id",
    "reference_acquisition_plan_frame",
    "reference_acquisition_plan_schema",
    "reference_media_candidate_schema",
    "reference_media_candidates_frame",
    "reference_media_object_schema",
    "reference_media_objects_frame",
    "reference_media_duplicate_relationship_schema",
    "reference_media_duplicate_relationships_frame",
    "reference_observation_schema",
    "reference_observations_frame",
    "reference_review_decision_schema",
    "reference_review_decisions_frame",
    "reference_review_queue_frame",
    "reference_review_queue_schema",
    "validate_reference_acquisition_plan",
    "validate_reference_acquisition_selections",
    "validate_reference_media_candidates",
    "validate_reference_media_objects",
    "validate_reference_media_duplicate_relationships",
    "validate_reference_observations",
    "validate_reference_review_decisions",
    "validate_reference_review_queue",
    "write_reference_acquisition_plan",
    "write_reference_acquisition_selections",
    "write_reference_media_candidates",
    "write_reference_media_objects",
    "write_reference_media_duplicate_relationships",
    "write_reference_observations",
    "write_reference_review_decisions",
    "write_reference_review_queue",
]
