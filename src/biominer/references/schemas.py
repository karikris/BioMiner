from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re

import polars as pl

from biominer.storage.parquet import write_parquet


REFERENCE_OBSERVATIONS_SCHEMA_VERSION = "reference-observations-v1.2.0"
REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION = "reference-media-candidates-v1.0.0"
REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION = "reference-acquisition-plan-v1.1.0"
REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION = (
    "reference-acquisition-selections-v1.0.0"
)

REFERENCE_OBSERVATIONS_FILE = "reference_observations.parquet"
REFERENCE_MEDIA_CANDIDATES_FILE = "reference_media_candidates.parquet"
REFERENCE_ACQUISITION_PLAN_FILE = "reference_acquisition_plan.parquet"
REFERENCE_ACQUISITION_SELECTIONS_FILE = "reference_acquisition_selections.parquet"

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
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


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


def reference_observations_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    frame = _frame(rows, schema=reference_observation_schema(), sort_by=_OBSERVATION_SORT)
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
        raise ValueError("reference media may be selected only once per acquisition plan")
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
    missing = sorted(set(schema) - set.intersection(*(set(row) for row in materialized)))
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


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _full_sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 digest")
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


__all__ = [
    "DOWNLOAD_STATUSES",
    "LICENCE_POLICY_STATUSES",
    "REFERENCE_ACQUISITION_PLAN_FILE",
    "REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION",
    "REFERENCE_ACQUISITION_SELECTIONS_FILE",
    "REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION",
    "REFERENCE_MEDIA_CANDIDATES_FILE",
    "REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION",
    "REFERENCE_OBSERVATIONS_FILE",
    "REFERENCE_OBSERVATIONS_SCHEMA_VERSION",
    "TAXON_RECONCILIATION_STATUSES",
    "VERIFICATION_STATUSES",
    "make_acquisition_plan_id",
    "make_reference_selection_id",
    "reference_acquisition_selection_schema",
    "reference_acquisition_selections_frame",
    "make_reference_media_id",
    "make_reference_observation_id",
    "reference_acquisition_plan_frame",
    "reference_acquisition_plan_schema",
    "reference_media_candidate_schema",
    "reference_media_candidates_frame",
    "reference_observation_schema",
    "reference_observations_frame",
    "validate_reference_acquisition_plan",
    "validate_reference_acquisition_selections",
    "validate_reference_media_candidates",
    "validate_reference_observations",
    "write_reference_acquisition_plan",
    "write_reference_acquisition_selections",
    "write_reference_media_candidates",
    "write_reference_observations",
]
