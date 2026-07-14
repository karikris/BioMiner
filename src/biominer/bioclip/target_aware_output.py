"""Versioned target-aware score artifacts and fail-closed validators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from math import isclose, isfinite
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.ml.nonmatch import CLASSIFICATION_OUTCOMES, TARGET_CONFIRMED
from biominer.storage.parquet import DEFAULT_PARQUET_COMPRESSION, write_parquet


TARGET_AWARE_OBJECT_SCORES_FILE = "target_aware_object_scores.parquet"
TARGET_AWARE_CANDIDATE_SCORES_FILE = "target_aware_candidate_scores.parquet"
TARGET_AWARE_OBJECT_SCORES_SCHEMA_VERSION = "target-aware-object-scores-v1.0.0"
TARGET_AWARE_CANDIDATE_SCORES_SCHEMA_VERSION = "target-aware-candidate-scores-v1.0.0"
TARGET_SCORE_ID_VERSION = "target-aware-score-id-v1.0.0"
TARGET_SCORE_FINGERPRINT_VERSION = "target-aware-score-fingerprint-v1.0.0"
TARGET_CANDIDATE_EVIDENCE_FINGERPRINT_VERSION = (
    "target-aware-candidate-evidence-fingerprint-v1.0.0"
)

_TARGET_SCORE_SORT = ["source", "flickr_photo_id", "route", "scoring_unit_id"]
_CANDIDATE_SCORE_SORT = [
    "target_score_id",
    "candidate_priority",
    "accepted_taxon_key",
]
_REVIEW_PRIORITIES = frozenset({"none", "normal", "high"})
_VISUAL_INPUT_KIND_ORDER = {
    "raw_full_image": 0,
    "focused_full_frame": 1,
    "masked_full_frame": 2,
    "multi_object_full_frame": 3,
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TARGET_SCORE_ID_PATTERN = re.compile(r"target-score:[0-9a-f]{64}\Z")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


VISUAL_INPUT_EVIDENCE_SCHEMA: dict[str, pl.DataType] = {
    "visual_input_id": pl.String,
    "visual_input_kind": pl.String,
    "transformation_fingerprint": pl.String,
    "embedding_id": pl.String,
    "embedding_fingerprint": pl.String,
    "source_fusion_fingerprint": pl.String,
    "target_classifier_decision_score": pl.Float32,
    "target_regional_classifier_decision_score": pl.Float32,
    "best_competitor_accepted_taxon_key": pl.String,
    "best_competitor_scientific_name": pl.String,
    "best_competitor_regional_classifier_decision_score": pl.Float32,
    "regional_competitor_margin": pl.Float32,
    "target_reference_centroid_similarity": pl.Float32,
    "best_competitor_reference_centroid_similarity": pl.Float32,
    "reference_competitor_margin": pl.Float32,
    "aggregation_weight": pl.Float32,
    "input_score_fingerprint": pl.String,
}


REGIONAL_CANDIDATE_EVIDENCE_SCHEMA: dict[str, pl.DataType] = {
    "accepted_taxon_key": pl.String,
    "scientific_name": pl.String,
    "target_candidate": pl.Boolean,
    "candidate_priority": pl.UInt32,
    "candidate_reasons": pl.List(pl.String),
    "text_ensemble_similarity": pl.Float32,
    "reference_centroid_similarity": pl.Float32,
    "nearest_reference_similarity": pl.Float32,
    "top_k_reference_similarity": pl.Float32,
    "reference_top_k": pl.UInt8,
    "local_prototype_similarity": pl.Float32,
    "global_prototype_similarity": pl.Float32,
    "classifier_task": pl.String,
    "classifier_decision_score": pl.Float32,
    "calibrated_probability": pl.Float64,
    "regional_classifier_decision_score": pl.Float32,
    "regional_calibrated_probability": pl.Float64,
    "regional_rank": pl.UInt32,
    "global_rank": pl.UInt32,
    "geographic_scope": pl.String,
    "geographic_evidence_score": pl.Float32,
    "occurrence_support": pl.UInt32,
    "life_stage": pl.String,
    "visual_domain": pl.String,
    "life_stage_compatible": pl.Boolean,
    "visual_domain_compatible": pl.Boolean,
    "best_competitor_accepted_taxon_key": pl.String,
    "best_competitor_scientific_name": pl.String,
    "competitor_margin": pl.Float32,
    "nonmatch_score": pl.Float32,
    "abstention_reason": pl.String,
    "prompt_result_fingerprint": pl.String,
    "reference_embedding_fingerprint": pl.String,
    "reference_prototype_fingerprint": pl.String,
    "support_manifest_fingerprint": pl.String,
    "classifier_fingerprint": pl.String,
    "calibration_fingerprint": pl.String,
    "structured_evidence_fingerprint": pl.String,
    "candidate_set_fingerprint": pl.String,
    "candidate_evidence_fingerprint": pl.String,
}


TARGET_AWARE_OBJECT_SCORE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "target_score_id": pl.String,
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "source_record_hash": pl.String,
    "scoring_unit_id": pl.String,
    "detection_id": pl.String,
    "route": pl.String,
    "geo_cluster_id": pl.String,
    "candidate_set_id": pl.String,
    "target_accepted_taxon_key": pl.String,
    "reference_bank_version": pl.String,
    "classifier_version": pl.String,
    "calibration_version": pl.String,
    "prompt_version": pl.String,
    "visual_input_version": pl.String,
    "classification_mode": pl.String,
    "registry_fingerprint": pl.String,
    "geographic_spread_fingerprint": pl.String,
    "flickr_cluster_fingerprint": pl.String,
    "candidate_set_fingerprint": pl.String,
    "reference_bank_fingerprint": pl.String,
    "reference_embedding_fingerprint": pl.String,
    "prototype_fingerprint": pl.String,
    "support_manifest_fingerprint": pl.String,
    "model_fingerprint": pl.String,
    "preprocessing_fingerprint": pl.String,
    "prompt_fingerprint": pl.String,
    "classifier_fingerprint": pl.String,
    "calibrator_fingerprint": pl.String,
    "regional_classifier_fingerprint": pl.String,
    "regional_calibrator_fingerprint": pl.String,
    "visual_input_fusion_fingerprint": pl.String,
    "decision_policy_fingerprint": pl.String,
    "scored_at": pl.Datetime("us", "UTC"),
    "producer_git_sha": pl.String,
    "target_raw_text_similarity": pl.Float32,
    "target_reference_centroid_similarity": pl.Float32,
    "target_nearest_reference_similarity": pl.Float32,
    "target_top_k_reference_similarity": pl.Float32,
    "target_local_prototype_similarity": pl.Float32,
    "target_global_prototype_similarity": pl.Float32,
    "target_classifier_margin": pl.Float32,
    "calibrated_target_probability": pl.Float64,
    "target_regional_rank": pl.UInt32,
    "target_global_rank": pl.UInt32,
    "best_competitor_accepted_taxon_key": pl.String,
    "best_competitor_scientific_name": pl.String,
    "calibrated_best_competitor_probability": pl.Float64,
    "best_competitor_reference_similarity": pl.Float32,
    "target_competitor_margin": pl.Float32,
    "best_domain_negative_similarity": pl.Float32,
    "calibrated_non_target_probability": pl.Float64,
    "nonmatch_score": pl.Float32,
    "nonmatch_margin": pl.Float32,
    "competitor_reason": pl.String,
    "yoloe_route": pl.String,
    "detector_score": pl.Float32,
    "subject_area_ratio": pl.Float32,
    "mask_coverage": pl.Float32,
    "visual_input_disagreement": pl.Float32,
    "reference_coverage": pl.Float32,
    "geo_evidence": pl.Float32,
    "ood_score": pl.Float32,
    "route_compatible": pl.Boolean,
    "reference_coverage_sufficient": pl.Boolean,
    "domain_negative_detected": pl.Boolean,
    "out_of_distribution": pl.Boolean,
    "visual_detail_sufficient": pl.Boolean,
    "no_geo_global_fallback": pl.Boolean,
    "quality_flags": pl.List(pl.String),
    "visual_input_evidence": pl.List(pl.Struct(VISUAL_INPUT_EVIDENCE_SCHEMA)),
    "regional_candidate_evidence": pl.List(
        pl.Struct(REGIONAL_CANDIDATE_EVIDENCE_SCHEMA)
    ),
    "classification_decision": pl.String,
    "abstained": pl.Boolean,
    "abstention_reason": pl.String,
    "review_priority": pl.String,
    "model_decision_threshold": pl.Float64,
    "competitor_margin_threshold": pl.Float32,
    "threshold_provenance": pl.String,
    "target_score_fingerprint": pl.String,
}


_CANDIDATE_PARENT_FIELDS = (
    "target_score_id",
    "target_score_fingerprint",
    "source",
    "flickr_photo_id",
    "scoring_unit_id",
    "route",
    "geo_cluster_id",
    "candidate_set_id",
    "target_accepted_taxon_key",
    "reference_bank_version",
    "classifier_version",
    "calibration_version",
    "prompt_version",
    "visual_input_version",
    "classification_mode",
    "registry_fingerprint",
    "geographic_spread_fingerprint",
    "flickr_cluster_fingerprint",
    "reference_bank_fingerprint",
    "model_fingerprint",
    "preprocessing_fingerprint",
    "visual_input_fusion_fingerprint",
    "decision_policy_fingerprint",
)


TARGET_AWARE_CANDIDATE_SCORE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    **{
        field: TARGET_AWARE_OBJECT_SCORE_SCHEMA[field]
        for field in _CANDIDATE_PARENT_FIELDS
    },
    **REGIONAL_CANDIDATE_EVIDENCE_SCHEMA,
}


_FINGERPRINT_FIELDS = tuple(
    field
    for field in TARGET_AWARE_OBJECT_SCORE_SCHEMA
    if field.endswith("_fingerprint") and field != "target_score_fingerprint"
)
_CANDIDATE_FINGERPRINT_FIELDS = tuple(
    field
    for field in REGIONAL_CANDIDATE_EVIDENCE_SCHEMA
    if field.endswith("_fingerprint") and field != "candidate_evidence_fingerprint"
)
_SIMILARITY_FIELDS = (
    "target_raw_text_similarity",
    "target_reference_centroid_similarity",
    "target_nearest_reference_similarity",
    "target_top_k_reference_similarity",
    "target_local_prototype_similarity",
    "target_global_prototype_similarity",
    "best_competitor_reference_similarity",
    "best_domain_negative_similarity",
)
_CANDIDATE_SIMILARITY_FIELDS = (
    "text_ensemble_similarity",
    "reference_centroid_similarity",
    "nearest_reference_similarity",
    "top_k_reference_similarity",
    "local_prototype_similarity",
    "global_prototype_similarity",
)
_PROBABILITY_FIELDS = (
    "calibrated_target_probability",
    "calibrated_best_competitor_probability",
    "calibrated_non_target_probability",
)


def empty_target_aware_object_scores_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=TARGET_AWARE_OBJECT_SCORE_SCHEMA)


def empty_target_aware_candidate_scores_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=TARGET_AWARE_CANDIDATE_SCORE_SCHEMA)


def make_target_score_id(row: Mapping[str, object]) -> str:
    """Return a stable work identity from all semantic input dependencies."""

    fingerprint = canonical_semantic_fingerprint(_target_score_identity(row))
    return "target-score:" + fingerprint.removeprefix("sha256:")


def target_aware_object_scores_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    materialized = list(rows)
    if not materialized:
        return empty_target_aware_object_scores_frame()
    prepared = [_prepare_object_row(row) for row in materialized]
    frame = pl.DataFrame(
        prepared,
        schema=TARGET_AWARE_OBJECT_SCORE_SCHEMA,
        orient="row",
        strict=True,
    )
    derived = [_derive_object_row(row) for row in frame.iter_rows(named=True)]
    result = pl.DataFrame(
        derived,
        schema=TARGET_AWARE_OBJECT_SCORE_SCHEMA,
        orient="row",
        strict=True,
    ).sort(_TARGET_SCORE_SORT)
    validate_target_aware_object_scores(result)
    return result


def target_aware_candidate_scores_frame(
    object_scores: pl.DataFrame,
) -> pl.DataFrame:
    validate_target_aware_object_scores(object_scores)
    frame = _candidate_frame_from_objects(object_scores)
    validate_target_aware_candidate_scores(frame, object_scores=object_scores)
    return frame


def validate_target_aware_object_scores(frame: pl.DataFrame) -> None:
    _validate_physical_frame(
        frame,
        schema=TARGET_AWARE_OBJECT_SCORE_SCHEMA,
        schema_version=TARGET_AWARE_OBJECT_SCORES_SCHEMA_VERSION,
        sort_by=_TARGET_SCORE_SORT,
        primary_key=["target_score_id"],
        artifact="target-aware object scores",
    )
    for row in frame.iter_rows(named=True):
        _validate_object_row(row)


def validate_target_aware_candidate_scores(
    frame: pl.DataFrame,
    *,
    object_scores: pl.DataFrame | None = None,
) -> None:
    _validate_physical_frame(
        frame,
        schema=TARGET_AWARE_CANDIDATE_SCORE_SCHEMA,
        schema_version=TARGET_AWARE_CANDIDATE_SCORES_SCHEMA_VERSION,
        sort_by=_CANDIDATE_SCORE_SORT,
        primary_key=["target_score_id", "accepted_taxon_key"],
        artifact="target-aware candidate scores",
    )
    for row in frame.iter_rows(named=True):
        _validate_candidate_row(row)
    if object_scores is None:
        return
    validate_target_aware_object_scores(object_scores)
    expected = _candidate_frame_from_objects(object_scores)
    if not frame.equals(expected):
        raise ValueError(
            "target-aware candidate scores are not the exact object-score projection"
        )


def write_target_aware_object_scores(
    frame: pl.DataFrame,
    path: str | Path,
    *,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> Path:
    validate_target_aware_object_scores(frame)
    return write_parquet(frame, path, compression=compression)


def write_target_aware_candidate_scores(
    frame: pl.DataFrame,
    path: str | Path,
    *,
    object_scores: pl.DataFrame | None = None,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> Path:
    validate_target_aware_candidate_scores(frame, object_scores=object_scores)
    return write_parquet(frame, path, compression=compression)


def _prepare_object_row(row: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise TypeError("target-aware object score rows must be mappings")
    values = dict(row)
    unknown = sorted(set(values) - set(TARGET_AWARE_OBJECT_SCORE_SCHEMA))
    if unknown:
        raise ValueError(
            f"target-aware object score rows have unknown fields: {unknown}"
        )
    derived_fields = {"schema_version", "target_score_id", "target_score_fingerprint"}
    missing = sorted(
        set(TARGET_AWARE_OBJECT_SCORE_SCHEMA) - set(values) - derived_fields
    )
    if missing:
        raise ValueError(
            f"target-aware object score rows are missing fields: {missing}"
        )
    values.setdefault("schema_version", TARGET_AWARE_OBJECT_SCORES_SCHEMA_VERSION)
    values.setdefault("target_score_id", None)
    values.setdefault("target_score_fingerprint", None)
    values["quality_flags"] = _sorted_unique_values(
        values["quality_flags"], field="quality_flags"
    )
    values["visual_input_evidence"] = _prepare_visual_inputs(
        values["visual_input_evidence"]
    )
    values["regional_candidate_evidence"] = _prepare_candidates(
        values["regional_candidate_evidence"]
    )
    return values


def _prepare_visual_inputs(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("visual_input_evidence must be a list")
    prepared: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("visual_input_evidence entries must be mappings")
        unknown = sorted(set(item) - set(VISUAL_INPUT_EVIDENCE_SCHEMA))
        missing = sorted(set(VISUAL_INPUT_EVIDENCE_SCHEMA) - set(item))
        if unknown or missing:
            raise ValueError(
                "visual_input_evidence fields mismatch: "
                f"unknown={unknown}, missing={missing}"
            )
        prepared.append(dict(item))
    return sorted(
        prepared,
        key=lambda item: (
            _VISUAL_INPUT_KIND_ORDER.get(str(item["visual_input_kind"]), 99),
            str(item["visual_input_id"]),
        ),
    )


def _prepare_candidates(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("regional_candidate_evidence must be a list")
    prepared: list[dict[str, object]] = []
    derived = {"candidate_evidence_fingerprint"}
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("regional_candidate_evidence entries must be mappings")
        unknown = sorted(set(item) - set(REGIONAL_CANDIDATE_EVIDENCE_SCHEMA))
        missing = sorted(set(REGIONAL_CANDIDATE_EVIDENCE_SCHEMA) - set(item) - derived)
        if unknown or missing:
            raise ValueError(
                "regional_candidate_evidence fields mismatch: "
                f"unknown={unknown}, missing={missing}"
            )
        candidate = dict(item)
        candidate.setdefault("candidate_evidence_fingerprint", None)
        candidate["candidate_reasons"] = _sorted_unique_values(
            candidate["candidate_reasons"], field="candidate_reasons"
        )
        prepared.append(candidate)
    try:
        return sorted(
            prepared,
            key=lambda item: (
                int(item["candidate_priority"]),
                str(item["accepted_taxon_key"]),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate_priority must be an integer") from exc


def _derive_object_row(row: Mapping[str, object]) -> dict[str, object]:
    values = dict(row)
    candidates: list[dict[str, object]] = []
    for raw in values["regional_candidate_evidence"]:
        candidate = dict(raw)
        supplied = candidate.get("candidate_evidence_fingerprint")
        expected = canonical_semantic_fingerprint(_candidate_semantics(candidate))
        if supplied is not None and supplied != expected:
            raise ValueError("candidate_evidence_fingerprint is inconsistent")
        candidate["candidate_evidence_fingerprint"] = expected
        candidates.append(candidate)
    values["regional_candidate_evidence"] = candidates

    expected_id = make_target_score_id(values)
    supplied_id = values.get("target_score_id")
    if supplied_id is not None and supplied_id != expected_id:
        raise ValueError("target_score_id does not match immutable inputs")
    values["target_score_id"] = expected_id
    expected_fingerprint = canonical_semantic_fingerprint(
        _target_score_semantics(values)
    )
    supplied_fingerprint = values.get("target_score_fingerprint")
    if (
        supplied_fingerprint is not None
        and supplied_fingerprint != expected_fingerprint
    ):
        raise ValueError("target_score_fingerprint is inconsistent")
    values["target_score_fingerprint"] = expected_fingerprint
    return values


def _candidate_frame_from_objects(object_scores: pl.DataFrame) -> pl.DataFrame:
    if object_scores.is_empty():
        return empty_target_aware_candidate_scores_frame()
    rows: list[dict[str, object]] = []
    for parent in object_scores.iter_rows(named=True):
        parent_values = {field: parent[field] for field in _CANDIDATE_PARENT_FIELDS}
        for candidate in parent["regional_candidate_evidence"]:
            rows.append(
                {
                    "schema_version": TARGET_AWARE_CANDIDATE_SCORES_SCHEMA_VERSION,
                    **parent_values,
                    **candidate,
                }
            )
    return pl.DataFrame(
        rows,
        schema=TARGET_AWARE_CANDIDATE_SCORE_SCHEMA,
        orient="row",
        strict=True,
    ).sort(_CANDIDATE_SCORE_SORT)


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
    if dict(frame.schema) != schema:
        raise ValueError(f"{artifact} frame does not match the physical schema")
    if not frame.equals(frame.sort(sort_by)):
        raise ValueError(f"{artifact} frame is not in deterministic sort order")
    if frame.height:
        versions = frame["schema_version"].unique().to_list()
        if versions != [schema_version]:
            raise ValueError(f"{artifact} schema version mismatch")
    duplicates = frame.group_by(primary_key).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError(f"{artifact} contains duplicate primary keys")


def _validate_object_row(row: Mapping[str, object]) -> None:
    required_text_fields = (
        "source",
        "flickr_photo_id",
        "scoring_unit_id",
        "route",
        "geo_cluster_id",
        "candidate_set_id",
        "target_accepted_taxon_key",
        "reference_bank_version",
        "classifier_version",
        "calibration_version",
        "prompt_version",
        "visual_input_version",
        "classification_mode",
        "competitor_reason",
        "yoloe_route",
    )
    for field in required_text_fields:
        _required_text(row[field], field=field)
    _optional_text(row["detection_id"], field="detection_id")
    _fingerprint(row["source_record_hash"], field="source_record_hash")
    for field in _FINGERPRINT_FIELDS:
        _fingerprint(row[field], field=field)
    _git_sha(row["producer_git_sha"])
    _utc_datetime(row["scored_at"], field="scored_at")
    for field in _SIMILARITY_FIELDS:
        _optional_bounded_float(row[field], field=field, minimum=-1.0, maximum=1.0)
    for field in _PROBABILITY_FIELDS:
        _optional_bounded_float(row[field], field=field, minimum=0.0, maximum=1.0)
    for field in ("target_regional_rank", "target_global_rank"):
        _optional_positive_integer(row[field], field=field)
    for field in ("target_classifier_margin", "target_competitor_margin"):
        _optional_finite_float(row[field], field=field)
    for field in ("nonmatch_score", "nonmatch_margin"):
        _optional_bounded_float(row[field], field=field, minimum=-1.0, maximum=1.0)
    _optional_bounded_float(
        row["detector_score"], field="detector_score", minimum=0.0, maximum=1.0
    )
    for field in (
        "subject_area_ratio",
        "mask_coverage",
        "reference_coverage",
        "geo_evidence",
    ):
        _optional_bounded_float(row[field], field=field, minimum=0.0, maximum=1.0)
    _nonnegative_float(
        row["visual_input_disagreement"], field="visual_input_disagreement"
    )
    if row["ood_score"] is not None:
        _nonnegative_float(row["ood_score"], field="ood_score")
    for field in (
        "route_compatible",
        "reference_coverage_sufficient",
        "domain_negative_detected",
        "out_of_distribution",
        "visual_detail_sufficient",
        "no_geo_global_fallback",
        "abstained",
    ):
        _boolean(row[field], field=field)
    _validate_sorted_unique_text(row["quality_flags"], field="quality_flags")
    _validate_visual_inputs(row["visual_input_evidence"])
    _validate_candidate_union(row)
    _validate_competitor(row)
    _validate_decision(row)

    expected_id = make_target_score_id(row)
    target_score_id = _required_text(row["target_score_id"], field="target_score_id")
    if _TARGET_SCORE_ID_PATTERN.fullmatch(target_score_id) is None:
        raise ValueError("target_score_id has an unsupported namespace")
    if target_score_id != expected_id:
        raise ValueError("target_score_id does not match immutable inputs")
    expected_fingerprint = canonical_semantic_fingerprint(_target_score_semantics(row))
    actual_fingerprint = _fingerprint(
        row["target_score_fingerprint"], field="target_score_fingerprint"
    )
    if actual_fingerprint != expected_fingerprint:
        raise ValueError("target_score_fingerprint is inconsistent")


def _validate_visual_inputs(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("visual_input_evidence must be a nonempty list")
    ids: list[str] = []
    weights: list[float] = []
    raw_count = 0
    for item in value:
        visual_input_id = _required_text(
            item["visual_input_id"], field="visual_input_id"
        )
        ids.append(visual_input_id)
        kind = _required_text(item["visual_input_kind"], field="visual_input_kind")
        if kind not in _VISUAL_INPUT_KIND_ORDER:
            raise ValueError(f"unsupported visual_input_kind: {kind}")
        raw_count += int(kind == "raw_full_image")
        for field in (
            "transformation_fingerprint",
            "embedding_fingerprint",
            "source_fusion_fingerprint",
            "input_score_fingerprint",
        ):
            _fingerprint(item[field], field=field)
        _required_text(item["embedding_id"], field="embedding_id")
        for field in (
            "target_classifier_decision_score",
            "target_regional_classifier_decision_score",
            "best_competitor_regional_classifier_decision_score",
            "regional_competitor_margin",
        ):
            _finite_float(item[field], field=field)
        _required_text(
            item["best_competitor_accepted_taxon_key"],
            field="best_competitor_accepted_taxon_key",
        )
        _required_text(
            item["best_competitor_scientific_name"],
            field="best_competitor_scientific_name",
        )
        for field in (
            "target_reference_centroid_similarity",
            "best_competitor_reference_centroid_similarity",
        ):
            _optional_bounded_float(item[field], field=field, minimum=-1.0, maximum=1.0)
        _optional_finite_float(
            item["reference_competitor_margin"], field="reference_competitor_margin"
        )
        weights.append(
            _nonnegative_float(item["aggregation_weight"], field="aggregation_weight")
        )
    if len(ids) != len(set(ids)):
        raise ValueError(
            "visual_input_evidence contains duplicate visual_input_id values"
        )
    if raw_count != 1:
        raise ValueError(
            "visual_input_evidence must contain exactly one raw full image"
        )
    if not isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("visual input aggregation weights must sum to one")
    expected_order = sorted(
        value,
        key=lambda item: (
            _VISUAL_INPUT_KIND_ORDER[str(item["visual_input_kind"])],
            str(item["visual_input_id"]),
        ),
    )
    if value != expected_order:
        raise ValueError("visual_input_evidence is not deterministically ordered")


def _validate_candidate_union(row: Mapping[str, object]) -> None:
    value = row["regional_candidate_evidence"]
    if not isinstance(value, list) or not value:
        raise ValueError("regional_candidate_evidence must be a nonempty list")
    keys: list[str] = []
    priorities: list[int] = []
    target_rows: list[Mapping[str, object]] = []
    regional_ranks: list[int] = []
    global_ranks: list[int] = []
    for candidate in value:
        _validate_candidate_row(candidate)
        key = str(candidate["accepted_taxon_key"])
        keys.append(key)
        priorities.append(int(candidate["candidate_priority"]))
        if bool(candidate["target_candidate"]):
            target_rows.append(candidate)
        if candidate["candidate_set_fingerprint"] != row["candidate_set_fingerprint"]:
            raise ValueError(
                "candidate evidence uses a different candidate set fingerprint"
            )
        if candidate["regional_rank"] is not None:
            regional_ranks.append(int(candidate["regional_rank"]))
        if candidate["global_rank"] is not None:
            global_ranks.append(int(candidate["global_rank"]))
    if len(keys) != len(set(keys)):
        raise ValueError(
            "regional_candidate_evidence contains duplicate candidate keys"
        )
    if len(target_rows) != 1:
        raise ValueError("regional_candidate_evidence must contain exactly one target")
    target = target_rows[0]
    if target["accepted_taxon_key"] != row["target_accepted_taxon_key"]:
        raise ValueError("target candidate does not match target_accepted_taxon_key")
    if priorities != list(range(len(value))):
        raise ValueError("candidate priorities must be contiguous from zero")
    if regional_ranks and sorted(regional_ranks) != list(range(1, len(value) + 1)):
        raise ValueError("regional ranks must be a complete one-based permutation")
    if global_ranks and len(global_ranks) != len(set(global_ranks)):
        raise ValueError("global ranks must be unique when populated")
    _validate_target_projection(row, target)


def _validate_candidate_row(row: Mapping[str, object]) -> None:
    for field in (
        "accepted_taxon_key",
        "scientific_name",
        "classifier_task",
        "geographic_scope",
        "life_stage",
        "visual_domain",
    ):
        _required_text(row[field], field=field)
    _boolean(row["target_candidate"], field="target_candidate")
    _nonnegative_integer(row["candidate_priority"], field="candidate_priority")
    reasons = _validate_sorted_unique_text(
        row["candidate_reasons"], field="candidate_reasons"
    )
    if not reasons:
        raise ValueError("candidate_reasons must not be empty")
    for field in _CANDIDATE_SIMILARITY_FIELDS:
        _optional_bounded_float(row[field], field=field, minimum=-1.0, maximum=1.0)
    _positive_integer(row["reference_top_k"], field="reference_top_k")
    _finite_float(row["classifier_decision_score"], field="classifier_decision_score")
    _optional_bounded_float(
        row["calibrated_probability"],
        field="calibrated_probability",
        minimum=0.0,
        maximum=1.0,
    )
    _finite_float(
        row["regional_classifier_decision_score"],
        field="regional_classifier_decision_score",
    )
    _optional_bounded_float(
        row["regional_calibrated_probability"],
        field="regional_calibrated_probability",
        minimum=0.0,
        maximum=1.0,
    )
    _optional_positive_integer(row["regional_rank"], field="regional_rank")
    _optional_positive_integer(row["global_rank"], field="global_rank")
    _optional_bounded_float(
        row["geographic_evidence_score"],
        field="geographic_evidence_score",
        minimum=0.0,
        maximum=1.0,
    )
    _nonnegative_integer(row["occurrence_support"], field="occurrence_support")
    _boolean(row["life_stage_compatible"], field="life_stage_compatible")
    _boolean(row["visual_domain_compatible"], field="visual_domain_compatible")
    competitor_key = _optional_text(
        row["best_competitor_accepted_taxon_key"],
        field="best_competitor_accepted_taxon_key",
    )
    competitor_name = _optional_text(
        row["best_competitor_scientific_name"],
        field="best_competitor_scientific_name",
    )
    if (competitor_key is None) != (competitor_name is None):
        raise ValueError("candidate competitor key and name must be populated together")
    _optional_finite_float(row["competitor_margin"], field="competitor_margin")
    _optional_bounded_float(
        row["nonmatch_score"], field="nonmatch_score", minimum=-1.0, maximum=1.0
    )
    _optional_text(row["abstention_reason"], field="abstention_reason")
    for field in _CANDIDATE_FINGERPRINT_FIELDS:
        _fingerprint(row[field], field=field)
    expected = canonical_semantic_fingerprint(_candidate_semantics(row))
    actual = _fingerprint(
        row["candidate_evidence_fingerprint"],
        field="candidate_evidence_fingerprint",
    )
    if actual != expected:
        raise ValueError("candidate_evidence_fingerprint is inconsistent")


def _validate_target_projection(
    row: Mapping[str, object], target: Mapping[str, object]
) -> None:
    projections = {
        "target_raw_text_similarity": "text_ensemble_similarity",
        "target_reference_centroid_similarity": "reference_centroid_similarity",
        "target_nearest_reference_similarity": "nearest_reference_similarity",
        "target_top_k_reference_similarity": "top_k_reference_similarity",
        "target_local_prototype_similarity": "local_prototype_similarity",
        "target_global_prototype_similarity": "global_prototype_similarity",
        "calibrated_target_probability": "calibrated_probability",
        "target_regional_rank": "regional_rank",
        "target_global_rank": "global_rank",
    }
    for object_field, candidate_field in projections.items():
        if row[object_field] != target[candidate_field]:
            raise ValueError(
                f"{object_field} does not match the complete target candidate evidence"
            )


def _validate_competitor(row: Mapping[str, object]) -> None:
    key = _optional_text(
        row["best_competitor_accepted_taxon_key"],
        field="best_competitor_accepted_taxon_key",
    )
    name = _optional_text(
        row["best_competitor_scientific_name"],
        field="best_competitor_scientific_name",
    )
    if (key is None) != (name is None):
        raise ValueError("best competitor key and name must be populated together")
    if key is None:
        return
    candidates = {
        str(item["accepted_taxon_key"]): item
        for item in row["regional_candidate_evidence"]
        if not bool(item["target_candidate"])
    }
    candidate = candidates.get(key)
    if candidate is None or candidate["scientific_name"] != name:
        raise ValueError(
            "best competitor must resolve to non-target candidate evidence"
        )
    probability = row["calibrated_best_competitor_probability"]
    if (
        probability is not None
        and probability != candidate["regional_calibrated_probability"]
    ):
        raise ValueError(
            "best competitor probability does not match candidate evidence"
        )
    similarity = row["best_competitor_reference_similarity"]
    if (
        similarity is not None
        and similarity != candidate["reference_centroid_similarity"]
    ):
        raise ValueError(
            "best competitor reference score does not match candidate evidence"
        )
    target_similarity = row["target_reference_centroid_similarity"]
    margin = row["target_competitor_margin"]
    if target_similarity is not None and similarity is not None:
        expected = float(target_similarity) - float(similarity)
        if margin is None or not isclose(float(margin), expected, abs_tol=1e-6):
            raise ValueError("target competitor margin is inconsistent")


def _validate_decision(row: Mapping[str, object]) -> None:
    decision = _required_text(
        row["classification_decision"], field="classification_decision"
    )
    if decision not in CLASSIFICATION_OUTCOMES:
        raise ValueError(f"unsupported classification_decision: {decision}")
    abstained = bool(row["abstained"])
    reason = _optional_text(row["abstention_reason"], field="abstention_reason")
    priority = _required_text(row["review_priority"], field="review_priority")
    if priority not in _REVIEW_PRIORITIES:
        raise ValueError(f"unsupported review_priority: {priority}")
    threshold = _optional_bounded_float(
        row["model_decision_threshold"],
        field="model_decision_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    margin_threshold = _optional_finite_float(
        row["competitor_margin_threshold"], field="competitor_margin_threshold"
    )
    provenance = _fingerprint(row["threshold_provenance"], field="threshold_provenance")
    if provenance != row["decision_policy_fingerprint"]:
        raise ValueError(
            "threshold provenance does not match decision policy fingerprint"
        )
    if abstained and reason is None:
        raise ValueError("abstained decisions require an abstention_reason")
    if not abstained and reason is not None:
        raise ValueError("non-abstained decisions cannot have an abstention_reason")
    if abstained and priority == "none":
        raise ValueError("abstained decisions must enter review")
    if not abstained and priority != "none":
        raise ValueError("non-abstained decisions cannot have review priority")
    if decision != TARGET_CONFIRMED:
        return
    if abstained:
        raise ValueError("target_confirmed cannot be abstained")
    if not bool(row["route_compatible"]):
        raise ValueError("target_confirmed requires a compatible route")
    if not bool(row["reference_coverage_sufficient"]):
        raise ValueError("target_confirmed requires sufficient reference coverage")
    if bool(row["domain_negative_detected"]):
        raise ValueError("target_confirmed cannot retain a domain negative")
    if bool(row["out_of_distribution"]):
        raise ValueError("target_confirmed cannot be out of distribution")
    if not bool(row["visual_detail_sufficient"]):
        raise ValueError("target_confirmed requires sufficient visual detail")
    if bool(row["no_geo_global_fallback"]):
        raise ValueError("target_confirmed cannot use no-geo review fallback")
    probability = row["calibrated_target_probability"]
    if probability is None or threshold is None or float(probability) < threshold:
        raise ValueError("target_confirmed does not meet its model decision threshold")
    competitor_margin = row["target_competitor_margin"]
    if (
        competitor_margin is None
        or margin_threshold is None
        or float(competitor_margin) < margin_threshold
    ):
        raise ValueError(
            "target_confirmed does not meet its competitor margin threshold"
        )


def _target_score_identity(row: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "schema_version",
        "source",
        "flickr_photo_id",
        "source_record_hash",
        "scoring_unit_id",
        "detection_id",
        "route",
        "geo_cluster_id",
        "candidate_set_id",
        "target_accepted_taxon_key",
        "reference_bank_version",
        "classifier_version",
        "calibration_version",
        "prompt_version",
        "visual_input_version",
        "classification_mode",
        *_FINGERPRINT_FIELDS,
    )
    return {
        "target_score_id_version": TARGET_SCORE_ID_VERSION,
        **{field: row[field] for field in fields},
    }


def _target_score_semantics(row: Mapping[str, object]) -> dict[str, object]:
    excluded = {"target_score_fingerprint", "scored_at"}
    return {
        "target_score_fingerprint_version": TARGET_SCORE_FINGERPRINT_VERSION,
        **{
            field: row[field]
            for field in TARGET_AWARE_OBJECT_SCORE_SCHEMA
            if field not in excluded
        },
    }


def _candidate_semantics(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "candidate_evidence_fingerprint_version": (
            TARGET_CANDIDATE_EVIDENCE_FINGERPRINT_VERSION
        ),
        **{
            field: row[field]
            for field in REGIONAL_CANDIDATE_EVIDENCE_SCHEMA
            if field != "candidate_evidence_fingerprint"
        },
    }


def _sorted_unique_values(value: object, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    result = sorted({_required_text(item, field=field) for item in value})
    return result


def _validate_sorted_unique_text(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a non-null list")
    normalized = [_required_text(item, field=field) for item in value]
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field} must be sorted and unique")
    return normalized


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be nonblank canonical text")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _fingerprint(value: object, *, field: str) -> str:
    parsed = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(parsed) is None:
        raise ValueError(f"{field} must be a canonical sha256 fingerprint")
    return parsed


def _git_sha(value: object) -> str:
    parsed = _required_text(value, field="producer_git_sha")
    if _GIT_SHA_PATTERN.fullmatch(parsed) is None:
        raise ValueError("producer_git_sha must be a full lowercase 40-character SHA")
    return parsed


def _utc_datetime(value: object, *, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be UTC")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _optional_finite_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field=field)


def _nonnegative_float(value: object, *, field: str) -> float:
    parsed = _finite_float(value, field=field)
    if parsed < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return parsed


def _optional_bounded_float(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    parsed = _finite_float(value, field=field)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def _nonnegative_integer(value: object, *, field: str) -> int:
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    parsed = _nonnegative_integer(value, field=field)
    if parsed == 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _optional_positive_integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, field=field)


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be Boolean")
    return value


__all__ = [
    "REGIONAL_CANDIDATE_EVIDENCE_SCHEMA",
    "TARGET_AWARE_CANDIDATE_SCORE_SCHEMA",
    "TARGET_AWARE_CANDIDATE_SCORES_FILE",
    "TARGET_AWARE_CANDIDATE_SCORES_SCHEMA_VERSION",
    "TARGET_AWARE_OBJECT_SCORE_SCHEMA",
    "TARGET_AWARE_OBJECT_SCORES_FILE",
    "TARGET_AWARE_OBJECT_SCORES_SCHEMA_VERSION",
    "VISUAL_INPUT_EVIDENCE_SCHEMA",
    "empty_target_aware_candidate_scores_frame",
    "empty_target_aware_object_scores_frame",
    "make_target_score_id",
    "target_aware_candidate_scores_frame",
    "target_aware_object_scores_frame",
    "validate_target_aware_candidate_scores",
    "validate_target_aware_object_scores",
    "write_target_aware_candidate_scores",
    "write_target_aware_object_scores",
]
