"""Raw dynamic-pool candidate score and photo-summary contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.dynamic_pool_contracts import (
    DYNAMIC_POOL_SCORING_STAGES,
    validate_dynamic_reference_pool_plans,
)
from biominer.bioclip.family_geo_candidates import (
    EVIDENCE_AVAILABILITY_STATES,
    validate_family_geo_candidate_sets,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.storage.parquet import write_parquet


DYNAMIC_POOL_CANDIDATE_SCORE_SCHEMA_VERSION = "dynamic-pool-candidate-score-v1.0.0"
DYNAMIC_POOL_PHOTO_SUMMARY_SCHEMA_VERSION = "dynamic-pool-photo-summary-v1.0.0"
DYNAMIC_POOL_CANDIDATE_SCORES_FILE = "dynamic_pool_candidate_scores.parquet"
DYNAMIC_POOL_PHOTO_SUMMARY_FILE = "dynamic_pool_photo_summary.parquet"

SCORE_AVAILABILITY_STATES = frozenset({"available", "unavailable"})
PROBABILITY_AVAILABILITY_STATES = frozenset(
    {"available", "unavailable", "not_applicable"}
)
STATISTICAL_SUPPORT_STATES = frozenset(
    {"not_evaluated", "insufficient_sample", "eligible", "ineligible"}
)

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PLAN_ID_PATTERN = re.compile(r"dynamic-pool-plan:[0-9a-f]{64}\Z")
_POOL_ID_PATTERN = re.compile(r"dynamic-reference-pool:[0-9a-f]{64}\Z")
_CANDIDATE_SET_ID_PATTERN = re.compile(r"family-geo-candidate-set:[0-9a-f]{64}\Z")
_SCORE_ID_PATTERN = re.compile(r"dynamic-pool-score:[0-9a-f]{64}\Z")
_SUMMARY_ID_PATTERN = re.compile(r"dynamic-pool-photo-summary:[0-9a-f]{64}\Z")

_GROUP_FIELDS = (
    "run_id",
    "flickr_query_id",
    "flickr_photo_id",
    "organism_unit_id",
    "visual_input_id",
    "scoring_stage",
    "query_route",
    "plan_id",
    "plan_fingerprint",
    "candidate_set_id",
    "candidate_set_fingerprint",
    "target_accepted_taxon_key",
    "target_scientific_name",
    "global_pool_ids",
    "local_pool_ids",
    "safety_pool_ids",
    "local_pool_status",
    "local_pool_unavailable_reason",
    "score_policy_version",
    "score_policy_fingerprint",
    "model_fingerprint",
)
_INPUT_FIELDS = (
    *_GROUP_FIELDS,
    "candidate_accepted_taxon_key",
    "candidate_scientific_name",
    "candidate_priority",
    "target_candidate",
    "target_preserved",
    "global_score_status",
    "global_score_unavailable_reason",
    "global_prototype_similarity",
    "global_nearest_reference_similarity",
    "global_top_k_mean_similarity",
    "global_raw_component_score",
    "global_configured_k",
    "global_effective_k",
    "global_reference_count",
    "global_independent_observation_count",
    "global_reference_shortfall_count",
    "local_score_status",
    "local_score_unavailable_reason",
    "local_prototype_similarity",
    "local_nearest_reference_similarity",
    "local_top_k_mean_similarity",
    "local_raw_component_score",
    "local_configured_k",
    "local_effective_k",
    "local_reference_count",
    "local_independent_observation_count",
    "local_reference_shortfall_count",
    "global_local_disagreement_status",
    "global_local_disagreement_reason",
    "global_local_raw_disagreement",
    "family_evidence_status",
    "family_evidence_reason",
    "family_evidence_rank",
    "family_evidence_raw_score",
    "family_priority_match",
    "family_changed_membership",
    "expansion_triggered",
    "expansion_rounds",
    "expansion_triggers",
    "expansion_stop_reason",
    "fused_raw_score",
    "probability_availability",
    "calibrated_probability",
    "probability_target",
    "calibrator_fingerprint",
    "probability_unavailable_reason",
    "human_review_required",
    "statistical_support_status",
    "statistical_support_report_fingerprint",
    "statistical_support_reason",
    "abstained",
    "abstention_reasons",
)
_SORT = (
    "run_id",
    "flickr_photo_id",
    "organism_unit_id",
    "scoring_stage",
    "candidate_rank",
    "candidate_accepted_taxon_key",
)
_SUMMARY_SORT = (
    "run_id",
    "flickr_photo_id",
    "organism_unit_id",
    "scoring_stage",
)


def dynamic_pool_candidate_score_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "schema_version": pl.String,
        "score_id": pl.String,
        "run_id": pl.String,
        "flickr_query_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "visual_input_id": pl.String,
        "scoring_stage": pl.String,
        "query_route": pl.String,
        "plan_id": pl.String,
        "plan_fingerprint": pl.String,
        "candidate_set_id": pl.String,
        "candidate_set_fingerprint": pl.String,
        "target_accepted_taxon_key": pl.String,
        "target_scientific_name": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "candidate_scientific_name": pl.String,
        "candidate_priority": pl.UInt32,
        "target_candidate": pl.Boolean,
        "target_preserved": pl.Boolean,
        "global_pool_ids": pl.List(pl.String),
        "local_pool_ids": pl.List(pl.String),
        "safety_pool_ids": pl.List(pl.String),
        "local_pool_status": pl.String,
        "local_pool_unavailable_reason": pl.String,
    }
    for prefix in ("global", "local"):
        schema.update(
            {
                f"{prefix}_score_status": pl.String,
                f"{prefix}_score_unavailable_reason": pl.String,
                f"{prefix}_prototype_similarity": pl.Float64,
                f"{prefix}_nearest_reference_similarity": pl.Float64,
                f"{prefix}_top_k_mean_similarity": pl.Float64,
                f"{prefix}_raw_component_score": pl.Float64,
                f"{prefix}_configured_k": pl.UInt32,
                f"{prefix}_effective_k": pl.UInt32,
                f"{prefix}_reference_count": pl.UInt32,
                f"{prefix}_independent_observation_count": pl.UInt32,
                f"{prefix}_reference_shortfall_count": pl.UInt32,
            }
        )
    schema.update(
        {
            "global_local_disagreement_status": pl.String,
            "global_local_disagreement_reason": pl.String,
            "global_local_raw_disagreement": pl.Float64,
            "family_evidence_status": pl.String,
            "family_evidence_reason": pl.String,
            "family_evidence_rank": pl.UInt32,
            "family_evidence_raw_score": pl.Float64,
            "family_priority_match": pl.Boolean,
            "family_changed_membership": pl.Boolean,
            "expansion_triggered": pl.Boolean,
            "expansion_rounds": pl.UInt16,
            "expansion_triggers": pl.List(pl.String),
            "expansion_stop_reason": pl.String,
            "score_policy_version": pl.String,
            "score_policy_fingerprint": pl.String,
            "model_fingerprint": pl.String,
            "fused_raw_score": pl.Float64,
            "candidate_rank": pl.UInt32,
            "margin_to_next_raw": pl.Float64,
            "probability_availability": pl.String,
            "calibrated_probability": pl.Float64,
            "probability_target": pl.String,
            "calibrator_fingerprint": pl.String,
            "probability_unavailable_reason": pl.String,
            "human_review_required": pl.Boolean,
            "statistical_support_status": pl.String,
            "statistical_support_report_fingerprint": pl.String,
            "statistical_support_reason": pl.String,
            "abstained": pl.Boolean,
            "abstention_reasons": pl.List(pl.String),
            "score_fingerprint": pl.String,
        }
    )
    return schema


def dynamic_pool_photo_summary_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "photo_summary_id": pl.String,
        "run_id": pl.String,
        "flickr_query_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "visual_input_id": pl.String,
        "scoring_stage": pl.String,
        "query_route": pl.String,
        "plan_id": pl.String,
        "plan_fingerprint": pl.String,
        "candidate_set_id": pl.String,
        "candidate_set_fingerprint": pl.String,
        "target_accepted_taxon_key": pl.String,
        "candidate_count": pl.UInt32,
        "complete_candidate_union": pl.Boolean,
        "target_preserved": pl.Boolean,
        "top_candidate_accepted_taxon_key": pl.String,
        "top_candidate_scientific_name": pl.String,
        "top_fused_raw_score": pl.Float64,
        "top_margin_raw": pl.Float64,
        "target_rank": pl.UInt32,
        "target_fused_raw_score": pl.Float64,
        "global_top_candidate_accepted_taxon_key": pl.String,
        "local_top_candidate_accepted_taxon_key": pl.String,
        "global_local_top1_agreement": pl.Boolean,
        "global_local_disagreement_status": pl.String,
        "global_local_disagreement_reason": pl.String,
        "expansion_triggered": pl.Boolean,
        "maximum_expansion_rounds": pl.UInt16,
        "abstained": pl.Boolean,
        "abstention_reasons": pl.List(pl.String),
        "decision_status": pl.String,
        "probability_availability": pl.String,
        "top_calibrated_probability": pl.Float64,
        "calibrator_fingerprint": pl.String,
        "probability_unavailable_reason": pl.String,
        "human_review_required": pl.Boolean,
        "statistical_support_status": pl.String,
        "statistical_support_report_fingerprint": pl.String,
        "statistical_support_reason": pl.String,
        "release_ready": pl.Boolean,
        "scientific_claim_allowed": pl.Boolean,
        "candidate_scores_fingerprint": pl.String,
        "summary_fingerprint": pl.String,
    }


def build_dynamic_pool_candidate_scores(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise TypeError("dynamic score rows must be a sequence")
    normalized: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("dynamic score rows must contain mappings")
        _require_exact_fields(row, set(_INPUT_FIELDS))
        normalized.append(_normalized_score(row))
    if not normalized:
        return pl.DataFrame(schema=dynamic_pool_candidate_score_schema())
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in normalized:
        groups.setdefault(_group_key(row), []).append(row)
    output: list[dict[str, object]] = []
    for group_key in sorted(groups, key=lambda key: tuple(str(item) for item in key)):
        group = sorted(
            groups[group_key],
            key=lambda row: (
                -float(row["fused_raw_score"]),
                str(row["candidate_accepted_taxon_key"]),
            ),
        )
        _validate_complete_score_set(group)
        for index, row in enumerate(group):
            margin = (
                float(row["fused_raw_score"])
                - float(group[index + 1]["fused_raw_score"])
                if index + 1 < len(group)
                else None
            )
            base = {
                "schema_version": DYNAMIC_POOL_CANDIDATE_SCORE_SCHEMA_VERSION,
                **row,
                "candidate_rank": index + 1,
                "margin_to_next_raw": margin,
            }
            score_digest = canonical_semantic_fingerprint(base).removeprefix("sha256:")
            complete = {"score_id": f"dynamic-pool-score:{score_digest}", **base}
            complete["score_fingerprint"] = canonical_semantic_fingerprint(complete)
            output.append(complete)
    frame = pl.DataFrame(
        output,
        schema=dynamic_pool_candidate_score_schema(),
        orient="row",
        strict=True,
    ).sort(*_SORT)
    validate_dynamic_pool_candidate_scores(frame)
    return frame


def _derive_summaries(scores: pl.DataFrame) -> pl.DataFrame:
    validate_dynamic_pool_candidate_scores(scores)
    if scores.is_empty():
        return pl.DataFrame(schema=dynamic_pool_photo_summary_schema())
    output: list[dict[str, object]] = []
    for plan_id in sorted(scores["plan_id"].unique().to_list()):
        group = scores.filter(pl.col("plan_id") == plan_id).sort("candidate_rank")
        rows = group.to_dicts()
        top = rows[0]
        target = next(row for row in rows if row["target_candidate"])
        global_top = _component_top(rows, prefix="global")
        local_top = _component_top(rows, prefix="local")
        disagreement_available = global_top is not None and local_top is not None
        candidate_scores_fingerprint = canonical_semantic_fingerprint(
            {
                "schema_version": "dynamic-pool-candidate-score-set-v1",
                "plan_id": plan_id,
                "score_fingerprints": group["score_fingerprint"].to_list(),
            }
        )
        base: dict[str, object] = {
            "schema_version": DYNAMIC_POOL_PHOTO_SUMMARY_SCHEMA_VERSION,
            "run_id": top["run_id"],
            "flickr_query_id": top["flickr_query_id"],
            "flickr_photo_id": top["flickr_photo_id"],
            "organism_unit_id": top["organism_unit_id"],
            "visual_input_id": top["visual_input_id"],
            "scoring_stage": top["scoring_stage"],
            "query_route": top["query_route"],
            "plan_id": plan_id,
            "plan_fingerprint": top["plan_fingerprint"],
            "candidate_set_id": top["candidate_set_id"],
            "candidate_set_fingerprint": top["candidate_set_fingerprint"],
            "target_accepted_taxon_key": top["target_accepted_taxon_key"],
            "candidate_count": len(rows),
            "complete_candidate_union": True,
            "target_preserved": True,
            "top_candidate_accepted_taxon_key": top["candidate_accepted_taxon_key"],
            "top_candidate_scientific_name": top["candidate_scientific_name"],
            "top_fused_raw_score": top["fused_raw_score"],
            "top_margin_raw": top["margin_to_next_raw"],
            "target_rank": target["candidate_rank"],
            "target_fused_raw_score": target["fused_raw_score"],
            "global_top_candidate_accepted_taxon_key": (
                global_top["candidate_accepted_taxon_key"] if global_top else None
            ),
            "local_top_candidate_accepted_taxon_key": (
                local_top["candidate_accepted_taxon_key"] if local_top else None
            ),
            "global_local_top1_agreement": (
                global_top["candidate_accepted_taxon_key"]
                == local_top["candidate_accepted_taxon_key"]
                if disagreement_available
                else None
            ),
            "global_local_disagreement_status": (
                "available" if disagreement_available else "unavailable"
            ),
            "global_local_disagreement_reason": (
                None if disagreement_available else "local_score_unavailable"
            ),
            "expansion_triggered": any(row["expansion_triggered"] for row in rows),
            "maximum_expansion_rounds": max(
                int(row["expansion_rounds"]) for row in rows
            ),
            "abstained": bool(top["abstained"]),
            "abstention_reasons": top["abstention_reasons"],
            "decision_status": "abstained"
            if top["abstained"]
            else "provisional_ranked",
            "probability_availability": top["probability_availability"],
            "top_calibrated_probability": top["calibrated_probability"],
            "calibrator_fingerprint": top["calibrator_fingerprint"],
            "probability_unavailable_reason": top["probability_unavailable_reason"],
            "human_review_required": bool(top["human_review_required"]),
            "statistical_support_status": top["statistical_support_status"],
            "statistical_support_report_fingerprint": top[
                "statistical_support_report_fingerprint"
            ],
            "statistical_support_reason": top["statistical_support_reason"],
            "release_ready": False,
            "scientific_claim_allowed": False,
            "candidate_scores_fingerprint": candidate_scores_fingerprint,
        }
        digest = canonical_semantic_fingerprint(base).removeprefix("sha256:")
        complete = {
            "photo_summary_id": f"dynamic-pool-photo-summary:{digest}",
            **base,
        }
        complete["summary_fingerprint"] = canonical_semantic_fingerprint(complete)
        output.append(complete)
    frame = pl.DataFrame(
        output,
        schema=dynamic_pool_photo_summary_schema(),
        orient="row",
        strict=True,
    ).sort(*_SUMMARY_SORT)
    validate_dynamic_pool_photo_summaries(frame)
    return frame


def build_dynamic_pool_photo_summaries(scores: pl.DataFrame) -> pl.DataFrame:
    """Derive photo summaries and cross-check them against every candidate row."""

    frame = _derive_summaries(scores)
    validate_dynamic_pool_score_artifacts(scores, frame)
    return frame


def validate_dynamic_pool_candidate_scores(frame: pl.DataFrame) -> None:
    _require_frame_schema(frame, dynamic_pool_candidate_score_schema(), "scores")
    if frame.is_empty():
        return
    if frame["score_id"].n_unique() != frame.height:
        raise ValueError("dynamic score IDs are not unique")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("dynamic score rows are not canonically sorted")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != DYNAMIC_POOL_CANDIDATE_SCORE_SCHEMA_VERSION:
            raise ValueError("unsupported dynamic candidate score schema version")
        if not _SCORE_ID_PATTERN.fullmatch(str(row["score_id"])):
            raise ValueError("dynamic score_id is invalid")
        normalized = _normalized_score(row)
        if any(normalized[field] != row[field] for field in _INPUT_FIELDS):
            raise ValueError("dynamic score fields are not canonical")
        base = {
            "schema_version": DYNAMIC_POOL_CANDIDATE_SCORE_SCHEMA_VERSION,
            **normalized,
            "candidate_rank": row["candidate_rank"],
            "margin_to_next_raw": row["margin_to_next_raw"],
        }
        expected_id = "dynamic-pool-score:" + canonical_semantic_fingerprint(
            base
        ).removeprefix("sha256:")
        if row["score_id"] != expected_id:
            raise ValueError("dynamic score identity mismatch")
        _validate_fingerprint(row, field="score_fingerprint")
        groups.setdefault(_group_key(row), []).append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: int(row["candidate_rank"]))
        _validate_complete_score_set(ordered)
        if [int(row["candidate_rank"]) for row in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise ValueError("dynamic candidate ranks must be contiguous from one")
        expected_order = sorted(
            ordered,
            key=lambda row: (
                -float(row["fused_raw_score"]),
                str(row["candidate_accepted_taxon_key"]),
            ),
        )
        if [row["score_id"] for row in ordered] != [
            row["score_id"] for row in expected_order
        ]:
            raise ValueError("dynamic candidate ranks conflict with fused raw scores")
        for index, row in enumerate(ordered):
            expected_margin = (
                float(row["fused_raw_score"])
                - float(ordered[index + 1]["fused_raw_score"])
                if index + 1 < len(ordered)
                else None
            )
            if row["margin_to_next_raw"] != expected_margin:
                raise ValueError("dynamic candidate rank margin mismatch")


def validate_dynamic_pool_photo_summaries(frame: pl.DataFrame) -> None:
    _require_frame_schema(frame, dynamic_pool_photo_summary_schema(), "summaries")
    if frame.is_empty():
        return
    if frame["photo_summary_id"].n_unique() != frame.height:
        raise ValueError("dynamic photo summary IDs are not unique")
    if frame["plan_id"].n_unique() != frame.height:
        raise ValueError("dynamic photo summaries must contain one row per plan")
    if not frame.equals(frame.sort(*_SUMMARY_SORT)):
        raise ValueError("dynamic photo summaries are not canonically sorted")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != DYNAMIC_POOL_PHOTO_SUMMARY_SCHEMA_VERSION:
            raise ValueError("unsupported dynamic photo summary schema version")
        if not _SUMMARY_ID_PATTERN.fullmatch(str(row["photo_summary_id"])):
            raise ValueError("dynamic photo_summary_id is invalid")
        payload = dict(row)
        payload.pop("summary_fingerprint")
        payload.pop("photo_summary_id")
        expected_id = "dynamic-pool-photo-summary:" + canonical_semantic_fingerprint(
            payload
        ).removeprefix("sha256:")
        if row["photo_summary_id"] != expected_id:
            raise ValueError("dynamic photo summary identity mismatch")
        if (
            row["complete_candidate_union"] is not True
            or row["target_preserved"] is not True
        ):
            raise ValueError(
                "dynamic photo summary requires a complete target-safe union"
            )
        if (
            row["release_ready"] is not False
            or row["scientific_claim_allowed"] is not False
        ):
            raise ValueError("dynamic model score summary cannot authorize release")
        _validate_summary_semantics(row)
        _validate_fingerprint(row, field="summary_fingerprint")


def validate_dynamic_pool_score_artifacts(
    scores: pl.DataFrame,
    summaries: pl.DataFrame,
    *,
    plans: pl.DataFrame | None = None,
    candidate_sets: pl.DataFrame | None = None,
) -> None:
    validate_dynamic_pool_candidate_scores(scores)
    validate_dynamic_pool_photo_summaries(summaries)
    rebuilt = _derive_summaries(scores)
    if not summaries.equals(rebuilt):
        raise ValueError("dynamic photo summaries do not match candidate scores")
    if plans is not None:
        validate_dynamic_reference_pool_plans(plans)
        plan_lookup = {str(row["plan_id"]): row for row in plans.iter_rows(named=True)}
        if set(scores["plan_id"].to_list()) != set(plan_lookup):
            raise ValueError("dynamic score/plan identity sets differ")
        for score in scores.iter_rows(named=True):
            plan = plan_lookup[str(score["plan_id"])]
            for field in (
                "plan_fingerprint",
                "candidate_set_id",
                "candidate_set_fingerprint",
                "global_pool_ids",
                "local_pool_ids",
                "safety_pool_ids",
                "local_pool_status",
                "local_pool_unavailable_reason",
                "model_fingerprint",
            ):
                if score[field] != plan[field]:
                    raise ValueError(f"dynamic score conflicts with plan field {field}")
    if candidate_sets is not None:
        validate_family_geo_candidate_sets(candidate_sets)
        expected = {
            (str(row["candidate_set_id"]), str(row["candidate_accepted_taxon_key"]))
            for row in candidate_sets.iter_rows(named=True)
        }
        actual = {
            (str(row["candidate_set_id"]), str(row["candidate_accepted_taxon_key"]))
            for row in scores.iter_rows(named=True)
        }
        if actual != expected:
            raise ValueError("dynamic score/candidate identity sets differ")


def write_dynamic_pool_score_artifacts(
    scores: pl.DataFrame,
    summaries: pl.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_dynamic_pool_score_artifacts(scores, summaries)
    destination = Path(output_dir)
    return {
        "candidate_scores": write_parquet(
            scores, destination / DYNAMIC_POOL_CANDIDATE_SCORES_FILE
        ),
        "photo_summary": write_parquet(
            summaries, destination / DYNAMIC_POOL_PHOTO_SUMMARY_FILE
        ),
    }


def _normalized_score(values: Mapping[str, object]) -> dict[str, object]:
    text_fields = (
        "run_id",
        "flickr_query_id",
        "flickr_photo_id",
        "organism_unit_id",
        "visual_input_id",
        "scoring_stage",
        "query_route",
        "plan_id",
        "plan_fingerprint",
        "candidate_set_id",
        "candidate_set_fingerprint",
        "target_accepted_taxon_key",
        "target_scientific_name",
        "candidate_accepted_taxon_key",
        "candidate_scientific_name",
        "local_pool_status",
        "global_score_status",
        "local_score_status",
        "global_local_disagreement_status",
        "family_evidence_status",
        "expansion_stop_reason",
        "score_policy_version",
        "score_policy_fingerprint",
        "model_fingerprint",
        "probability_availability",
        "statistical_support_status",
        "statistical_support_reason",
    )
    row: dict[str, object] = {
        field: _required_text(values[field], field=field) for field in text_fields
    }
    optional_text_fields = (
        "local_pool_unavailable_reason",
        "global_score_unavailable_reason",
        "local_score_unavailable_reason",
        "global_local_disagreement_reason",
        "family_evidence_reason",
        "probability_target",
        "calibrator_fingerprint",
        "probability_unavailable_reason",
        "statistical_support_report_fingerprint",
    )
    row.update(
        {
            field: _optional_text(values[field], field=field)
            for field in optional_text_fields
        }
    )
    for field in ("global_pool_ids", "local_pool_ids", "safety_pool_ids"):
        row[field] = _canonical_pool_ids(values[field], field=field)
    row["candidate_priority"] = _nonnegative_int(
        values["candidate_priority"], field="candidate_priority", maximum=2**32 - 1
    )
    for field in (
        "target_candidate",
        "target_preserved",
        "family_changed_membership",
        "expansion_triggered",
        "human_review_required",
        "abstained",
    ):
        row[field] = _boolean(values[field], field=field)
    row["family_priority_match"] = _optional_boolean(
        values["family_priority_match"], field="family_priority_match"
    )
    for prefix in ("global", "local"):
        for suffix in (
            "prototype_similarity",
            "nearest_reference_similarity",
            "top_k_mean_similarity",
            "raw_component_score",
        ):
            field = f"{prefix}_{suffix}"
            row[field] = _optional_score(values[field], field=field)
        for suffix in (
            "configured_k",
            "effective_k",
            "reference_count",
            "independent_observation_count",
            "reference_shortfall_count",
        ):
            field = f"{prefix}_{suffix}"
            row[field] = _nonnegative_int(values[field], field=field, maximum=2**32 - 1)
        _validate_component(row, prefix=prefix)
    row["global_local_raw_disagreement"] = _optional_nonnegative_float(
        values["global_local_raw_disagreement"],
        field="global_local_raw_disagreement",
    )
    row["family_evidence_rank"] = _optional_positive_int(
        values["family_evidence_rank"], field="family_evidence_rank"
    )
    row["family_evidence_raw_score"] = _optional_score(
        values["family_evidence_raw_score"], field="family_evidence_raw_score"
    )
    row["expansion_rounds"] = _nonnegative_int(
        values["expansion_rounds"], field="expansion_rounds", maximum=2**16 - 1
    )
    row["expansion_triggers"] = _canonical_strings(
        values["expansion_triggers"], field="expansion_triggers"
    )
    row["fused_raw_score"] = _score(values["fused_raw_score"], field="fused_raw_score")
    row["calibrated_probability"] = _optional_probability(
        values["calibrated_probability"]
    )
    row["abstention_reasons"] = _canonical_strings(
        values["abstention_reasons"], field="abstention_reasons"
    )
    _validate_score_semantics(row)
    return row


def _validate_score_semantics(row: Mapping[str, object]) -> None:
    if row["scoring_stage"] not in DYNAMIC_POOL_SCORING_STAGES:
        raise ValueError("unsupported dynamic score scoring_stage")
    if not _PLAN_ID_PATTERN.fullmatch(str(row["plan_id"])):
        raise ValueError("dynamic score plan_id is invalid")
    if not _CANDIDATE_SET_ID_PATTERN.fullmatch(str(row["candidate_set_id"])):
        raise ValueError("dynamic score candidate_set_id is invalid")
    if not row["global_pool_ids"]:
        raise ValueError("dynamic score requires global pool identities")
    if row["global_score_status"] != "available":
        raise ValueError("dynamic score requires an available global component")
    if row["local_pool_status"] == "available":
        if (
            not row["local_pool_ids"]
            or row["local_pool_unavailable_reason"] is not None
        ):
            raise ValueError("available local score requires pool IDs")
    elif row["local_pool_status"] == "unavailable":
        if row["local_pool_ids"] or row["local_pool_unavailable_reason"] is None:
            raise ValueError("unavailable local score requires an exact reason")
    else:
        raise ValueError("unsupported local_pool_status")
    if row["family_changed_membership"]:
        raise ValueError("family evidence cannot change scored candidate membership")
    family_values = (
        row["family_evidence_rank"],
        row["family_evidence_raw_score"],
        row["family_priority_match"],
    )
    if row["family_evidence_status"] not in EVIDENCE_AVAILABILITY_STATES:
        raise ValueError("unsupported family score evidence status")
    if row["family_evidence_status"] == "available":
        if (
            any(value is None for value in family_values)
            or row["family_evidence_reason"] is not None
        ):
            raise ValueError("available family score evidence is incomplete")
    elif (
        any(value is not None for value in family_values)
        or row["family_evidence_reason"] is None
    ):
        raise ValueError("unavailable family score evidence is inconsistent")
    disagreement_available = row["global_local_disagreement_status"] == "available"
    if disagreement_available:
        if (
            row["global_local_raw_disagreement"] is None
            or row["global_local_disagreement_reason"] is not None
        ):
            raise ValueError("available global/local disagreement is incomplete")
    elif row["global_local_disagreement_status"] == "unavailable":
        if (
            row["global_local_raw_disagreement"] is not None
            or row["global_local_disagreement_reason"] is None
        ):
            raise ValueError("unavailable global/local disagreement is inconsistent")
    else:
        raise ValueError("unsupported global/local disagreement status")
    if disagreement_available and (
        row["global_score_status"] != "available"
        or row["local_score_status"] != "available"
        or row["global_local_raw_disagreement"]
        != abs(
            float(row["global_raw_component_score"])
            - float(row["local_raw_component_score"])
        )
    ):
        raise ValueError("global/local disagreement does not match components")
    if row["expansion_triggered"] != bool(row["expansion_rounds"]):
        raise ValueError("expansion flag and rounds are inconsistent")
    if row["expansion_triggered"] != bool(row["expansion_triggers"]):
        raise ValueError("expansion flag and triggers are inconsistent")
    _validate_probability(row)
    if row["statistical_support_status"] not in STATISTICAL_SUPPORT_STATES:
        raise ValueError("unsupported statistical support status")
    report = row["statistical_support_report_fingerprint"]
    if row["statistical_support_status"] in {"eligible", "ineligible"}:
        _sha256(report, field="statistical_support_report_fingerprint")
    elif report is not None:
        raise ValueError("unevaluated statistical support cannot have a report")
    if row["abstained"] != bool(row["abstention_reasons"]):
        raise ValueError("abstention state and reasons are inconsistent")
    for field in (
        "visual_input_id",
        "plan_fingerprint",
        "candidate_set_fingerprint",
        "score_policy_fingerprint",
        "model_fingerprint",
    ):
        _sha256(row[field], field=field)


def _validate_component(row: Mapping[str, object], *, prefix: str) -> None:
    status = row[f"{prefix}_score_status"]
    reason = row[f"{prefix}_score_unavailable_reason"]
    similarities = tuple(
        row[f"{prefix}_{suffix}"]
        for suffix in (
            "prototype_similarity",
            "nearest_reference_similarity",
            "top_k_mean_similarity",
            "raw_component_score",
        )
    )
    configured = int(row[f"{prefix}_configured_k"])
    effective = int(row[f"{prefix}_effective_k"])
    references = int(row[f"{prefix}_reference_count"])
    independent = int(row[f"{prefix}_independent_observation_count"])
    shortfall = int(row[f"{prefix}_reference_shortfall_count"])
    if effective > configured:
        raise ValueError(f"available {prefix} effective k is inconsistent")
    if shortfall != configured - effective:
        raise ValueError(f"{prefix} score shortfall is inconsistent")
    if status == "available":
        if any(value is None for value in similarities) or reason is not None:
            raise ValueError(f"available {prefix} score component is incomplete")
        if not 1 <= effective <= configured or references < effective:
            raise ValueError(f"available {prefix} effective k is inconsistent")
        if independent > references:
            raise ValueError(f"{prefix} independence count exceeds references")
    elif status == "unavailable":
        counts = (effective, references, independent)
        if (
            any(value is not None for value in similarities)
            or reason is None
            or any(counts)
        ):
            raise ValueError(f"unavailable {prefix} score component is inconsistent")
    else:
        raise ValueError(f"unsupported {prefix} score status")


def _validate_summary_semantics(row: Mapping[str, object]) -> None:
    candidate_count = _nonnegative_int(
        row["candidate_count"], field="candidate_count", maximum=2**32 - 1
    )
    target_rank = _nonnegative_int(
        row["target_rank"], field="target_rank", maximum=2**32 - 1
    )
    if candidate_count == 0 or not 1 <= target_rank <= candidate_count:
        raise ValueError("dynamic photo summary candidate counts are inconsistent")
    for field in (
        "run_id",
        "flickr_query_id",
        "flickr_photo_id",
        "organism_unit_id",
        "query_route",
        "target_accepted_taxon_key",
        "top_candidate_accepted_taxon_key",
        "top_candidate_scientific_name",
    ):
        _required_text(row[field], field=field)
    if row["scoring_stage"] not in DYNAMIC_POOL_SCORING_STAGES:
        raise ValueError("unsupported dynamic photo summary scoring_stage")
    if not _PLAN_ID_PATTERN.fullmatch(str(row["plan_id"])):
        raise ValueError("dynamic photo summary plan_id is invalid")
    if not _CANDIDATE_SET_ID_PATTERN.fullmatch(str(row["candidate_set_id"])):
        raise ValueError("dynamic photo summary candidate_set_id is invalid")
    _score(row["top_fused_raw_score"], field="top_fused_raw_score")
    _score(row["target_fused_raw_score"], field="target_fused_raw_score")
    _optional_nonnegative_float(row["top_margin_raw"], field="top_margin_raw")
    expansion_rounds = _nonnegative_int(
        row["maximum_expansion_rounds"],
        field="maximum_expansion_rounds",
        maximum=2**16 - 1,
    )
    if row["expansion_triggered"] != bool(expansion_rounds):
        raise ValueError("summary expansion state and rounds are inconsistent")
    for field in (
        "visual_input_id",
        "plan_fingerprint",
        "candidate_set_fingerprint",
        "candidate_scores_fingerprint",
    ):
        _sha256(row[field], field=field)

    disagreement_status = row["global_local_disagreement_status"]
    if disagreement_status == "available":
        if (
            row["global_top_candidate_accepted_taxon_key"] is None
            or row["local_top_candidate_accepted_taxon_key"] is None
            or row["global_local_top1_agreement"] is None
            or row["global_local_disagreement_reason"] is not None
        ):
            raise ValueError("available summary disagreement evidence is incomplete")
        expected_agreement = (
            row["global_top_candidate_accepted_taxon_key"]
            == row["local_top_candidate_accepted_taxon_key"]
        )
        if row["global_local_top1_agreement"] != expected_agreement:
            raise ValueError("summary top-one agreement is inconsistent")
    elif disagreement_status == "unavailable":
        if (
            row["global_top_candidate_accepted_taxon_key"] is None
            or row["local_top_candidate_accepted_taxon_key"] is not None
            or row["global_local_top1_agreement"] is not None
            or row["global_local_disagreement_reason"] is None
        ):
            raise ValueError(
                "unavailable summary disagreement evidence is inconsistent"
            )
    else:
        raise ValueError("unsupported summary disagreement status")

    if row["abstained"] != bool(row["abstention_reasons"]):
        raise ValueError("summary abstention state and reasons are inconsistent")
    expected_decision = "abstained" if row["abstained"] else "provisional_ranked"
    if row["decision_status"] != expected_decision:
        raise ValueError("summary decision status is inconsistent")

    probability_status = row["probability_availability"]
    probability_values = (
        row["top_calibrated_probability"],
        row["calibrator_fingerprint"],
    )
    if probability_status == "available":
        if (
            any(value is None for value in probability_values)
            or row["probability_unavailable_reason"] is not None
        ):
            raise ValueError("available summary probability evidence is incomplete")
        _optional_probability(row["top_calibrated_probability"])
        _sha256(row["calibrator_fingerprint"], field="calibrator_fingerprint")
    elif probability_status in {"unavailable", "not_applicable"}:
        if (
            any(value is not None for value in probability_values)
            or row["probability_unavailable_reason"] is None
        ):
            raise ValueError("unavailable summary probability evidence is inconsistent")
    else:
        raise ValueError("unsupported summary probability availability")

    support_status = row["statistical_support_status"]
    if support_status not in STATISTICAL_SUPPORT_STATES:
        raise ValueError("unsupported summary statistical support status")
    support_report = row["statistical_support_report_fingerprint"]
    if support_status in {"eligible", "ineligible"}:
        _sha256(support_report, field="statistical_support_report_fingerprint")
    elif support_report is not None:
        raise ValueError("unevaluated summary support cannot have a report")
    _required_text(
        row["statistical_support_reason"], field="statistical_support_reason"
    )


def _validate_probability(row: Mapping[str, object]) -> None:
    status = row["probability_availability"]
    if status not in PROBABILITY_AVAILABILITY_STATES:
        raise ValueError("unsupported probability availability")
    values = (
        row["calibrated_probability"],
        row["probability_target"],
        row["calibrator_fingerprint"],
    )
    reason = row["probability_unavailable_reason"]
    if status == "available":
        if any(value is None for value in values) or reason is not None:
            raise ValueError("available probability evidence is incomplete")
        _sha256(values[2], field="calibrator_fingerprint")
    elif any(value is not None for value in values) or reason is None:
        raise ValueError("unavailable probability evidence requires null values")


def _validate_complete_score_set(group: Sequence[Mapping[str, object]]) -> None:
    keys = [str(row["candidate_accepted_taxon_key"]) for row in group]
    if len(keys) != len(set(keys)):
        raise ValueError("dynamic score set contains duplicate candidate taxa")
    targets = [row for row in group if row["target_candidate"]]
    if (
        len(targets) != 1
        or targets[0]["candidate_accepted_taxon_key"]
        != targets[0]["target_accepted_taxon_key"]
    ):
        raise ValueError("dynamic score set must contain exactly one matching target")
    if not all(row["target_preserved"] for row in group):
        raise ValueError("dynamic score set did not preserve the target")
    shared = ("abstained", "abstention_reasons", "human_review_required")
    if any(len({_hashable(row[field]) for row in group}) != 1 for field in shared):
        raise ValueError("dynamic score decision fields conflict within a plan")


def _component_top(
    rows: Sequence[Mapping[str, object]], *, prefix: str
) -> Mapping[str, object] | None:
    available = [row for row in rows if row[f"{prefix}_score_status"] == "available"]
    return (
        min(
            available,
            key=lambda row: (
                -float(row[f"{prefix}_raw_component_score"]),
                str(row["candidate_accepted_taxon_key"]),
            ),
        )
        if available
        else None
    )


def _group_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(_hashable(row[field]) for field in _GROUP_FIELDS)


def _hashable(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _require_frame_schema(
    frame: pl.DataFrame, schema: dict[str, pl.DataType], label: str
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"dynamic {label} must be a Polars DataFrame")
    if frame.schema != schema:
        raise ValueError(f"dynamic {label} schema mismatch")


def _require_exact_fields(values: Mapping[str, object], expected: set[str]) -> None:
    missing = expected - set(values)
    unexpected = set(values) - expected
    if missing or unexpected:
        raise ValueError(
            f"dynamic score fields mismatch: missing={sorted(missing)}, "
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


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be Boolean")
    return value


def _optional_boolean(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, field=field)


def _nonnegative_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{field} must be an integer in [0, {maximum}]")
    return value


def _optional_positive_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    result = _nonnegative_int(value, field=field, maximum=2**32 - 1)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _score(value: object, *, field: str) -> float:
    result = _optional_score(value, field=field)
    if result is None:
        raise ValueError(f"{field} must be available")
    return result


def _optional_score(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric or null")
    result = float(value)
    if not isfinite(result) or not -1 <= result <= 1:
        raise ValueError(f"{field} must be finite and in [-1, 1]")
    return result


def _optional_nonnegative_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric or null")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _optional_probability(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("calibrated_probability must be numeric or null")
    result = float(value)
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError("calibrated_probability must be finite and in [0, 1]")
    return result


def _canonical_strings(value: object, *, field: str) -> list[str]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return sorted({_required_text(item, field=field) for item in value})


def _canonical_pool_ids(value: object, *, field: str) -> list[str]:
    values = _canonical_strings(value, field=field)
    if any(not _POOL_ID_PATTERN.fullmatch(item) for item in values):
        raise ValueError(f"{field} contains an invalid pool ID")
    return values


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical SHA-256 fingerprint")
    return text


def _validate_fingerprint(row: Mapping[str, object], *, field: str) -> None:
    payload = dict(row)
    fingerprint = payload.pop(field)
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError(f"dynamic {field} mismatch")


__all__ = [
    "DYNAMIC_POOL_CANDIDATE_SCORES_FILE",
    "DYNAMIC_POOL_CANDIDATE_SCORE_SCHEMA_VERSION",
    "DYNAMIC_POOL_PHOTO_SUMMARY_FILE",
    "DYNAMIC_POOL_PHOTO_SUMMARY_SCHEMA_VERSION",
    "PROBABILITY_AVAILABILITY_STATES",
    "SCORE_AVAILABILITY_STATES",
    "STATISTICAL_SUPPORT_STATES",
    "build_dynamic_pool_candidate_scores",
    "build_dynamic_pool_photo_summaries",
    "dynamic_pool_candidate_score_schema",
    "dynamic_pool_photo_summary_schema",
    "validate_dynamic_pool_candidate_scores",
    "validate_dynamic_pool_photo_summaries",
    "validate_dynamic_pool_score_artifacts",
    "write_dynamic_pool_score_artifacts",
]
