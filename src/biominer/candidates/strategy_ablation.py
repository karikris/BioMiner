"""Deterministic candidate-strategy plans over the complete safety union."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.family_geo_candidates import (
    validate_family_geo_candidate_sets,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.storage.parquet import write_parquet


CANDIDATE_STRATEGY_PLAN_SCHEMA_VERSION = "candidate-strategy-plan-v1.0.0"
CANDIDATE_STRATEGY_PLANS_FILE = "candidate_strategy_plans.parquet"

GEOGRAPHY_FIRST_STRATEGY = "geography_first"
GEOGRAPHY_FIRST_STRATEGY_VERSION = "geography-first-complete-union-v1.0.0"
CANDIDATE_STRATEGIES = frozenset({GEOGRAPHY_FIRST_STRATEGY})

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PLAN_ID_PATTERN = re.compile(r"candidate-strategy-plan:[0-9a-f]{64}\Z")
_SORT = (
    "run_id",
    "flickr_photo_id",
    "organism_unit_id",
    "scoring_stage",
    "strategy_name",
    "strategy_priority",
    "candidate_accepted_taxon_key",
)


def candidate_strategy_plan_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "strategy_plan_id": pl.String,
        "strategy_plan_fingerprint": pl.String,
        "strategy_name": pl.String,
        "strategy_version": pl.String,
        "source_candidate_set_id": pl.String,
        "source_candidate_set_fingerprint": pl.String,
        "run_id": pl.String,
        "flickr_query_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "scoring_stage": pl.String,
        "target_accepted_taxon_key": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "candidate_scientific_name": pl.String,
        "source_candidate_priority": pl.UInt32,
        "strategy_priority": pl.UInt32,
        "strategy_stage": pl.String,
        "strategy_stage_rank": pl.UInt32,
        "inclusion_axes": pl.List(pl.String),
        "geographic_evidence_status": pl.String,
        "geographic_evidence_score": pl.Float64,
        "family_evidence_status": pl.String,
        "family_evidence_rank": pl.UInt32,
        "family_evidence_raw_score": pl.Float64,
        "query_associated": pl.Boolean,
        "visual_neighbour": pl.Boolean,
        "safety_union_membership": pl.Boolean,
        "target_candidate": pl.Boolean,
        "target_preserved": pl.Boolean,
        "complete_union_preserved": pl.Boolean,
        "family_changed_membership": pl.Boolean,
        "strategy_policy_fingerprint": pl.String,
        "source_candidate_row_fingerprint": pl.String,
        "strategy_row_fingerprint": pl.String,
    }


def build_candidate_strategy_plans(
    candidate_sets: pl.DataFrame,
    *,
    strategy: str,
) -> pl.DataFrame:
    """Build one scheduling row per source complete-union candidate."""

    validate_family_geo_candidate_sets(candidate_sets)
    normalized_strategy = _required_text(strategy, field="strategy").casefold()
    if normalized_strategy not in CANDIDATE_STRATEGIES:
        raise ValueError(f"unsupported candidate strategy {normalized_strategy!r}")
    rows = _build_rows(candidate_sets, strategy=normalized_strategy)
    frame = (
        pl.DataFrame(
            rows,
            schema=candidate_strategy_plan_schema(),
            orient="row",
            strict=True,
        ).sort(*_SORT)
        if rows
        else pl.DataFrame(schema=candidate_strategy_plan_schema())
    )
    validate_candidate_strategy_plans(frame, candidate_sets)
    return frame


def validate_candidate_strategy_plans(
    frame: pl.DataFrame,
    candidate_sets: pl.DataFrame,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("candidate strategy plans must be a Polars DataFrame")
    validate_family_geo_candidate_sets(candidate_sets)
    if frame.schema != candidate_strategy_plan_schema():
        raise ValueError("candidate strategy plan schema mismatch")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("candidate strategy plans are not canonically sorted")
    if frame.is_empty():
        if not candidate_sets.is_empty():
            raise ValueError("candidate strategy plans do not cover source candidates")
        return
    if frame.height != frame.select(
        "strategy_plan_id", "candidate_accepted_taxon_key"
    ).n_unique():
        raise ValueError("candidate strategy plan grain is not unique")
    if frame.height != candidate_sets.height:
        raise ValueError("candidate strategy plans do not preserve the complete union")
    strategies = set(frame["strategy_name"].to_list())
    if len(strategies) != 1:
        raise ValueError("one strategy artifact must contain exactly one strategy")
    strategy = next(iter(strategies))
    if strategy not in CANDIDATE_STRATEGIES:
        raise ValueError("unsupported candidate strategy")
    expected = _build_rows(candidate_sets, strategy=strategy)
    expected_frame = pl.DataFrame(
        expected,
        schema=candidate_strategy_plan_schema(),
        orient="row",
        strict=True,
    ).sort(*_SORT)
    if not frame.equals(expected_frame):
        raise ValueError("candidate strategy plans do not match source evidence")
    for (plan_id,), group in frame.group_by("strategy_plan_id"):
        if not _PLAN_ID_PATTERN.fullmatch(str(plan_id)):
            raise ValueError("candidate strategy plan ID is invalid")
        if group["candidate_accepted_taxon_key"].n_unique() != group.height:
            raise ValueError("candidate strategy plan contains duplicate taxa")
        priorities = group.sort("strategy_priority")["strategy_priority"].to_list()
        if priorities != list(range(group.height)):
            raise ValueError("candidate strategy priorities are not contiguous")
        if group["target_candidate"].sum() != 1:
            raise ValueError("candidate strategy plan requires exactly one target")
        if not all(group["target_preserved"].to_list()):
            raise ValueError("candidate strategy plan lost target preservation")
        if not all(group["complete_union_preserved"].to_list()):
            raise ValueError("candidate strategy plan pruned the complete union")
        if any(group["family_changed_membership"].to_list()):
            raise ValueError("family evidence changed strategy membership")
        for field in (
            "strategy_plan_fingerprint",
            "source_candidate_set_fingerprint",
            "strategy_policy_fingerprint",
            "source_candidate_row_fingerprint",
            "strategy_row_fingerprint",
        ):
            if any(not _is_sha256(value) for value in group[field].to_list()):
                raise ValueError(f"invalid {field}")


def write_candidate_strategy_plans(
    frame: pl.DataFrame,
    candidate_sets: pl.DataFrame,
    output_path: str | Path,
) -> Path:
    validate_candidate_strategy_plans(frame, candidate_sets)
    destination = Path(output_path)
    if destination.suffix.casefold() != ".parquet":
        destination /= CANDIDATE_STRATEGY_PLANS_FILE
    return write_parquet(frame, destination)


def _build_rows(
    candidate_sets: pl.DataFrame,
    *,
    strategy: str,
) -> list[dict[str, object]]:
    strategy_version = _strategy_version(strategy)
    policy_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": CANDIDATE_STRATEGY_PLAN_SCHEMA_VERSION,
            "strategy": strategy,
            "strategy_version": strategy_version,
            "membership": "complete_source_union",
            "family_membership_changes_allowed": False,
            "stages": list(_stage_order(strategy)),
        }
    )
    output: list[dict[str, object]] = []
    for (candidate_set_id,), group_frame in candidate_sets.group_by(
        "candidate_set_id", maintain_order=True
    ):
        group = group_frame.sort(
            "candidate_priority", "candidate_accepted_taxon_key"
        ).to_dicts()
        scheduled = _schedule_geography_first(group)
        schedule_specs = [
            {
                "candidate_accepted_taxon_key": row[
                    "candidate_accepted_taxon_key"
                ],
                "source_candidate_row_fingerprint": row[
                    "candidate_row_fingerprint"
                ],
                "strategy_priority": priority,
                "strategy_stage": stage,
                "strategy_stage_rank": stage_rank,
                "inclusion_axes": _inclusion_axes(row),
            }
            for priority, (row, stage, stage_rank) in enumerate(scheduled)
        ]
        first = group[0]
        plan_fingerprint = canonical_semantic_fingerprint(
            {
                "schema_version": CANDIDATE_STRATEGY_PLAN_SCHEMA_VERSION,
                "strategy": strategy,
                "strategy_version": strategy_version,
                "source_candidate_set_id": candidate_set_id,
                "source_candidate_set_fingerprint": first[
                    "candidate_set_fingerprint"
                ],
                "policy_fingerprint": policy_fingerprint,
                "schedule": schedule_specs,
            }
        )
        plan_id = _prefixed_id("candidate-strategy-plan", plan_fingerprint)
        for spec, (source, _stage, _stage_rank) in zip(
            schedule_specs, scheduled, strict=True
        ):
            base = {
                "schema_version": CANDIDATE_STRATEGY_PLAN_SCHEMA_VERSION,
                "strategy_plan_id": plan_id,
                "strategy_plan_fingerprint": plan_fingerprint,
                "strategy_name": strategy,
                "strategy_version": strategy_version,
                "source_candidate_set_id": candidate_set_id,
                "source_candidate_set_fingerprint": source[
                    "candidate_set_fingerprint"
                ],
                "run_id": source["run_id"],
                "flickr_query_id": source["flickr_query_id"],
                "flickr_photo_id": source["flickr_photo_id"],
                "organism_unit_id": source["organism_unit_id"],
                "scoring_stage": source["scoring_stage"],
                "target_accepted_taxon_key": source[
                    "target_accepted_taxon_key"
                ],
                "candidate_accepted_taxon_key": source[
                    "candidate_accepted_taxon_key"
                ],
                "candidate_scientific_name": source[
                    "candidate_scientific_name"
                ],
                "source_candidate_priority": source["candidate_priority"],
                "strategy_priority": spec["strategy_priority"],
                "strategy_stage": spec["strategy_stage"],
                "strategy_stage_rank": spec["strategy_stage_rank"],
                "inclusion_axes": spec["inclusion_axes"],
                "geographic_evidence_status": source[
                    "geographic_evidence_status"
                ],
                "geographic_evidence_score": source[
                    "geographic_evidence_score"
                ],
                "family_evidence_status": source["family_evidence_status"],
                "family_evidence_rank": source["family_evidence_rank"],
                "family_evidence_raw_score": source[
                    "family_evidence_raw_score"
                ],
                "query_associated": source["query_associated"],
                "visual_neighbour": source["visual_neighbour"],
                "safety_union_membership": source["safety_union_membership"],
                "target_candidate": source["target_candidate"],
                "target_preserved": source["target_preserved"],
                "complete_union_preserved": source[
                    "included_in_complete_union"
                ],
                "family_changed_membership": source[
                    "family_changed_membership"
                ],
                "strategy_policy_fingerprint": policy_fingerprint,
                "source_candidate_row_fingerprint": source[
                    "candidate_row_fingerprint"
                ],
            }
            output.append(
                {
                    **base,
                    "strategy_row_fingerprint": canonical_semantic_fingerprint(
                        base
                    ),
                }
            )
    return output


def _schedule_geography_first(
    rows: Sequence[Mapping[str, object]],
) -> list[tuple[Mapping[str, object], str, int]]:
    stages: dict[str, list[Mapping[str, object]]] = {
        stage: [] for stage in _stage_order(GEOGRAPHY_FIRST_STRATEGY)
    }
    for row in rows:
        if row["geographic_evidence_status"] == "available":
            stage = "geographic_union"
        elif (
            row["target_candidate"]
            or row["query_associated"]
            or row["visual_neighbour"]
            or row["safety_union_membership"]
        ):
            stage = "required_safety_union"
        elif row["family_evidence_status"] == "available":
            stage = "family_expansion"
        else:
            stage = "complete_union_remainder"
        stages[stage].append(row)
    output: list[tuple[Mapping[str, object], str, int]] = []
    for stage in _stage_order(GEOGRAPHY_FIRST_STRATEGY):
        ordered = sorted(stages[stage], key=lambda row: _geography_first_key(row, stage))
        output.extend((row, stage, rank) for rank, row in enumerate(ordered))
    return output


def _geography_first_key(
    row: Mapping[str, object],
    stage: str,
) -> tuple[object, ...]:
    source_priority = int(row["candidate_priority"])
    candidate_key = str(row["candidate_accepted_taxon_key"])
    if stage == "geographic_union":
        score = row["geographic_evidence_score"]
        return (
            0 if row["target_candidate"] else 1,
            -(float(score) if score is not None else -1.0),
            -int(row["occurrence_support"]),
            source_priority,
            candidate_key,
        )
    if stage == "required_safety_union":
        return (
            0 if row["target_candidate"] else 1,
            0 if row["query_associated"] else 1,
            0 if row["visual_neighbour"] else 1,
            source_priority,
            candidate_key,
        )
    if stage == "family_expansion":
        rank = row["family_evidence_rank"]
        score = row["family_evidence_raw_score"]
        return (
            int(rank) if rank is not None else 2**32,
            -(float(score) if score is not None else -1.0),
            source_priority,
            candidate_key,
        )
    return (source_priority, candidate_key)


def _inclusion_axes(row: Mapping[str, object]) -> list[str]:
    axes: list[str] = []
    if row["target_candidate"]:
        axes.append("target")
    if row["geographic_evidence_status"] == "available":
        axes.append("geography")
    if row["family_evidence_status"] == "available":
        axes.append("family")
    if row["query_associated"]:
        axes.append("query")
    if row["visual_neighbour"]:
        axes.append("visual")
    if row["safety_union_membership"]:
        axes.append("safety")
    return sorted(set(axes))


def _stage_order(strategy: str) -> tuple[str, ...]:
    if strategy == GEOGRAPHY_FIRST_STRATEGY:
        return (
            "geographic_union",
            "required_safety_union",
            "family_expansion",
            "complete_union_remainder",
        )
    raise ValueError(f"unsupported candidate strategy {strategy!r}")


def _strategy_version(strategy: str) -> str:
    if strategy == GEOGRAPHY_FIRST_STRATEGY:
        return GEOGRAPHY_FIRST_STRATEGY_VERSION
    raise ValueError(f"unsupported candidate strategy {strategy!r}")


def _prefixed_id(prefix: str, fingerprint: str) -> str:
    return f"{prefix}:{fingerprint.removeprefix('sha256:')}"


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _is_sha256(value: object) -> bool:
    return bool(_SHA256_PATTERN.fullmatch(str(value)))


__all__ = [
    "CANDIDATE_STRATEGIES",
    "CANDIDATE_STRATEGY_PLANS_FILE",
    "CANDIDATE_STRATEGY_PLAN_SCHEMA_VERSION",
    "GEOGRAPHY_FIRST_STRATEGY",
    "GEOGRAPHY_FIRST_STRATEGY_VERSION",
    "build_candidate_strategy_plans",
    "candidate_strategy_plan_schema",
    "validate_candidate_strategy_plans",
    "write_candidate_strategy_plans",
]
