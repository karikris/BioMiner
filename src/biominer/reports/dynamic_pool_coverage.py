"""Coverage, fallback and shortfall reports for dynamic reference pools."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.dynamic_pool_contracts import (
    validate_dynamic_reference_pool_members,
    validate_dynamic_reference_pool_plans,
)
from biominer.bioclip.dynamic_pool_policy import DynamicReferencePoolPolicy
from biominer.bioclip.family_geo_candidates import (
    validate_family_geo_candidate_sets,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.storage.parquet import write_parquet


DYNAMIC_POOL_COVERAGE_SCHEMA_VERSION = "dynamic-pool-coverage-v1.0.0"
DYNAMIC_POOL_COVERAGE_REPORT_SCHEMA_VERSION = (
    "dynamic-pool-coverage-report-v1.0.0"
)
DYNAMIC_POOL_COVERAGE_FILE = "dynamic_pool_coverage_shortfalls.parquet"
DYNAMIC_POOL_COVERAGE_REPORT_FILE = "dynamic_pool_coverage_report.json"
DYNAMIC_POOL_COVERAGE_SUMMARY_FILE = "dynamic_pool_coverage_report.md"
DYNAMIC_POOL_COVERAGE_STATUSES = frozenset(
    {
        "complete",
        "usable_with_shortfalls",
        "insufficient_global",
        "no_reference_support",
    }
)

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SORT = ("run_id", "plan_id", "candidate_accepted_taxon_key")
_NO_GEO_QUALITIES = frozenset(
    {"no_geo", "unassigned_geo", "withheld", "invalid"}
)


def dynamic_pool_coverage_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "plan_id": pl.String,
        "run_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "scoring_stage": pl.String,
        "candidate_set_id": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "candidate_scientific_name": pl.String,
        "target_candidate": pl.Boolean,
        "safety_union_membership": pl.Boolean,
        "query_coordinate_quality": pl.String,
        "no_geo": pl.Boolean,
        "global_minimum": pl.UInt32,
        "global_requested": pl.UInt32,
        "global_effective": pl.UInt32,
        "global_minimum_shortfall": pl.UInt32,
        "global_unfilled_to_requested": pl.UInt32,
        "local_pool_status": pl.String,
        "local_pool_unavailable_reason": pl.String,
        "local_minimum": pl.UInt32,
        "local_requested": pl.UInt32,
        "local_effective": pl.UInt32,
        "local_minimum_shortfall": pl.UInt32,
        "local_unfilled_to_requested": pl.UInt32,
        "safety_requested": pl.UInt32,
        "safety_effective": pl.UInt32,
        "safety_unfilled_to_requested": pl.UInt32,
        "total_effective": pl.UInt32,
        "selected_reference_media_count": pl.UInt32,
        "selected_reference_observation_count": pl.UInt32,
        "independent_observation_count": pl.UInt32,
        "minimum_independent_observation_count": pl.UInt32,
        "independent_observation_shortfall": pl.UInt32,
        "independent_observer_count": pl.UInt32,
        "source_dataset_count": pl.UInt32,
        "country_count": pl.UInt32,
        "maximum_fallback_level": pl.UInt8,
        "fallback_reasons": pl.List(pl.String),
        "coverage_status": pl.String,
        "shortfall_reasons": pl.List(pl.String),
        "selection_policy_fingerprint": pl.String,
        "reference_geography_index_fingerprint": pl.String,
        "coverage_fingerprint": pl.String,
    }


def build_dynamic_pool_coverage(
    plans: pl.DataFrame,
    members: pl.DataFrame,
    candidate_sets: pl.DataFrame,
    *,
    policy: DynamicReferencePoolPolicy,
) -> pl.DataFrame:
    """Build one denominator-explicit coverage row per plan candidate."""

    if not isinstance(policy, DynamicReferencePoolPolicy):
        raise TypeError("policy must be a DynamicReferencePoolPolicy")
    validate_dynamic_reference_pool_plans(plans)
    validate_dynamic_reference_pool_members(members)
    validate_family_geo_candidate_sets(candidate_sets)
    if set(plans["plan_id"].to_list()) != set(members["plan_id"].to_list()):
        raise ValueError("dynamic pool plan/member identity sets differ")
    if plans.height and set(plans["selection_policy_fingerprint"].to_list()) != {
        policy.fingerprint
    }:
        raise ValueError("dynamic pool plans do not use the supplied policy")
    output: list[dict[str, object]] = []
    for plan in plans.iter_rows(named=True):
        candidates = candidate_sets.filter(
            pl.col("candidate_set_id") == plan["candidate_set_id"]
        ).sort("candidate_priority", "candidate_accepted_taxon_key")
        if candidates.is_empty():
            raise ValueError("dynamic pool plan references an unknown candidate set")
        if set(candidates["candidate_set_fingerprint"].to_list()) != {
            plan["candidate_set_fingerprint"]
        }:
            raise ValueError("dynamic pool plan candidate-set fingerprint mismatch")
        plan_members = members.filter(pl.col("plan_id") == plan["plan_id"])
        for candidate in candidates.iter_rows(named=True):
            selected = plan_members.filter(
                pl.col("candidate_accepted_taxon_key")
                == candidate["candidate_accepted_taxon_key"]
            )
            output.append(
                _coverage_row(
                    plan=plan,
                    candidate=candidate,
                    selected=selected,
                    policy=policy,
                )
            )
    frame = (
        pl.DataFrame(
            output,
            schema=dynamic_pool_coverage_schema(),
            orient="row",
            strict=True,
        ).sort(*_SORT)
        if output
        else pl.DataFrame(schema=dynamic_pool_coverage_schema())
    )
    validate_dynamic_pool_coverage(frame)
    return frame


def validate_dynamic_pool_coverage(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("dynamic pool coverage must be a Polars DataFrame")
    if frame.schema != dynamic_pool_coverage_schema():
        raise ValueError("dynamic pool coverage schema mismatch")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("dynamic pool coverage is not canonically sorted")
    if frame.select("plan_id", "candidate_accepted_taxon_key").n_unique() != frame.height:
        raise ValueError("dynamic pool coverage grain is not unique")
    for row in frame.iter_rows(named=True):
        _validate_coverage_row(row)


def summarize_dynamic_pool_coverage(frame: pl.DataFrame) -> dict[str, object]:
    validate_dynamic_pool_coverage(frame)
    status_counts = {
        status: frame.filter(pl.col("coverage_status") == status).height
        for status in sorted(DYNAMIC_POOL_COVERAGE_STATUSES)
    }
    report: dict[str, object] = {
        "schema_version": DYNAMIC_POOL_COVERAGE_REPORT_SCHEMA_VERSION,
        "plan_count": frame["plan_id"].n_unique(),
        "candidate_count": frame.height,
        "coverage_status_counts": status_counts,
        "complete_candidate_count": status_counts["complete"],
        "candidate_with_shortfall_count": frame.filter(
            pl.col("coverage_status") != "complete"
        ).height,
        "zero_reference_candidate_count": frame.filter(
            pl.col("total_effective") == 0
        ).height,
        "no_geo_candidate_count": frame.filter(pl.col("no_geo")).height,
        "fallback_candidate_count": frame.filter(
            pl.col("fallback_reasons").list.len() > 0
        ).height,
        "global_minimum_shortfall": int(frame["global_minimum_shortfall"].sum()),
        "local_minimum_shortfall": int(frame["local_minimum_shortfall"].sum()),
        "independent_observation_shortfall": int(
            frame["independent_observation_shortfall"].sum()
        ),
        "production_release_authorized": False,
        "coverage_artifact_fingerprint": canonical_semantic_fingerprint(
            frame.to_dicts()
        ),
    }
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    return report


def dynamic_pool_coverage_markdown(
    frame: pl.DataFrame,
    report: Mapping[str, object] | None = None,
) -> str:
    validate_dynamic_pool_coverage(frame)
    summary = dict(report or summarize_dynamic_pool_coverage(frame))
    lines = [
        "# Dynamic Reference-Pool Coverage",
        "",
        f"- Plans: {summary['plan_count']}",
        f"- Candidate rows: {summary['candidate_count']}",
        f"- Complete candidates: {summary['complete_candidate_count']}",
        f"- Candidates with shortfalls: {summary['candidate_with_shortfall_count']}",
        f"- Zero-reference candidates: {summary['zero_reference_candidate_count']}",
        f"- No-geo candidates: {summary['no_geo_candidate_count']}",
        "- Production release authorized: `false`",
        "",
        "| Candidate | Status | Global effective/min | Local status | Local effective/min | Independent effective/min | Fallback and shortfall reasons |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for row in frame.iter_rows(named=True):
        reasons = sorted(set([*row["fallback_reasons"], *row["shortfall_reasons"]]))
        lines.append(
            f"| {row['candidate_scientific_name']} | {row['coverage_status']} | "
            f"{row['global_effective']}/{row['global_minimum']} | "
            f"{row['local_pool_status']} | "
            f"{row['local_effective']}/{row['local_minimum']} | "
            f"{row['independent_observation_count']}/"
            f"{row['minimum_independent_observation_count']} | "
            f"{'; '.join(reasons) if reasons else 'none'} |"
        )
    lines.extend(
        [
            "",
            "Missing geography is an evidence-availability state, not evidence that a taxon is absent.",
            "",
        ]
    )
    return "\n".join(lines)


def write_dynamic_pool_coverage_report(
    frame: pl.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_dynamic_pool_coverage(frame)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = summarize_dynamic_pool_coverage(frame)
    paths = {
        "coverage": destination / DYNAMIC_POOL_COVERAGE_FILE,
        "report": destination / DYNAMIC_POOL_COVERAGE_REPORT_FILE,
        "summary": destination / DYNAMIC_POOL_COVERAGE_SUMMARY_FILE,
    }
    write_parquet(frame, paths["coverage"])
    paths["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["summary"].write_text(
        dynamic_pool_coverage_markdown(frame, report), encoding="utf-8"
    )
    return paths


def _coverage_row(
    *,
    plan: Mapping[str, object],
    candidate: Mapping[str, object],
    selected: pl.DataFrame,
    policy: DynamicReferencePoolPolicy,
) -> dict[str, object]:
    global_count = selected.filter(pl.col("pool_scope") == "global").height
    local_count = selected.filter(pl.col("pool_scope") == "local").height
    safety_count = selected.filter(pl.col("pool_scope") == "safety_expansion").height
    global_minimum = policy.minimum_global_per_candidate
    local_minimum = policy.minimum_local_per_candidate
    global_requested = policy.maximum_global_per_candidate
    local_requested = policy.maximum_local_per_candidate
    safety_requested = (
        policy.maximum_safety_per_candidate
        if plan["scoring_stage"] != "initial"
        and bool(candidate["safety_union_membership"])
        else 0
    )
    independent_count = selected["independent_observation_group"].n_unique()
    fallback_reasons = sorted(
        set(
            str(value)
            for value in selected.filter(pl.col("fallback_level") > 0)[
                "inclusion_reason"
            ]
        )
    )
    local_reason = plan["local_pool_unavailable_reason"]
    if local_reason is not None:
        fallback_reasons.append(f"local_unavailable:{local_reason}")
    fallback_reasons = sorted(set(fallback_reasons))
    global_shortfall = max(0, global_minimum - global_count)
    local_shortfall = max(0, local_minimum - local_count)
    independent_shortfall = max(
        0,
        policy.minimum_independent_observation_groups_per_candidate
        - independent_count,
    )
    shortfall_reasons: list[str] = []
    if global_shortfall:
        shortfall_reasons.append("global_minimum_unmet")
    if local_shortfall:
        shortfall_reasons.append("local_minimum_unmet")
    if independent_shortfall:
        shortfall_reasons.append("independent_observation_minimum_unmet")
    if safety_count < safety_requested:
        shortfall_reasons.append("safety_requested_count_unmet")
    total = selected.height
    status = _coverage_status(
        total=total,
        global_shortfall=global_shortfall,
        local_shortfall=local_shortfall,
        independent_shortfall=independent_shortfall,
        safety_shortfall=max(0, safety_requested - safety_count),
    )
    no_geo = str(plan["query_coordinate_quality"]) in _NO_GEO_QUALITIES
    base: dict[str, object] = {
        "schema_version": DYNAMIC_POOL_COVERAGE_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "run_id": plan["run_id"],
        "flickr_photo_id": plan["flickr_photo_id"],
        "organism_unit_id": plan["organism_unit_id"],
        "scoring_stage": plan["scoring_stage"],
        "candidate_set_id": plan["candidate_set_id"],
        "candidate_accepted_taxon_key": candidate["candidate_accepted_taxon_key"],
        "candidate_scientific_name": candidate["candidate_scientific_name"],
        "target_candidate": candidate["target_candidate"],
        "safety_union_membership": candidate["safety_union_membership"],
        "query_coordinate_quality": plan["query_coordinate_quality"],
        "no_geo": no_geo,
        "global_minimum": global_minimum,
        "global_requested": global_requested,
        "global_effective": global_count,
        "global_minimum_shortfall": global_shortfall,
        "global_unfilled_to_requested": max(0, global_requested - global_count),
        "local_pool_status": plan["local_pool_status"],
        "local_pool_unavailable_reason": local_reason,
        "local_minimum": local_minimum,
        "local_requested": local_requested,
        "local_effective": local_count,
        "local_minimum_shortfall": local_shortfall,
        "local_unfilled_to_requested": max(0, local_requested - local_count),
        "safety_requested": safety_requested,
        "safety_effective": safety_count,
        "safety_unfilled_to_requested": max(0, safety_requested - safety_count),
        "total_effective": total,
        "selected_reference_media_count": selected["reference_media_id"].n_unique(),
        "selected_reference_observation_count": selected[
            "reference_observation_id"
        ].n_unique(),
        "independent_observation_count": independent_count,
        "minimum_independent_observation_count": (
            policy.minimum_independent_observation_groups_per_candidate
        ),
        "independent_observation_shortfall": independent_shortfall,
        "independent_observer_count": selected["observer_id_hash"].drop_nulls().n_unique(),
        "source_dataset_count": selected["source_dataset_key"].n_unique(),
        "country_count": selected["reference_country_code"].drop_nulls().n_unique(),
        "maximum_fallback_level": (
            max(selected["fallback_level"].to_list()) if total else 0
        ),
        "fallback_reasons": fallback_reasons,
        "coverage_status": status,
        "shortfall_reasons": sorted(shortfall_reasons),
        "selection_policy_fingerprint": plan["selection_policy_fingerprint"],
        "reference_geography_index_fingerprint": plan[
            "reference_geography_index_fingerprint"
        ],
    }
    base["coverage_fingerprint"] = canonical_semantic_fingerprint(base)
    return base


def _validate_coverage_row(row: Mapping[str, object]) -> None:
    if row["schema_version"] != DYNAMIC_POOL_COVERAGE_SCHEMA_VERSION:
        raise ValueError("unsupported dynamic pool coverage schema")
    if row["coverage_status"] not in DYNAMIC_POOL_COVERAGE_STATUSES:
        raise ValueError("unsupported dynamic pool coverage status")
    for field in (
        "selection_policy_fingerprint",
        "reference_geography_index_fingerprint",
        "coverage_fingerprint",
    ):
        if not _SHA256_PATTERN.fullmatch(str(row[field])):
            raise ValueError(f"invalid dynamic pool coverage {field}")
    if int(row["total_effective"]) != (
        int(row["global_effective"])
        + int(row["local_effective"])
        + int(row["safety_effective"])
    ):
        raise ValueError("dynamic pool coverage total is inconsistent")
    for prefix in ("global", "local"):
        expected_minimum = max(
            0, int(row[f"{prefix}_minimum"]) - int(row[f"{prefix}_effective"])
        )
        if int(row[f"{prefix}_minimum_shortfall"]) != expected_minimum:
            raise ValueError(f"dynamic pool {prefix} minimum shortfall is inconsistent")
        expected_requested = max(
            0, int(row[f"{prefix}_requested"]) - int(row[f"{prefix}_effective"])
        )
        if int(row[f"{prefix}_unfilled_to_requested"]) != expected_requested:
            raise ValueError(f"dynamic pool {prefix} requested shortfall is inconsistent")
    if bool(row["no_geo"]) and row["local_pool_status"] != "unavailable":
        raise ValueError("no-geo coverage must have an unavailable local pool")
    for field in ("fallback_reasons", "shortfall_reasons"):
        values = row[field]
        if values != sorted(set(values)):
            raise ValueError(f"dynamic pool coverage {field} is not canonical")
    payload = dict(row)
    fingerprint = payload.pop("coverage_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("dynamic pool coverage fingerprint mismatch")


def _coverage_status(
    *,
    total: int,
    global_shortfall: int,
    local_shortfall: int,
    independent_shortfall: int,
    safety_shortfall: int,
) -> str:
    if total == 0:
        return "no_reference_support"
    if global_shortfall:
        return "insufficient_global"
    if local_shortfall or independent_shortfall or safety_shortfall:
        return "usable_with_shortfalls"
    return "complete"


__all__ = [
    "DYNAMIC_POOL_COVERAGE_FILE",
    "DYNAMIC_POOL_COVERAGE_REPORT_FILE",
    "DYNAMIC_POOL_COVERAGE_REPORT_SCHEMA_VERSION",
    "DYNAMIC_POOL_COVERAGE_SCHEMA_VERSION",
    "DYNAMIC_POOL_COVERAGE_STATUSES",
    "DYNAMIC_POOL_COVERAGE_SUMMARY_FILE",
    "build_dynamic_pool_coverage",
    "dynamic_pool_coverage_markdown",
    "dynamic_pool_coverage_schema",
    "summarize_dynamic_pool_coverage",
    "validate_dynamic_pool_coverage",
    "write_dynamic_pool_coverage_report",
]
