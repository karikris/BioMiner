"""Deterministic family/geography schedule ablation for the bounded pilot."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.family_geo_candidates import (
    build_family_geo_candidate_sets,
    validate_family_geo_candidate_sets,
)
from biominer.candidates.strategy_ablation import (
    build_candidate_strategy_plans,
    validate_candidate_strategy_plans,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_pilot_plan import (
    PILOT_CANDIDATE_STRATEGIES,
    PILOT_CASE_EVIDENCE_BASIS,
    validate_dynamic_pool_pilot_plan,
)


DYNAMIC_POOL_PILOT_ABLATION_VERSION = "dynamic-pool-pilot-candidate-ablation-v1.0.0"
DYNAMIC_POOL_PILOT_ABLATION_REPORT_VERSION = (
    "dynamic-pool-pilot-candidate-ablation-report-v1.0.0"
)
DYNAMIC_POOL_PILOT_ABLATION_REPORT_FILE = "candidate_pooling_ablation.json"
PILOT_ABLATION_CUTOFFS = (1, 3, 5)

_SORT = ("case_id", "strategy_name")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def pilot_candidate_ablation_schema() -> dict[str, pl.DataType]:
    """Return the canonical structural-ablation result schema."""

    return {
        "schema_version": pl.String,
        "pilot_id": pl.String,
        "plan_fingerprint": pl.String,
        "case_id": pl.String,
        "fixture_media_id": pl.String,
        "target_accepted_taxon_key": pl.String,
        "country_code": pl.String,
        "region_id": pl.String,
        "geographic_evidence_status": pl.String,
        "no_geo": pl.Boolean,
        "candidate_set_id": pl.String,
        "candidate_set_fingerprint": pl.String,
        "strategy_name": pl.String,
        "strategy_version": pl.String,
        "strategy_plan_id": pl.String,
        "strategy_plan_fingerprint": pl.String,
        "candidate_set_size": pl.UInt32,
        "target_rank": pl.UInt32,
        "target_candidate_recall_at_1": pl.Float64,
        "target_candidate_recall_at_3": pl.Float64,
        "target_candidate_recall_at_5": pl.Float64,
        "target_preserved": pl.Boolean,
        "complete_union_preserved": pl.Boolean,
        "membership_fingerprint": pl.String,
        "ordered_candidate_keys": pl.List(pl.String),
        "ordered_strategy_stages": pl.List(pl.String),
        "order_differs_from_source": pl.Boolean,
        "expected_label_basis": pl.String,
        "classification_accuracy_status": pl.String,
        "timing_status": pl.String,
        "production_default_eligible": pl.Boolean,
        "result_fingerprint": pl.String,
    }


def build_pilot_family_geo_candidate_sets(
    plan: Mapping[str, object],
) -> pl.DataFrame:
    """Build all frozen cases through the production complete-union contract."""

    validate_dynamic_pool_pilot_plan(plan)
    catalog = {
        str(taxon["accepted_taxon_key"]): taxon for taxon in plan["taxon_catalog"]
    }
    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(plan["cases"]):
        rows.extend(
            _candidate_rows(
                plan=plan,
                case=case,
                case_index=case_index,
                catalog=catalog,
            )
        )
    frame = build_family_geo_candidate_sets(rows)
    validate_family_geo_candidate_sets(frame)
    return frame


def build_dynamic_pool_pilot_candidate_ablation(
    plan: Mapping[str, object],
) -> pl.DataFrame:
    """Execute every frozen candidate schedule and retain structural evidence."""

    validate_dynamic_pool_pilot_plan(plan)
    candidate_sets = build_pilot_family_geo_candidate_sets(plan)
    cases = {str(case["fixture_media_id"]): case for case in plan["cases"]}
    source_by_set = {
        str(set_id): group.sort("candidate_priority", "candidate_accepted_taxon_key")
        for (set_id,), group in candidate_sets.group_by(
            "candidate_set_id", maintain_order=True
        )
    }
    rows: list[dict[str, object]] = []
    for strategy in PILOT_CANDIDATE_STRATEGIES:
        strategy_plan = build_candidate_strategy_plans(
            candidate_sets, strategy=strategy
        )
        validate_candidate_strategy_plans(strategy_plan, candidate_sets)
        for (strategy_plan_id,), group in strategy_plan.group_by(
            "strategy_plan_id", maintain_order=True
        ):
            ordered = group.sort("strategy_priority")
            first = ordered.row(0, named=True)
            case = cases[str(first["flickr_photo_id"])]
            source = source_by_set[str(first["source_candidate_set_id"])]
            rows.append(
                _ablation_row(
                    plan=plan,
                    case=case,
                    source=source,
                    strategy_plan_id=str(strategy_plan_id),
                    strategy_plan=ordered,
                )
            )
    frame = pl.DataFrame(
        rows,
        schema=pilot_candidate_ablation_schema(),
        orient="row",
        strict=True,
    ).sort(*_SORT)
    validate_dynamic_pool_pilot_candidate_ablation(frame, plan)
    return frame


def validate_dynamic_pool_pilot_candidate_ablation(
    frame: pl.DataFrame,
    plan: Mapping[str, object],
) -> None:
    """Validate complete coverage, identity, and fail-closed evidence maturity."""

    validate_dynamic_pool_pilot_plan(plan)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("pilot candidate ablation must be a Polars DataFrame")
    if frame.schema != pilot_candidate_ablation_schema():
        raise ValueError("pilot candidate ablation schema mismatch")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("pilot candidate ablation is not canonically sorted")
    expected_rows = len(plan["cases"]) * len(PILOT_CANDIDATE_STRATEGIES)
    if frame.height != expected_rows:
        raise ValueError("pilot candidate ablation coverage is incomplete")
    if frame.select("case_id", "strategy_name").n_unique() != frame.height:
        raise ValueError("pilot candidate ablation grain is not unique")
    if set(frame["case_id"]) != {str(case["case_id"]) for case in plan["cases"]}:
        raise ValueError("pilot candidate ablation cases differ from the plan")
    if set(frame["strategy_name"]) != set(PILOT_CANDIDATE_STRATEGIES):
        raise ValueError("pilot candidate ablation strategies differ from the plan")

    for row in frame.to_dicts():
        if row["schema_version"] != DYNAMIC_POOL_PILOT_ABLATION_VERSION:
            raise ValueError("unsupported pilot candidate ablation version")
        if row["pilot_id"] != plan["pilot_id"]:
            raise ValueError("pilot candidate ablation pilot identity differs")
        if row["plan_fingerprint"] != plan["plan_fingerprint"]:
            raise ValueError("pilot candidate ablation plan identity differs")
        for field in (
            "plan_fingerprint",
            "candidate_set_fingerprint",
            "strategy_plan_fingerprint",
            "membership_fingerprint",
            "result_fingerprint",
        ):
            if not _is_sha256(row[field]):
                raise ValueError(f"pilot candidate ablation {field} is invalid")
        keys = list(row["ordered_candidate_keys"])
        stages = list(row["ordered_strategy_stages"])
        if len(keys) != row["candidate_set_size"] or len(stages) != len(keys):
            raise ValueError("pilot candidate order and stages are inconsistent")
        if len(set(keys)) != len(keys):
            raise ValueError("pilot candidate order contains duplicate taxa")
        rank = int(row["target_rank"])
        if not 1 <= rank <= len(keys):
            raise ValueError("pilot target rank is outside the candidate union")
        if keys[rank - 1] != row["target_accepted_taxon_key"]:
            raise ValueError("pilot target rank does not locate the target")
        for cutoff in PILOT_ABLATION_CUTOFFS:
            expected_recall = 1.0 if rank <= cutoff else 0.0
            if row[f"target_candidate_recall_at_{cutoff}"] != expected_recall:
                raise ValueError("pilot target candidate recall is inconsistent")
        expected_membership = canonical_semantic_fingerprint(sorted(keys))
        if row["membership_fingerprint"] != expected_membership:
            raise ValueError("pilot candidate membership fingerprint differs")
        if row["target_preserved"] is not True:
            raise ValueError("pilot candidate ablation lost its target")
        if row["complete_union_preserved"] is not True:
            raise ValueError("pilot candidate ablation pruned its complete union")
        if row["expected_label_basis"] != PILOT_CASE_EVIDENCE_BASIS:
            raise ValueError("pilot candidate ablation promoted fixture labels")
        if row["classification_accuracy_status"] != "unavailable_fixture_only":
            raise ValueError("pilot candidate ablation fabricated accuracy")
        if row["timing_status"] != "not_instrumented":
            raise ValueError("pilot candidate ablation fabricated timing")
        if row["production_default_eligible"] is not False:
            raise ValueError("pilot candidate ablation authorized a default")
        payload = {
            key: value for key, value in row.items() if key != "result_fingerprint"
        }
        if row["result_fingerprint"] != canonical_semantic_fingerprint(payload):
            raise ValueError("pilot candidate ablation result fingerprint differs")

    for (_case_id,), group in frame.group_by("case_id"):
        if group["membership_fingerprint"].n_unique() != 1:
            raise ValueError("pilot strategies changed candidate membership")
        if group["candidate_set_id"].n_unique() != 1:
            raise ValueError("pilot strategies do not share one candidate set")


def build_dynamic_pool_pilot_candidate_ablation_report(
    plan: Mapping[str, object], frame: pl.DataFrame
) -> dict[str, object]:
    """Summarize schedule behavior without making an accuracy claim."""

    validate_dynamic_pool_pilot_candidate_ablation(frame, plan)
    report = _candidate_ablation_report_payload(plan, frame)
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    validate_dynamic_pool_pilot_candidate_ablation_report(report, plan, frame)
    return report


def _candidate_ablation_report_payload(
    plan: Mapping[str, object], frame: pl.DataFrame
) -> dict[str, object]:
    strategy_metrics = [
        _strategy_summary(frame.filter(pl.col("strategy_name") == strategy), strategy)
        for strategy in PILOT_CANDIDATE_STRATEGIES
    ]
    row_fingerprints = frame["result_fingerprint"].to_list()
    report: dict[str, object] = {
        "schema_version": DYNAMIC_POOL_PILOT_ABLATION_REPORT_VERSION,
        "pilot_id": plan["pilot_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "evidence_basis": "current_fixture_backed_execution",
        "historical_real_source_inventory_used_as_current_results": False,
        "case_count": len(plan["cases"]),
        "strategy_count": len(PILOT_CANDIDATE_STRATEGIES),
        "result_row_count": frame.height,
        "candidate_taxon_count_per_case": int(frame["candidate_set_size"].min()),
        "located_case_count": frame.filter(~pl.col("no_geo"))["case_id"].n_unique(),
        "no_geo_case_count": frame.filter(pl.col("no_geo"))["case_id"].n_unique(),
        "australian_case_count": frame.filter(pl.col("country_code") == "AU")[
            "case_id"
        ].n_unique(),
        "target_preserved_result_count": int(frame["target_preserved"].sum()),
        "complete_union_preserved_result_count": int(
            frame["complete_union_preserved"].sum()
        ),
        "strategy_metrics": strategy_metrics,
        "result_set_fingerprint": canonical_semantic_fingerprint(row_fingerprints),
        "classification_accuracy": {
            "status": "unavailable",
            "reason": "fixture_expected_taxa_are_not_source_bound_human_labels",
        },
        "timing": {
            "status": "not_instrumented",
            "reason": "structural_schedule_ablation_only",
        },
        "selection": {
            "status": "insufficient_evidence",
            "selected_candidate_strategy": None,
            "production_default_eligible": False,
            "reason": "fixture_evidence_cannot_select_a_production_default",
        },
        "scientific_claims": {
            "raw_scores_are_probabilities": False,
            "fixture_labels_are_human_reviews": False,
            "missing_geography_is_biological_absence": False,
            "candidate_order_is_classification_accuracy": False,
            "occurrence_release_authorized": False,
        },
    }
    return report


def validate_dynamic_pool_pilot_candidate_ablation_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    frame: pl.DataFrame,
) -> None:
    """Require the report to equal a fresh deterministic summary."""

    if report.get("schema_version") != DYNAMIC_POOL_PILOT_ABLATION_REPORT_VERSION:
        raise ValueError("unsupported pilot candidate ablation report version")
    validate_dynamic_pool_pilot_candidate_ablation(frame, plan)
    expected = _candidate_ablation_report_payload(plan, frame)
    expected["report_fingerprint"] = canonical_semantic_fingerprint(expected)
    if dict(report) != expected:
        raise ValueError("pilot candidate ablation report differs from its inputs")


def write_dynamic_pool_pilot_candidate_ablation_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    """Atomically write one validated pilot candidate-ablation report."""

    validate_dynamic_pool_pilot_candidate_ablation_report(report, plan, frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".json":
        destination /= DYNAMIC_POOL_PILOT_ABLATION_REPORT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _candidate_rows(
    *,
    plan: Mapping[str, object],
    case: Mapping[str, object],
    case_index: int,
    catalog: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    catalog_keys = list(catalog)
    target_key = str(case["accepted_taxon_key"])
    source_order = [target_key, *(key for key in catalog_keys if key != target_key)]
    family_order = (
        catalog_keys[case_index % len(catalog_keys) :]
        + catalog_keys[: case_index % len(catalog_keys)]
    )
    family_rank = {key: index + 1 for index, key in enumerate(family_order)}
    geographic_keys = (
        {
            target_key,
            catalog_keys[(catalog_keys.index(target_key) + 1) % len(catalog_keys)],
        }
        if case["geographic_evidence_status"] == "located_fixture_context"
        else set()
    )
    target_genus = str(catalog[target_key]["genus_key"])
    same_genus = [
        key
        for key in catalog_keys
        if key != target_key and catalog[key]["genus_key"] == target_genus
    ]
    visual_key = (
        same_genus[0]
        if same_genus
        else next(key for key in catalog_keys if key != target_key)
    )
    visual_graph = canonical_semantic_fingerprint(
        {
            "schema_version": "dynamic-pool-pilot-visual-neighbour-fixture-v1.0.0",
            "plan_fingerprint": plan["plan_fingerprint"],
            "case_id": case["case_id"],
            "visual_neighbour_key": visual_key,
        }
    )
    rows: list[dict[str, object]] = []
    for priority, candidate_key in enumerate(source_order):
        candidate = catalog[candidate_key]
        target = candidate_key == target_key
        visual = candidate_key == visual_key
        located = candidate_key in geographic_keys
        rank = family_rank[candidate_key]
        family_priority = rank <= 2
        safety_reasons = sorted(
            reason
            for reason, included in (
                ("target", target),
                ("visual_neighbour", visual),
            )
            if included
        )
        rows.append(
            {
                "run_id": f"pilot-fixture-run:{plan['pilot_id']}",
                "flickr_query_id": f"pilot-query:{case['case_id']}",
                "flickr_photo_id": case["fixture_media_id"],
                "organism_unit_id": f"pilot-organism:{case['case_id']}",
                "scoring_stage": "pilot_candidate_ablation",
                "registry_version": "butterflies-v2-20260712",
                "target_accepted_taxon_key": target_key,
                "target_scientific_name": catalog[target_key]["scientific_name"],
                "query_geo_cluster_id": case["region_id"],
                "query_coordinate_quality": (
                    "fixture_region"
                    if case["geographic_evidence_status"] == "located_fixture_context"
                    else "no_geo"
                ),
                "candidate_accepted_taxon_key": candidate_key,
                "candidate_scientific_name": candidate["scientific_name"],
                "family_key": candidate["family_key"],
                "family_name": candidate["family"],
                "genus_key": candidate["genus_key"],
                "genus_name": candidate["genus"],
                "candidate_priority": priority,
                "candidate_reasons": sorted(
                    [
                        "complete_union",
                        *(["fixture_expected_target"] if target else []),
                        *(["fixture_geographic_context"] if located else []),
                        *(["fixture_visual_neighbour"] if visual else []),
                    ]
                ),
                "family_evidence_status": "available",
                "family_evidence_reason": None,
                "family_evidence_rank": rank,
                "family_evidence_raw_score": 1.0 - rank / 10,
                "family_priority_match": family_priority,
                "family_changed_membership": False,
                "geographic_evidence_status": "available" if located else "unavailable",
                "geographic_evidence_reason": (
                    None
                    if located
                    else (
                        "missing_source_geography_global_only"
                        if case["geographic_evidence_status"]
                        == "missing_source_geography"
                        else "outside_fixture_local_context"
                    )
                ),
                "geographic_scopes": [str(case["region_id"])] if located else [],
                "geographic_evidence_score": (
                    0.9 if target and located else 0.7 if located else None
                ),
                "occurrence_support": 1 if located else 0,
                "query_evidence_status": "available" if target else "not_applicable",
                "query_evidence_reason": None if target else "not_query_associated",
                "query_evidence_ids": (
                    [f"fixture-query-evidence:{case['case_id']}"] if target else []
                ),
                "query_associated": target,
                "visual_neighbour_evidence_status": (
                    "available" if visual else "not_applicable"
                ),
                "visual_neighbour_evidence_reason": (
                    None if visual else "not_fixture_visual_neighbour"
                ),
                "visual_neighbour_graph_fingerprint": visual_graph if visual else None,
                "visual_neighbour_rank": 1 if visual else None,
                "visual_neighbour_raw_similarity": 0.75 if visual else None,
                "visual_neighbour": visual,
                "safety_union_membership": bool(safety_reasons),
                "safety_union_reasons": safety_reasons,
                "target_candidate": target,
                "target_preserved": True,
                "included_in_complete_union": True,
                "source_versions": [
                    "fixture-candidate-evidence-v1",
                    "registry:butterflies-v2-20260712",
                ],
            }
        )
    return rows


def _ablation_row(
    *,
    plan: Mapping[str, object],
    case: Mapping[str, object],
    source: pl.DataFrame,
    strategy_plan_id: str,
    strategy_plan: pl.DataFrame,
) -> dict[str, object]:
    first = strategy_plan.row(0, named=True)
    ordered_keys = [
        str(value) for value in strategy_plan["candidate_accepted_taxon_key"]
    ]
    target_key = str(first["target_accepted_taxon_key"])
    target_rank = ordered_keys.index(target_key) + 1
    source_keys = [str(value) for value in source["candidate_accepted_taxon_key"]]
    membership_fingerprint = canonical_semantic_fingerprint(sorted(ordered_keys))
    base: dict[str, object] = {
        "schema_version": DYNAMIC_POOL_PILOT_ABLATION_VERSION,
        "pilot_id": plan["pilot_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "case_id": case["case_id"],
        "fixture_media_id": case["fixture_media_id"],
        "target_accepted_taxon_key": target_key,
        "country_code": case["country_code"],
        "region_id": case["region_id"],
        "geographic_evidence_status": case["geographic_evidence_status"],
        "no_geo": case["geographic_evidence_status"] == "missing_source_geography",
        "candidate_set_id": first["source_candidate_set_id"],
        "candidate_set_fingerprint": first["source_candidate_set_fingerprint"],
        "strategy_name": first["strategy_name"],
        "strategy_version": first["strategy_version"],
        "strategy_plan_id": strategy_plan_id,
        "strategy_plan_fingerprint": first["strategy_plan_fingerprint"],
        "candidate_set_size": len(ordered_keys),
        "target_rank": target_rank,
        **{
            f"target_candidate_recall_at_{cutoff}": (
                1.0 if target_rank <= cutoff else 0.0
            )
            for cutoff in PILOT_ABLATION_CUTOFFS
        },
        "target_preserved": bool(strategy_plan["target_preserved"].all()),
        "complete_union_preserved": (
            set(ordered_keys) == set(source_keys)
            and bool(strategy_plan["complete_union_preserved"].all())
        ),
        "membership_fingerprint": membership_fingerprint,
        "ordered_candidate_keys": ordered_keys,
        "ordered_strategy_stages": [
            str(value) for value in strategy_plan["strategy_stage"]
        ],
        "order_differs_from_source": ordered_keys != source_keys,
        "expected_label_basis": case["expected_label_basis"],
        "classification_accuracy_status": "unavailable_fixture_only",
        "timing_status": "not_instrumented",
        "production_default_eligible": False,
    }
    return {**base, "result_fingerprint": canonical_semantic_fingerprint(base)}


def _strategy_summary(frame: pl.DataFrame, strategy: str) -> dict[str, object]:
    located = frame.filter(~pl.col("no_geo"))
    no_geo = frame.filter(pl.col("no_geo"))
    return {
        "strategy_name": strategy,
        "case_count": frame.height,
        "mean_target_rank": float(frame["target_rank"].mean()),
        "maximum_target_rank": int(frame["target_rank"].max()),
        **{
            f"target_candidate_recall_at_{cutoff}": float(
                frame[f"target_candidate_recall_at_{cutoff}"].mean()
            )
            for cutoff in PILOT_ABLATION_CUTOFFS
        },
        "located_target_candidate_recall_at_1": float(
            located["target_candidate_recall_at_1"].mean()
        ),
        "no_geo_target_candidate_recall_at_1": float(
            no_geo["target_candidate_recall_at_1"].mean()
        ),
        "target_preserved_count": int(frame["target_preserved"].sum()),
        "complete_union_preserved_count": int(frame["complete_union_preserved"].sum()),
        "order_differs_from_source_count": int(
            frame["order_differs_from_source"].sum()
        ),
        "accuracy_interpretation": "unavailable_fixture_only",
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "DYNAMIC_POOL_PILOT_ABLATION_REPORT_FILE",
    "DYNAMIC_POOL_PILOT_ABLATION_REPORT_VERSION",
    "DYNAMIC_POOL_PILOT_ABLATION_VERSION",
    "PILOT_ABLATION_CUTOFFS",
    "build_dynamic_pool_pilot_candidate_ablation",
    "build_dynamic_pool_pilot_candidate_ablation_report",
    "build_pilot_family_geo_candidate_sets",
    "pilot_candidate_ablation_schema",
    "validate_dynamic_pool_pilot_candidate_ablation",
    "validate_dynamic_pool_pilot_candidate_ablation_report",
    "write_dynamic_pool_pilot_candidate_ablation_report",
]
