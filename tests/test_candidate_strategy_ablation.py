"""Tests for target-preserving candidate strategy schedules."""

from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.family_geo_candidates import build_family_geo_candidate_sets
from biominer.candidates.strategy_ablation import (
    CANDIDATE_STRATEGY_PLANS_FILE,
    GEOGRAPHY_FIRST_STRATEGY,
    build_candidate_strategy_plans,
    candidate_strategy_plan_schema,
    validate_candidate_strategy_plans,
    write_candidate_strategy_plans,
)


TARGET = "gbif:target"


def test_geography_first_schedules_regions_before_safe_expansion() -> None:
    source = build_family_geo_candidate_sets(_rows())

    plan = build_candidate_strategy_plans(
        source,
        strategy=GEOGRAPHY_FIRST_STRATEGY,
    )

    assert plan.schema == candidate_strategy_plan_schema()
    assert plan.height == source.height == 5
    assert plan["candidate_accepted_taxon_key"].to_list() == [
        TARGET,
        "gbif:geographic",
        "gbif:visual",
        "gbif:family",
        "gbif:remainder",
    ]
    assert plan["strategy_stage"].to_list() == [
        "geographic_union",
        "geographic_union",
        "required_safety_union",
        "family_expansion",
        "complete_union_remainder",
    ]
    assert plan["strategy_priority"].to_list() == list(range(5))
    assert plan["strategy_stage_rank"].to_list() == [0, 1, 0, 0, 0]
    assert plan["source_candidate_priority"].to_list() == [0, 1, 2, 3, 4]
    assert plan["target_candidate"].sum() == 1
    assert all(plan["target_preserved"].to_list())
    assert all(plan["complete_union_preserved"].to_list())
    assert not any(plan["family_changed_membership"].to_list())
    target = plan.filter(pl.col("target_candidate")).row(0, named=True)
    assert target["inclusion_axes"] == [
        "family",
        "geography",
        "query",
        "safety",
        "target",
    ]


def test_geography_first_no_geo_keeps_target_and_complete_union() -> None:
    rows = _rows()
    for row in rows:
        row.update(
            geographic_evidence_status="unavailable",
            geographic_evidence_reason="no_geo_global_fallback",
            geographic_scopes=[],
            geographic_evidence_score=None,
            occurrence_support=0,
        )
    source = build_family_geo_candidate_sets(rows)

    plan = build_candidate_strategy_plans(
        source,
        strategy=GEOGRAPHY_FIRST_STRATEGY,
    )

    assert "geographic_union" not in set(plan["strategy_stage"])
    target = plan.filter(pl.col("target_candidate")).row(0, named=True)
    assert target["strategy_stage"] == "required_safety_union"
    assert plan.height == source.height
    assert set(plan["candidate_accepted_taxon_key"]) == set(
        source["candidate_accepted_taxon_key"]
    )


def test_geography_first_is_deterministic_and_round_trips(tmp_path) -> None:
    forward = build_family_geo_candidate_sets(_rows())
    reverse = build_family_geo_candidate_sets(list(reversed(_rows())))

    first = build_candidate_strategy_plans(
        forward,
        strategy=GEOGRAPHY_FIRST_STRATEGY,
    )
    second = build_candidate_strategy_plans(
        reverse,
        strategy=GEOGRAPHY_FIRST_STRATEGY,
    )

    assert first.equals(second)
    assert first["strategy_plan_id"].n_unique() == 1
    assert first["strategy_plan_fingerprint"].n_unique() == 1
    path = write_candidate_strategy_plans(first, forward, tmp_path)
    assert path.name == CANDIDATE_STRATEGY_PLANS_FILE
    persisted = pl.read_parquet(path)
    validate_candidate_strategy_plans(persisted, forward)
    assert persisted.equals(first)


def test_strategy_validation_rejects_pruning_and_tampering() -> None:
    source = build_family_geo_candidate_sets(_rows())
    plan = build_candidate_strategy_plans(
        source,
        strategy=GEOGRAPHY_FIRST_STRATEGY,
    )

    with pytest.raises(ValueError, match="unsupported candidate strategy"):
        build_candidate_strategy_plans(source, strategy="hard_family_pruning")
    with pytest.raises(ValueError, match="do not match source evidence"):
        validate_candidate_strategy_plans(
            plan.with_columns(
                pl.when(pl.col("target_candidate"))
                .then(pl.lit(False))
                .otherwise(pl.col("complete_union_preserved"))
                .alias("complete_union_preserved")
            ),
            source,
        )


def _rows() -> list[dict[str, object]]:
    return [
        _row(
            key=TARGET,
            name="Papilio target",
            priority=0,
            target=True,
            geography=True,
            family=True,
            query=True,
            safety_reasons=["target", "query_associated"],
            geographic_score=0.7,
            occurrence_support=3,
        ),
        _row(
            key="gbif:geographic",
            name="Papilio geographic",
            priority=1,
            geography=True,
            family=True,
            geographic_score=0.9,
            occurrence_support=10,
        ),
        _row(
            key="gbif:visual",
            name="Papilio visual",
            priority=2,
            family=True,
            visual=True,
            safety_reasons=["visual_neighbour"],
        ),
        _row(
            key="gbif:family",
            name="Papilio family",
            priority=3,
            family=True,
        ),
        _row(
            key="gbif:remainder",
            name="Pieris remainder",
            priority=4,
        ),
    ]


def _row(
    *,
    key: str,
    name: str,
    priority: int,
    target: bool = False,
    geography: bool = False,
    family: bool = False,
    query: bool = False,
    visual: bool = False,
    safety_reasons: list[str] | None = None,
    geographic_score: float | None = None,
    occurrence_support: int = 0,
) -> dict[str, object]:
    safety = safety_reasons or []
    return {
        "run_id": "run-strategy",
        "flickr_query_id": "query-target",
        "flickr_photo_id": "photo-1",
        "organism_unit_id": "organism-1",
        "scoring_stage": "initial",
        "registry_version": "registry-v1",
        "target_accepted_taxon_key": TARGET,
        "target_scientific_name": "Papilio target",
        "query_geo_cluster_id": "geo-au-qld",
        "query_coordinate_quality": "local",
        "candidate_accepted_taxon_key": key,
        "candidate_scientific_name": name,
        "family_key": "family-papilionidae" if "Papilio" in name else "family-pieridae",
        "family_name": "Papilionidae" if "Papilio" in name else "Pieridae",
        "genus_key": "genus-papilio" if "Papilio" in name else "genus-pieris",
        "genus_name": "Papilio" if "Papilio" in name else "Pieris",
        "candidate_priority": priority,
        "candidate_reasons": safety or (["geographic"] if geography else ["complete_union"]),
        "family_evidence_status": "available" if family else "unavailable",
        "family_evidence_reason": None if family else "outside_family_priority",
        "family_evidence_rank": priority + 1 if family else None,
        "family_evidence_raw_score": 0.9 - priority / 10 if family else None,
        "family_priority_match": family if family else None,
        "family_changed_membership": False,
        "geographic_evidence_status": "available" if geography else "unavailable",
        "geographic_evidence_reason": None if geography else "no_local_support",
        "geographic_scopes": ["exact_local_cell"] if geography else [],
        "geographic_evidence_score": geographic_score if geography else None,
        "occurrence_support": occurrence_support if geography else 0,
        "query_evidence_status": "available" if query else "not_applicable",
        "query_evidence_reason": None if query else "not_query_associated",
        "query_evidence_ids": ["query-evidence-1"] if query else [],
        "query_associated": query,
        "visual_neighbour_evidence_status": "available" if visual else "not_applicable",
        "visual_neighbour_evidence_reason": None if visual else "not_visual_neighbour",
        "visual_neighbour_graph_fingerprint": "sha256:" + "a" * 64 if visual else None,
        "visual_neighbour_rank": 1 if visual else None,
        "visual_neighbour_raw_similarity": 0.75 if visual else None,
        "visual_neighbour": visual,
        "safety_union_membership": bool(safety),
        "safety_union_reasons": safety,
        "target_candidate": target,
        "target_preserved": True,
        "included_in_complete_union": True,
        "source_versions": ["registry:v1", "regional-candidate:v1"],
    }
