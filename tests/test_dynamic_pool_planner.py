"""Tests for deterministic dynamic reference-observation planning."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from biominer.bioclip.dynamic_pool_contracts import (
    validate_dynamic_reference_pool_artifacts,
)
from biominer.bioclip.dynamic_pool_planner import plan_dynamic_reference_pools
from biominer.bioclip.dynamic_pool_policy import default_dynamic_reference_pool_policy
from biominer.bioclip.family_geo_candidates import build_family_geo_candidate_sets
from biominer.bioclip.global_reference_anchors import select_global_reference_anchors
from biominer.bioclip.reference_geography_index import (
    build_reference_geography_index,
    reference_geography_index_artifact_fingerprint,
)


TARGET = "gbif:1938069"


def test_planner_selects_observations_into_global_and_exact_local_pools() -> None:
    candidate_sets = _candidate_sets()
    index = _reference_index()
    anchors = select_global_reference_anchors(index)
    policy = _small_policy()

    plans, members, summaries = plan_dynamic_reference_pools(
        [_request(candidate_sets, index)],
        candidate_sets,
        index,
        anchors,
        policy=policy,
    )

    validate_dynamic_reference_pool_artifacts(plans, members, summaries)
    assert plans.height == 1
    assert members.height == 3
    assert members["reference_observation_id"].n_unique() == members.height
    assert members.select("plan_id", "reference_observation_id").n_unique() == 3
    assert members["reference_embedding_fingerprint"].n_unique() == 3
    assert set(members["pool_scope"]) == {"global", "local"}
    global_members = members.filter(pl.col("pool_scope") == "global")
    local_members = members.filter(pl.col("pool_scope") == "local")
    assert global_members.height == 2
    assert local_members.height == 1
    assert global_members["selection_rank"].to_list() == [1, 2]
    assert local_members["selection_rank"].to_list() == [1]
    assert set(global_members["inclusion_reason"]) == {"global_reference_anchor"}
    assert local_members["inclusion_reason"].to_list() == [
        "exact_workload_cluster"
    ]
    assert local_members["geographic_distance_status"].to_list() == ["unavailable"]
    assert local_members["geographic_distance_reason"].to_list() == [
        "query_distance_not_materialized"
    ]
    plan = plans.row(0, named=True)
    assert plan["local_pool_status"] == "available"
    assert plan["local_pool_unavailable_reason"] is None
    assert plan["selection_policy_fingerprint"] == policy.fingerprint
    assert plan["configured_global_per_candidate"] == 2
    assert plan["configured_local_per_candidate"] == 1
    assert summaries["effective_reference_count"].sum() == 3


def test_planner_is_input_order_independent() -> None:
    candidate_sets = _candidate_sets()
    rows = _reference_rows()
    forward_index = build_reference_geography_index(rows)
    reverse_index = build_reference_geography_index(list(reversed(rows)))
    forward_anchors = select_global_reference_anchors(forward_index)
    reverse_anchors = select_global_reference_anchors(reverse_index)
    policy = _small_policy()

    forward = plan_dynamic_reference_pools(
        [_request(candidate_sets, forward_index)],
        candidate_sets,
        forward_index,
        forward_anchors,
        policy=policy,
    )
    reverse = plan_dynamic_reference_pools(
        [_request(candidate_sets, reverse_index)],
        candidate_sets,
        reverse_index,
        reverse_anchors,
        policy=policy,
    )

    assert all(left.equals(right) for left, right in zip(forward, reverse, strict=True))


def test_planner_no_geo_request_is_explicitly_global_only() -> None:
    candidate_sets = _candidate_sets(local=False)
    index = _reference_index()
    anchors = select_global_reference_anchors(index)

    plans, members, summaries = plan_dynamic_reference_pools(
        [_request(candidate_sets, index, local=False)],
        candidate_sets,
        index,
        anchors,
        policy=_small_policy(),
    )

    plan = plans.row(0, named=True)
    assert plan["local_pool_status"] == "unavailable"
    assert plan["local_pool_unavailable_reason"] == "no_geo_global_fallback"
    assert plan["local_pool_ids"] == []
    assert set(members["pool_scope"]) == {"global"}
    assert summaries["distance_available_count"].sum() == 0


def test_planner_rejects_candidate_reference_name_conflict() -> None:
    candidate_sets = _candidate_sets()
    rows = _reference_rows()
    rows[0]["scientific_name"] = "Papilio conflicting"
    index = build_reference_geography_index(rows)

    with pytest.raises(ValueError, match="scientific names conflict"):
        plan_dynamic_reference_pools(
            [_request(candidate_sets, index)],
            candidate_sets,
            index,
            select_global_reference_anchors(index),
            policy=_small_policy(),
        )


def _small_policy():
    return replace(
        default_dynamic_reference_pool_policy(),
        minimum_global_per_candidate=1,
        maximum_global_per_candidate=2,
        minimum_local_per_candidate=1,
        maximum_local_per_candidate=1,
        maximum_safety_per_candidate=1,
        minimum_independent_observation_groups_per_candidate=2,
        minimum_global_countries_per_candidate=1,
    )


def _request(
    candidate_sets: pl.DataFrame,
    index: pl.DataFrame,
    *,
    local: bool = True,
) -> dict[str, object]:
    first = candidate_sets.row(0, named=True)
    return {
        "run_id": "run-dynamic-planner",
        "flickr_query_id": first["flickr_query_id"],
        "flickr_photo_id": first["flickr_photo_id"],
        "organism_unit_id": first["organism_unit_id"],
        "visual_input_id": _sha("a"),
        "query_embedding_fingerprint": _sha("b"),
        "scoring_stage": "initial",
        "query_route": "adult_field",
        "registry_version": first["registry_version"],
        "reference_bank_version": "reference-bank-v3",
        "reference_geography_index_fingerprint": (
            reference_geography_index_artifact_fingerprint(index)
        ),
        "candidate_set_id": first["candidate_set_id"],
        "candidate_set_fingerprint": first["candidate_set_fingerprint"],
        "query_geo_cluster_id": "cluster-query" if local else None,
        "query_coordinate_quality": "local" if local else "no_geo",
        "model_id": "bioclip-2.5",
        "model_revision": "revision-1",
        "model_weights_sha256": _sha("c"),
        "model_fingerprint": _sha("d"),
        "preprocessing_fingerprint": _sha("e"),
    }


def _candidate_sets(*, local: bool = True) -> pl.DataFrame:
    return build_family_geo_candidate_sets(
        [
            {
                "run_id": "run-dynamic-planner",
                "flickr_query_id": "query-target",
                "flickr_photo_id": "photo-1",
                "organism_unit_id": "organism-1",
                "scoring_stage": "initial",
                "registry_version": "butterflies-v2-20260718",
                "target_accepted_taxon_key": TARGET,
                "target_scientific_name": "Papilio demoleus",
                "query_geo_cluster_id": "cluster-query" if local else None,
                "query_coordinate_quality": "local" if local else "no_geo",
                "candidate_accepted_taxon_key": TARGET,
                "candidate_scientific_name": "Papilio demoleus",
                "family_key": "gbif:9417",
                "family_name": "Papilionidae",
                "genus_key": "gbif:1920494",
                "genus_name": "Papilio",
                "candidate_priority": 0,
                "candidate_reasons": ["target", "query_associated"],
                "family_evidence_status": "available",
                "family_evidence_reason": None,
                "family_evidence_rank": 1,
                "family_evidence_raw_score": 0.9,
                "family_priority_match": True,
                "family_changed_membership": False,
                "geographic_evidence_status": "available" if local else "unavailable",
                "geographic_evidence_reason": None if local else "no_geo",
                "geographic_scopes": ["exact_local_cell"] if local else [],
                "geographic_evidence_score": 0.8 if local else None,
                "occurrence_support": 3 if local else 0,
                "query_evidence_status": "available",
                "query_evidence_reason": None,
                "query_evidence_ids": ["query-evidence-1"],
                "query_associated": True,
                "visual_neighbour_evidence_status": "not_applicable",
                "visual_neighbour_evidence_reason": "not_visual_neighbour",
                "visual_neighbour_graph_fingerprint": None,
                "visual_neighbour_rank": None,
                "visual_neighbour_raw_similarity": None,
                "visual_neighbour": False,
                "safety_union_membership": True,
                "safety_union_reasons": ["target", "query_associated"],
                "target_candidate": True,
                "target_preserved": True,
                "included_in_complete_union": True,
                "source_versions": ["registry:v1", "candidate:v1"],
            }
        ]
    )


def _reference_index() -> pl.DataFrame:
    return build_reference_geography_index(_reference_rows())


def _reference_rows() -> list[dict[str, object]]:
    rows = [_reference_row(str(index)) for index in range(1, 5)]
    rows[3].update(
        reference_observation_id=rows[0]["reference_observation_id"],
        duplicate_group_id=rows[0]["duplicate_group_id"],
        visual_input_kind="focused_full_frame",
    )
    return rows


def _reference_row(suffix: str) -> dict[str, object]:
    return {
        "registry_version": "butterflies-v2-20260718",
        "reference_bank_version": "reference-bank-v3",
        "reference_media_id": f"reference-media:{suffix * 64}",
        "reference_observation_id": f"reference-observation:{suffix * 64}",
        "source": "gbif",
        "source_dataset_key": f"dataset-{suffix}",
        "accepted_taxon_key": TARGET,
        "scientific_name": "Papilio demoleus",
        "family_key": "gbif:9417",
        "family_name": "Papilionidae",
        "genus_key": "gbif:1920494",
        "genus_name": "Papilio",
        "route": "adult_field",
        "life_stage": "adult",
        "visual_domain": "live_field",
        "visual_input_kind": "raw_full_image",
        "country_code": "AU",
        "admin1": "Queensland",
        "bioregion": "Wet Tropics",
        "geo_cluster_id": "cluster-query",
        "coarse_cell_id": f"coarse-{suffix}",
        "regional_cell_id": f"regional-{suffix}",
        "local_cell_id": f"local-{suffix}",
        "latitude": -16.9 - int(suffix) / 100,
        "longitude": 145.7 + int(suffix) / 100,
        "coordinate_uncertainty_m": 25.0,
        "coordinate_quality": "local",
        "global_anchor_eligible": True,
        "local_anchor_eligible": True,
        "duplicate_group_id": f"reference-duplicate-group:{suffix * 32}",
        "observer_id_hash": _sha(suffix),
        "observation_date": date(2026, 1, int(suffix)),
        "admission_mode": "adaptive_gbif_fast_start",
        "admission_policy_fingerprint": _sha("f"),
        "reference_quality_flags": ["provisional"],
        "embedding_fingerprint": _sha(suffix),
    }


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"
