"""Tests for the frozen bounded dynamic-pooling pilot plan."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import polars as pl
import pytest

from biominer.bioclip.dynamic_pool_fusion import RAW_FUSION_METHODS
from biominer.evaluation.dynamic_pool_pilot_plan import (
    PILOT_CANDIDATE_STRATEGIES,
    PILOT_POOL_VARIANTS,
    load_dynamic_pool_pilot_plan,
    validate_dynamic_pool_pilot_inputs,
    validate_dynamic_pool_pilot_plan,
)


ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "config/pilot/geography_conditioned_dynamic_pool_pilot_v1.json"
TAXA_PATH = ROOT / "data/registry/butterflies-v2-20260712/taxa.parquet"


def _plan() -> dict[str, object]:
    return load_dynamic_pool_pilot_plan(PLAN_PATH, repository_root=ROOT)


def test_plan_freezes_fixture_execution_separately_from_real_inventory() -> None:
    plan = _plan()
    boundary = plan["evidence_boundary"]

    assert boundary == {
        "current_execution_basis": "fixture_backed",
        "historical_real_source_inventory_present": True,
        "historical_outputs_count_as_current_execution": False,
        "fixture_expected_taxa_are_human_labels": False,
        "current_human_reviewed_label_count": 0,
        "live_network_calls_planned": False,
        "live_model_execution_planned": False,
        "production_default_authorized": False,
        "occurrence_release_authorized": False,
    }
    assert len(plan["durable_inputs"]) == 7
    assert all(
        descriptor["scientific_authority"]
        in {
            "taxonomy_identity_only",
            "discovery_metadata_only",
            "historical_execution_evidence_only",
        }
        for descriptor in plan["durable_inputs"]
    )


def test_plan_covers_target_australian_competitors_regions_and_no_geo() -> None:
    plan = _plan()
    taxa = {row["accepted_taxon_key"]: row for row in plan["taxon_catalog"]}
    cases = plan["cases"]

    assert taxa["gbif:1938069"]["scientific_name"] == "Papilio demoleus"
    assert sum("australian_scope" in row["pilot_roles"] for row in taxa.values()) == 4
    assert (
        sum(
            "stress_same_genus_competitor" in row["pilot_roles"]
            for row in taxa.values()
        )
        == 2
    )
    assert {row["family"] for row in taxa.values()} == {"Papilionidae"}
    assert (
        len({row["accepted_taxon_key"] for row in cases if row["country_code"] == "AU"})
        == 5
    )
    assert (
        len(
            {
                row["region_id"]
                for row in cases
                if row["geographic_evidence_status"] == "located_fixture_context"
            }
        )
        == 6
    )
    no_geo = [
        row
        for row in cases
        if row["geographic_evidence_status"] == "missing_source_geography"
    ]
    assert len(no_geo) == 1
    assert no_geo[0]["country_code"] is None
    assert no_geo[0]["region_id"] is None
    assert all(row["biological_occurrence_claim"] is False for row in cases)
    assert all(row["review_status"] == "not_human_reviewed_fixture" for row in cases)


def test_catalog_taxa_match_the_frozen_registry() -> None:
    plan = _plan()
    expected = pl.DataFrame(plan["taxon_catalog"]).select(
        "accepted_taxon_key",
        "scientific_name",
        "family_key",
        "family",
        "genus_key",
        "genus",
    )
    observed = (
        pl.read_parquet(TAXA_PATH)
        .filter(
            pl.col("accepted_taxon_key").is_in(expected["accepted_taxon_key"].to_list())
        )
        .select(expected.columns)
    )

    assert observed.sort("accepted_taxon_key").equals(
        expected.sort("accepted_taxon_key")
    )


def test_plan_preregisters_all_comparable_variants_and_limits() -> None:
    plan = _plan()
    ablations = plan["ablations"]
    limits = plan["execution_limits"]

    assert tuple(ablations["candidate_strategies"]) == PILOT_CANDIDATE_STRATEGIES
    assert tuple(ablations["pool_variants"]) == PILOT_POOL_VARIANTS
    assert tuple(ablations["fusion_methods"]) == RAW_FUSION_METHODS
    assert ablations["variant_count"] == 24
    assert ablations["comparability"]["same_candidate_union"] is True
    assert ablations["comparability"]["target_pruning_allowed"] is False
    assert ablations["comparability"]["raw_scores_are_probabilities"] is False
    assert limits["fixture_case_count"] == limits["maximum_unique_fixture_media"] == 7
    assert limits["encode_each_unique_media_once"] is True
    assert limits["reuse_reference_embeddings"] is True
    assert limits["reuse_candidate_and_pool_matrices"] is True
    assert limits["live_network_calls_allowed"] is False
    assert limits["source_media_bytes_in_artifacts"] is False


def test_acceptance_policy_blocks_fixture_default_selection_and_release() -> None:
    plan = _plan()
    policy = plan["acceptance_policy"]

    assert policy["eligible_evidence_basis"] == "real_source_bound_human_review"
    assert policy["fixture_evidence_can_select_default"] is False
    assert policy["fixture_forced_decision"] == "insufficient_evidence"
    assert policy["minimum_target_candidate_recall"] == 1.0
    assert policy["minimum_reviewed_precision_lower_bound"] == 0.95
    assert policy["minimum_effective_reviewed_records"] == 86
    assert policy["minimum_subgroup_independent_records"] == 30
    assert policy["no_target_pruning_regressions_required"] is True
    assert policy["unsupported_statistical_claims_allowed"] is False
    assert plan["evidence_boundary"]["production_default_authorized"] is False
    assert plan["evidence_boundary"]["occurrence_release_authorized"] is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("evidence_boundary", "current_execution_basis", "real"),
        ("evidence_boundary", "fixture_expected_taxa_are_human_labels", True),
        ("evidence_boundary", "production_default_authorized", True),
        ("ablations", "variant_count", 23),
        ("execution_limits", "encode_each_unique_media_once", False),
        ("acceptance_policy", "fixture_evidence_can_select_default", True),
        ("acceptance_policy", "minimum_reviewed_precision_lower_bound", 0.5),
    ),
)
def test_plan_rejects_weakened_boundaries(
    section: str, field: str, value: object
) -> None:
    tampered = deepcopy(_plan())
    tampered[section][field] = value

    with pytest.raises(ValueError):
        validate_dynamic_pool_pilot_plan(tampered)


def test_plan_rejects_missing_geo_as_located_and_occurrence_claims() -> None:
    plan = _plan()
    no_geo_index = next(
        index
        for index, row in enumerate(plan["cases"])
        if row["geographic_evidence_status"] == "missing_source_geography"
    )
    for field, value in (
        ("region_id", "fixture-region:invented"),
        ("biological_occurrence_claim", True),
        ("review_status", "human_reviewed"),
    ):
        tampered = deepcopy(plan)
        tampered["cases"][no_geo_index][field] = value
        with pytest.raises(ValueError):
            validate_dynamic_pool_pilot_plan(tampered)


def test_durable_input_validation_detects_byte_drift(tmp_path: Path) -> None:
    plan = _plan()
    for descriptor in plan["durable_inputs"]:
        source = ROOT / descriptor["relative_path"]
        destination = tmp_path / descriptor["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    descriptor = plan["durable_inputs"][0]
    target = tmp_path / descriptor["relative_path"]
    target.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 differs"):
        validate_dynamic_pool_pilot_inputs(plan, tmp_path)
