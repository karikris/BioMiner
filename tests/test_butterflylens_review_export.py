"""Tests for ButterflyLens review, maturity, and release projections."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest

from biominer.integration.butterflylens_review_export import (
    ASSIGNMENT_POLICY_VERSION,
    BUTTERFLYLENS_REVIEW_ROLES,
    build_butterflylens_review_layer,
    export_butterflylens_review_evidence,
    validate_butterflylens_review_export,
    validate_butterflylens_review_layer,
)
from helpers.butterflylens_handoff_fixture import (
    build_butterflylens_review_fixture,
)
from helpers.dynamic_pool_handoff_fixture import build_review_selection_fixture


def _layer():
    fixture = build_butterflylens_review_fixture()
    layer = build_butterflylens_review_layer(
        project=fixture["project"],
        run=fixture["run"],
        model_layer=fixture["layer"],
        selection=fixture["selection"],
        sampling_policy=fixture["sampling_policy"],
        target={
            "accepted_taxon_key": "gbif:5131359",
            "scientific_name": "Papilio demoleus",
            "rank": "species",
        },
        observed_at="2026-07-18T12:03:00+10:00",
    )
    return fixture, layer


def test_campaign_preserves_representative_blind_review_design() -> None:
    _, layer = _layer()
    campaign = layer.campaign

    assert campaign["target_schema_version"] == (
        "butterflylens-verification-campaign:v1.0.0"
    )
    assert campaign["status"] == "draft"
    assert campaign["sampling_plan"]["purpose"] == "quality_estimation"
    assert campaign["sampling_plan"]["design"] == "stratified_random"
    assert campaign["sampling_plan"]["representative"] is True
    assert campaign["sampling_plan"]["inclusion_probabilities_recorded"] is True
    assert campaign["review_requirement"]["required_independent_reviewers"] == 2
    assert campaign["review_requirement"]["second_review_policy"] == "always"
    assert campaign["blind_policy"]["enabled"] is True
    assert "model_score" in campaign["blind_policy"]["hidden_fields"]
    assert campaign["assignment_authority"] is False
    assert campaign["reviewer_identity_included"] is False
    assert campaign["scientific_claim_allowed"] is False


def test_pre_assignment_items_retain_sampling_and_exclude_reviewer_identity() -> None:
    _, layer = _layer()
    frame = layer.assignment_inputs

    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["assignment_policy_version"] == ASSIGNMENT_POLICY_VERSION
    assert row["assignment_status"] == "unassigned"
    assert row["review_round"] == 1
    assert row["blind"] is True
    assert row["required_independent_reviewers"] == 2
    assert row["inclusion_probability"] == 1.0
    assert row["sampling_weight"] == 1.0
    assert row["representative"] is True
    assert row["raw_score_is_probability"] is False
    assert row["reviewer_identity_included"] is False
    assert row["assignment_created"] is False
    assert row["occurrence_release_authorized"] is False
    assert "reviewer_id" not in frame.columns


def test_maturity_only_claims_available_species_candidate_evidence() -> None:
    _, layer = _layer()
    maturity = layer.classification_maturity

    assert maturity.height == 1
    row = maturity.row(0, named=True)
    assert row["species_candidate_available_status"] == "available"
    assert row["species_candidate_available_value"] is True
    assert row["species_candidate_available_evidence_fingerprints"]
    for name in (
        "butterfly_detected",
        "community_reviewed",
        "quality_estimate_available",
        "expert_reviewed",
        "release_ready",
    ):
        assert row[f"{name}_status"] == "unavailable"
        assert row[f"{name}_value"] is None
        assert row[f"{name}_reason"]
        assert row[f"{name}_evidence_fingerprints"] == []
    assert row["scientific_claim_allowed"] is False


def test_release_projection_is_blocked_with_no_human_or_database_authority() -> None:
    _, layer = _layer()
    release = layer.release_state

    assert release["candidate_state"] == "blocked"
    assert release["release_ready"] is False
    assert release["all_release_gates_passed"] is False
    assert release["release_blockers"]
    assert set(release["gate_states"].values()) == {False}
    assert release["review_event_count"] == 0
    assert release["reviewer_identity_included"] is False
    assert release["authorization_included"] is False
    assert release["downstream_authorization_required"] is True
    assert release["database_primary_key_included"] is False
    assert release["scientific_claim_allowed"] is False


def test_review_layer_rejects_unlinked_selection_and_target_fields() -> None:
    fixture = build_butterflylens_review_fixture()
    unrelated_selection, unrelated_policy = build_review_selection_fixture()
    common = {
        "project": fixture["project"],
        "run": fixture["run"],
        "model_layer": fixture["layer"],
        "target": {
            "accepted_taxon_key": "gbif:5131359",
            "scientific_name": "Papilio demoleus",
            "rank": "species",
        },
        "observed_at": "2026-07-18T12:03:00+10:00",
    }
    with pytest.raises(ValueError, match="source is unavailable"):
        build_butterflylens_review_layer(
            selection=unrelated_selection,
            sampling_policy=unrelated_policy,
            **common,
        )

    target = {**common["target"], "reviewer_id": "must-not-cross-boundary"}
    with pytest.raises(ValueError, match="target fields differ"):
        build_butterflylens_review_layer(
            selection=fixture["selection"],
            sampling_policy=fixture["sampling_policy"],
            **{**common, "target": target},
        )


def test_review_export_is_deterministic_create_only_and_round_trips(
    tmp_path: Path,
) -> None:
    _, layer = _layer()
    first = export_butterflylens_review_evidence(
        layer=layer, output_root=tmp_path / "first"
    )
    second = export_butterflylens_review_evidence(
        layer=layer, output_root=tmp_path / "second"
    )

    validate_butterflylens_review_export(first.root, first.artifacts)
    assert [row["role"] for row in first.artifacts] == list(BUTTERFLYLENS_REVIEW_ROLES)
    assert [row["sha256"] for row in first.artifacts] == [
        row["sha256"] for row in second.artifacts
    ]
    assert all(
        str(row["relative_path"]).startswith("artifacts/review/")
        for row in first.artifacts
    )
    with pytest.raises(FileExistsError, match="create-only"):
        export_butterflylens_review_evidence(layer=layer, output_root=first.root)


def test_review_export_rejects_lineage_and_content_tampering(tmp_path: Path) -> None:
    _, layer = _layer()
    exported = export_butterflylens_review_evidence(layer=layer, output_root=tmp_path)
    descriptors = deepcopy(exported.artifacts)
    descriptors[0]["parent_fingerprints"] = []
    with pytest.raises(ValueError, match="semantics differ"):
        validate_butterflylens_review_export(exported.root, descriptors)

    assignment_path = (
        exported.review_directory / "butterflylens_review_assignment_inputs.parquet"
    )
    tampered = pl.read_parquet(assignment_path).with_columns(
        pl.lit(True).alias("assignment_created")
    )
    tampered.write_parquet(assignment_path)
    with pytest.raises(ValueError, match="identity differs"):
        validate_butterflylens_review_export(exported.root, exported.artifacts)


def test_review_layer_validator_rejects_release_escalation() -> None:
    _, layer = _layer()
    release = deepcopy(layer.release_state)
    release["candidate_state"] = "approved"

    with pytest.raises(ValueError, match="release fingerprint differs"):
        validate_butterflylens_review_layer(
            type(layer)(
                campaign=layer.campaign,
                assignment_inputs=layer.assignment_inputs,
                classification_maturity=layer.classification_maturity,
                release_state=release,
            )
        )
