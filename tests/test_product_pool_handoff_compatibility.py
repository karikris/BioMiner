"""Pinned compatibility checks for both product handoff manifests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from biominer.integration.butterflylens_pool_handoff import (
    BUTTERFLYLENS_PINNED_COMMIT,
    BUTTERFLYLENS_PREVIOUS_AUDITED_COMMIT,
    BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES,
    BUTTERFLYLENS_TARGET_CONTRACTS,
    build_butterflylens_pool_handoff,
    validate_butterflylens_pool_handoff,
)
from biominer.integration.taxalens_pool_handoff import (
    TAXALENS_PINNED_COMMIT,
    TAXALENS_REQUIRED_ARTIFACT_ROLES,
    TAXALENS_TARGET_CONTRACTS,
    build_taxalens_pool_handoff,
    validate_taxalens_pool_handoff,
)


ROOT = Path(__file__).parents[1]
COMPATIBILITY_FIXTURE = ROOT / "tests/fixtures/product_pool_handoff_compatibility.json"
PIN_FIXTURE = ROOT / "tests/fixtures/downstream_pooling_contract_pins.json"


def _fixture() -> dict[str, object]:
    return json.loads(COMPATIBILITY_FIXTURE.read_text(encoding="utf-8"))


def _common(value: dict[str, object]) -> dict[str, object]:
    return {
        "producer_commit": value["producer_commit"],
        "created_at": value["created_at"],
        "run_id": value["run_id"],
        "registry_version": value["registry_version"],
        "source_snapshot_fingerprints": value["source_snapshot_fingerprints"],
        "model_fingerprint": value["model_fingerprint"],
        "preprocessing_fingerprint": value["preprocessing_fingerprint"],
    }


def _build_manifests() -> tuple[dict[str, object], dict[str, object]]:
    fixture = _fixture()
    taxalens = fixture["taxalens"]
    butterflylens = fixture["butterflylens"]
    taxalens_manifest = build_taxalens_pool_handoff(
        **_common(fixture),
        artifacts=taxalens["artifacts"],
        completed_review_count=taxalens["completed_review_count"],
        quality_estimate_available=taxalens["quality_estimate_available"],
        quality_unavailable_reason=taxalens["quality_unavailable_reason"],
    )
    butterflylens_manifest = build_butterflylens_pool_handoff(
        **_common(fixture),
        project_id=butterflylens["project_id"],
        artifacts=butterflylens["artifacts"],
    )
    return taxalens_manifest, butterflylens_manifest


def test_representative_fixture_has_frozen_production_identities() -> None:
    fixture = _fixture()
    taxalens, butterflylens = _build_manifests()

    assert fixture["schema_version"] == (
        "biominer-product-pool-handoff-compatibility-v1.0.0"
    )
    assert taxalens["handoff_id"] == fixture["taxalens"]["expected_handoff_id"]
    assert (
        taxalens["manifest_fingerprint"]
        == (fixture["taxalens"]["expected_manifest_fingerprint"])
    )
    assert (
        butterflylens["handoff_id"] == (fixture["butterflylens"]["expected_handoff_id"])
    )
    assert (
        butterflylens["manifest_fingerprint"]
        == (fixture["butterflylens"]["expected_manifest_fingerprint"])
    )
    validate_taxalens_pool_handoff(taxalens)
    validate_butterflylens_pool_handoff(butterflylens)


def test_shared_pin_fixture_matches_production_consumers_and_targets() -> None:
    pins = json.loads(PIN_FIXTURE.read_text(encoding="utf-8"))
    fixture = _fixture()

    assert pins["taxalens"]["audited_commit"] == TAXALENS_PINNED_COMMIT
    assert fixture["taxalens"]["consumer_commit"] == TAXALENS_PINNED_COMMIT
    assert pins["butterflylens"]["previous_audited_commit"] == (
        BUTTERFLYLENS_PREVIOUS_AUDITED_COMMIT
    )
    assert pins["butterflylens"]["audited_commit"] == (BUTTERFLYLENS_PINNED_COMMIT)
    assert fixture["butterflylens"]["consumer_commit"] == (BUTTERFLYLENS_PINNED_COMMIT)
    assert fixture["butterflylens"]["previous_consumer_commit"] == (
        BUTTERFLYLENS_PREVIOUS_AUDITED_COMMIT
    )
    for name in (
        "geographic_impact_export",
        "verification_campaign",
        "flickr_verification_source",
        "quality_snapshot",
        "reviewed_labels",
        "storage_handoff_inventory",
        "storage_handoff_inventory_path",
    ):
        assert pins["taxalens"]["contracts"][name] == (TAXALENS_TARGET_CONTRACTS[name])
    for name in (
        "project",
        "run",
        "classification_maturity",
        "evidence_fingerprint",
        "geographic_impact_cell",
        "verification_campaign",
        "verification_assignment",
        "verification_adjudication",
        "verification_event",
        "verification_consensus",
        "layered_consensus_policy",
        "reviewer_reliability",
        "quality_snapshot",
        "quality_policy",
        "quality_estimator",
        "flickr_public_display_policy",
        "sensitive_location_policy",
        "public_location_decision",
        "occurrence_release_policy",
        "occurrence_release_decision",
        "darwin_core_export",
        "darwin_core_export_policy",
        "ala_contribution",
        "ala_contribution_policy",
    ):
        assert (
            pins["butterflylens"]["contracts"][name]
            == (BUTTERFLYLENS_TARGET_CONTRACTS[name])
        )


def test_both_manifests_are_complete_and_fail_closed() -> None:
    taxalens, butterflylens = _build_manifests()

    assert taxalens["artifact_roles"] == list(TAXALENS_REQUIRED_ARTIFACT_ROLES)
    assert butterflylens["artifact_roles"] == list(
        BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES
    )
    for manifest in (taxalens, butterflylens):
        for artifact in manifest["artifacts"]:
            if artifact["availability"] == "available":
                assert artifact["relative_path"]
                assert artifact["semantic_fingerprint"].startswith("sha256:")
                assert artifact["sha256"].startswith("sha256:")
            else:
                assert artifact["unavailable_reason"]
                assert artifact["relative_path"] is None
                assert artifact["semantic_fingerprint"] is None
                assert artifact["sha256"] is None
            assert artifact["scientific_claim_allowed"] is False
    assert taxalens["evidence_maturity"]["quality_estimate"]["status"] == (
        "unavailable"
    )
    assert (
        taxalens["evidence_maturity"]["quality_estimate"]["zero_review_is_zero_quality"]
        is False
    )
    assert butterflylens["release_state"]["release_ready"] is False
    assert (
        butterflylens["authority_boundary"]["biominer_database_writes_allowed"] is False
    )
    assert butterflylens["authority_boundary"]["reviewer_identity_in_handoff"] is (
        False
    )
    assert (
        butterflylens["authority_boundary"]["sensitive_location_decisions_in_handoff"]
        is False
    )
    assert (
        butterflylens["authority_boundary"]["occurrence_release_decisions_in_handoff"]
        is False
    )
    assert butterflylens["authority_boundary"]["ala_submission_authority"] is False


def test_field_parity_covers_product_specific_scientific_boundaries() -> None:
    taxalens, butterflylens = _build_manifests()

    taxalens_projections = taxalens["field_projections"]
    assert "inclusion_probability" in taxalens_projections["review_sampling"]["fields"]
    assert "baseline_union_count" in taxalens_projections["geographic_impact"]["fields"]
    assert taxalens_projections["quality"]["zero_review_is_zero_quality"] is False
    butterflylens_projections = butterflylens["field_projections"]
    assert (
        butterflylens_projections["model_evidence"]["raw_score_is_probability"] is False
    )
    assert (
        "reviewer_id" in butterflylens_projections["review_inputs"]["excluded_fields"]
    )
    assert (
        butterflylens_projections["geographic_impact"]["candidate_only_is_occurrence"]
        is False
    )
    assert (
        butterflylens_projections["geographic_impact"]["raw_coordinates_excluded"]
        is True
    )
    assert butterflylens_projections["review_inputs"]["quality_estimate_authority"] is (
        False
    )
    assert butterflylens_projections["release"]["producer_release_authority"] is False
    assert (
        butterflylens_projections["publication"]["darwin_core_preparation_authority"]
        is False
    )
    assert butterflylens_projections["publication"]["ala_submission_authority"] is (
        False
    )


@pytest.mark.parametrize(
    ("repository_name", "commit", "path", "required_values"),
    [
        (
            "taxalens",
            TAXALENS_PINNED_COMMIT,
            "taxalens/product/biominer_handoff.py",
            (
                TAXALENS_TARGET_CONTRACTS["storage_handoff_inventory"],
                TAXALENS_TARGET_CONTRACTS["storage_handoff_inventory_path"],
            ),
        ),
        (
            "taxalens",
            TAXALENS_PINNED_COMMIT,
            "apps/web/src/review/domain/verificationContracts.ts",
            (
                TAXALENS_TARGET_CONTRACTS["verification_campaign"],
                TAXALENS_TARGET_CONTRACTS["flickr_verification_source"],
            ),
        ),
        (
            "taxalens",
            TAXALENS_PINNED_COMMIT,
            "apps/web/src/impact/geographicImpactExport.ts",
            (TAXALENS_TARGET_CONTRACTS["geographic_impact_export"],),
        ),
        (
            "taxalens",
            TAXALENS_PINNED_COMMIT,
            "apps/web/src/review/domain/verificationQualitySnapshot.ts",
            (TAXALENS_TARGET_CONTRACTS["quality_snapshot"],),
        ),
        (
            "taxalens",
            TAXALENS_PINNED_COMMIT,
            "apps/web/src/review/exports/flickrReviewedLabels.ts",
            (
                TAXALENS_TARGET_CONTRACTS["reviewed_labels"],
                TAXALENS_TARGET_CONTRACTS["reviewed_labels_filename"],
            ),
        ),
        (
            "ButterflyLens",
            BUTTERFLYLENS_PINNED_COMMIT,
            "packages/contracts/tests/fixtures/parity-cases.json",
            (
                BUTTERFLYLENS_TARGET_CONTRACTS["project"],
                BUTTERFLYLENS_TARGET_CONTRACTS["run"],
                BUTTERFLYLENS_TARGET_CONTRACTS["classification_maturity"],
                BUTTERFLYLENS_TARGET_CONTRACTS["evidence_fingerprint"],
                BUTTERFLYLENS_TARGET_CONTRACTS["geographic_impact_cell"],
                BUTTERFLYLENS_TARGET_CONTRACTS["verification_campaign"],
                BUTTERFLYLENS_TARGET_CONTRACTS["verification_assignment"],
                BUTTERFLYLENS_TARGET_CONTRACTS["verification_adjudication"],
                BUTTERFLYLENS_TARGET_CONTRACTS["reviewer_reliability"],
                BUTTERFLYLENS_TARGET_CONTRACTS["quality_snapshot"],
            ),
        ),
        (
            "ButterflyLens",
            BUTTERFLYLENS_PINNED_COMMIT,
            "packages/contracts/python/butterflylens/contracts/occurrence_release.py",
            (
                BUTTERFLYLENS_TARGET_CONTRACTS["occurrence_release_policy"],
                BUTTERFLYLENS_TARGET_CONTRACTS["occurrence_release_decision"],
            ),
        ),
        (
            "ButterflyLens",
            BUTTERFLYLENS_PINNED_COMMIT,
            "packages/contracts/python/butterflylens/contracts/sensitive_locations.py",
            (
                BUTTERFLYLENS_TARGET_CONTRACTS["sensitive_location_policy"],
                BUTTERFLYLENS_TARGET_CONTRACTS["public_location_decision"],
            ),
        ),
        (
            "ButterflyLens",
            BUTTERFLYLENS_PINNED_COMMIT,
            "packages/contracts/python/butterflylens/contracts/darwin_core_export.py",
            (
                BUTTERFLYLENS_TARGET_CONTRACTS["darwin_core_export"],
                BUTTERFLYLENS_TARGET_CONTRACTS["darwin_core_export_policy"],
            ),
        ),
        (
            "ButterflyLens",
            BUTTERFLYLENS_PINNED_COMMIT,
            "packages/contracts/python/butterflylens/contracts/ala_contribution.py",
            (
                BUTTERFLYLENS_TARGET_CONTRACTS["ala_contribution"],
                BUTTERFLYLENS_TARGET_CONTRACTS["ala_contribution_policy"],
            ),
        ),
        (
            "ButterflyLens",
            BUTTERFLYLENS_PINNED_COMMIT,
            "packages/flickr/public-display-policy.v1.json",
            (BUTTERFLYLENS_TARGET_CONTRACTS["flickr_public_display_policy"],),
        ),
    ],
)
def test_pinned_sibling_objects_contain_target_contracts(
    repository_name: str,
    commit: str,
    path: str,
    required_values: tuple[str, ...],
) -> None:
    repository = ROOT.parent / repository_name
    if not (repository / ".git").exists():
        pytest.skip(f"{repository_name} sibling checkout is unavailable")
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    for value in required_values:
        assert value in result.stdout


def test_pinned_butterflylens_review_migrations_match_adapter_policy() -> None:
    repository = ROOT.parent / "ButterflyLens"
    if not (repository / ".git").exists():
        pytest.skip("ButterflyLens sibling checkout is unavailable")
    paths = (
        "supabase/migrations/20260718013600_repeated_independent_assignments.sql",
        "supabase/migrations/20260718014600_blind_review_disclosure.sql",
        "supabase/migrations/20260718015500_append_only_review_submission.sql",
    )
    committed = "\n".join(
        subprocess.run(
            ["git", "show", f"{BUTTERFLYLENS_PINNED_COMMIT}:{path}"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for path in paths
    )

    for term in (
        "repeated-independent-v1",
        "minimum_review_count >= 2",
        "required_reviewer_role",
        "security_invoker",
        "supersedes_event_pk",
    ):
        assert term in committed


def test_pinned_butterflylens_release_migrations_match_authority_boundary() -> None:
    repository = ROOT.parent / "ButterflyLens"
    if not (repository / ".git").exists():
        pytest.skip("ButterflyLens sibling checkout is unavailable")
    paths = tuple(
        BUTTERFLYLENS_TARGET_CONTRACTS[name]
        for name in (
            "conflict_adjudication_migration",
            "reviewer_control_migration",
            "reviewer_reliability_migration",
            "layered_consensus_migration",
            "dataset_quality_migration",
            "flickr_public_display_migration",
            "sensitive_location_migration",
            "media_takedown_migration",
            "occurrence_release_migration",
        )
    )
    committed = "\n".join(
        subprocess.run(
            [
                "git",
                "show",
                f"{BUTTERFLYLENS_PINNED_COMMIT}:supabase/migrations/{path}",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for path in paths
    )

    for term in (
        "security_invoker",
        "butterflylens-reviewer-reliability-policy:v1.0.0",
        "butterflylens-sensitive-location-policy:v1.0.0",
        "butterflylens-media-rights:v1.0.0",
        "butterflylens-occurrence-release:v1.0.0",
    ):
        assert term in committed


def test_pinned_butterflylens_rejects_retired_fingerprint_schema() -> None:
    repository = ROOT.parent / "ButterflyLens"
    if not (repository / ".git").exists():
        pytest.skip("ButterflyLens sibling checkout is unavailable")
    fingerprint = subprocess.run(
        [
            "git",
            "show",
            (
                f"{BUTTERFLYLENS_PINNED_COMMIT}:packages/contracts/python/"
                "butterflylens/contracts/fingerprint.py"
            ),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parity = subprocess.run(
        [
            "git",
            "show",
            (
                f"{BUTTERFLYLENS_PINNED_COMMIT}:packages/contracts/tests/"
                "fixtures/parity-cases.json"
            ),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "EVIDENCE_FINGERPRINT_LEGACY_SCHEMA_VERSION" not in fingerprint
    assert '"butterflylens-evidence-fingerprint:v1.1.0"' in fingerprint
    assert "fingerprint-rejects-retired-v1.0-version" in parity
