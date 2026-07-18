"""Tests for the ButterflyLens dynamic-pool handoff boundary."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from biominer.integration.butterflylens_pool_handoff import (
    BUTTERFLYLENS_PINNED_COMMIT,
    BUTTERFLYLENS_POOL_HANDOFF_FILE,
    BUTTERFLYLENS_POOL_HANDOFF_SCHEMA_VERSION,
    BUTTERFLYLENS_PREVIOUS_AUDITED_COMMIT,
    BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES,
    BUTTERFLYLENS_ROLE_DEFAULTS,
    BUTTERFLYLENS_TARGET_CONTRACTS,
    build_butterflylens_pool_handoff,
    validate_butterflylens_pool_handoff,
    write_butterflylens_pool_handoff,
)


PRODUCER_COMMIT = "1" * 40
SOURCE_FINGERPRINT = "sha256:" + "2" * 64
MODEL_FINGERPRINT = "sha256:" + "3" * 64
PREPROCESSING_FINGERPRINT = "sha256:" + "4" * 64
COMPATIBILITY_REVIEW = (
    Path(__file__).parents[1]
    / "docs/architecture/butterflylens_1cea643_compatibility_review.md"
)


def _artifacts() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for index, role in enumerate(BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES, start=5):
        filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS[role]
        available = role != "release_state"
        values.append(
            {
                "role": role,
                "availability": "available" if available else "unavailable",
                "unavailable_reason": (
                    None
                    if available
                    else "release gates have not been evaluated by ButterflyLens"
                ),
                "relative_path": f"artifacts/{filename}" if available else None,
                "media_type": (
                    "application/vnd.apache.parquet"
                    if filename.endswith(".parquet")
                    else "application/json"
                ),
                "schema_version": schema_version,
                "semantic_fingerprint": (
                    "sha256:" + f"{index:x}" * 64 if available else None
                ),
                "sha256": "sha256:" + f"{index + 1:x}" * 64 if available else None,
                "byte_count": 200 + index if available else None,
                "row_count": index if available else None,
                "parent_fingerprints": [],
                "evidence_maturity_label": (
                    "provisional_raw_score" if role == "model_evidence" else None
                ),
            }
        )
    return values


def _manifest(
    artifacts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return build_butterflylens_pool_handoff(
        producer_commit=PRODUCER_COMMIT,
        created_at="2026-07-18T12:00:00+10:00",
        project_id="australian-butterflies",
        run_id="run-papilio-demoleus-001",
        registry_version="registry-2026-07-18",
        source_snapshot_fingerprints=[SOURCE_FINGERPRINT],
        model_fingerprint=MODEL_FINGERPRINT,
        preprocessing_fingerprint=PREPROCESSING_FINGERPRINT,
        artifacts=artifacts or _artifacts(),
    )


def test_manifest_records_deliberate_pin_movement_and_all_roles() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == BUTTERFLYLENS_POOL_HANDOFF_SCHEMA_VERSION
    assert manifest["consumer"] == {
        "repository": "karikris/ButterflyLens",
        "commit": BUTTERFLYLENS_PINNED_COMMIT,
    }
    assert manifest["consumer_compatibility"] == {
        "previous_audited_commit": BUTTERFLYLENS_PREVIOUS_AUDITED_COMMIT,
        "current_audited_commit": BUTTERFLYLENS_PINNED_COMMIT,
        "decision": "compatible_additive_with_stricter_review_controls",
        "wire_schema_breaking_changes": False,
        "database_adapter_changes_required": True,
        "required_adapter_changes": [
            "create repeated independent assignments under downstream policy",
            "keep model evidence and peer decisions blind until review submission",
            "submit append-only review events with correction lineage",
        ],
        "silent_pin_movement": False,
    }
    assert [row["role"] for row in manifest["artifacts"]] == list(
        BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES
    )
    assert manifest["target_contracts"] == BUTTERFLYLENS_TARGET_CONTRACTS


def test_manifest_normalizes_role_order_deterministically() -> None:
    artifacts = _artifacts()

    assert _manifest(artifacts) == _manifest(list(reversed(artifacts)))


def test_model_evidence_remains_raw_and_preserves_unfinished_states() -> None:
    projection = _manifest()["field_projections"]["model_evidence"]

    assert projection["raw_score_is_probability"] is False
    assert projection["statuses"] == [
        "completed",
        "failed",
        "blocked",
        "skipped_unfinished",
    ]
    assert set(projection["evidence_kinds"]) == {
        "yoloe_route",
        "bioclip_embedding",
        "prototype",
        "candidate_score",
    }


def test_review_inputs_cannot_allocate_reviewers_or_database_ids() -> None:
    manifest = _manifest()
    policy = manifest["review_assignment_policy"]
    projection = manifest["field_projections"]["review_inputs"]

    assert policy["policy_version"] == "repeated-independent-v1"
    assert policy["minimum_independent_reviewers"] == 2
    assert policy["blind_until_decision"] is True
    assert policy["assignments_created_by_biominer"] is False
    assert policy["review_events_are_append_only"] is True
    assert "reviewer_id" in projection["excluded_fields"]
    assert "assignment_database_pk" in projection["excluded_fields"]


def test_database_and_release_authority_stay_downstream() -> None:
    manifest = _manifest()
    boundary = manifest["authority_boundary"]

    assert boundary["database_primary_keys_in_handoff"] is False
    assert boundary["reviewer_identity_in_handoff"] is False
    assert boundary["service_role_credentials_in_handoff"] is False
    assert boundary["biominer_database_writes_allowed"] is False
    assert boundary["service_role_ingestion_enforced_downstream"] is True
    assert boundary["row_level_security_enforced_downstream"] is True
    assert manifest["release_state"]["candidate_state"] == "blocked"
    assert manifest["release_state"]["release_ready"] is False
    assert manifest["release_state"]["scientific_claim_allowed"] is False


def test_final_release_maturity_is_rejected_from_producer_artifact() -> None:
    artifacts = _artifacts()
    release = next(row for row in artifacts if row["role"] == "release_state")
    release["evidence_maturity_label"] = "final_release_status"

    with pytest.raises(ValueError, match="cannot hand off final"):
        _manifest(artifacts)


def test_validation_rejects_pin_authority_and_content_tampering() -> None:
    manifest = _manifest()
    moved_pin = deepcopy(manifest)
    moved_pin["consumer"]["commit"] = "f" * 40
    with pytest.raises(ValueError, match="consumer pin moved"):
        validate_butterflylens_pool_handoff(moved_pin)

    escalated = deepcopy(manifest)
    escalated["authority_boundary"]["biominer_database_writes_allowed"] = True
    with pytest.raises(ValueError, match="authority boundary differs"):
        validate_butterflylens_pool_handoff(escalated)

    tampered = deepcopy(manifest)
    tampered["run_id"] = "changed-run"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_butterflylens_pool_handoff(tampered)


def test_writer_uses_contract_filename_and_round_trips(tmp_path: Path) -> None:
    manifest = _manifest()

    output = write_butterflylens_pool_handoff(manifest, tmp_path)

    assert output == tmp_path / BUTTERFLYLENS_POOL_HANDOFF_FILE
    assert json.loads(output.read_text(encoding="utf-8")) == manifest


def test_compatibility_review_documents_supabase_and_pin_boundaries() -> None:
    review = COMPATIBILITY_REVIEW.read_text(encoding="utf-8")

    for term in (
        BUTTERFLYLENS_PREVIOUS_AUDITED_COMMIT,
        BUTTERFLYLENS_PINNED_COMMIT,
        "repeated-independent-v1",
        "append-only",
        "service-role",
        "row-level security",
        "compatible_additive_with_stricter_review_controls",
        "https://supabase.com/docs/guides/database/postgres/row-level-security",
    ):
        assert term in review
