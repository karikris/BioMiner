"""Tests for the TaxaLens dynamic-pool handoff boundary."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from biominer.integration.taxalens_pool_handoff import (
    TAXALENS_PINNED_COMMIT,
    TAXALENS_POOL_HANDOFF_FILE,
    TAXALENS_POOL_HANDOFF_SCHEMA_VERSION,
    TAXALENS_REQUIRED_ARTIFACT_ROLES,
    TAXALENS_ROLE_DEFAULTS,
    TAXALENS_TARGET_CONTRACTS,
    build_taxalens_pool_handoff,
    validate_taxalens_pool_handoff,
    write_taxalens_pool_handoff,
)


PRODUCER_COMMIT = "1" * 40
SOURCE_FINGERPRINT = "sha256:" + "2" * 64
MODEL_FINGERPRINT = "sha256:" + "3" * 64
PREPROCESSING_FINGERPRINT = "sha256:" + "4" * 64


def _artifacts(*, quality_available: bool = False) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for index, role in enumerate(TAXALENS_REQUIRED_ARTIFACT_ROLES, start=5):
        filename, schema_version = TAXALENS_ROLE_DEFAULTS[role]
        available = role != "quality_sidecar" or quality_available
        maturity = (
            "provisional_raw_score"
            if role in {"candidate_scores", "photo_summaries", "candidate_sets"}
            else "provider_asserted_provisional_support"
            if role in {"pool_plans", "pool_members", "pool_summaries"}
            else "human_reviewed_flickr_labels"
            if role == "quality_sidecar" and quality_available
            else None
        )
        values.append(
            {
                "role": role,
                "availability": "available" if available else "unavailable",
                "unavailable_reason": (
                    None if available else "no reviewed quality estimate was produced"
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
                "byte_count": 100 + index if available else None,
                "row_count": index if available else None,
                "parent_fingerprints": [],
                "evidence_maturity_label": maturity,
            }
        )
    return values


def _manifest(
    *,
    artifacts: list[dict[str, object]] | None = None,
    completed_review_count: int = 0,
    quality_estimate_available: bool = False,
    quality_unavailable_reason: str | None = "no completed human review",
) -> dict[str, object]:
    return build_taxalens_pool_handoff(
        producer_commit=PRODUCER_COMMIT,
        created_at="2026-07-18T12:00:00+10:00",
        run_id="run-papilio-demoleus-001",
        registry_version="registry-2026-07-18",
        source_snapshot_fingerprints=[SOURCE_FINGERPRINT],
        model_fingerprint=MODEL_FINGERPRINT,
        preprocessing_fingerprint=PREPROCESSING_FINGERPRINT,
        artifacts=artifacts or _artifacts(quality_available=quality_estimate_available),
        completed_review_count=completed_review_count,
        quality_estimate_available=quality_estimate_available,
        quality_unavailable_reason=quality_unavailable_reason,
    )


def test_manifest_pins_taxalens_and_preserves_all_evidence_roles() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == TAXALENS_POOL_HANDOFF_SCHEMA_VERSION
    assert manifest["consumer"] == {
        "repository": "karikris/taxalens",
        "commit": TAXALENS_PINNED_COMMIT,
    }
    assert manifest["target_contracts"] == TAXALENS_TARGET_CONTRACTS
    assert [row["role"] for row in manifest["artifacts"]] == list(
        TAXALENS_REQUIRED_ARTIFACT_ROLES
    )
    assert all(
        row["producer_commit"] == PRODUCER_COMMIT
        and row["scientific_claim_allowed"] is False
        for row in manifest["artifacts"]
    )
    assert manifest["handoff_id"].startswith("taxalens-pool-handoff:")
    assert manifest["manifest_fingerprint"].startswith("sha256:")


def test_manifest_is_deterministic_and_normalizes_artifact_order() -> None:
    artifacts = _artifacts()

    assert _manifest(artifacts=artifacts) == _manifest(
        artifacts=list(reversed(artifacts))
    )


def test_zero_reviews_are_unavailable_quality_not_zero_quality() -> None:
    maturity = _manifest()["evidence_maturity"]

    assert maturity["human_review"]["status"] == "not_evaluated"
    assert maturity["human_review"]["completed_review_count"] == 0
    assert maturity["quality_estimate"]["status"] == "unavailable"
    assert maturity["quality_estimate"]["zero_review_is_zero_quality"] is False
    assert maturity["release"] == {
        "status": "not_evaluated",
        "label": None,
        "release_ready": False,
        "scientific_claim_allowed": False,
        "unavailable_reason": (
            "BioMiner evidence handoff cannot authorize TaxaLens occurrence release"
        ),
    }


def test_reviewed_quality_can_be_available_without_authorizing_release() -> None:
    manifest = _manifest(
        completed_review_count=12,
        quality_estimate_available=True,
        quality_unavailable_reason=None,
    )

    maturity = manifest["evidence_maturity"]
    assert maturity["human_review"]["status"] == "available"
    assert maturity["quality_estimate"]["status"] == "available"
    assert maturity["release"]["release_ready"] is False
    assert manifest["authority_boundary"]["review_occurrence_is_release"] is False


def test_insufficient_reviewed_quality_sidecar_blocks_release() -> None:
    manifest = _manifest(
        artifacts=_artifacts(quality_available=True),
        completed_review_count=4,
        quality_estimate_available=False,
        quality_unavailable_reason="representative evidence is insufficient",
    )

    maturity = manifest["evidence_maturity"]
    quality = next(
        row for row in manifest["artifacts"] if row["role"] == "quality_sidecar"
    )
    assert quality["availability"] == "available"
    assert maturity["human_review"]["status"] == "available"
    assert maturity["quality_estimate"]["status"] == "unavailable"
    assert maturity["release"]["release_ready"] is False


def test_quality_estimate_without_reviews_fails_closed() -> None:
    with pytest.raises(ValueError, match="require completed human reviews"):
        _manifest(
            completed_review_count=0,
            quality_estimate_available=True,
            quality_unavailable_reason=None,
        )


def test_nonavailable_artifact_cannot_claim_physical_identity() -> None:
    artifacts = _artifacts()
    quality = next(row for row in artifacts if row["role"] == "quality_sidecar")
    quality["sha256"] = "sha256:" + "a" * 64

    with pytest.raises(ValueError, match="cannot claim physical identity"):
        _manifest(artifacts=artifacts)


def test_artifact_paths_cannot_escape_the_handoff_root() -> None:
    artifacts = _artifacts()
    artifacts[0]["relative_path"] = "../candidate_scores.parquet"

    with pytest.raises(ValueError, match="canonical and relative"):
        _manifest(artifacts=artifacts)


def test_candidate_scores_cannot_be_labelled_as_probabilities() -> None:
    artifacts = _artifacts()
    artifacts[0]["evidence_maturity_label"] = "calibrated_probability"

    with pytest.raises(ValueError, match="provisional raw scores"):
        _manifest(artifacts=artifacts)


def test_validation_detects_pin_release_and_content_tampering() -> None:
    manifest = _manifest()
    moved_pin = deepcopy(manifest)
    moved_pin["consumer"]["commit"] = "f" * 40
    with pytest.raises(ValueError, match="consumer pin moved"):
        validate_taxalens_pool_handoff(moved_pin)

    released = deepcopy(manifest)
    released["evidence_maturity"]["release"]["release_ready"] = True
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_taxalens_pool_handoff(released)

    tampered = deepcopy(manifest)
    tampered["run_id"] = "changed-run"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_taxalens_pool_handoff(tampered)


def test_writer_uses_the_contract_filename_and_round_trips(tmp_path: Path) -> None:
    manifest = _manifest()

    output = write_taxalens_pool_handoff(manifest, tmp_path)

    assert output == tmp_path / TAXALENS_POOL_HANDOFF_FILE
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert output.read_bytes().endswith(b"\n")
