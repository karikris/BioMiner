"""Behaviour tests for local TaxaLens handoff publication."""

from __future__ import annotations

from pathlib import Path

import pytest

from biominer.integration.taxalens_pool_handoff import (
    TAXALENS_POOL_HANDOFF_FILE,
)
from biominer.integration.taxalens_pool_publish import (
    publish_taxalens_pool_handoff,
    validate_published_taxalens_pool_handoff,
)
from biominer.storage.handoff import verify_handoff_bundle
from helpers.dynamic_pool_handoff_fixture import (
    build_dynamic_pool_handoff_fixture,
    build_quality_report_fixture,
    build_review_selection_fixture,
)

PRODUCER_COMMIT = "1" * 40


def test_complete_publication_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    # Arrange
    quality_report = build_quality_report_fixture()

    # Act
    first = _publish(tmp_path / "first", quality_report=quality_report)
    second = _publish(
        tmp_path / "second",
        quality_report=quality_report,
    )

    # Assert
    assert first.manifest == second.manifest
    assert first.bundle.sha256 == second.bundle.sha256
    assert (
        first.bundle.archive_path.read_bytes()
        == second.bundle.archive_path.read_bytes()
    )


def test_complete_publication_bundle_is_self_verifying(tmp_path: Path) -> None:
    # Arrange
    published = _publish(
        tmp_path / "published",
        quality_report=build_quality_report_fixture(),
    )

    # Act
    verification = verify_handoff_bundle(
        published.bundle.archive_path,
        expected_sha256=published.bundle.sha256,
    )

    # Assert
    assert verification.source_git_sha == PRODUCER_COMMIT
    assert len(verification.files) == published.bundle.file_count == 9
    assert TAXALENS_POOL_HANDOFF_FILE in {
        item.relative_path for item in verification.files
    }


def test_publication_without_review_outcomes_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    # Arrange / Act
    published = _publish(tmp_path / "no-quality", quality_report=None)

    # Assert
    quality = next(
        row
        for row in published.manifest["artifacts"]
        if row["role"] == "quality_sidecar"
    )
    assert quality["availability"] == "unavailable"
    assert published.bundle.file_count == 8
    assert not (
        published.root / "artifacts/review/taxalens_quality_sidecar.json"
    ).exists()
    assert published.manifest["evidence_maturity"]["human_review"]["status"] == (
        "not_evaluated"
    )
    assert published.manifest["evidence_maturity"]["quality_estimate"] == {
        "status": "unavailable",
        "label": None,
        "zero_review_is_zero_quality": False,
        "release_authorizing": False,
        "unavailable_reason": "no validated reviewed quality report supplied",
    }
    validate_published_taxalens_pool_handoff(
        published.root,
        published.manifest,
    )


def test_publication_refuses_to_replace_existing_output(
    tmp_path: Path,
) -> None:
    # Arrange
    base = tmp_path / "create-only"
    published = _publish(base, quality_report=None)

    # Act / Assert
    with pytest.raises(FileExistsError, match="create-only"):
        _publish(base, quality_report=None)
    assert published.manifest_path.is_file()


def test_publication_rejects_archive_inside_publication_root(
    tmp_path: Path,
) -> None:
    # Arrange
    selection, policy = build_review_selection_fixture()
    nested_root = tmp_path / "nested"

    # Act / Assert
    with pytest.raises(ValueError, match="outside the publication root"):
        publish_taxalens_pool_handoff(
            **_publication_inputs(
                selection=selection,
                policy=policy,
                quality_report=None,
            ),
            output_root=nested_root,
            archive_output_dir=nested_root / "archives",
        )
    assert not nested_root.exists()


def test_directory_validator_detects_artifact_tampering(tmp_path: Path) -> None:
    # Arrange
    published = _publish(tmp_path / "tamper", quality_report=None)
    score_path = published.root / "artifacts/dynamic_pool_candidate_scores.parquet"
    with score_path.open("ab") as output:
        output.write(b"tampered")

    # Act / Assert
    with pytest.raises(ValueError, match="byte count differs"):
        validate_published_taxalens_pool_handoff(
            published.root,
            published.manifest,
        )


def _publish(base: Path, *, quality_report):
    selection, policy = build_review_selection_fixture()
    return publish_taxalens_pool_handoff(
        **_publication_inputs(
            selection=selection,
            policy=policy,
            quality_report=quality_report,
        ),
        output_root=base / "handoff",
        archive_output_dir=base / "archives",
    )


def _publication_inputs(*, selection, policy, quality_report) -> dict[str, object]:
    return {
        "score_pool_frames": build_dynamic_pool_handoff_fixture(),
        "review_selection": selection,
        "review_sampling_policy": policy,
        "quality_report": quality_report,
        "producer_commit": PRODUCER_COMMIT,
        "created_at": "2026-07-18T12:00:00+10:00",
        "run_id": "run-papilio-demoleus-001",
        "registry_version": "registry-2026-07-18",
        "source_snapshot_fingerprints": ["sha256:" + "2" * 64],
        "model_fingerprint": "sha256:" + "3" * 64,
        "preprocessing_fingerprint": "sha256:" + "4" * 64,
    }
