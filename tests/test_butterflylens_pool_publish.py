"""Tests for complete ButterflyLens handoff publication."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from biominer.integration.butterflylens_pool_handoff import (
    BUTTERFLYLENS_POOL_HANDOFF_FILE,
    BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES,
)
from biominer.integration.butterflylens_pool_publish import (
    publish_butterflylens_pool_handoff,
    validate_published_butterflylens_pool_handoff,
)
from biominer.storage.handoff import verify_handoff_bundle
from helpers.butterflylens_handoff_fixture import (
    build_butterflylens_complete_fixture,
)


def _inputs() -> dict[str, object]:
    return build_butterflylens_complete_fixture()


def test_complete_publication_contains_all_roles_and_verified_archive(
    tmp_path: Path,
) -> None:
    values = _inputs()
    published = publish_butterflylens_pool_handoff(
        project=values["project"],
        run=values["run"],
        model_layer=values["layer"],
        geographic_impact=values["geographic"],
        review_layer=values["review"],
        output_root=tmp_path / "published",
        archive_output_dir=tmp_path / "archives",
        producer_commit="1" * 40,
        created_at="2026-07-18T12:04:00+10:00",
        registry_version="butterflies-v2-20260712",
    )

    validate_published_butterflylens_pool_handoff(published.root, published.manifest)
    assert [row["role"] for row in published.manifest["artifacts"]] == list(
        BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES
    )
    assert all(
        row["availability"] == "available" for row in published.manifest["artifacts"]
    )
    assert published.manifest["release_state"]["release_ready"] is False
    assert published.manifest["release_state"]["scientific_claim_allowed"] is False
    assert (
        published.manifest["authority_boundary"]["biominer_database_writes_allowed"]
        is False
    )
    assert (
        json.loads((published.root / BUTTERFLYLENS_POOL_HANDOFF_FILE).read_text())
        == published.manifest
    )
    verification = verify_handoff_bundle(
        published.bundle.archive_path, expected_sha256=published.bundle.sha256
    )
    assert verification.file_count == 11
    assert verification.source_git_sha == "1" * 40


def test_publication_is_create_only_and_requires_matching_run_commit(
    tmp_path: Path,
) -> None:
    values = _inputs()
    common = {
        "project": values["project"],
        "run": values["run"],
        "model_layer": values["layer"],
        "geographic_impact": values["geographic"],
        "review_layer": values["review"],
        "output_root": tmp_path / "published",
        "archive_output_dir": tmp_path / "archives",
        "created_at": "2026-07-18T12:04:00+10:00",
        "registry_version": "butterflies-v2-20260712",
    }
    with pytest.raises(ValueError, match="producer and run commit differ"):
        publish_butterflylens_pool_handoff(producer_commit="2" * 40, **common)

    publish_butterflylens_pool_handoff(producer_commit="1" * 40, **common)
    with pytest.raises(FileExistsError, match="create-only"):
        publish_butterflylens_pool_handoff(producer_commit="1" * 40, **common)


def test_published_validator_rejects_manifest_and_artifact_tampering(
    tmp_path: Path,
) -> None:
    values = _inputs()
    published = publish_butterflylens_pool_handoff(
        project=values["project"],
        run=values["run"],
        model_layer=values["layer"],
        geographic_impact=values["geographic"],
        review_layer=values["review"],
        output_root=tmp_path / "published",
        archive_output_dir=tmp_path / "archives",
        producer_commit="1" * 40,
        created_at="2026-07-18T12:04:00+10:00",
        registry_version="butterflies-v2-20260712",
    )
    tampered_manifest = deepcopy(published.manifest)
    tampered_manifest["run_id"] = "changed-run"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_published_butterflylens_pool_handoff(published.root, tampered_manifest)

    model_path = published.root / "artifacts/model/butterflylens_model_evidence.parquet"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="physical identity differs"):
        validate_published_butterflylens_pool_handoff(
            published.root, published.manifest
        )
