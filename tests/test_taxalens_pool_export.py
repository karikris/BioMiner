"""Tests for immutable TaxaLens score and pool evidence export."""

from __future__ import annotations

from copy import deepcopy

import polars as pl
import pytest

from biominer.integration.taxalens_pool_export import (
    TAXALENS_SCORE_POOL_ROLES,
    export_taxalens_score_pool_evidence,
    validate_taxalens_score_pool_export,
)
from biominer.integration.taxalens_pool_handoff import TAXALENS_ROLE_DEFAULTS
from tests.helpers.dynamic_pool_handoff_fixture import (
    build_dynamic_pool_handoff_fixture,
)


def test_export_writes_six_canonical_fingerprinted_artifacts(tmp_path) -> None:
    frames = build_dynamic_pool_handoff_fixture()

    exported = export_taxalens_score_pool_evidence(
        **frames,
        output_root=tmp_path / "handoff",
    )

    assert [row["role"] for row in exported.artifacts] == list(
        TAXALENS_SCORE_POOL_ROLES
    )
    assert exported.artifact_directory == tmp_path / "handoff" / "artifacts"
    for descriptor in exported.artifacts:
        role = descriptor["role"]
        filename, schema_version = TAXALENS_ROLE_DEFAULTS[role]
        path = exported.artifact_directory / filename
        assert path.is_file()
        assert descriptor["schema_version"] == schema_version
        assert descriptor["semantic_fingerprint"].startswith("sha256:")
        assert descriptor["sha256"].startswith("sha256:")
        assert descriptor["byte_count"] == path.stat().st_size
        assert descriptor["row_count"] == frames[role].height
        assert "scientific_claim_allowed" not in descriptor
    validate_taxalens_score_pool_export(exported.root, exported.artifacts)


def test_export_preserves_raw_score_and_pool_maturity(tmp_path) -> None:
    exported = export_taxalens_score_pool_evidence(
        **build_dynamic_pool_handoff_fixture(),
        output_root=tmp_path / "handoff",
    )
    by_role = {row["role"]: row for row in exported.artifacts}

    assert by_role["candidate_scores"]["evidence_maturity_label"] == (
        "provisional_raw_score"
    )
    assert by_role["pool_members"]["evidence_maturity_label"] == (
        "provider_asserted_provisional_support"
    )
    scores = pl.read_parquet(
        exported.artifact_directory / TAXALENS_ROLE_DEFAULTS["candidate_scores"][0]
    )
    assert scores["probability_availability"].unique().to_list() == ["unavailable"]
    assert scores["calibrated_probability"].null_count() == scores.height
    assert scores["human_review_required"].to_list() == [True, True]


def test_export_bytes_and_descriptors_are_deterministic(tmp_path) -> None:
    frames = build_dynamic_pool_handoff_fixture()
    first = export_taxalens_score_pool_evidence(
        **frames,
        output_root=tmp_path / "first",
    )
    second = export_taxalens_score_pool_evidence(
        **frames,
        output_root=tmp_path / "second",
    )

    assert first.artifacts == second.artifacts


def test_export_is_create_only_and_does_not_replace_existing_evidence(tmp_path) -> None:
    root = tmp_path / "handoff"
    frames = build_dynamic_pool_handoff_fixture()
    first = export_taxalens_score_pool_evidence(**frames, output_root=root)
    checksums = {row["role"]: row["sha256"] for row in first.artifacts}

    with pytest.raises(FileExistsError, match="create-only"):
        export_taxalens_score_pool_evidence(**frames, output_root=root)

    validate_taxalens_score_pool_export(root, first.artifacts)
    assert {row["role"]: row["sha256"] for row in first.artifacts} == checksums


def test_export_rejects_empty_or_cross_artifact_mismatch_before_publication(
    tmp_path,
) -> None:
    frames = build_dynamic_pool_handoff_fixture()
    empty = dict(frames)
    empty["candidate_scores"] = frames["candidate_scores"].clear()
    with pytest.raises(ValueError, match="must not be empty"):
        export_taxalens_score_pool_evidence(
            **empty,
            output_root=tmp_path / "empty",
        )
    assert not (tmp_path / "empty" / "artifacts").exists()

    mismatched = dict(frames)
    mismatched["candidate_sets"] = frames["candidate_sets"].with_columns(
        pl.col("candidate_set_id").str.replace("family-geo", "other")
    )
    with pytest.raises(ValueError):
        export_taxalens_score_pool_evidence(
            **mismatched,
            output_root=tmp_path / "mismatched",
        )
    assert not (tmp_path / "mismatched" / "artifacts").exists()


def test_validator_detects_descriptor_and_physical_tampering(tmp_path) -> None:
    exported = export_taxalens_score_pool_evidence(
        **build_dynamic_pool_handoff_fixture(),
        output_root=tmp_path / "handoff",
    )
    tampered = deepcopy(exported.artifacts)
    tampered[0]["semantic_fingerprint"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="identity differs"):
        validate_taxalens_score_pool_export(exported.root, tampered)

    score_path = (
        exported.artifact_directory / TAXALENS_ROLE_DEFAULTS["candidate_scores"][0]
    )
    with score_path.open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ValueError, match="byte count differs"):
        validate_taxalens_score_pool_export(exported.root, exported.artifacts)
