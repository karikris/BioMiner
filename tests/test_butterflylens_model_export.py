"""Tests for the ButterflyLens model-evidence anti-corruption export."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.integration.butterflylens_model_export import (
    BUTTERFLYLENS_MODEL_EVIDENCE_VERSION,
    BUTTERFLYLENS_MODEL_ROLES,
    build_butterflylens_model_layer,
    export_butterflylens_model_evidence,
    validate_butterflylens_model_export,
    validate_butterflylens_project_projection,
    validate_butterflylens_run_projection,
)
from biominer.integration.butterflylens_pool_handoff import (
    BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES,
    BUTTERFLYLENS_ROLE_DEFAULTS,
    build_butterflylens_pool_handoff,
    validate_butterflylens_pool_handoff,
)
from helpers.butterflylens_handoff_fixture import (
    build_butterflylens_model_fixture,
    sha,
)


def _unavailable_artifacts() -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for role in BUTTERFLYLENS_REQUIRED_ARTIFACT_ROLES:
        if role in BUTTERFLYLENS_MODEL_ROLES:
            continue
        filename, schema_version = BUTTERFLYLENS_ROLE_DEFAULTS[role]
        artifacts.append(
            {
                "role": role,
                "availability": "unavailable",
                "unavailable_reason": f"{role} has not been produced",
                "relative_path": None,
                "media_type": (
                    "application/vnd.apache.parquet"
                    if filename.endswith(".parquet")
                    else "application/json"
                ),
                "schema_version": schema_version,
                "semantic_fingerprint": None,
                "sha256": None,
                "byte_count": None,
                "row_count": None,
                "parent_fingerprints": [],
                "evidence_maturity_label": None,
            }
        )
    return artifacts


def test_project_and_run_projections_are_target_shaped_and_source_bound() -> None:
    fixture = build_butterflylens_model_fixture()
    project = fixture["project"]
    run = fixture["run"]

    validate_butterflylens_project_projection(project)
    validate_butterflylens_run_projection(run)

    assert project["target_schema_version"] == "butterflylens-project:v1.0.0"
    assert project["geographic_scope"]["country_code"] == "AU"
    assert project["scientific_claim_allowed"] is False
    assert run["target_schema_version"] == "butterflylens-run:v1.0.0"
    assert run["project_fingerprint"] == project["project_fingerprint"]
    assert run["requested_by"] == {"actor_type": "system", "actor_id": None}
    assert run["database_primary_key_included"] is False
    assert "database_id" not in json.dumps({"project": project, "run": run})


def test_model_layer_preserves_identity_lineage_and_unfinished_authority() -> None:
    fixture = build_butterflylens_model_fixture()
    layer = fixture["layer"]

    assert layer.flickr_source_records.height == 1
    assert layer.media_objects.height == 1
    assert layer.model_evidence.height == 2
    source = layer.flickr_source_records.row(0, named=True)
    media = layer.media_objects.row(0, named=True)
    evidence = layer.model_evidence.row(0, named=True)

    assert media["flickr_record_id"] == source["flickr_record_id"]
    assert evidence["flickr_record_id"] == source["flickr_record_id"]
    assert evidence["media_object_id"] == media["media_object_id"]
    assert evidence["schema_version"] == BUTTERFLYLENS_MODEL_EVIDENCE_VERSION
    assert evidence["model_id"] == "bioclip-2.5"
    assert evidence["model_revision"] == "revision-1"
    assert evidence["model_weights_sha256"] == sha("5")
    assert evidence["preprocessing_fingerprint"] == sha("7")
    assert evidence["output_content_sha256"] == sha("f")
    assert evidence["probability_availability"] == "unavailable"
    assert evidence["calibrated_probability"] is None
    assert evidence["raw_score_is_probability"] is False
    assert evidence["human_verified"] is False
    assert evidence["occurrence_release_authorized"] is False
    assert media["media_payload_included"] is False
    assert media["storage_location_included"] is False


def test_export_is_deterministic_create_only_and_round_trips(tmp_path: Path) -> None:
    fixture = build_butterflylens_model_fixture()
    first = export_butterflylens_model_evidence(
        project=fixture["project"],
        run=fixture["run"],
        layer=fixture["layer"],
        output_root=tmp_path / "first",
    )
    second = export_butterflylens_model_evidence(
        project=fixture["project"],
        run=fixture["run"],
        layer=fixture["layer"],
        output_root=tmp_path / "second",
    )

    validate_butterflylens_model_export(first.root, first.artifacts)
    assert [row["role"] for row in first.artifacts] == list(BUTTERFLYLENS_MODEL_ROLES)
    assert [row["sha256"] for row in first.artifacts] == [
        row["sha256"] for row in second.artifacts
    ]
    assert all(
        str(row["relative_path"]).startswith("artifacts/model/")
        for row in first.artifacts
    )
    assert (
        next(row for row in first.artifacts if row["role"] == "model_evidence")[
            "evidence_maturity_label"
        ]
        == "provisional_raw_score"
    )
    with pytest.raises(FileExistsError, match="create-only"):
        export_butterflylens_model_evidence(
            project=fixture["project"],
            run=fixture["run"],
            layer=fixture["layer"],
            output_root=first.root,
        )


def test_export_validation_rejects_content_and_lineage_tampering(
    tmp_path: Path,
) -> None:
    fixture = build_butterflylens_model_fixture()
    exported = export_butterflylens_model_evidence(
        project=fixture["project"],
        run=fixture["run"],
        layer=fixture["layer"],
        output_root=tmp_path,
    )
    descriptors = deepcopy(exported.artifacts)
    descriptors[0]["parent_fingerprints"] = []
    with pytest.raises(ValueError, match="lineage differs"):
        validate_butterflylens_model_export(exported.root, descriptors)

    model_path = exported.model_directory / "butterflylens_model_evidence.parquet"
    model = pl.read_parquet(model_path).with_columns(
        pl.lit(True).alias("occurrence_release_authorized")
    )
    model.write_parquet(model_path)
    with pytest.raises(ValueError, match="physical identity differs"):
        validate_butterflylens_model_export(exported.root, exported.artifacts)


def test_input_boundary_rejects_missing_extra_and_zero_byte_source() -> None:
    fixture = build_butterflylens_model_fixture()
    common = {
        "project": fixture["project"],
        "run": fixture["run"],
        "candidate_scores": fixture["candidate_scores"],
        "pool_plans": fixture["pool_plans"],
        "source_score_artifact_sha256": sha("f"),
    }
    with pytest.raises(ValueError, match="must be nonempty"):
        build_butterflylens_model_layer(source_media_records=[], **common)

    extra = deepcopy(fixture["source_media_records"])
    extra[0]["api_key"] = "must-not-cross-boundary"
    with pytest.raises(ValueError, match="input fields differ"):
        build_butterflylens_model_layer(source_media_records=extra, **common)

    empty_media = deepcopy(fixture["source_media_records"])
    empty_media[0]["media_byte_count"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        build_butterflylens_model_layer(source_media_records=empty_media, **common)


def test_model_export_combines_with_fail_closed_pool_manifest(tmp_path: Path) -> None:
    fixture = build_butterflylens_model_fixture()
    exported = export_butterflylens_model_evidence(
        project=fixture["project"],
        run=fixture["run"],
        layer=fixture["layer"],
        output_root=tmp_path,
    )
    manifest = build_butterflylens_pool_handoff(
        producer_commit="1" * 40,
        created_at="2026-07-18T12:02:00+10:00",
        project_id=fixture["project"]["project_id"],
        run_id=fixture["run"]["run_id"],
        registry_version="butterflies-v2-20260712",
        source_snapshot_fingerprints=[sha("9")],
        model_fingerprint=sha("6"),
        preprocessing_fingerprint=sha("7"),
        artifacts=[*exported.artifacts, *_unavailable_artifacts()],
    )

    validate_butterflylens_pool_handoff(manifest)
    assert manifest["release_state"]["release_ready"] is False
    assert manifest["release_state"]["scientific_claim_allowed"] is False
    assert manifest["authority_boundary"]["biominer_database_writes_allowed"] is False


def test_exported_frames_exclude_payload_credentials_and_database_keys() -> None:
    layer = build_butterflylens_model_fixture()["layer"]

    excluded = {
        "api_key",
        "service_role_key",
        "reviewer_id",
        "database_id",
        "storage_object_key",
        "media_bytes",
    }
    for frame in (
        layer.flickr_source_records,
        layer.media_objects,
        layer.model_evidence,
    ):
        assert excluded.isdisjoint(frame.columns)
