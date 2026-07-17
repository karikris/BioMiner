from __future__ import annotations

import polars as pl
import pytest

from biominer.run.reference_model_rebuild import (
    REFERENCE_MODEL_REBUILD_PLAN_FILE,
    calculate_reference_model_rebuild_plan,
    calibrator_training_change_frame,
    reference_model_artifact_ids_to_rebuild,
    reference_model_rebuild_metrics,
    validate_reference_model_rebuild_plan,
    write_reference_model_rebuild_plan,
)
from biominer.run.reference_revision_impact import (
    calculate_reference_revision_impact,
    reference_artifact_catalog_frame,
    reference_artifact_dependencies_frame,
)
from test_reference_revision_impact import _revision_and_graph
from test_targeted_reference_review import SHA_A, SHA_B, SHA_C


def _impact_with_reference_models() -> pl.DataFrame:
    revision, catalog, edges, _media_ids = _revision_and_graph()
    catalog = reference_artifact_catalog_frame(
        [
            *catalog.to_dicts(),
            {
                "artifact_id": "calibrator:changed",
                "artifact_type": "calibrator",
                "artifact_fingerprint": SHA_A,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio demoleus"],
                "routes": ["adult_field"],
                "region": "geo:qld",
                "record_ids": [],
            },
            {
                "artifact_id": "calibrator:stable",
                "artifact_type": "calibrator",
                "artifact_fingerprint": SHA_B,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio demoleus"],
                "routes": ["adult_field"],
                "region": "geo:qld",
                "record_ids": [],
            },
            {
                "artifact_id": "prototype:global",
                "artifact_type": "reference_prototype",
                "artifact_fingerprint": SHA_C,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio demoleus"],
                "routes": ["adult_field"],
                "region": None,
                "record_ids": [],
            },
        ]
    )
    edges = reference_artifact_dependencies_frame(
        [
            *edges.to_dicts(),
            {
                "upstream_artifact_id": "embedding:a",
                "downstream_artifact_id": "prototype:global",
                "dependency_kind": "embedding_prototype",
            },
            {
                "upstream_artifact_id": "classifier:target",
                "downstream_artifact_id": "calibrator:changed",
                "dependency_kind": "classifier_calibration_predictions",
            },
            {
                "upstream_artifact_id": "classifier:target",
                "downstream_artifact_id": "calibrator:stable",
                "dependency_kind": "classifier_calibration_predictions",
            },
        ]
    )
    return calculate_reference_revision_impact(revision, catalog, edges)


def _training_changes() -> pl.DataFrame:
    return calibrator_training_change_frame(
        [
            {
                "artifact_id": "calibrator:changed",
                "old_training_data_fingerprint": SHA_A,
                "new_training_data_fingerprint": SHA_B,
            },
            {
                "artifact_id": "calibrator:stable",
                "old_training_data_fingerprint": SHA_C,
                "new_training_data_fingerprint": SHA_C,
            },
        ]
    )


def test_rebuild_plan_selects_only_affected_model_inputs(tmp_path) -> None:
    plan = calculate_reference_model_rebuild_plan(
        _impact_with_reference_models(),
        _training_changes(),
    )
    by_id = {
        str(row["artifact_id"]): row for row in plan.iter_rows(named=True)
    }

    assert by_id["prototype:global"]["rebuild_action"] == (
        "rebuild_affected_species_prototypes"
    )
    assert by_id["prototype:target"]["rebuild_action"] == (
        "rebuild_affected_regional_prototypes"
    )
    assert by_id["candidate:qld"]["rebuild_action"] == (
        "rebuild_affected_regional_candidate_index"
    )
    assert by_id["classifier:target"]["rebuild_action"] == (
        "rebuild_affected_classifier_rows_and_model"
    )
    assert by_id["calibrator:changed"]["rebuild_status"] == "rebuild"
    assert by_id["calibrator:stable"]["rebuild_status"] == "reuse"
    assert by_id["calibrator:stable"]["rebuild_action"] == (
        "reuse_calibrator_training_data_unchanged"
    )
    assert by_id["unrelated:model"]["rebuild_action"] == (
        "reuse_unchanged_artifact"
    )
    assert reference_model_artifact_ids_to_rebuild(
        plan,
        artifact_type="calibrator",
    ) == ("calibrator:changed",)
    assert reference_model_artifact_ids_to_rebuild(
        plan,
        artifact_type="candidate_set",
    ) == ("candidate:qld",)
    assert by_id["candidate:qld"]["affected_regions"] == ["geo:qld"]
    assert "Papilio demoleus" in by_id["classifier:target"][
        "affected_species"
    ]
    metrics = reference_model_rebuild_metrics(plan)
    assert metrics["artifact_count"].sum() == plan.height
    output = write_reference_model_rebuild_plan(plan, tmp_path)
    assert output.name == REFERENCE_MODEL_REBUILD_PLAN_FILE
    assert pl.read_parquet(output).equals(plan)


def test_affected_calibrator_requires_training_change_evidence() -> None:
    with pytest.raises(ValueError, match="lack training-data change evidence"):
        calculate_reference_model_rebuild_plan(
            _impact_with_reference_models(),
            calibrator_training_change_frame(),
        )


def test_training_change_evidence_must_name_a_known_calibrator() -> None:
    changes = calibrator_training_change_frame(
        [
            *_training_changes().to_dicts(),
            {
                "artifact_id": "calibrator:unknown",
                "old_training_data_fingerprint": SHA_A,
                "new_training_data_fingerprint": SHA_B,
            },
        ]
    )

    with pytest.raises(ValueError, match="unknown calibrators"):
        calculate_reference_model_rebuild_plan(
            _impact_with_reference_models(),
            changes,
        )


def test_rebuild_plan_rejects_tampered_action() -> None:
    plan = calculate_reference_model_rebuild_plan(
        _impact_with_reference_models(),
        _training_changes(),
    )
    tampered = plan.with_columns(
        pl.when(pl.col("artifact_id") == "classifier:target")
        .then(pl.lit("reuse_unchanged_artifact"))
        .otherwise(pl.col("rebuild_action"))
        .alias("rebuild_action")
    )

    with pytest.raises(ValueError, match="action mismatch|plan_fingerprint"):
        validate_reference_model_rebuild_plan(tampered)
