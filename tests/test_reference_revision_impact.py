from __future__ import annotations

import polars as pl
import pytest

from biominer.references.adaptive_bank_revision import revise_adaptive_support_bank
from biominer.run.reference_revision_impact import (
    REFERENCE_REVISION_IMPACT_FILE,
    calculate_reference_revision_impact,
    reference_artifact_catalog_frame,
    reference_artifact_dependencies_frame,
    validate_reference_revision_impact,
    write_reference_revision_impact,
)
from test_adaptive_bank_revision import (
    _dependencies,
    _review_with_verify_and_exclude,
    _support_manifest,
)
from test_targeted_reference_review import SHA_A, SHA_B, SHA_C


def _revision_and_graph():  # noqa: ANN202
    inputs, review = _review_with_verify_and_exclude()
    queue = inputs[0]
    revision = revise_adaptive_support_bank(
        _support_manifest(queue),
        review,
        _dependencies(queue),
    )
    media_ids = queue["reference_media_id"].to_list()
    catalog = reference_artifact_catalog_frame(
        [
            {
                "artifact_id": "candidate:qld",
                "artifact_type": "candidate_set",
                "artifact_fingerprint": SHA_A,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio demoleus"],
                "routes": ["adult_field"],
                "region": "geo:qld",
                "record_ids": [],
            },
            {
                "artifact_id": "classifier:target",
                "artifact_type": "classifier",
                "artifact_fingerprint": SHA_B,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio demoleus"],
                "routes": ["adult_field"],
                "region": None,
                "record_ids": [],
            },
            {
                "artifact_id": "embedding:a",
                "artifact_type": "reference_embedding",
                "artifact_fingerprint": SHA_A,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio demoleus"],
                "routes": ["adult_field"],
                "region": None,
                "record_ids": [],
            },
            {
                "artifact_id": "embedding:unflagged",
                "artifact_type": "reference_embedding",
                "artifact_fingerprint": SHA_B,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio machaon"],
                "routes": ["adult_field"],
                "region": None,
                "record_ids": [],
            },
            {
                "artifact_id": "flickr:qld",
                "artifact_type": "flickr_score_partition",
                "artifact_fingerprint": SHA_C,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio demoleus"],
                "routes": ["adult_field"],
                "region": "geo:qld",
                "record_ids": ["flickr:1", "flickr:2"],
            },
            {
                "artifact_id": "prototype:target",
                "artifact_type": "regional_prototype",
                "artifact_fingerprint": SHA_C,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio demoleus"],
                "routes": ["adult_field"],
                "region": "geo:qld",
                "record_ids": [],
            },
            {
                "artifact_id": "unrelated:model",
                "artifact_type": "classifier",
                "artifact_fingerprint": SHA_A,
                "reference_bank_fingerprint": SHA_C,
                "species": ["Papilio machaon"],
                "routes": ["adult_field"],
                "region": None,
                "record_ids": [],
            },
        ]
    )
    edges = reference_artifact_dependencies_frame(
        [
            {
                "upstream_artifact_id": "prototype:target",
                "downstream_artifact_id": "candidate:qld",
                "dependency_kind": "prototype_candidate_index",
            },
            {
                "upstream_artifact_id": "prototype:target",
                "downstream_artifact_id": "classifier:target",
                "dependency_kind": "prototype_classifier_training",
            },
            {
                "upstream_artifact_id": "candidate:qld",
                "downstream_artifact_id": "flickr:qld",
                "dependency_kind": "candidate_union_scoring",
            },
            {
                "upstream_artifact_id": "classifier:target",
                "downstream_artifact_id": "flickr:qld",
                "dependency_kind": "classifier_scoring",
            },
            {
                "upstream_artifact_id": "embedding:unflagged",
                "downstream_artifact_id": "unrelated:model",
                "dependency_kind": "unrelated_training",
            },
        ]
    )
    return revision, catalog, edges, media_ids


def test_impact_propagates_and_persists_unaffected_reuse(tmp_path) -> None:
    revision, catalog, edges, _media_ids = _revision_and_graph()

    impact = calculate_reference_revision_impact(revision, catalog, edges)

    by_id = {
        str(row["artifact_id"]): row for row in impact.iter_rows(named=True)
    }
    assert by_id["prototype:target"]["directly_affected"] is True
    assert by_id["prototype:target"]["impact_depth"] == 0
    assert by_id["classifier:target"]["impact_depth"] == 1
    assert by_id["candidate:qld"]["impact_depth"] == 1
    assert by_id["flickr:qld"]["impact_depth"] == 2
    assert by_id["flickr:qld"]["affected_record_ids"] == [
        "flickr:1",
        "flickr:2",
    ]
    assert by_id["flickr:qld"]["expected_action"] == (
        "selectively_rescore_affected_records"
    )
    assert by_id["embedding:unflagged"]["impact_status"] == "reusable_as_is"
    assert by_id["unrelated:model"]["impact_status"] == "reusable_as_is"
    path = write_reference_revision_impact(impact, tmp_path)
    assert path.name == REFERENCE_REVISION_IMPACT_FILE
    assert pl.read_parquet(path).equals(impact)


def test_impact_carries_species_routes_regions_and_changed_references() -> None:
    revision, catalog, edges, _media_ids = _revision_and_graph()

    row = calculate_reference_revision_impact(revision, catalog, edges).filter(
        pl.col("artifact_id") == "flickr:qld"
    ).row(0, named=True)

    assert row["affected_species"] == ["Papilio demoleus"]
    assert "adult_field" in row["affected_routes"]
    assert "pinned_specimen" in row["affected_routes"]
    assert row["affected_regions"] == ["geo:qld"]
    assert len(row["affected_reference_media_ids"]) == 2


def test_dependency_cycles_fail_closed() -> None:
    revision, catalog, edges, _media_ids = _revision_and_graph()
    cycle = reference_artifact_dependencies_frame(
        [
            *edges.to_dicts(),
            {
                "upstream_artifact_id": "flickr:qld",
                "downstream_artifact_id": "prototype:target",
                "dependency_kind": "invalid_cycle",
            },
        ]
    )

    with pytest.raises(ValueError, match="contains a cycle"):
        calculate_reference_revision_impact(revision, catalog, cycle)


def test_impact_validator_rejects_tampering() -> None:
    revision, catalog, edges, _media_ids = _revision_and_graph()
    impact = calculate_reference_revision_impact(revision, catalog, edges)
    tampered = impact.with_columns(
        pl.when(pl.col("artifact_id") == "classifier:target")
        .then(pl.lit("reusable_as_is"))
        .otherwise(pl.col("impact_status"))
        .alias("impact_status")
    )

    with pytest.raises(ValueError, match="fingerprint mismatch|contains impact"):
        validate_reference_revision_impact(tampered)


def test_dependency_fingerprint_tampering_is_rejected() -> None:
    _revision, _catalog, edges, _media_ids = _revision_and_graph()
    tampered = edges.with_columns(
        pl.when(pl.col("upstream_artifact_id") == "prototype:target")
        .then(pl.lit("tampered:upstream"))
        .otherwise(pl.col("upstream_artifact_id"))
        .alias("upstream_artifact_id")
    )

    with pytest.raises(ValueError, match="dependency fingerprint mismatch"):
        reference_artifact_dependencies_frame(tampered.to_dicts())
