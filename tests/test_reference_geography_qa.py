"""Cross-artifact QA tests for the geographic reference index."""

from __future__ import annotations

import json

import polars as pl
import pytest

from biominer.bioclip.geographic_reference_neighbours import (
    build_geographic_reference_neighbours,
)
from biominer.bioclip.global_reference_anchors import (
    select_global_reference_anchors,
)
from biominer.bioclip.reference_geography_index import (
    build_reference_geography_index,
)
from biominer.bioclip.reference_geography_qa import (
    REFERENCE_GEOGRAPHY_INDEX_MANIFEST_FILE,
    REFERENCE_GEOGRAPHY_INDEX_MANIFEST_SCHEMA_VERSION,
    build_reference_geography_index_manifest,
    validate_reference_geography_index_manifest,
    write_reference_geography_index_manifest,
)
from support.reference_geography_fixtures import (
    FixtureGrid,
    complete_artifacts,
    index_row,
    normalized,
    observation,
    sha,
)


PRODUCER_SHA = "e4245a6d652aca8ff20a198d957897d1c91c00fc"


def _manifest(
    *,
    observations: tuple[dict[str, object], ...] | None = None,
    physical_sha256s: dict[str, str] | None = None,
) -> dict[str, object]:
    selected = observations if observations is not None else (observation(),)
    index, geography, anchors, neighbours = complete_artifacts(*selected)
    return build_reference_geography_index_manifest(
        index,
        geography,
        anchors,
        neighbours,
        producer_git_sha=PRODUCER_SHA,
        physical_sha256s=physical_sha256s,
    )


def test_passing_manifest_reports_independence_precision_and_fallback_metrics() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == (
        REFERENCE_GEOGRAPHY_INDEX_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["qa_status"] == "passed"
    assert manifest["fatal_finding_count"] == 0
    assert manifest["findings"] == []
    assert manifest["source_snapshot_versions"] == ["gbif-occurrence-2026-07-18"]
    metrics = manifest["metrics"]
    assert metrics["reference_media_count"] == 1
    assert metrics["reference_embedding_count"] == 1
    assert metrics["reference_observation_count"] == 1
    assert metrics["reference_duplicate_group_count"] == 1
    assert metrics["global_anchor_observation_inflation_count"] == 0
    assert metrics["global_anchor_duplicate_group_inflation_count"] == 0
    assert metrics["false_local_membership_count"] == 0
    assert metrics["precision_violation_count"] == 0
    assert metrics["missing_exact_membership_count"] == 0
    assert metrics["missing_fallback_membership_count"] == 0
    assert metrics["unexpected_global_membership_count"] == 0
    assert metrics["lookup_independence_counting_unit"] == (
        "distinct_reference_observation_id"
    )
    assert metrics["neighbour_scope_counts"] == {
        "bioregion": 1,
        "continent": 1,
        "country": 1,
        "exact_supported_cell": 1,
        "global": 1,
        "neighbouring_supported_cell": 2,
        "parent_coarse_cell": 1,
        "parent_regional_cell": 1,
    }


def test_manifest_is_deterministic_for_canonical_equivalent_inputs() -> None:
    first = observation("1")
    second = observation(
        "2",
        latitude=-37.81,
        longitude=144.96,
        geo_cluster_id="cluster-vic",
    )

    forward = _manifest(observations=(first, second))
    reverse = _manifest(observations=(second, first))

    assert forward == reverse
    assert forward["manifest_fingerprint"] == reverse["manifest_fingerprint"]


def test_manifest_reports_physical_checksum_availability_without_inventing_it() -> None:
    unavailable = _manifest()
    assert unavailable["physical_checksum_status"] == "unavailable_not_supplied"
    assert all(
        record["physical_sha256"] is None
        for record in unavailable["artifacts"].values()
    )

    checksums = {
        "normalized_reference_geography": sha("1"),
        "reference_geography_index": sha("2"),
        "global_reference_anchors": sha("3"),
        "geographic_reference_neighbours": sha("4"),
    }
    complete = _manifest(physical_sha256s=checksums)
    assert complete["physical_checksum_status"] == "complete"
    assert {
        name: record["physical_sha256"]
        for name, record in complete["artifacts"].items()
    } == checksums


def test_write_round_trip_uses_required_filename_and_canonical_json(tmp_path) -> None:
    manifest = _manifest()
    path = write_reference_geography_index_manifest(manifest, tmp_path / "index")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == REFERENCE_GEOGRAPHY_INDEX_MANIFEST_FILE
    assert loaded == manifest
    assert path.read_text(encoding="utf-8").endswith("\n")
    validate_reference_geography_index_manifest(loaded)


def test_audit_fails_closed_on_stale_anchor_and_neighbour_lineage() -> None:
    geography = normalized(observation())
    geography_row = geography.row(0, named=True)
    original_index, _, original_anchors, original_neighbours = complete_artifacts(
        observation()
    )
    expanded_index = build_reference_geography_index(
        [
            index_row(geography_row, "1"),
            index_row(
                geography_row,
                "2",
                reference_media_id=f"reference-media:{'2' * 64}",
                visual_input_kind="focused_full_frame",
                embedding_fingerprint=sha("2"),
            ),
        ]
    )

    manifest = build_reference_geography_index_manifest(
        expanded_index,
        geography,
        original_anchors,
        original_neighbours,
        producer_git_sha=PRODUCER_SHA,
    )

    assert original_index.height == 1
    assert manifest["qa_status"] == "failed"
    assert {finding["code"] for finding in manifest["findings"]} >= {
        "global_anchor_index_lineage_mismatch",
        "neighbour_input_lineage_mismatch",
    }


@pytest.mark.parametrize(
    ("scope", "expected_code"),
    [
        ("exact_supported_cell", "local_reference_exact_membership_missing"),
        ("country", "named_fallback_membership_missing"),
        ("global", "global_fallback_membership_missing"),
    ],
)
def test_audit_detects_missing_required_fallback_provenance(
    scope: str,
    expected_code: str,
) -> None:
    index, geography, anchors, neighbours = complete_artifacts(observation())
    incomplete = neighbours.filter(pl.col("lookup_scope") != scope)

    manifest = build_reference_geography_index_manifest(
        index,
        geography,
        anchors,
        incomplete,
        producer_git_sha=PRODUCER_SHA,
    )

    assert manifest["qa_status"] == "failed"
    assert expected_code in {finding["code"] for finding in manifest["findings"]}


def test_audit_reports_invalid_artifact_contract_instead_of_claiming_success() -> None:
    index, geography, anchors, neighbours = complete_artifacts(observation())
    tampered = neighbours.with_columns(pl.lit(7).cast(pl.UInt8).alias("fallback_level"))

    manifest = build_reference_geography_index_manifest(
        index,
        geography,
        anchors,
        tampered,
        producer_git_sha=PRODUCER_SHA,
    )

    assert manifest["qa_status"] == "failed"
    assert manifest["metrics"]["cross_artifact_audit_status"] == (
        "unavailable_invalid_contract"
    )
    assert (
        manifest["artifacts"]["geographic_reference_neighbours"]["semantic_fingerprint"]
        is None
    )
    assert any(
        finding["code"] == "artifact_contract_invalid"
        and finding["subject"] == "geographic_reference_neighbours"
        for finding in manifest["findings"]
    )


def test_multiple_embeddings_are_reported_but_not_counted_as_independent() -> None:
    geography = normalized(observation())
    row = geography.row(0, named=True)
    index = build_reference_geography_index(
        [
            index_row(row, "1"),
            index_row(
                row,
                "2",
                reference_media_id=f"reference-media:{'2' * 64}",
                visual_input_kind="focused_full_frame",
                embedding_fingerprint=sha("2"),
            ),
        ]
    )
    anchors = select_global_reference_anchors(index)
    neighbours = build_geographic_reference_neighbours(
        index, geography, anchors, grid=FixtureGrid()
    )

    manifest = build_reference_geography_index_manifest(
        index,
        geography,
        anchors,
        neighbours,
        producer_git_sha=PRODUCER_SHA,
    )

    metrics = manifest["metrics"]
    assert manifest["qa_status"] == "passed"
    assert metrics["reference_embedding_count"] == 2
    assert metrics["reference_observation_count"] == 1
    assert metrics["global_anchor_observation_inflation_count"] == 0
    assert metrics["maximum_embedding_rows_per_observation_lookup"] == 2
    assert metrics["lookup_independence_counting_unit"] == (
        "distinct_reference_observation_id"
    )


def test_empty_artifacts_have_reproducible_passing_audit() -> None:
    index, geography, anchors, neighbours = complete_artifacts()

    manifest = build_reference_geography_index_manifest(
        index,
        geography,
        anchors,
        neighbours,
        producer_git_sha=PRODUCER_SHA,
    )

    assert manifest["qa_status"] == "passed"
    assert manifest["source_snapshot_versions"] == []
    assert manifest["metrics"]["reference_observation_count"] == 0
    assert manifest["metrics"]["neighbour_membership_count"] == 0


def test_manifest_validator_rejects_fingerprint_and_status_tampering() -> None:
    manifest = _manifest()
    tampered_fingerprint = {**manifest, "qa_policy_fingerprint": sha("f")}
    with pytest.raises(ValueError, match="policy fingerprint drifted"):
        validate_reference_geography_index_manifest(tampered_fingerprint)

    tampered_status = {**manifest, "qa_status": "failed"}
    with pytest.raises(ValueError, match="QA status conflicts"):
        validate_reference_geography_index_manifest(tampered_status)


@pytest.mark.parametrize("producer", ["abc", "A" * 40, "1" * 39, None])
def test_rejects_invalid_producer_git_sha(producer: object) -> None:
    index, geography, anchors, neighbours = complete_artifacts()
    with pytest.raises(ValueError, match="40-character Git SHA"):
        build_reference_geography_index_manifest(
            index,
            geography,
            anchors,
            neighbours,
            producer_git_sha=producer,  # type: ignore[arg-type]
        )


def test_rejects_unknown_or_invalid_physical_checksums() -> None:
    with pytest.raises(ValueError, match="unknown artifacts"):
        _manifest(physical_sha256s={"other": sha("1")})
    with pytest.raises(ValueError, match="physical checksum"):
        _manifest(physical_sha256s={"reference_geography_index": "abc"})
