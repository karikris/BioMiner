from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.detection.detector_base import (
    DecodedImage,
    DetectionCandidate,
    FakeObjectDetector,
)
from biominer.references.admission import default_reference_admission_policy
from biominer.references.admission_compiler import (
    REFERENCE_ADMISSION_DECISION_SCHEMA,
    REFERENCE_PROVISIONAL_SUPPORT_SCHEMA,
    compile_provisional_gbif_support_bank,
    validate_reference_provisional_support,
)
from biominer.references.deduplication import _duplicate_group_id
from biominer.references.provisional_selection import (
    PROVISIONAL_SELECTION_CANDIDATE_SCHEMA,
    ProvisionalSelectionPolicy,
    select_independent_provisional_support,
)
from biominer.references.schemas import (
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_observation_id,
    reference_media_candidates_frame,
    reference_media_duplicate_relationships_frame,
    reference_media_objects_frame,
    reference_observations_frame,
)
from biominer.references.yoloe_routing import run_reference_yoloe_routing


NOW = datetime(2026, 7, 17, 2, 0, tzinfo=UTC)


def test_compiler_emits_five_provenance_complete_artifacts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    result = compile_provisional_gbif_support_bank(
        **inputs,
        admission_policy=default_reference_admission_policy(),
        prototype_scope="regional",
        output_dir=tmp_path / "compiled",
        created_at=NOW,
    )

    assert result.decisions.schema == REFERENCE_ADMISSION_DECISION_SCHEMA
    assert result.provisional_support.schema == REFERENCE_PROVISIONAL_SUPPORT_SCHEMA
    assert result.decisions.height == 3
    assert result.provisional_support.height == 1
    assert result.report["counts"] == {
        "candidates": 3,
        "admitted": 2,
        "review_required": 0,
        "excluded": 1,
        "selected": 1,
        "provisional_support": 1,
    }
    for path in (
        result.decisions_path,
        result.provisional_support_path,
        result.summary_path,
        result.report_path,
        result.markdown_path,
    ):
        assert path.exists()
    assert pl.read_parquet(result.decisions_path).equals(result.decisions)
    assert pl.read_parquet(result.provisional_support_path).equals(
        result.provisional_support
    )
    assert json.loads(result.report_path.read_text(encoding="utf-8")) == result.report
    assert result.markdown_path.read_text(encoding="utf-8") == result.markdown


def test_decisions_separate_admission_selection_and_provisional_support(
    tmp_path: Path,
) -> None:
    result = compile_provisional_gbif_support_bank(
        **_inputs(tmp_path),
        admission_policy=default_reference_admission_policy(),
        prototype_scope="regional",
        output_dir=tmp_path / "compiled",
        created_at=NOW,
    )
    rows = {
        row["reference_media_id"]: row for row in result.decisions.to_dicts()
    }

    selected = rows[_media_id(1)]
    assert selected["admission_decision"] == "admitted"
    assert selected["selection_decision"] == "selected"
    assert selected["provisional_support"] is True
    assert selected["provisional_status"] == "provisional_admitted_selected"
    assert len(selected["automated_gate_ids"]) == 24
    assert set(selected["automated_gate_dispositions"]) == {"passed"}

    quota_skip = rows[_media_id(2)]
    assert quota_skip["admission_decision"] == "admitted"
    assert quota_skip["selection_decision"] == "skipped"
    assert quota_skip["selection_reason"] == "species_quota_reached"
    assert quota_skip["provisional_support"] is False
    assert quota_skip["provisional_status"] == (
        "provisional_admitted_not_selected"
    )

    absent = rows[_media_id(3)]
    assert absent["admission_decision"] == "excluded"
    assert "occurrence_is_absent" in absent["admission_reason_codes"]
    assert absent["provisional_support"] is False


def test_support_rows_carry_source_provider_qa_route_duplicate_and_policy_evidence(
    tmp_path: Path,
) -> None:
    policy = default_reference_admission_policy()
    result = compile_provisional_gbif_support_bank(
        **_inputs(tmp_path),
        admission_policy=policy,
        prototype_scope="regional",
        output_dir=tmp_path / "compiled",
        created_at=NOW,
    )
    row = result.provisional_support.row(0, named=True)

    assert row["source"] == "gbif"
    assert row["source_record_hash"].startswith("sha256:")
    assert row["provider_assertion_status"] == "provider_asserted_unreviewed"
    assert row["provider_assertion_identity_basis"] == "gbif_provider_asserted"
    assert row["human_verified"] is False
    assert row["admission_decision"] == "admitted"
    assert row["admission_reason_codes"] == []
    assert len(row["automated_gate_ids"]) == 24
    assert set(row["automated_gate_dispositions"]) == {"passed"}
    assert row["route"] == "adult_field"
    assert row["subject_area_ratio"] == pytest.approx(0.25)
    assert row["duplicate_type"] == "unique"
    assert row["canonical_reference_media_id"] == row["reference_media_id"]
    assert row["duplicate_resolution_status"] == "resolved"
    assert row["reference_admission_policy_fingerprint"] == policy.fingerprint
    assert row["route_evidence_fingerprint"].startswith("sha256:")
    assert row["eligibility_result_fingerprint"].startswith("sha256:")
    assert row["selection_decision_fingerprint"].startswith("sha256:")
    assert row["support_row_fingerprint"].startswith("sha256:")
    validate_reference_provisional_support(result.provisional_support)

    tampered = result.provisional_support.with_columns(
        pl.lit("failed").alias("admission_decision")
    )
    with pytest.raises(ValueError, match="semantics are inconsistent"):
        validate_reference_provisional_support(tampered)


def test_compiler_rejects_stale_selection_after_admission_policy_change(
    tmp_path: Path,
) -> None:
    stricter = replace(
        default_reference_admission_policy(), minimum_subject_area_ratio=0.30
    )

    with pytest.raises(ValueError, match="selection admission decision is stale"):
        compile_provisional_gbif_support_bank(
            **_inputs(tmp_path),
            admission_policy=stricter,
            prototype_scope="regional",
            output_dir=tmp_path / "compiled",
            created_at=NOW,
        )


def test_compiler_rejects_tampered_route_and_incomplete_coverage(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    routes = inputs["yoloe_routes"].with_columns(
        pl.when(pl.col("reference_media_id") == _media_id(1))
        .then(pl.lit("tampered"))
        .otherwise(pl.col("routing_reason"))
        .alias("routing_reason")
    )
    with pytest.raises(ValueError, match="route evidence fingerprint mismatch"):
        compile_provisional_gbif_support_bank(
            **{**inputs, "yoloe_routes": routes},
            admission_policy=default_reference_admission_policy(),
            prototype_scope="regional",
            output_dir=tmp_path / "tampered",
            created_at=NOW,
        )

    incomplete = inputs["selection_decisions"].head(2)
    with pytest.raises(ValueError, match="cover every media candidate"):
        compile_provisional_gbif_support_bank(
            **{**inputs, "selection_decisions": incomplete},
            admission_policy=default_reference_admission_policy(),
            prototype_scope="regional",
            output_dir=tmp_path / "incomplete",
            created_at=NOW,
        )


def _inputs(tmp_path: Path) -> dict[str, pl.DataFrame]:
    observation_rows = [
        _observation(index, occurrence_absent=index == 3) for index in range(1, 4)
    ]
    candidate_rows = [
        _candidate(observation, index)
        for index, observation in enumerate(observation_rows, start=1)
    ]
    object_rows = [
        _media_object(candidate, index)
        for index, candidate in enumerate(candidate_rows, start=1)
    ]
    observations = reference_observations_frame(observation_rows)
    media_candidates = reference_media_candidates_frame(candidate_rows)
    media_objects = reference_media_objects_frame(object_rows)
    relationships = reference_media_duplicate_relationships_frame([])
    route_result = run_reference_yoloe_routing(
        records=[
            {
                "reference_media_id": candidate["reference_media_id"],
                "source": "gbif",
                "source_record_hash": observation["source_record_hash"],
                "source_object_uri": media_object["source_object_uri"],
                "source_record_url": observation["source_record_url"],
            }
            for observation, candidate, media_object in zip(
                observation_rows, candidate_rows, object_rows, strict=True
            )
        ],
        detector=FakeObjectDetector(
            [
                [
                    DetectionCandidate(
                        label="adult_butterfly",
                        score=0.95,
                        bbox_xyxy=(100, 100, 400, 400),
                        detector_prompt="butterfly",
                    )
                ]
                for _ in candidate_rows
            ]
        ),
        output_dir=tmp_path / "routes",
        image_loader=lambda _row: _image(),
    )
    selection_candidates = pl.DataFrame(
        [
            {
                "reference_media_id": candidate["reference_media_id"],
                "accepted_taxon_key": observation["accepted_taxon_key"],
                "reference_observation_id": candidate[
                    "reference_observation_id"
                ],
                "observer_id": observation["observer_id"],
                "duplicate_group_id": media_object["duplicate_group_id"],
                "canonical_reference_media_id": media_object[
                    "canonical_reference_media_id"
                ],
                "route": "adult_field",
                "documented_view": "unknown",
                "distinct_view_evidence": "none",
                "quality_score": 1.0 - index / 10,
                "admission_decision": (
                    "excluded" if observation["occurrence_absent"] else "admitted"
                ),
            }
            for index, (observation, candidate, media_object) in enumerate(
                zip(observation_rows, candidate_rows, object_rows, strict=True),
                start=1,
            )
        ],
        schema=PROVISIONAL_SELECTION_CANDIDATE_SCHEMA,
    )
    selection = select_independent_provisional_support(
        selection_candidates,
        output_dir=tmp_path / "selection",
        policy=ProvisionalSelectionPolicy(quota_per_species=1),
    )
    return {
        "observations": observations,
        "media_candidates": media_candidates,
        "media_objects": media_objects,
        "duplicate_relationships": relationships,
        "yoloe_routes": route_result.routes,
        "selection_decisions": selection.decisions,
    }


def _observation(index: int, *, occurrence_absent: bool) -> dict[str, object]:
    observation_id = make_reference_observation_id("gbif", f"occurrence-{index}")
    return {
        "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        "reference_observation_id": observation_id,
        "source": "gbif",
        "source_observation_id": f"occurrence-{index}",
        "source_taxon_id": "1938069",
        "supplied_scientific_name": "Papilio demoleus",
        "accepted_taxon_key": "gbif:1938069",
        "reconciled_scientific_name": "Papilio demoleus",
        "registry_version": "butterflies-v2-20260717",
        "taxon_reconciliation_status": "accepted_key_exact",
        "identification_quality": "research_grade",
        "community_taxon_status": "species",
        "identification_disagreement": False,
        "captive_or_cultivated": False,
        "observer_id": f"observer-{index}",
        "locality": "Sydney",
        "life_stage": "adult",
        "sex": None,
        "observed_at": datetime(2025, 1, index, tzinfo=UTC),
        "latitude": -33.87,
        "longitude": 151.21,
        "coordinate_uncertainty": 50.0,
        "coordinates_obscured": False,
        "country": "Australia",
        "country_code": "AU",
        "geo_cluster_id": "cluster-au",
        "distance_to_cluster_medoid_km": 4.2,
        "source_dataset_key": "dataset-gbif",
        "source_dataset_doi": None,
        "source_record_url": f"https://example.test/gbif/{index}",
        "source_record_hash": _sha(f"record:{index}"),
        "retrieved_at": NOW,
        "source_snapshot_version": "gbif-2026-07-17",
        "source_query_fingerprint": _sha("query:gbif"),
        "fallback_level": 0,
        "geospatial_issue": False,
        "preserved_specimen": False,
        "fossil": False,
        "occurrence_absent": occurrence_absent,
        "uncertain_taxon_match": False,
        "basis_of_record_suitable": True,
    }


def _candidate(
    observation: dict[str, object], index: int
) -> dict[str, object]:
    media_id = make_reference_media_id(
        "gbif", f"media-{index}", str(observation["reference_observation_id"])
    )
    return {
        "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "reference_observation_id": observation["reference_observation_id"],
        "provider_media_id": f"media-{index}",
        "source": "gbif",
        "media_identifier": f"https://media.example.test/{index}.jpg",
        "media_type": "StillImage",
        "width": 600,
        "height": 600,
        "creator": f"Observer {index}",
        "rights_holder": f"Observer {index}",
        "licence": "CC-BY-4.0",
        "licence_uri": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": f"Observer {index} / CC BY 4.0",
        "occurrence_licence": "CC0-1.0",
        "original_provider": "gbif",
        "media_position": 0,
        "source_checksum": None,
        "source_checksum_algorithm": None,
        "download_status": "complete",
        "verification_status": "unreviewed",
        "exclusion_reason": None,
        "licence_policy_status": "allowed",
        "retrieved_at": NOW,
        "source_snapshot_version": "gbif-2026-07-17",
    }


def _media_object(candidate: dict[str, object], index: int) -> dict[str, object]:
    media_id = str(candidate["reference_media_id"])
    image_sha = _sha(f"image:{index}")
    digest = image_sha.removeprefix("sha256:")
    return {
        "schema_version": REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "source_object_uri": f"s3://references/source_objects/{digest}.jpg",
        "content_type": "image/jpeg",
        "source_byte_count": 1_000_000,
        "decoded_width": 600,
        "decoded_height": 600,
        "sha256": image_sha,
        "perceptual_hash": f"dhash128-v1:{index:032x}",
        "duplicate_group_id": _duplicate_group_id([media_id]),
        "duplicate_type": "unique",
        "canonical_reference_media_id": media_id,
        "provider_mirror_ids": [],
        "downloaded_at": NOW,
        "download_attempt_count": 1,
        "licence_policy_status": "allowed",
        "decode_status": "valid",
        "quarantine_reason": None,
        "object_fingerprint": _sha(f"object:{index}"),
    }


def _image() -> DecodedImage:
    return DecodedImage(
        width=600,
        height=600,
        mode="RGB",
        data=bytes([100, 120, 140]) * 360_000,
        source_uri="memory://reference",
    )


def _media_id(index: int) -> str:
    observation_id = make_reference_observation_id("gbif", f"occurrence-{index}")
    return make_reference_media_id("gbif", f"media-{index}", observation_id)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
