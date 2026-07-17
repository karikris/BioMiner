from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Sequence

import polars as pl
import pytest

from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy
from biominer.references.yoloe_routing import (
    REFERENCE_YOLOE_ROUTE_SCHEMA,
    run_reference_yoloe_routing,
)
from biominer.vision.full_frame_attention import (
    AttentionQualityPolicy,
    FullFrameTransformPolicy,
)


class RecordingDetector:
    backend = "fake-yoloe"
    model_id = "yoloe-reference-test"
    model_version = "2026.07"
    checkpoint = "sha256:" + "d" * 64
    prompt_set_fingerprint = "sha256:" + "e" * 64

    def __init__(self, outputs: Sequence[Sequence[DetectionCandidate]]) -> None:
        self.outputs = [list(items) for items in outputs]
        self.images_seen = 0

    def detect_batch(
        self, images: Sequence[DecodedImage]
    ) -> list[list[DetectionCandidate]]:
        start = self.images_seen
        self.images_seen += len(images)
        return self.outputs[start : self.images_seen]


def test_reference_yoloe_routes_life_stages_and_excludes_bad_domains(
    tmp_path: Path,
) -> None:
    records = [_record(index) for index in range(5)]
    before = deepcopy(records)
    detector = RecordingDetector(
        [
            [_candidate("adult_butterfly", "butterfly")],
            [_candidate("caterpillar", "caterpillar")],
            [_candidate("pinned_specimen", "pinned butterfly specimen")],
            [_candidate("artifact", "butterfly logo")],
            [],
        ]
    )

    result = run_reference_yoloe_routing(
        records=records,
        detector=detector,
        output_dir=tmp_path,
        image_loader=lambda _row: _image(),
        detection_policy=DetectionPolicy(backend=detector.backend),
        run_policy=DetectionRunPolicy(detector_batch_size=8),
    )

    assert records == before
    assert detector.images_seen == len(records)
    assert result.records_seen == result.images_loaded == 5
    assert result.image_failures == 0
    assert result.routes.schema == REFERENCE_YOLOE_ROUTE_SCHEMA
    assert result.detections_path.exists()
    assert result.routes_path.exists()
    assert pl.read_parquet(result.routes_path).equals(result.routes)
    assert result.detections["crop_storage_policy"].unique().to_list() == [
        "not_created"
    ]

    rows = {
        row["reference_media_id"]: row for row in result.routes.to_dicts()
    }
    assert rows[_media_id(0)]["route"] == "adult_field"
    assert rows[_media_id(0)]["provisional_life_stage"] == "adult"
    assert rows[_media_id(1)]["route"] == "larval"
    assert rows[_media_id(1)]["provisional_life_stage"] == "larva"
    assert rows[_media_id(2)]["route"] == "pinned_specimen"
    assert rows[_media_id(2)]["provisional_visual_domain"] == "pinned_specimen"
    assert rows[_media_id(3)]["routing_action"] == "exclude"
    assert rows[_media_id(3)]["provisional_visual_domain"] == "logo"
    assert "artifact_detected" in rows[_media_id(3)]["domain_flags"]
    assert rows[_media_id(4)]["routing_action"] == "exclude"
    assert "no_relevant_organism_detected" in rows[_media_id(4)]["domain_flags"]


def test_routing_persists_detector_area_and_shared_full_frame_policy(
    tmp_path: Path,
) -> None:
    detector = RecordingDetector(
        [[_candidate("adult_butterfly", "butterfly", bbox=(10, 10, 30, 30))]]
    )

    result = run_reference_yoloe_routing(
        records=[_record(1)],
        detector=detector,
        output_dir=tmp_path,
        image_loader=lambda _row: _image(),
        detection_policy=DetectionPolicy(backend=detector.backend),
    )
    row = result.routes.row(0, named=True)

    assert row["detector_backend"] == detector.backend
    assert row["detector_model_id"] == detector.model_id
    assert row["detector_model_version"] == detector.model_version
    assert row["detector_checkpoint"] == detector.checkpoint
    assert row["detector_prompt_set_fingerprint"] == detector.prompt_set_fingerprint
    assert row["subject_area_ratio"] == pytest.approx(0.04)
    assert row["attention_transform_policy_fingerprint"] == (
        FullFrameTransformPolicy().fingerprint
    )
    assert row["attention_quality_policy_fingerprint"] == (
        AttentionQualityPolicy().fingerprint
    )
    assert row["full_frame_input_generation_succeeded"] is True
    assert row["raw_visual_input_id"].startswith("sha256:")
    assert row["route_evidence_fingerprint"].startswith("sha256:")
    assert row["species_identity_decision"] == "not_assessed_by_yoloe"
    assert not any(
        "species" in column and column != "species_identity_decision"
        for column in result.routes.columns
    )


def test_one_reference_can_persist_separate_adult_larval_and_specimen_routes(
    tmp_path: Path,
) -> None:
    detector = RecordingDetector(
        [
            [
                _candidate("adult_butterfly", "butterfly", bbox=(0, 0, 20, 20)),
                _candidate("caterpillar", "caterpillar", bbox=(30, 0, 50, 20)),
                _candidate(
                    "pinned_specimen",
                    "pinned butterfly specimen",
                    bbox=(60, 0, 80, 20),
                ),
            ]
        ]
    )

    result = run_reference_yoloe_routing(
        records=[_record(2)],
        detector=detector,
        output_dir=tmp_path,
        image_loader=lambda _row: _image(),
        detection_policy=DetectionPolicy(backend=detector.backend),
    )

    assert result.routes["route"].to_list() == [
        "adult_field",
        "larval",
        "pinned_specimen",
    ]
    assert set(result.routes["routing_action"].to_list()) == {"review"}
    assert all(
        "multiple_biological_routes" in flags
        for flags in result.routes["domain_flags"].to_list()
    )


def test_mixed_biological_and_artifact_evidence_cannot_score_automatically(
    tmp_path: Path,
) -> None:
    detector = RecordingDetector(
        [
            [
                _candidate("adult_butterfly", "butterfly"),
                _candidate("artifact", "painting", bbox=(50, 50, 80, 80)),
            ]
        ]
    )

    result = run_reference_yoloe_routing(
        records=[_record(3)],
        detector=detector,
        output_dir=tmp_path,
        image_loader=lambda _row: _image(),
        detection_policy=DetectionPolicy(backend=detector.backend),
    )
    row = result.routes.row(0, named=True)

    assert row["route"] == "adult_field"
    assert row["routing_action"] == "review"
    assert row["provisional_visual_domain"] == "ambiguous"
    assert "mixed_biological_and_excluded_domain_evidence" in row["domain_flags"]


def _record(index: int) -> dict[str, object]:
    return {
        "reference_media_id": _media_id(index),
        "source": "gbif",
        "source_record_hash": "sha256:" + f"{index + 1:x}" * 64,
        "source_object_uri": f"memory://reference/{index}",
        "source_record_url": f"https://example.test/occurrence/{index}",
    }


def _media_id(index: int) -> str:
    return "reference-media:" + f"{index + 1:x}" * 64


def _image() -> DecodedImage:
    return DecodedImage(
        width=100,
        height=100,
        mode="RGB",
        data=bytes([120, 80, 40]) * 10_000,
        source_uri="memory://reference-image",
    )


def _candidate(
    label: str,
    prompt: str,
    *,
    bbox: tuple[float, float, float, float] = (10, 10, 40, 40),
) -> DetectionCandidate:
    return DetectionCandidate(
        label=label,
        score=0.95,
        bbox_xyxy=bbox,
        detector_prompt=prompt,
    )
