from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import pytest

from biominer.detection.detector_base import (
    DecodedImage,
    DetectionCandidate,
    FakeObjectDetector,
)
from biominer.detection.pipeline import run_detection_pipeline
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
    TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    TARGET_FULL_FRAME_PREPROCESSING,
    AttentionRegion,
    generate_full_frame_attention_variants,
)
from biominer.vision.target_full_frame import (
    build_target_full_frame_plan,
    encode_target_full_frame_plan,
    generate_target_full_frame_attention_variants,
)


_MODEL_FINGERPRINT = "sha256:" + "b" * 64
_PREPROCESSING_FINGERPRINT = "sha256:" + "c" * 64


class RecordingFullFrameEncoder:
    image_resize_mode = TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    preprocessing_contract_fingerprint = TARGET_FULL_FRAME_PREPROCESSING.fingerprint
    preprocessing_fingerprint = _PREPROCESSING_FINGERPRINT

    def __init__(self) -> None:
        self.batches: list[tuple[DecodedImage, ...]] = []

    def encode_images(
        self,
        images: Sequence[DecodedImage],
    ) -> list[list[float]]:
        batch = tuple(images)
        self.batches.append(batch)
        return [[1.0, float(image.width), float(image.height)] for image in batch]


def test_full_frame_yoloe_domain_routing_is_no_crop_and_reference_symmetric(
    tmp_path,
) -> None:
    image = _image()
    detector = FakeObjectDetector(
        [
            [
                _candidate(
                    label="adult_butterfly",
                    prompt="butterfly",
                    bbox=(0.0, 0.0, 5.0, 5.0),
                    polygon=((0.0, 0.0), (0.05, 0.0), (0.05, 0.05), (0.0, 0.05)),
                ),
                _candidate(
                    label="adult_butterfly",
                    prompt="live adult butterfly",
                    bbox=(30.0, 30.0, 50.0, 50.0),
                    polygon=((0.3, 0.3), (0.5, 0.3), (0.5, 0.5), (0.3, 0.5)),
                ),
                _candidate(
                    label="caterpillar",
                    prompt="caterpillar",
                    bbox=(10.0, 0.0, 30.0, 20.0),
                ),
                _candidate(
                    label="pinned_specimen",
                    prompt="pinned butterfly specimen",
                    bbox=(40.0, 0.0, 60.0, 20.0),
                ),
                _candidate(
                    label="insect_like",
                    prompt="insect",
                    bbox=(70.0, 0.0, 90.0, 20.0),
                ),
                _candidate(
                    label="hard_negative",
                    prompt="flower",
                    bbox=(0.0, 30.0, 20.0, 50.0),
                ),
            ],
            [],
        ]
    )
    result = run_detection_pipeline(
        records=[
            {
                "source": "flickr",
                "flickr_photo_id": "photo-domains",
                "source_record_hash": "sha256:" + "1" * 64,
                "image_url": "memory://photo-domains",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-no-butterfly",
                "source_record_hash": "sha256:" + "2" * 64,
                "image_url": "memory://photo-no-butterfly",
            },
        ],
        detector=detector,
        output_path=tmp_path / "target_object_detections.parquet",
        image_loader=lambda _record: image,
        detection_policy=DetectionPolicy(
            backend="fake",
            min_box_area_ratio=0.0,
        ),
        run_policy=DetectionRunPolicy(detector_batch_size=2),
    )

    rows = result.frame.to_dicts()
    assert all(row["crop_hash"] is None for row in rows)
    assert "crop_path" not in result.frame.columns
    assert all(row["crop_storage_policy"] == "not_created" for row in rows)

    rows_by_prompt = {
        str(row["detector_prompt"]): row
        for row in rows
        if row["detector_prompt"] is not None
    }
    assert _route_tuple(rows_by_prompt["butterfly"]) == (
        "adult_butterfly_field",
        "score",
        "adult_field",
        "standard",
    )
    assert _route_tuple(rows_by_prompt["live adult butterfly"]) == (
        "adult_butterfly_field",
        "score",
        "adult_field",
        "standard",
    )
    assert _route_tuple(rows_by_prompt["caterpillar"]) == (
        "caterpillar_field",
        "score",
        "larval",
        "standard",
    )
    assert _route_tuple(rows_by_prompt["pinned butterfly specimen"]) == (
        "pinned_specimen",
        "score",
        "pinned_specimen",
        "standard",
    )
    assert _route_tuple(rows_by_prompt["insect"]) == (
        "ambiguous_visual_domain",
        "review",
        "adult_field",
        "low",
    )
    assert _route_tuple(rows_by_prompt["flower"]) == (
        "no_relevant_organism",
        "exclude",
        None,
        "none",
    )
    no_butterfly = next(
        row for row in rows if row["flickr_photo_id"] == "photo-no-butterfly"
    )
    assert no_butterfly["detection_status"] == "no_detection"
    assert _route_tuple(no_butterfly) == (
        "no_relevant_organism",
        "exclude",
        None,
        "none",
    )

    plan = build_target_full_frame_plan(
        detection_rows=rows,
        image_loader=lambda _row: image,
    )
    units_by_route = {unit.route: unit for unit in plan.scoring_units}
    assert set(units_by_route) == {"adult_field", "larval", "pinned_specimen"}
    assert units_by_route["adult_field"].detection_ids == tuple(
        sorted(
            (
                rows_by_prompt["butterfly"]["detection_id"],
                rows_by_prompt["live adult butterfly"]["detection_id"],
            )
        )
    )
    assert units_by_route["larval"].detection_ids == (
        rows_by_prompt["caterpillar"]["detection_id"],
    )
    assert units_by_route["pinned_specimen"].detection_ids == (
        rows_by_prompt["pinned butterfly specimen"]["detection_id"],
    )
    assert (
        rows_by_prompt["caterpillar"]["detection_id"]
        not in units_by_route["adult_field"].detection_ids
    )
    automatically_scored_ids = {
        detection_id
        for unit in plan.scoring_units
        for detection_id in unit.detection_ids
    }
    assert rows_by_prompt["insect"]["detection_id"] not in automatically_scored_ids
    assert rows_by_prompt["flower"]["detection_id"] not in automatically_scored_ids
    assert no_butterfly["detection_id"] not in automatically_scored_ids

    encoder = RecordingFullFrameEncoder()
    embedded = encode_target_full_frame_plan(
        plan,
        encoder=encoder,
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
    )
    assert encoder.batches == [(image,)]
    assert len(embedded.embeddings) == 1
    assert len(embedded.scoring_unit_references) == 3
    assert (
        len({reference.embedding_id for reference in embedded.scoring_unit_references})
        == 1
    )

    adult_unit = units_by_route["adult_field"]
    query = generate_target_full_frame_attention_variants(
        plan,
        scoring_unit_id=adult_unit.scoring_unit_id,
    )
    focused_evidence = next(
        evidence
        for evidence in query.evidence
        if evidence.visual_input_kind == FOCUSED_FULL_FRAME_KIND
        and evidence.source_detection_ids
        == (rows_by_prompt["butterfly"]["detection_id"],)
    )
    assert focused_evidence.subject_area_ratio == pytest.approx(0.0025)
    assert "small_subject" in focused_evidence.visual_input_quality_flags
    assert focused_evidence.quality_policy_fingerprint.startswith("sha256:")

    reference = generate_full_frame_attention_variants(
        replace(image, source_uri="reference://adult"),
        tuple(
            AttentionRegion(
                source_detection_id=detection.detection_id,
                route=detection.route,
                bbox_xyxyn=detection.bbox_xyxyn,
                mask_polygon_xyn=detection.mask_polygon_xyn,
                detector_score=detection.detector_score,
            )
            for detection in adult_unit.detections
        ),
        source_type="reference",
        source_record_id="reference-adult",
    )
    assert {variant.visual_input_kind for variant in query.variants} == {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
        MULTI_OBJECT_FULL_FRAME_KIND,
    }
    assert _visual_signatures(reference) == _visual_signatures(query)
    assert {evidence.source_type for evidence in reference.evidence} == {"reference"}
    assert {evidence.source_type for evidence in query.evidence} == {"flickr"}
    assert [evidence.evidence_id for evidence in reference.evidence] != [
        evidence.evidence_id for evidence in query.evidence
    ]


def _image() -> DecodedImage:
    pixels = bytes(
        channel
        for y in range(100)
        for x in range(100)
        for channel in (x % 256, y % 256, (x + y) % 256)
    )
    return DecodedImage(
        width=100,
        height=100,
        mode="RGB",
        data=pixels,
        source_uri="memory://domain-routing",
    )


def _candidate(
    *,
    label: str,
    prompt: str,
    bbox: tuple[float, float, float, float],
    polygon: tuple[tuple[float, float], ...] | None = None,
) -> DetectionCandidate:
    return DetectionCandidate(
        label=label,
        score=0.9,
        bbox_xyxy=bbox,
        detector_prompt=prompt,
        mask_polygon_xyn=polygon,
    )


def _route_tuple(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["detection_route"],
        row["routing_action"],
        row["bioclip_route"],
        row["routing_priority"],
    )


def _visual_signatures(result) -> tuple[tuple[object, ...], ...]:  # noqa: ANN001
    return tuple(
        (
            variant.visual_input_kind,
            variant.visual_input_id,
            variant.visual_content_hash,
            variant.transformation_applied,
            variant.transformation_version,
            variant.transformation_fingerprint,
            variant.width,
            variant.height,
            variant.image.data,
        )
        for variant in result.variants
    )
