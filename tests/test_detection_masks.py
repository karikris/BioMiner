from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.policy import DetectionPolicy
from biominer.detection.schema import (
    DETECTION_OUTPUT_SCHEMA,
    build_detection_rows,
    empty_detection_frame,
)
from biominer.detection.yolo26_detector import (
    _candidate_to_payload as yolo26_candidate_to_payload,
    _detections_from_payload as yolo26_detections_from_payload,
    detections_from_yolo26_result,
)
from biominer.detection.yoloe26_detector import (
    _candidate_to_payload as yoloe26_candidate_to_payload,
    _detections_from_payload as yoloe26_detections_from_payload,
    detections_from_yoloe_result,
)


def _candidate(*, mask_polygon_xyn=None) -> DetectionCandidate:  # noqa: ANN001
    return DetectionCandidate(
        label="butterfly_like",
        score=0.91,
        bbox_xyxy=(1.0, 2.0, 9.0, 8.0),
        objectness_score=0.88,
        mask_polygon_xyn=mask_polygon_xyn,
    )


def test_detection_candidate_canonicalizes_mask_polygon_as_immutable_normalized_metadata() -> (
    None
):
    mutable_polygon = [[0.1000004, 0.2], [0.9, 0.2], [0.5, 0.8]]

    candidate = _candidate(mask_polygon_xyn=mutable_polygon)
    mutable_polygon[0][0] = 0.7

    assert candidate.mask_polygon_xyn == ((0.1, 0.2), (0.9, 0.2), (0.5, 0.8))
    with pytest.raises(FrozenInstanceError):
        candidate.mask_polygon_xyn = None  # type: ignore[misc]


@pytest.mark.parametrize(
    "polygon",
    (
        [[0.1, 0.2], [0.9, 0.2]],
        [[-0.1, 0.2], [0.9, 0.2], [0.5, 0.8]],
        [[0.1, 0.2], [0.9, float("inf")], [0.5, 0.8]],
    ),
)
def test_detection_candidate_rejects_invalid_normalized_mask_polygon(polygon) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="mask_polygon_xyn"):
        _candidate(mask_polygon_xyn=polygon)


def test_detection_rows_preserve_nullable_mask_polygon_in_original_canvas_coordinates() -> (
    None
):
    image = DecodedImage(width=10, height=10, mode="RGB", data=b"\xff" * 300)
    record = {
        "source": "flickr",
        "flickr_photo_id": "photo-mask",
        "image_url": "https://example.test/photo-mask.jpg",
    }
    policy = DetectionPolicy(backend="fake", min_box_area_ratio=0.0)
    polygon = ((0.1, 0.2), (0.9, 0.2), (0.5, 0.8))

    masked = build_detection_rows(
        record=record,
        image=image,
        detections=[_candidate(mask_polygon_xyn=polygon)],
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="v1",
        detector_checkpoint="checkpoint-a",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        policy=policy,
    )
    unmasked = build_detection_rows(
        record={**record, "flickr_photo_id": "photo-no-mask"},
        image=image,
        detections=[_candidate()],
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="v1",
        detector_checkpoint="checkpoint-a",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        policy=policy,
    )

    assert masked[0]["bbox_xyxyn"] == [0.1, 0.2, 0.9, 0.8]
    assert masked[0]["mask_polygon_xyn"] == [[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]]
    assert unmasked[0]["mask_polygon_xyn"] is None
    assert empty_detection_frame().schema["mask_polygon_xyn"] == pl.List(
        pl.List(pl.Float64)
    )
    assert pl.DataFrame(masked, schema=DETECTION_OUTPUT_SCHEMA)[
        "mask_polygon_xyn"
    ].to_list() == [[[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]]]


def test_yoloe26_result_conversion_preserves_box_mask_instance_alignment() -> None:
    result = SimpleNamespace(
        names={0: "butterfly", 1: "moth"},
        boxes=[
            SimpleNamespace(xyxy=[[1.0, 2.0, 9.0, 8.0]], cls=[0], conf=[0.91]),
            SimpleNamespace(xyxy=[[2.0, 1.0, 8.0, 9.0]], cls=[1], conf=[0.82]),
        ],
        masks=SimpleNamespace(
            xyn=[
                [[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]],
                [[0.2, 0.1], [0.8, 0.1], [0.5, 0.9]],
            ]
        ),
    )

    detections = detections_from_yoloe_result(result)

    assert detections[0].label == "adult_butterfly"
    assert detections[0].mask_polygon_xyn == ((0.1, 0.2), (0.9, 0.2), (0.5, 0.8))
    assert detections[1].label == "moth_like"
    assert detections[1].mask_polygon_xyn == ((0.2, 0.1), (0.8, 0.1), (0.5, 0.9))


def test_yolo26_result_conversion_rejects_misaligned_box_and_mask_counts() -> None:
    result = SimpleNamespace(
        names={0: "butterfly"},
        boxes=[SimpleNamespace(xyxy=[[1.0, 2.0, 9.0, 8.0]], cls=[0], conf=[0.91])],
        masks=SimpleNamespace(xyn=[]),
    )

    with pytest.raises(ValueError, match="align one-to-one"):
        detections_from_yolo26_result(result)


@pytest.mark.parametrize(
    ("to_payload", "from_payload"),
    (
        (yolo26_candidate_to_payload, yolo26_detections_from_payload),
        (yoloe26_candidate_to_payload, yoloe26_detections_from_payload),
    ),
)
def test_detector_sidecar_round_trip_preserves_mask_polygon_order(
    to_payload, from_payload
) -> None:  # noqa: ANN001
    candidate = _candidate(mask_polygon_xyn=((0.4, 0.1), (0.8, 0.7), (0.2, 0.9)))

    restored = from_payload({"detections": [[to_payload(candidate)]]})[0][0]

    assert restored.mask_polygon_xyn == ((0.4, 0.1), (0.8, 0.7), (0.2, 0.9))
