from __future__ import annotations

from datetime import UTC, datetime

import pytest

from biominer.detection.cropper import crop_with_padding
from biominer.detection.detector_base import DecodedImage, DetectionCandidate, FakeObjectDetector
from biominer.detection.evaluate import evaluate_xie_style, iou_xyxy, joint_detection_species_correct
from biominer.detection.pipeline import run_detection_pipeline
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy
from biominer.detection.schema import build_detection_rows, detection_id_for


def _image() -> DecodedImage:
    pixels = bytes(
        value
        for y in range(4)
        for x in range(4)
        for value in ((x * 40) % 256, (y * 40) % 256, ((x + y) * 20) % 256)
    )
    return DecodedImage(width=4, height=4, mode="RGB", data=pixels, source_uri="memory://edge")


def test_detection_policy_defaults_match_object_pipeline_profile() -> None:
    policy = DetectionPolicy()
    run_policy = DetectionRunPolicy()

    assert policy.backend == "yolo"
    assert policy.box_score_threshold == 0.20
    assert policy.nms_iou_threshold == 0.50
    assert policy.max_boxes_per_image == 8
    assert policy.crop_target_px == 336
    assert run_policy.download_workers == 4
    assert run_policy.detector_workers == 1
    assert run_policy.max_inflight_images == 32
    assert run_policy.crop_batch_size == 24


def test_detection_rows_keep_join_keys_and_stable_detection_id() -> None:
    image = _image()
    record = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "source_record_hash": "sha256:source",
        "image_url": "https://live.staticflickr.com/photo-1.jpg",
        "photo_page_url": "https://www.flickr.com/photos/u/photo-1",
    }
    candidate = DetectionCandidate(label="butterfly", score=0.91, bbox_xyxy=(0.5, 0.5, 3.5, 3.5), objectness_score=0.88)

    rows = build_detection_rows(
        record=record,
        image=image,
        detections=[candidate],
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="v1",
        detector_checkpoint="checkpoint-a",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "flickr"
    assert row["flickr_photo_id"] == "photo-1"
    assert row["bbox_xyxy"] == [0.5, 0.5, 3.5, 3.5]
    assert row["bbox_xyxyn"] == [0.125, 0.125, 0.875, 0.875]
    assert row["bbox_xywhn"] == [0.5, 0.5, 0.75, 0.75]
    assert row["box_area_ratio"] == pytest.approx(0.5625)
    assert row["detection_status"] == "detected"
    assert row["failure_reason"] is None
    assert row["detection_id"] == detection_id_for(
        source="flickr",
        flickr_photo_id="photo-1",
        detector_checkpoint="checkpoint-a",
        bbox_xyxyn=row["bbox_xyxyn"],
        detector_label="butterfly",
    )


def test_detection_rows_write_image_level_failure_when_no_objects_are_found() -> None:
    rows = build_detection_rows(
        record={"source": "flickr", "flickr_photo_id": "photo-2", "image_url": "https://example.test/2.jpg"},
        image=_image(),
        detections=[],
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="v1",
        detector_checkpoint="checkpoint-a",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(rows) == 1
    assert rows[0]["detection_status"] == "no_detection"
    assert rows[0]["failure_reason"] == "no_butterfly_like_object"
    assert rows[0]["detection_id"].startswith("sha256:")
    assert rows[0]["source"] == "flickr"
    assert rows[0]["flickr_photo_id"] == "photo-2"


def test_cropper_clamps_edge_bbox_adds_padding_and_hashes_deterministically() -> None:
    crop = crop_with_padding(_image(), bbox_xyxy=(-1.0, -1.0, 2.0, 2.0), padding_ratio=0.25, target_px=3)
    same = crop_with_padding(_image(), bbox_xyxy=(-1.0, -1.0, 2.0, 2.0), padding_ratio=0.25, target_px=3)

    assert crop.crop_width == 3
    assert crop.crop_height == 3
    assert crop.clamped_bbox_xyxy == [0.0, 0.0, 2.0, 2.0]
    assert crop.padded_bbox_xyxy == [0.0, 0.0, 2.75, 2.75]
    assert crop.crop_hash == same.crop_hash
    assert crop.storage_policy == "ephemeral"
    assert len(crop.encoded_bytes) == 3 * 3 * 3


def test_fake_detector_returns_multiple_rows_for_one_photo() -> None:
    detector = FakeObjectDetector(
        detections=[
            [DetectionCandidate(label="butterfly", score=0.9, bbox_xyxy=(0, 0, 2, 2))],
            [
                DetectionCandidate(label="butterfly", score=0.8, bbox_xyxy=(1, 1, 3, 3)),
                DetectionCandidate(label="life_stage", score=0.7, bbox_xyxy=(0, 2, 2, 4)),
            ],
        ]
    )

    detections = detector.detect_batch([_image(), _image()])

    assert detector.backend == "fake"
    assert [len(batch) for batch in detections] == [1, 2]
    assert detections[1][1].label == "life_stage"


def test_detection_pipeline_writes_ephemeral_crop_metadata_for_each_detection(tmp_path) -> None:
    output = tmp_path / "object_detections.parquet"
    records = [
        {
            "source": "flickr",
            "flickr_photo_id": "photo-1",
            "source_record_hash": "sha256:source-1",
            "image_url": "memory://photo-1",
            "photo_page_url": "https://www.flickr.com/photos/u/photo-1",
        }
    ]
    detector = FakeObjectDetector(
        [
            [
                DetectionCandidate(label="butterfly", score=0.9, bbox_xyxy=(0, 0, 3, 3)),
                DetectionCandidate(label="butterfly", score=0.8, bbox_xyxy=(1, 1, 4, 4)),
            ]
        ]
    )

    result = run_detection_pipeline(
        records=records,
        detector=detector,
        output_path=output,
        image_loader=lambda record: _image(),
        detection_policy=DetectionPolicy(backend="fake", crop_padding_ratio=0.25, crop_target_px=3),
    )

    rows = result.frame.sort("detector_score", descending=True).to_dicts()
    assert output.exists()
    assert result.records_seen == 1
    assert result.images_loaded == 1
    assert result.detections_written == 2
    assert result.crops_created == 2
    assert all(row["source"] == "flickr" and row["flickr_photo_id"] == "photo-1" for row in rows)
    assert all(row["crop_hash"].startswith("sha256:") for row in rows)
    assert all(row["crop_width"] == 3 and row["crop_height"] == 3 for row in rows)
    assert all(row["crop_padding_ratio"] == 0.25 for row in rows)
    assert all(row["crop_storage_policy"] == "ephemeral" for row in rows)
    assert "encoded_bytes" not in result.frame.columns
    assert len({row["detection_id"] for row in rows}) == 2
    assert len({row["crop_hash"] for row in rows}) == 2


def test_detection_pipeline_uses_bounded_map_buffersize(tmp_path) -> None:
    calls: list[int | None] = []

    class RecordingExecutor:
        def __init__(self, max_workers):  # noqa: ANN001 - mirrors executor constructor.
            self.max_workers = max_workers

        def __enter__(self):  # noqa: ANN204 - mirrors executor context manager.
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204 - mirrors executor context manager.
            return None

        def map(self, fn, iterable, *, buffersize=None):  # noqa: ANN001, ANN202 - mirrors Executor.map.
            calls.append(buffersize)
            return [fn(item) for item in iterable]

    run_detection_pipeline(
        records=[
            {"source": "flickr", "flickr_photo_id": "photo-1", "image_url": "memory://photo-1"},
            {"source": "flickr", "flickr_photo_id": "photo-2", "image_url": "memory://photo-2"},
        ],
        detector=FakeObjectDetector([[DetectionCandidate(label="butterfly", score=0.9, bbox_xyxy=(0, 0, 2, 2))], []]),
        output_path=tmp_path / "object_detections.parquet",
        image_loader=lambda record: _image(),
        run_policy=DetectionRunPolicy(download_workers=2, max_inflight_images=7),
        executor_factory=RecordingExecutor,
    )

    assert calls == [7]


def test_xie_style_evaluation_uses_iou_and_species_correctness() -> None:
    assert iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert joint_detection_species_correct(
        prediction={"bbox_xyxy": [0, 0, 10, 10], "species_top1_scientific_name": "Danaus plexippus", "species_top1_score": 0.91},
        truth={"bbox_xyxy": [1, 1, 9, 9], "scientific_name": "Danaus plexippus"},
        iou_threshold=0.5,
        score_threshold=0.35,
    )
    assert not joint_detection_species_correct(
        prediction={"bbox_xyxy": [0, 0, 10, 10], "species_top1_scientific_name": "Danaus gilippus", "species_top1_score": 0.91},
        truth={"bbox_xyxy": [1, 1, 9, 9], "scientific_name": "Danaus plexippus"},
        iou_threshold=0.5,
        score_threshold=0.35,
    )

    report = evaluate_xie_style(
        predictions=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [0, 0, 10, 10],
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top5": ["Danaus plexippus", "Danaus gilippus"],
                "family_top3": ["Nymphalidae"],
                "genus_top8": ["Danaus"],
                "species_top1_score": 0.9,
            },
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [20, 20, 30, 30],
                "species_top1_scientific_name": "Danaus gilippus",
                "species_top5": ["Danaus gilippus"],
                "family_top3": ["Nymphalidae"],
                "genus_top8": ["Danaus"],
                "species_top1_score": 0.8,
            },
        ],
        ground_truth=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [1, 1, 9, 9],
                "scientific_name": "Danaus plexippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
            }
        ],
    )

    assert report["ground_truth_available"] is True
    assert report["species_top1_accuracy"] == pytest.approx(1.0)
    assert report["species_top5_accuracy"] == pytest.approx(1.0)
    assert report["family_top3_accuracy"] == pytest.approx(1.0)
    assert report["genus_top8_accuracy"] == pytest.approx(1.0)
    assert report["joint_map50"] == pytest.approx(1.0)


def test_xie_style_detector_metrics_use_detector_scores_and_ap50_95() -> None:
    report = evaluate_xie_style(
        predictions=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [40, 40, 50, 50],
                "detector_score": 0.99,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top5": ["Danaus plexippus"],
                "species_top1_score": 0.2,
            },
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [0, 0, 10, 10],
                "detector_score": 0.40,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top5": ["Danaus plexippus"],
                "species_top1_score": 0.9,
            },
        ],
        ground_truth=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [1, 1, 9, 9],
                "scientific_name": "Danaus plexippus",
            }
        ],
    )

    assert report["detector_ap50"] == pytest.approx(0.5)
    assert report["detector_ap50_95"] is not None
    assert report["detector_ap50_95"] == pytest.approx(0.15)


def test_xie_style_evaluation_without_ground_truth_reports_qa_only() -> None:
    report = evaluate_xie_style(
        predictions=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [0, 0, 10, 10],
                "detector_score": 0.90,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top1_score": 0.95,
            }
        ],
        ground_truth=None,
    )

    assert report["ground_truth_available"] is False
    assert report["predictions_seen"] == 1
    assert report["detector_ap50"] is None
    assert report["detector_ap50_95"] is None
    assert report["joint_map50"] is None
