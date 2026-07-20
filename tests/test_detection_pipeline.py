from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.detection import yoloe26_detector
from biominer.detection.detector_base import COARSE_DETECTOR_LABELS, DecodedImage, DetectionCandidate, FakeObjectDetector, normalize_detector_label
from biominer.detection.evaluate import evaluate_xie_style, iou_xyxy, joint_detection_species_correct
from biominer.detection.pipeline import run_detection_pipeline
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy
from biominer.detection.routing import DetectionRoutingPolicy
from biominer.detection.schema import build_detection_rows, detection_id_for
from biominer.detection.yoloe26_detector import YoloE26SidecarObjectDetector


def _image() -> DecodedImage:
    pixels = bytes(
        value
        for y in range(4)
        for x in range(4)
        for value in ((x * 40) % 256, (y * 40) % 256, ((x + y) * 20) % 256)
    )
    return DecodedImage(width=4, height=4, mode="RGB", data=pixels, source_uri="memory://edge")


def _wide_white_image() -> DecodedImage:
    return DecodedImage(width=4, height=2, mode="RGB", data=bytes([255, 255, 255] * 8), source_uri="memory://wide")


def test_detection_policy_defaults_match_full_frame_pipeline() -> None:
    policy = DetectionPolicy()
    run_policy = DetectionRunPolicy()

    assert policy.backend == "yoloe26"
    assert policy.box_score_threshold == 0.20
    assert policy.nms_iou_threshold == 0.50
    assert policy.max_boxes_per_image == 8
    assert policy.image_max_side_px == 1280
    assert run_policy.download_workers == 4
    assert run_policy.max_inflight_images == 32
    assert run_policy.detector_batch_size == 4
    assert run_policy.adaptive_batching is False
    assert run_policy.min_detector_batch_size == 1


def test_detection_and_run_sources_do_not_create_reviewed_box_training_artifacts() -> None:
    source_paths = (
        *sorted(Path("src/biominer/detection").glob("*.py")),
        *sorted(Path("src/biominer/run").glob("*.py")),
        Path("src/biominer/cli.py"),
    )
    forbidden_tokens = (
        "reviewed_boxes",
        "reviewed-boxes",
        "reviewed_box_dataset",
        "reviewed box dataset",
        "training_dataset",
        "training-dataset",
        "training dataset",
        "fine_tune",
        "fine-tune",
        "finetune",
        "fine tuning",
        "yolo_train",
        "train_yolo",
        "label_studio",
        "label-studio",
        "cvat",
        "/annotations",
        "annotations/",
        "/labels",
        "labels/",
    )

    violations: dict[str, list[str]] = {}
    for path in source_paths:
        text = path.read_text(encoding="utf-8").casefold()
        matches = [token for token in forbidden_tokens if token in text]
        if matches:
            violations[str(path)] = matches

    assert violations == {}


def test_detection_candidate_contract_normalizes_legacy_labels_and_rejects_taxa() -> None:
    assert {
        "adult_butterfly",
        "possible_adult_butterfly",
        "pinned_specimen",
        "artifact",
        "no_relevant_organism",
    }.issubset(COARSE_DETECTOR_LABELS)
    assert normalize_detector_label("butterfly") == "butterfly_like"
    assert normalize_detector_label("life stage") == "caterpillar"
    assert normalize_detector_label("museum label") == "artifact"
    assert DetectionCandidate(label="butterfly", score=0.9, bbox_xyxy=(0, 0, 1, 1)).label == "butterfly_like"
    with pytest.raises(ValueError, match="taxonomic"):
        DetectionCandidate(label="Papilio demoleus", score=0.9, bbox_xyxy=(0, 0, 1, 1))


def test_detection_rows_keep_join_keys_and_stable_detection_id() -> None:
    image = _image()
    record = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "source_record_hash": "sha256:source",
        "image_url": "https://live.staticflickr.com/photo-1.jpg",
        "photo_page_url": "https://www.flickr.com/photos/u/photo-1",
    }
    candidate = DetectionCandidate(
        label="adult_butterfly",
        score=0.91,
        bbox_xyxy=(0.5, 0.5, 3.5, 3.5),
        objectness_score=0.88,
        detector_prompt="butterfly",
        detector_class_id=0,
        detector_prompt_set_fingerprint="sha256:" + "a" * 64,
    )

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
    assert row["prediction_source"] == "object_detector:fake"
    assert row["bbox_xyxy"] == [0.5, 0.5, 3.5, 3.5]
    assert row["bbox_xyxyn"] == [0.125, 0.125, 0.875, 0.875]
    assert row["bbox_xywhn"] == [0.5, 0.5, 0.75, 0.75]
    assert row["box_area_ratio"] == pytest.approx(0.5625)
    assert row["detection_status"] == "detected"
    assert row["failure_reason"] is None
    assert row["detector_prompt"] == "butterfly"
    assert row["detector_class_id"] == 0
    assert row["detector_prompt_set_fingerprint"] == "sha256:" + "a" * 64
    assert row["schema_version"] == "object-detection-v2"
    assert row["detection_id"] == detection_id_for(
        source="flickr",
        flickr_photo_id="photo-1",
        detector_checkpoint="checkpoint-a",
        bbox_xyxyn=row["bbox_xyxyn"],
        detector_label="adult_butterfly",
        detector_prompt="butterfly",
    )


def test_detection_id_includes_normalized_detector_prompt() -> None:
    base = {
        "source": "flickr",
        "flickr_photo_id": "photo-prompt",
        "detector_checkpoint": "checkpoint-a",
        "bbox_xyxyn": (0.0, 0.0, 1.0, 1.0),
        "detector_label": "possible_adult_butterfly",
    }

    assert detection_id_for(**base, detector_prompt="butterfly wing") != detection_id_for(
        **base, detector_prompt="lepidoptera"
    )
    assert detection_id_for(**base, detector_prompt=" Butterfly   Wing ") == detection_id_for(
        **base, detector_prompt="butterfly wing"
    )


def test_detection_rows_apply_nms_iou_threshold_before_max_boxes() -> None:
    image = _image()
    record = {"source": "flickr", "flickr_photo_id": "photo-nms", "image_url": "https://example.test/nms.jpg"}

    rows = build_detection_rows(
        record=record,
        image=image,
        detections=[
            DetectionCandidate(label="butterfly", score=0.95, bbox_xyxy=(0, 0, 3, 3)),
            DetectionCandidate(label="butterfly", score=0.80, bbox_xyxy=(0.2, 0.2, 3.2, 3.2)),
            DetectionCandidate(label="butterfly", score=0.70, bbox_xyxy=(3, 3, 4, 4)),
        ],
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="v1",
        detector_checkpoint="checkpoint-a",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        policy=DetectionPolicy(backend="fake", nms_iou_threshold=0.5, min_box_area_ratio=0.0, max_boxes_per_image=8),
    )

    assert [row["detector_score"] for row in rows] == [0.95, 0.70]


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
    assert rows[0]["detector_label"] == "no_relevant_organism"
    assert rows[0]["failure_reason"] == "no_relevant_organism"
    assert rows[0]["detection_id"].startswith("sha256:")
    assert rows[0]["source"] == "flickr"
    assert rows[0]["flickr_photo_id"] == "photo-2"


def test_detection_pipeline_empty_input_writes_stable_schema(tmp_path) -> None:
    output = tmp_path / "object_detections.parquet"

    result = run_detection_pipeline(
        records=[],
        detector=FakeObjectDetector(),
        output_path=output,
        image_loader=lambda record: _image(),
    )

    frame = pl.read_parquet(output)
    assert result.records_seen == 0
    assert frame.height == 0
    assert {
        "source",
        "flickr_photo_id",
        "source_record_hash",
        "image_url",
        "photo_page_url",
        "detection_id",
        "detector_backend",
        "prediction_source",
        "detector_model_id",
        "detector_model_version",
        "detector_checkpoint",
        "detected_at",
        "bbox_xyxy",
        "bbox_xyxyn",
        "bbox_xywhn",
        "box_area_ratio",
        "detector_label",
        "detector_score",
        "objectness_score",
        "detector_prompt",
        "detector_class_id",
        "detector_prompt_set_fingerprint",
        "detection_route",
        "routing_action",
        "bioclip_route",
        "routing_priority",
        "routing_reason",
        "routing_policy_version",
        "routing_policy_fingerprint",
        "nms_group_id",
        "crop_padding_ratio",
        "crop_hash",
        "crop_width",
        "crop_height",
        "crop_storage_policy",
        "detection_status",
        "failure_reason",
        "schema_version",
    }.issubset(frame.columns)


def test_detection_pipeline_buffer_uses_schema_beyond_polars_inference_window(tmp_path) -> None:
    records = [
        {
            "source": "flickr",
            "flickr_photo_id": f"photo-{index:03d}",
            "image_url": f"https://example.test/{index}.jpg",
        }
        for index in range(101)
    ]
    detected = [DetectionCandidate(label="butterfly_like", score=0.9, bbox_xyxy=(0, 0, 4, 4))]
    detector = FakeObjectDetector([detected] * 100 + [[]])

    result = run_detection_pipeline(
        records=records,
        detector=detector,
        output_path=tmp_path / "mixed-null-buffer.parquet",
        image_loader=lambda _record: _image(),
        run_policy=DetectionRunPolicy(detector_batch_size=101, parquet_batch_rows=500),
    )

    assert result.frame.height == 101
    assert result.frame["failure_reason"].tail(1).item() == "no_relevant_organism"


def test_detection_rows_extra_nms_suppresses_only_within_same_route() -> None:
    rows = build_detection_rows(
        record={"source": "flickr", "flickr_photo_id": "photo-route-nms", "image_url": "memory://route-nms"},
        image=_image(),
        detections=[
            DetectionCandidate(
                label="adult_butterfly",
                score=0.95,
                bbox_xyxy=(0, 0, 3, 3),
                detector_prompt="butterfly",
            ),
            DetectionCandidate(
                label="possible_adult_butterfly",
                score=0.90,
                bbox_xyxy=(0.1, 0.1, 3.1, 3.1),
                detector_prompt="butterfly wing",
            ),
            DetectionCandidate(
                label="caterpillar",
                score=0.85,
                bbox_xyxy=(0.1, 0.1, 3.1, 3.1),
                detector_prompt="caterpillar",
            ),
        ],
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="v1",
        detector_checkpoint="checkpoint-a",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        policy=DetectionPolicy(backend="fake", nms_iou_threshold=0.5, min_box_area_ratio=0.0),
    )

    assert [row["detector_label"] for row in rows] == ["adult_butterfly", "caterpillar"]


def test_detection_rows_nms_preserves_review_over_overlapping_exclusion() -> None:
    policy = DetectionPolicy(
        backend="fake",
        nms_iou_threshold=0.5,
        min_box_area_ratio=0.0,
        routing_policy=DetectionRoutingPolicy(
            possible_adult_route_threshold=0.70,
            ambiguous_insect_review_threshold=0.30,
        ),
    )
    candidates = [
        DetectionCandidate(
            label="possible_adult_butterfly",
            score=0.60,
            bbox_xyxy=(0, 0, 3, 3),
            detector_prompt="butterfly wing",
        ),
        DetectionCandidate(
            label="insect_like",
            score=0.50,
            bbox_xyxy=(0.1, 0.1, 3.1, 3.1),
            detector_prompt="insect",
        ),
    ]

    rows = build_detection_rows(
        record={
            "source": "flickr",
            "flickr_photo_id": "photo-mixed-action-nms",
            "image_url": "memory://mixed-action-nms",
        },
        image=_image(),
        detections=candidates,
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="v1",
        detector_checkpoint="checkpoint-a",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        policy=policy,
    )

    assert [(row["detector_prompt"], row["routing_action"]) for row in rows] == [
        ("insect", "review"),
        ("butterfly wing", "exclude"),
    ]

    capped_rows = build_detection_rows(
        record={
            "source": "flickr",
            "flickr_photo_id": "photo-mixed-action-cap",
            "image_url": "memory://mixed-action-cap",
        },
        image=_image(),
        detections=candidates,
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="v1",
        detector_checkpoint="checkpoint-a",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        policy=DetectionPolicy(
            backend="fake",
            nms_iou_threshold=0.5,
            min_box_area_ratio=0.0,
            max_boxes_per_image=1,
            routing_policy=policy.routing_policy,
        ),
    )

    assert capped_rows[0]["detector_prompt"] == "insect"
    assert capped_rows[0]["routing_action"] == "review"


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
    assert detections[1][1].label == "caterpillar"


def _fake_yoloe26_popen(calls: dict[str, object], response: dict[str, object] | None = None):
    output_lines: list[str] = []
    sidecar_response = response or {
        "metadata": {
            "backend": "yoloe26",
            "model_id": "yoloe26:yoloe-26s-seg",
            "model_version": "ultralytics:test",
            "checkpoint": "yoloe-26s-seg.pt",
        },
        "detections": [[]],
    }

    class FakeStdin:
        def write(self, text: str) -> int:
            payload = json.loads(text)
            calls["payload"] = payload
            image_paths = [Path(path) for path in payload.get("image_paths", [])]
            calls["paths_existed_during_write"] = [path.exists() for path in image_paths]
            if image_paths:
                calls["ppm_prefix"] = image_paths[0].read_bytes()[: len(b"P6\n4 2\n255\n")]
            output_lines.append(json.dumps(sidecar_response, sort_keys=True) + "\n")
            return len(text)

        def flush(self) -> None:
            return None

    class FakeStdout:
        def readline(self) -> str:
            return output_lines.pop(0)

    class FakeProcess:
        stdin = FakeStdin()
        stdout = FakeStdout()
        stderr = None
        returncode = None

        def poll(self):  # noqa: ANN202
            return self.returncode

        def wait(self, timeout=None):  # noqa: ANN001, ANN202
            self.returncode = 0
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

    def fake_popen(command, **kwargs):  # noqa: ANN001, ANN003, ANN202 - mirrors subprocess.Popen.
        calls["command"] = command
        calls["env"] = kwargs["env"]
        return FakeProcess()

    return fake_popen


def test_yoloe26_sidecar_detector_serializes_rgb_images_without_importing_ultralytics(tmp_path) -> None:
    calls: dict[str, object] = {}
    output_lines: list[str] = []

    class FakeStdin:
        def write(self, text: str) -> int:
            payload = json.loads(text)
            calls["payload"] = payload
            output_lines.append(
                json.dumps(
                    {
                        "metadata": {
                            "backend": "yoloe26",
                            "model_id": "yoloe26:yoloe-26s-seg",
                            "model_version": "ultralytics:test",
                            "checkpoint": "yoloe-26s-seg.pt",
                        },
                        "detections": [
                            [
                                {
                                    "label": "butterfly",
                                    "score": 0.91,
                                    "bbox_xyxy": [0.0, 0.0, 4.0, 2.0],
                                    "objectness_score": 0.88,
                                }
                            ]
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            return len(text)

        def flush(self) -> None:
            return None

    class FakeStdout:
        def readline(self) -> str:
            return output_lines.pop(0)

    class FakeProcess:
        stdin = FakeStdin()
        stdout = FakeStdout()
        stderr = None
        returncode = None

        def poll(self):  # noqa: ANN202
            return self.returncode

        def wait(self, timeout=None):  # noqa: ANN001, ANN202
            self.returncode = 0
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

    def fake_popen(command, **kwargs):  # noqa: ANN001, ANN003, ANN202 - mirrors subprocess.Popen.
        calls["command"] = command
        calls["env"] = kwargs["env"]
        return FakeProcess()

    runtime_python = str(tmp_path / "YOLO26" / "venv" / "bin" / "python")
    detector = YoloE26SidecarObjectDetector(runtime_python=runtime_python, device="mps", popen=fake_popen)

    detections = detector.detect_batch([_wide_white_image()])

    payload = calls["payload"]
    assert calls["command"] == [runtime_python, "-m", "biominer.detection.yoloe26_detector", "--persistent"]
    assert payload["device"] == "mps"
    assert payload["checkpoint"] == "yoloe-26s-seg.pt"
    assert payload["transport"] == "json_b64"
    assert "butterfly" in payload["prompt_classes"]
    assert payload["images"][0]["width"] == 4
    assert payload["images"][0]["height"] == 2
    assert "src" in calls["env"]["PYTHONPATH"]
    assert detector.model_id == "yoloe26:yoloe-26s-seg"
    assert detector.model_version == "ultralytics:test"
    assert detections == [[DetectionCandidate(label="butterfly_like", score=0.91, bbox_xyxy=(0.0, 0.0, 4.0, 2.0), objectness_score=0.88)]]


def test_yoloe26_sidecar_detector_can_send_temp_image_paths_and_cleanup(tmp_path) -> None:
    calls: dict[str, object] = {}
    detector = YoloE26SidecarObjectDetector(
        runtime_python=str(tmp_path / "YOLO26" / "venv" / "bin" / "python"),
        transport="image_path",
        temp_dir=tmp_path / "sidecar-images",
        popen=_fake_yoloe26_popen(calls),
    )

    detector.detect_batch([_wide_white_image()])

    payload = calls["payload"]
    image_paths = [Path(path) for path in payload["image_paths"]]
    assert payload["transport"] == "image_path"
    assert calls["paths_existed_during_write"] == [True]
    assert calls["ppm_prefix"] == b"P6\n4 2\n255\n"
    assert image_paths and not image_paths[0].exists()


def test_yoloe26_sidecar_detector_cleans_temp_image_paths_after_sidecar_error(tmp_path) -> None:
    calls: dict[str, object] = {}
    detector = YoloE26SidecarObjectDetector(
        runtime_python=str(tmp_path / "YOLO26" / "venv" / "bin" / "python"),
        transport="image_path",
        temp_dir=tmp_path / "sidecar-images",
        popen=_fake_yoloe26_popen(calls, response={"error": "boom", "error_type": "RuntimeError"}),
    )

    with pytest.raises(RuntimeError, match="boom"):
        detector.detect_batch([_wide_white_image()])

    image_paths = [Path(path) for path in calls["payload"]["image_paths"]]
    assert image_paths and not image_paths[0].exists()


def test_yoloe26_sidecar_detector_can_retain_temp_image_paths_for_debug(tmp_path) -> None:
    calls: dict[str, object] = {}
    detector = YoloE26SidecarObjectDetector(
        runtime_python=str(tmp_path / "YOLO26" / "venv" / "bin" / "python"),
        transport="image_path",
        temp_dir=tmp_path / "sidecar-images",
        retain_temp_images=True,
        popen=_fake_yoloe26_popen(calls),
    )

    detector.detect_batch([_wide_white_image()])

    image_paths = [Path(path) for path in calls["payload"]["image_paths"]]
    assert image_paths and image_paths[0].exists()


def test_yoloe26_sidecar_worker_missing_image_path_fails_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="YOLOE-26 sidecar image path does not exist"):
        yoloe26_detector._images_from_request(  # noqa: SLF001 - worker request parser contract.
            {"image_paths": [str(tmp_path / "missing.ppm")]}
        )


def test_detection_pipeline_resizes_loaded_images_before_detection(tmp_path) -> None:
    seen_dimensions: list[tuple[int, int]] = []

    class RecordingDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "v1"
        checkpoint = "checkpoint-a"

        def detect_batch(self, images):  # noqa: ANN001, ANN201 - mirrors detector protocol.
            seen_dimensions.extend((image.width, image.height) for image in images)
            return [[DetectionCandidate(label="butterfly", score=0.9, bbox_xyxy=(0, 0, 2, 1))] for _image in images]

    result = run_detection_pipeline(
        records=[{"source": "flickr", "flickr_photo_id": "photo-resize", "image_url": "memory://wide"}],
        detector=RecordingDetector(),
        output_path=tmp_path / "object_detections.parquet",
        image_loader=lambda record: _wide_white_image(),
        detection_policy=DetectionPolicy(
            backend="fake", image_max_side_px=2, min_box_area_ratio=0.0
        ),
        run_policy=DetectionRunPolicy(detector_batch_size=1),
    )

    row = result.frame.to_dicts()[0]
    assert seen_dimensions == [(2, 1)]
    assert row["bbox_xyxy"] == [0.0, 0.0, 2.0, 1.0]
    assert row["bbox_xyxyn"] == [0.0, 0.0, 1.0, 1.0]
    assert row["bbox_xywhn"] == [0.5, 0.5, 1.0, 1.0]


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


def test_detection_pipeline_image_load_failures_use_stable_detection_id(tmp_path) -> None:
    result = run_detection_pipeline(
        records=[
            {
                "source": "flickr",
                "flickr_photo_id": "photo-failed",
                "image_url": "memory://missing",
            }
        ],
        detector=FakeObjectDetector(),
        output_path=tmp_path / "object_detections.parquet",
        image_loader=lambda record: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )

    row = result.frame.to_dicts()[0]
    assert row["detection_status"] == "failed_image_load"
    assert row["failure_reason"] == "decode failed"
    assert row["prediction_source"] == "object_detector:fake"
    assert isinstance(row["detected_at"], str)
    assert row["detected_at"].endswith("+00:00")
    assert row["detection_id"] == detection_id_for(
        source="flickr",
        flickr_photo_id="photo-failed",
        detector_checkpoint="fake-checkpoint",
        bbox_xyxyn=(None, None, None, None),
        detector_label="failed_image_load",
    )


def test_detection_pipeline_streams_loaded_images_into_detector_batches(tmp_path) -> None:
    consumed = {"records": 0}
    consumed_at_detect: list[int] = []

    def records():
        for index in range(5):
            consumed["records"] += 1
            yield {"source": "flickr", "flickr_photo_id": f"photo-{index}", "image_url": f"memory://photo-{index}"}

    class LazyExecutor:
        def __init__(self, max_workers):  # noqa: ANN001 - mirrors executor constructor.
            self.max_workers = max_workers

        def __enter__(self):  # noqa: ANN204 - mirrors executor context manager.
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204 - mirrors executor context manager.
            return None

        def map(self, fn, iterable, *, buffersize=None):  # noqa: ANN001, ANN202 - mirrors Executor.map.
            for item in iterable:
                yield fn(item)

    class RecordingDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "v1"
        checkpoint = "checkpoint-a"

        def detect_batch(self, images):  # noqa: ANN001, ANN201 - mirrors detector protocol.
            consumed_at_detect.append(consumed["records"])
            return [[DetectionCandidate(label="butterfly", score=0.9, bbox_xyxy=(0, 0, 2, 2))] for _image in images]

    result = run_detection_pipeline(
        records=records(),
        detector=RecordingDetector(),
        output_path=tmp_path / "object_detections.parquet",
        image_loader=lambda record: _image(),
        run_policy=DetectionRunPolicy(download_workers=1, max_inflight_images=2, detector_batch_size=2),
        executor_factory=LazyExecutor,
    )

    assert consumed_at_detect[0] == 2
    assert consumed_at_detect == [2, 4, 5]
    assert result.records_seen == 5
    assert result.images_loaded == 5
    assert result.detections_written == 5


def test_detection_pipeline_adaptive_detector_batching_halves_after_memory_error(tmp_path) -> None:
    class AdaptiveDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "v1"
        checkpoint = "checkpoint-a"

        def __init__(self) -> None:
            self.batches: list[tuple[str, ...]] = []

        def detect_batch(self, images):  # noqa: ANN001, ANN201 - mirrors detector protocol.
            self.batches.append(tuple(str(image.source_uri) for image in images))
            if len(images) > 8:
                raise RuntimeError("MPS memory allocation failed during YOLO inference")
            return [[DetectionCandidate(label="butterfly", score=0.9, bbox_xyxy=(0, 0, 2, 2))] for _image in images]

    def image_loader(record):  # noqa: ANN001, ANN202 - mirrors test image loader protocol.
        return DecodedImage(width=4, height=4, mode="RGB", data=_image().data, source_uri=str(record["image_url"]))

    detector = AdaptiveDetector()
    result = run_detection_pipeline(
        records=[
            {"source": "flickr", "flickr_photo_id": f"photo-{index}", "image_url": f"memory://photo-{index}"}
            for index in range(16)
        ],
        detector=detector,
        output_path=tmp_path / "object_detections.parquet",
        image_loader=image_loader,
        run_policy=DetectionRunPolicy(detector_batch_size=16, adaptive_batching=True, min_detector_batch_size=1),
    )

    assert [len(batch) for batch in detector.batches] == [16, 8, 8]
    assert result.adaptive_batching_enabled is True
    assert result.detector_batch_retries == 1
    assert result.detector_batch_size_initial == 16
    assert result.detector_batch_size_final == 8
    assert result.detector_batch_size_min == 1
    assert result.detections_written == 16
    assert result.frame.get_column("flickr_photo_id").to_list() == [f"photo-{index}" for index in range(16)]


def test_detection_pipeline_adaptive_detector_batching_does_not_retry_non_memory_error(tmp_path) -> None:
    class NonMemoryDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "v1"
        checkpoint = "checkpoint-a"

        def detect_batch(self, images):  # noqa: ANN001, ANN201 - mirrors detector protocol.
            raise RuntimeError("invalid YOLO tensor shape")

    with pytest.raises(RuntimeError, match="invalid YOLO tensor shape"):
        run_detection_pipeline(
            records=[
                {"source": "flickr", "flickr_photo_id": "photo-1", "image_url": "memory://photo-1"},
                {"source": "flickr", "flickr_photo_id": "photo-2", "image_url": "memory://photo-2"},
            ],
            detector=NonMemoryDetector(),
            output_path=tmp_path / "object_detections.parquet",
            image_loader=lambda record: _image(),
            run_policy=DetectionRunPolicy(detector_batch_size=2, adaptive_batching=True, min_detector_batch_size=1),
        )


def test_detection_pipeline_adaptive_detector_batching_reports_min_batch_failure(tmp_path) -> None:
    class AlwaysMemoryDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "v1"
        checkpoint = "checkpoint-a"

        def detect_batch(self, images):  # noqa: ANN001, ANN201 - mirrors detector protocol.
            raise RuntimeError(f"CUDA out of memory at detector batch size {len(images)}")

    with pytest.raises(RuntimeError, match="CUDA out of memory at detector batch size 1"):
        run_detection_pipeline(
            records=[
                {"source": "flickr", "flickr_photo_id": "photo-1", "image_url": "memory://photo-1"},
                {"source": "flickr", "flickr_photo_id": "photo-2", "image_url": "memory://photo-2"},
            ],
            detector=AlwaysMemoryDetector(),
            output_path=tmp_path / "object_detections.parquet",
            image_loader=lambda record: _image(),
            run_policy=DetectionRunPolicy(detector_batch_size=2, adaptive_batching=True, min_detector_batch_size=1),
        )


def test_detection_pipeline_flushes_detection_rows_in_parquet_batches(tmp_path) -> None:
    output = tmp_path / "object_detections.parquet"

    def image_loader(record):  # noqa: ANN001, ANN202 - mirrors test image loader protocol.
        if record["flickr_photo_id"] == "photo-1":
            raise RuntimeError("decode failed")
        return _image()

    result = run_detection_pipeline(
        records=[
            {"source": "flickr", "flickr_photo_id": f"photo-{index}", "image_url": f"memory://photo-{index}"}
            for index in range(3)
        ],
        detector=FakeObjectDetector(
            [[DetectionCandidate(label="butterfly", score=0.9, bbox_xyxy=(0, 0, 2, 2))] for _index in range(3)]
        ),
        output_path=output,
        image_loader=image_loader,
        run_policy=DetectionRunPolicy(detector_batch_size=1, parquet_batch_rows=1),
    )

    frame = pl.read_parquet(output)
    assert result.parquet_batches_written == 3
    assert result.records_seen == 3
    assert result.image_failures == 1
    assert result.detections_written == 2
    assert frame.height == 3
    assert sorted(frame["flickr_photo_id"].to_list()) == ["photo-0", "photo-1", "photo-2"]
    assert sorted(frame["detection_status"].to_list()) == ["detected", "detected", "failed_image_load"]
    assert not (tmp_path / ".object_detections.parquet.batches.tmp").exists()


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
                "genus_top3": ["Danaus"],
                "species_top1_score": 0.9,
            },
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [20, 20, 30, 30],
                "species_top1_scientific_name": "Danaus gilippus",
                "species_top5": ["Danaus gilippus"],
                "family_top3": ["Nymphalidae"],
                "genus_top3": ["Danaus"],
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
    assert report["genus_top3_accuracy"] == pytest.approx(1.0)
    assert report["joint_map50"] == pytest.approx(1.0)


def test_xie_style_evaluation_counts_accepted_taxon_key_species_matches() -> None:
    report = evaluate_xie_style(
        predictions=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [0, 0, 10, 10],
                "species_top1_scientific_name": "Anosia plexippus",
                "species_top1_accepted_taxon_key": "gbif:5131654",
                "species_top5": ["Anosia plexippus"],
                "species_top5_accepted_taxon_keys": ["gbif:5131654"],
                "species_top1_score": 0.9,
            }
        ],
        ground_truth=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [1, 1, 9, 9],
                "scientific_name": "Danaus plexippus",
                "accepted_taxon_key": "gbif:5131654",
            }
        ],
    )

    assert report["species_top1_accuracy"] == pytest.approx(1.0)
    assert report["species_top5_accuracy"] == pytest.approx(1.0)
    assert report["joint_map50"] == pytest.approx(1.0)
    assert report["joint_top5_map50"] == pytest.approx(1.0)


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


def test_xie_style_joint_map_penalizes_high_scoring_false_positive() -> None:
    report = evaluate_xie_style(
        predictions=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [40, 40, 50, 50],
                "detector_score": 0.99,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top5": ["Danaus plexippus"],
                "species_top1_score": 0.99,
            },
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [0, 0, 10, 10],
                "detector_score": 0.40,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top5": ["Danaus plexippus"],
                "species_top1_score": 0.90,
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

    assert report["joint_map50"] == pytest.approx(0.5)
    assert report["joint_top5_map50"] == pytest.approx(0.5)


def test_xie_style_joint_map_allows_correct_species_after_wrong_species_match() -> None:
    report = evaluate_xie_style(
        predictions=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [0, 0, 10, 10],
                "detector_score": 0.99,
                "species_top1_scientific_name": "Danaus gilippus",
                "species_top5": ["Danaus gilippus"],
                "species_top1_score": 0.99,
            },
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [0, 0, 10, 10],
                "detector_score": 0.90,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top5": ["Danaus plexippus"],
                "species_top1_score": 0.90,
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

    assert report["joint_map50"] == pytest.approx(0.5)
    assert report["joint_top5_map50"] == pytest.approx(0.5)


def test_xie_style_report_includes_evaluation_thresholds() -> None:
    report = evaluate_xie_style(
        predictions=[],
        ground_truth=None,
        iou_threshold=0.6,
        score_threshold=0.45,
    )

    assert report["iou_threshold"] == 0.6
    assert report["score_threshold"] == 0.45


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


def test_xie_style_taxonomic_accuracy_counts_missed_ground_truth_as_wrong() -> None:
    report = evaluate_xie_style(
        predictions=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [0, 0, 10, 10],
                "detector_score": 0.90,
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top5": ["Danaus plexippus"],
                "family_top3": ["Nymphalidae"],
                "genus_top3": ["Danaus"],
                "species_top1_score": 0.95,
            }
        ],
        ground_truth=[
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [0, 0, 10, 10],
                "scientific_name": "Danaus plexippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "p1",
                "bbox_xyxy": [20, 20, 30, 30],
                "scientific_name": "Danaus gilippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
            },
        ],
    )

    assert report["ground_truth_seen"] == 2
    assert report["matched_ground_truth"] == 1
    assert report["species_top1_accuracy"] == pytest.approx(0.5)
    assert report["species_top5_accuracy"] == pytest.approx(0.5)
    assert report["family_top3_accuracy"] == pytest.approx(0.5)
    assert report["genus_top3_accuracy"] == pytest.approx(0.5)
