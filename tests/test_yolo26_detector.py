from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.yolo26_detector import Yolo26SidecarObjectDetector, detections_from_yolo26_result, yolo26_coarse_label


def test_yolo26_module_imports_without_ultralytics_runtime() -> None:
    module = importlib.import_module("biominer.detection.yolo26_detector")

    assert module.DEFAULT_YOLO26_IMGSZ == 640


def test_yolo26_requires_user_checkpoint_before_optional_runtime_import() -> None:
    module = importlib.import_module("biominer.detection.yolo26_detector")

    with pytest.raises(ValueError, match="explicit user-provided checkpoint"):
        module.Yolo26ObjectDetector(checkpoint="")


def test_yolo26_labels_are_coarsened_for_inference_only() -> None:
    assert yolo26_coarse_label("butterfly") == "butterfly_like"
    assert yolo26_coarse_label("flower") == "hard_negative"
    with pytest.raises(ValueError, match="taxonomic"):
        yolo26_coarse_label("Papilio demoleus")
    with pytest.raises(ValueError, match="coarse object label"):
        yolo26_coarse_label("custom species class")


def test_yolo26_result_conversion_emits_only_coarse_labels() -> None:
    result = _FakeResult(
        names={0: "butterfly", 1: "insect", 2: "flower"},
        boxes=[
            _FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87]),
            _FakeBox(xyxy=[[0.0, 0.0, 4.0, 5.0]], cls=[1], conf=[0.42]),
            _FakeBox(xyxy=[[3.0, 3.0, 8.0, 9.0]], cls=[2], conf=[0.33]),
        ],
    )

    detections = detections_from_yolo26_result(result)

    assert [item.label for item in detections] == ["butterfly_like", "insect_like", "hard_negative"]
    assert detections[0].objectness_score == 0.87


def test_yolo26_result_conversion_rejects_species_classifier_labels() -> None:
    result = _FakeResult(
        names={0: "Papilio demoleus"},
        boxes=[_FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87])],
    )

    with pytest.raises(ValueError, match="taxonomic"):
        detections_from_yolo26_result(result)


def test_yolo26_sidecar_detector_serializes_rgb_images_without_importing_ultralytics(tmp_path, monkeypatch) -> None:
    runtime_python = tmp_path / "YOLO26" / "venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003, ANN202 - mirrors subprocess.run.
        calls["command"] = command
        calls["payload"] = json.loads(kwargs["input"])
        calls["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "metadata": {
                        "backend": "yolo26",
                        "model_id": "yolo26:coarse-objects",
                        "model_version": "ultralytics:test",
                        "checkpoint": "coarse-objects.pt",
                    },
                    "detections": [
                        [
                            {
                                "label": "butterfly_like",
                                "score": 0.91,
                                "bbox_xyxy": [0.0, 0.0, 4.0, 2.0],
                                "objectness_score": 0.88,
                            }
                        ]
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("biominer.detection.yolo26_detector.subprocess.run", fake_run)
    detector = Yolo26SidecarObjectDetector(
        runtime_python=str(runtime_python),
        checkpoint="coarse-objects.pt",
        device="mps",
    )

    image = DecodedImage(width=4, height=2, mode="RGB", data=bytes([255, 255, 255] * 8), source_uri="memory://wide")
    detections = detector.detect_batch([image])

    payload = calls["payload"]
    assert calls["command"] == [str(runtime_python), "-m", "biominer.detection.yolo26_detector"]
    assert payload["device"] == "mps"
    assert payload["checkpoint"] == "coarse-objects.pt"
    assert payload["images"][0]["width"] == 4
    assert payload["images"][0]["height"] == 2
    assert "src" in calls["env"]["PYTHONPATH"]
    assert detector.model_id == "yolo26:coarse-objects"
    assert detector.model_version == "ultralytics:test"
    assert detections == [
        [DetectionCandidate(label="butterfly_like", score=0.91, bbox_xyxy=(0.0, 0.0, 4.0, 2.0), objectness_score=0.88)]
    ]


class _FakeResult:
    def __init__(self, *, names, boxes) -> None:  # noqa: ANN001
        self.names = names
        self.boxes = boxes


class _FakeBox:
    def __init__(self, *, xyxy, cls, conf) -> None:  # noqa: ANN001
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf
