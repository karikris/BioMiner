from __future__ import annotations

import importlib

import pytest

from biominer.detection.yolo26_detector import detections_from_yolo26_result, yolo26_coarse_label


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


class _FakeResult:
    def __init__(self, *, names, boxes) -> None:  # noqa: ANN001
        self.names = names
        self.boxes = boxes


class _FakeBox:
    def __init__(self, *, xyxy, cls, conf) -> None:  # noqa: ANN001
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf
