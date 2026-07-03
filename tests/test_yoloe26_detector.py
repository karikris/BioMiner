from __future__ import annotations

import importlib

from biominer.detection.detector_base import COARSE_DETECTOR_LABELS
from biominer.detection.yoloe26_detector import (
    DEFAULT_YOLOE26_PROMPTS,
    detections_from_yoloe_result,
    default_yoloe26_prompts,
    yoloe26_coarse_label,
)


def test_yoloe26_module_imports_without_ultralytics_runtime() -> None:
    module = importlib.import_module("biominer.detection.yoloe26_detector")

    assert module.DEFAULT_YOLOE26_CHECKPOINT == "yoloe-26s-seg.pt"


def test_yoloe26_default_prompts_and_coarse_labels_are_stable() -> None:
    assert default_yoloe26_prompts() == DEFAULT_YOLOE26_PROMPTS
    assert yoloe26_coarse_label("butterfly") == "butterfly_like"
    assert yoloe26_coarse_label("butterfly wing") == "butterfly_like"
    assert yoloe26_coarse_label("moth") == "moth_like"
    assert yoloe26_coarse_label("caterpillar") == "caterpillar"
    assert yoloe26_coarse_label("chrysalis") == "pupa"
    assert yoloe26_coarse_label("pupa") == "pupa"
    assert yoloe26_coarse_label("flower") == "hard_negative"
    assert yoloe26_coarse_label("museum label") == "hard_negative"
    assert yoloe26_coarse_label("custom proposal") == "insect_like"
    assert "flower" not in default_yoloe26_prompts(include_hard_negative_prompts=False)
    assert "butterfly" in default_yoloe26_prompts(include_hard_negative_prompts=False)


def test_yoloe26_result_conversion_maps_prompts_to_detection_candidates() -> None:
    result = _FakeResult(
        names={0: "butterfly", 1: "flower"},
        boxes=[
            _FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87]),
            _FakeBox(xyxy=[[0.0, 0.0, 4.0, 5.0]], cls=[1], conf=[0.42]),
        ],
    )

    detections = detections_from_yoloe_result(result)

    assert [item.label for item in detections] == ["butterfly_like", "hard_negative"]
    assert detections[0].bbox_xyxy == (1.0, 2.0, 9.0, 12.0)
    assert detections[0].score == 0.87
    assert detections[0].objectness_score == 0.87


def test_yoloe26_result_conversion_keeps_taxonomic_custom_prompts_coarse() -> None:
    result = _FakeResult(
        names={0: "Papilio demoleus"},
        boxes=[_FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87])],
    )

    detections = detections_from_yoloe_result(result)

    assert [item.label for item in detections] == ["insect_like"]
    assert set(item.label for item in detections).issubset(set(COARSE_DETECTOR_LABELS))


class _FakeResult:
    def __init__(self, *, names, boxes) -> None:  # noqa: ANN001
        self.names = names
        self.boxes = boxes


class _FakeBox:
    def __init__(self, *, xyxy, cls, conf) -> None:  # noqa: ANN001
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf
