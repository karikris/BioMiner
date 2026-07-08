from __future__ import annotations

import io
import importlib
import json
import sys

import pytest

import biominer.detection.yoloe26_detector as yoloe26_module
from biominer.detection.detector_base import COARSE_DETECTOR_LABELS, DecodedImage, DetectionCandidate
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


def test_yoloe26_result_conversion_keeps_nontaxonomic_custom_prompts_coarse() -> None:
    result = _FakeResult(
        names={0: "custom proposal"},
        boxes=[_FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87])],
    )

    detections = detections_from_yoloe_result(result)

    assert [item.label for item in detections] == ["insect_like"]
    assert set(item.label for item in detections).issubset(set(COARSE_DETECTOR_LABELS))


def test_yoloe26_result_conversion_rejects_taxonomic_custom_prompts() -> None:
    result = _FakeResult(
        names={0: "Papilio demoleus"},
        boxes=[_FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87])],
    )

    with pytest.raises(ValueError, match="object proposals"):
        detections_from_yoloe_result(result)


def test_yoloe26_persistent_sidecar_reuses_detector_for_same_settings(monkeypatch) -> None:
    loads: list[dict[str, object]] = []

    class FakeDetector:
        backend = "yoloe26"

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            loads.append(dict(kwargs))
            self.checkpoint = str(kwargs["checkpoint"])
            self.model_id = "fake-model"
            self.model_version = "fake-version"

        def detect_batch(self, images) -> list[list[DetectionCandidate]]:  # noqa: ANN001
            return [[DetectionCandidate(label="butterfly_like", score=0.9, bbox_xyxy=(0, 0, 1, 1))] for _image in images]

    request = _persistent_request(imgsz=768)
    stdout = _run_persistent_worker_with_requests(monkeypatch, FakeDetector, [request, request, {"shutdown": True}])
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert len(loads) == 1
    assert loads[0]["imgsz"] == 768
    assert responses[0]["backend"] == "yoloe26"
    assert responses[0]["model_id"] == "fake-model"
    assert responses[0]["model_version"] == "fake-version"
    assert responses[0]["checkpoint"] == "yoloe-26s-seg.pt"
    assert responses[0]["detections"][0][0]["label"] == "butterfly_like"
    assert responses[0]["metadata"]["model_id"] == "fake-model"
    assert len(responses) == 2


def test_yoloe26_persistent_sidecar_reloads_when_settings_change(monkeypatch) -> None:
    loads: list[dict[str, object]] = []

    class FakeDetector:
        backend = "yoloe26"

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            loads.append(dict(kwargs))
            self.checkpoint = str(kwargs["checkpoint"])
            self.model_id = f"fake-{kwargs['imgsz']}"
            self.model_version = "fake-version"

        def detect_batch(self, images) -> list[list[DetectionCandidate]]:  # noqa: ANN001
            return [[] for _image in images]

    stdout = _run_persistent_worker_with_requests(
        monkeypatch,
        FakeDetector,
        [_persistent_request(imgsz=768), _persistent_request(imgsz=640), {"shutdown": True}],
    )
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert [load["imgsz"] for load in loads] == [768, 640]
    assert [response["model_id"] for response in responses] == ["fake-768", "fake-640"]


def test_yoloe26_persistent_sidecar_reports_json_errors(monkeypatch) -> None:
    class FailingDetector:
        backend = "yoloe26"

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.checkpoint = str(kwargs["checkpoint"])
            self.model_id = "fake-model"
            self.model_version = "fake-version"

        def detect_batch(self, images) -> list[list[DetectionCandidate]]:  # noqa: ANN001, ARG002
            raise RuntimeError("worker boom")

    stdout = _run_persistent_worker_with_requests(monkeypatch, FailingDetector, [_persistent_request(), {"shutdown": True}])
    payload = json.loads(stdout.getvalue().strip())

    assert payload["error"] == "worker boom"
    assert payload["error_type"] == "RuntimeError"


def _persistent_request(**overrides: object) -> dict[str, object]:
    image = DecodedImage(width=1, height=1, mode="RGB", data=b"\x00\x00\x00", source_uri="memory://image")
    request: dict[str, object] = {
        "checkpoint": "yoloe-26s-seg.pt",
        "device": "mps",
        "imgsz": 768,
        "conf": 0.20,
        "iou": 0.50,
        "max_det": 8,
        "prompt_classes": ["butterfly"],
        "images": [yoloe26_module._image_to_payload(image)],
    }
    request.update(overrides)
    return request


def _run_persistent_worker_with_requests(monkeypatch, detector_class, requests: list[dict[str, object]]) -> io.StringIO:  # noqa: ANN001
    stdin = io.StringIO("".join(json.dumps(request, sort_keys=True) + "\n" for request in requests))
    stdout = io.StringIO()
    monkeypatch.setattr(yoloe26_module, "YoloE26ObjectDetector", detector_class)
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    yoloe26_module._run_persistent_sidecar()
    return stdout


class _FakeResult:
    def __init__(self, *, names, boxes) -> None:  # noqa: ANN001
        self.names = names
        self.boxes = boxes


class _FakeBox:
    def __init__(self, *, xyxy, cls, conf) -> None:  # noqa: ANN001
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf
