from __future__ import annotations

import io
import importlib
import json
import sys

import pytest

import biominer.detection.yoloe26_detector as yoloe26_module
from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.yoloe26_detector import (
    DEFAULT_YOLOE26_PROMPTS,
    YoloE26ObjectDetector,
    YoloE26SidecarObjectDetector,
    detections_from_yoloe_result,
    default_yoloe26_prompts,
    yoloe26_prompt_set_fingerprint,
    yoloe26_coarse_label,
)


def test_yoloe26_module_imports_without_ultralytics_runtime() -> None:
    module = importlib.import_module("biominer.detection.yoloe26_detector")

    assert module.DEFAULT_YOLOE26_CHECKPOINT == "yoloe-26s-seg.pt"


def test_yoloe26_default_prompts_and_coarse_labels_are_stable() -> None:
    assert default_yoloe26_prompts() == DEFAULT_YOLOE26_PROMPTS
    assert yoloe26_coarse_label("butterfly") == "adult_butterfly"
    assert yoloe26_coarse_label("butterfly wing") == "possible_adult_butterfly"
    assert yoloe26_coarse_label("pinned butterfly specimen") == "pinned_specimen"
    assert yoloe26_coarse_label("moth") == "moth_like"
    assert yoloe26_coarse_label("caterpillar") == "caterpillar"
    assert yoloe26_coarse_label("chrysalis") == "pupa"
    assert yoloe26_coarse_label("pupa") == "pupa"
    assert yoloe26_coarse_label("flower") == "no_relevant_organism"
    assert yoloe26_coarse_label("museum label") == "artifact"
    assert "flower" not in default_yoloe26_prompts(include_hard_negative_prompts=False)
    assert "butterfly" in default_yoloe26_prompts(include_hard_negative_prompts=False)

    with pytest.raises(ValueError, match="unsupported YOLOE-26 object prompt"):
        yoloe26_coarse_label("custom proposal")


def test_yoloe26_prompt_set_fingerprint_is_normalized_and_order_sensitive() -> None:
    fingerprint = yoloe26_prompt_set_fingerprint((" Butterfly  ", "MOTH"))

    assert fingerprint == yoloe26_prompt_set_fingerprint(("butterfly", "moth"))
    assert fingerprint.startswith("sha256:")
    assert fingerprint != yoloe26_prompt_set_fingerprint(("moth", "butterfly"))


def test_yoloe26_result_conversion_maps_prompts_to_detection_candidates() -> None:
    result = _FakeResult(
        names={0: "butterfly", 1: "flower"},
        boxes=[
            _FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87]),
            _FakeBox(xyxy=[[0.0, 0.0, 4.0, 5.0]], cls=[1], conf=[0.42]),
        ],
    )

    detections = detections_from_yoloe_result(result)

    assert [item.label for item in detections] == ["adult_butterfly", "no_relevant_organism"]
    assert detections[0].bbox_xyxy == (1.0, 2.0, 9.0, 12.0)
    assert detections[0].score == 0.87
    assert detections[0].objectness_score == 0.87
    assert detections[0].detector_prompt == "butterfly"
    assert detections[0].detector_class_id == 0
    assert detections[0].detector_prompt_set_fingerprint == yoloe26_prompt_set_fingerprint(("butterfly", "flower"))


def test_yoloe26_result_conversion_rejects_unknown_nontaxonomic_prompts() -> None:
    result = _FakeResult(
        names={0: "custom proposal"},
        boxes=[_FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87])],
    )

    with pytest.raises(ValueError, match="unsupported YOLOE-26 object prompt"):
        detections_from_yoloe_result(result)


def test_yoloe26_result_conversion_rejects_taxonomic_custom_prompts() -> None:
    result = _FakeResult(
        names={0: "Papilio demoleus"},
        boxes=[_FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87])],
    )

    with pytest.raises(ValueError, match="object proposals"):
        detections_from_yoloe_result(result)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ({}, "at least one"),
        ({0: "butterfly", 2: "moth"}, "contiguous"),
        ({0: "butterfly", 1: " Butterfly "}, "duplicate"),
        ({1: "butterfly"}, "contiguous"),
    ],
)
def test_yoloe26_result_conversion_rejects_invalid_actual_name_maps(names, message: str) -> None:  # noqa: ANN001
    result = _FakeResult(names=names, boxes=[])

    with pytest.raises(ValueError, match=message):
        detections_from_yoloe_result(result)


def test_yoloe26_result_conversion_uses_actual_name_order_not_requested_order() -> None:
    result = _FakeResult(
        names={0: "moth", 1: "butterfly"},
        boxes=[_FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[0], conf=[0.87])],
    )

    detections = detections_from_yoloe_result(result, prompt_classes=("butterfly", "moth"))

    assert detections[0].label == "moth_like"
    assert detections[0].detector_prompt == "moth"
    assert detections[0].detector_class_id == 0
    assert detections[0].detector_prompt_set_fingerprint == yoloe26_prompt_set_fingerprint(("moth", "butterfly"))


def test_yoloe26_result_conversion_rejects_class_id_outside_actual_name_map() -> None:
    result = _FakeResult(
        names={0: "butterfly"},
        boxes=[_FakeBox(xyxy=[[1.0, 2.0, 9.0, 12.0]], cls=[1], conf=[0.87])],
    )

    with pytest.raises(ValueError, match="class id 1"):
        detections_from_yoloe_result(result)


def test_yoloe26_direct_and_sidecar_share_checkpoint_validation(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported YOLOE-26 checkpoint"):
        YoloE26ObjectDetector(checkpoint="not-a-yoloe-checkpoint.pt")
    with pytest.raises(ValueError, match="unsupported YOLOE-26 checkpoint"):
        YoloE26SidecarObjectDetector(
            runtime_python=str(tmp_path / "YOLO26" / "venv" / "bin" / "python"),
            checkpoint="not-a-yoloe-checkpoint.pt",
        )


def test_yoloe26_direct_and_sidecar_share_prompt_validation(tmp_path) -> None:
    with pytest.raises(ValueError, match="object proposals"):
        YoloE26ObjectDetector(prompt_classes=("Papilio demoleus",))
    with pytest.raises(ValueError, match="object proposals"):
        YoloE26SidecarObjectDetector(
            runtime_python=str(tmp_path / "YOLO26" / "venv" / "bin" / "python"),
            prompt_classes=("Papilio demoleus",),
        )


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
            return [
                [
                    DetectionCandidate(
                        label="adult_butterfly",
                        score=0.9,
                        bbox_xyxy=(0, 0, 1, 1),
                        detector_prompt="butterfly",
                        detector_class_id=0,
                        detector_prompt_set_fingerprint="sha256:" + "a" * 64,
                    )
                ]
                for _image in images
            ]

    request = _persistent_request(imgsz=768)
    stdout = _run_persistent_worker_with_requests(monkeypatch, FakeDetector, [request, request, {"shutdown": True}])
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert len(loads) == 1
    assert loads[0]["imgsz"] == 768
    assert responses[0]["backend"] == "yoloe26"
    assert responses[0]["model_id"] == "fake-model"
    assert responses[0]["model_version"] == "fake-version"
    assert responses[0]["checkpoint"] == "yoloe-26s-seg.pt"
    assert responses[0]["detections"][0][0]["label"] == "adult_butterfly"
    assert responses[0]["detections"][0][0]["detector_prompt"] == "butterfly"
    assert responses[0]["detections"][0][0]["detector_class_id"] == 0
    assert responses[0]["detections"][0][0]["detector_prompt_set_fingerprint"] == "sha256:" + "a" * 64
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


def test_yoloe26_sidecar_detector_reuses_one_process_for_multiple_batches(tmp_path) -> None:
    factory = _FakePopenFactory()
    detector = YoloE26SidecarObjectDetector(
        runtime_python=str(tmp_path / "YOLO26" / "venv" / "bin" / "python"),
        checkpoint="yoloe-26s-seg.pt",
        device="mps",
        imgsz=768,
        popen=factory,
    )

    first = detector.detect_batch([_decoded_image()])
    second = detector.detect_batch([_decoded_image()])
    detector.close()

    assert len(factory.processes) == 1
    assert factory.processes[0].args[-1] == "--persistent"
    assert factory.processes[0].kwargs["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert first[0][0].label == "adult_butterfly"
    assert first[0][0].detector_prompt == "butterfly"
    assert first[0][0].detector_class_id == 0
    assert first[0][0].detector_prompt_set_fingerprint == "sha256:" + "b" * 64
    assert second[0][0].score == 0.91
    assert detector.model_id == "yoloe26:yoloe-26s-seg"
    assert detector.model_version == "ultralytics:fake"
    assert any(json.loads(line).get("shutdown") is True for line in factory.processes[0].writes)
    assert factory.processes[0].returncode == 0


def test_yoloe26_sidecar_detector_closes_from_context_manager(tmp_path) -> None:
    factory = _FakePopenFactory()
    with YoloE26SidecarObjectDetector(
        runtime_python=str(tmp_path / "YOLO26" / "venv" / "bin" / "python"),
        popen=factory,
    ) as detector:
        detector.detect_batch([_decoded_image()])

    assert len(factory.processes) == 1
    assert any(json.loads(line).get("shutdown") is True for line in factory.processes[0].writes)


def test_yoloe26_sidecar_detector_raises_worker_errors(tmp_path) -> None:
    factory = _FakePopenFactory(error_payload={"error": "bad request", "error_type": "ValueError"})
    detector = YoloE26SidecarObjectDetector(
        runtime_python=str(tmp_path / "YOLO26" / "venv" / "bin" / "python"),
        popen=factory,
    )

    with pytest.raises(RuntimeError, match="ValueError.*bad request"):
        detector.detect_batch([_decoded_image()])

    detector.close()


def test_yoloe26_sidecar_detector_fails_clearly_when_process_exits(tmp_path) -> None:
    factory = _FakePopenFactory(exited=True)
    detector = YoloE26SidecarObjectDetector(
        runtime_python=str(tmp_path / "YOLO26" / "venv" / "bin" / "python"),
        popen=factory,
    )

    with pytest.raises(RuntimeError, match="exited early"):
        detector.detect_batch([_decoded_image()])


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


def _decoded_image() -> DecodedImage:
    return DecodedImage(width=1, height=1, mode="RGB", data=b"\x00\x00\x00", source_uri="memory://image")


def _run_persistent_worker_with_requests(monkeypatch, detector_class, requests: list[dict[str, object]]) -> io.StringIO:  # noqa: ANN001
    stdin = io.StringIO("".join(json.dumps(request, sort_keys=True) + "\n" for request in requests))
    stdout = io.StringIO()
    monkeypatch.setattr(yoloe26_module, "YoloE26ObjectDetector", detector_class)
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    yoloe26_module._run_persistent_sidecar()
    return stdout


class _FakePopenFactory:
    def __init__(self, *, error_payload: dict[str, object] | None = None, exited: bool = False) -> None:
        self.error_payload = error_payload
        self.exited = exited
        self.processes: list[_FakeProcess] = []

    def __call__(self, args, **kwargs) -> "_FakeProcess":  # noqa: ANN001, ANN003
        process = _FakeProcess(args=list(args), kwargs=kwargs, error_payload=self.error_payload, exited=self.exited)
        self.processes.append(process)
        return process


class _FakeProcess:
    def __init__(self, *, args: list[str], kwargs: dict[str, object], error_payload: dict[str, object] | None, exited: bool) -> None:
        self.args = args
        self.kwargs = kwargs
        self.error_payload = error_payload
        self.returncode = 17 if exited else None
        self.writes: list[str] = []
        self.output_lines: list[str] = []
        self.stdin = _FakeStdin(self)
        self.stdout = _FakeStdout(self)
        self.stderr = io.StringIO("sidecar stderr tail\n" if exited else "")

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout=None) -> int:  # noqa: ANN001
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15


class _FakeStdin:
    def __init__(self, process: _FakeProcess) -> None:
        self.process = process

    def write(self, text: str) -> int:
        for raw_line in text.splitlines():
            self.process.writes.append(raw_line)
            payload = json.loads(raw_line)
            if payload.get("shutdown"):
                self.process.returncode = 0
                continue
            if self.process.error_payload is not None:
                self.process.output_lines.append(json.dumps(self.process.error_payload, sort_keys=True) + "\n")
                continue
            self.process.output_lines.append(
                json.dumps(
                    {
                        "backend": "yoloe26",
                        "model_id": "yoloe26:yoloe-26s-seg",
                        "model_version": "ultralytics:fake",
                        "checkpoint": payload["checkpoint"],
                        "metadata": {
                            "backend": "yoloe26",
                            "model_id": "yoloe26:yoloe-26s-seg",
                            "model_version": "ultralytics:fake",
                            "checkpoint": payload["checkpoint"],
                        },
                        "detections": [
                            [
                                {
                                    "label": "adult_butterfly",
                                    "score": 0.91,
                                    "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                                    "objectness_score": 0.91,
                                    "detector_prompt": "butterfly",
                                    "detector_class_id": 0,
                                    "detector_prompt_set_fingerprint": "sha256:" + "b" * 64,
                                }
                            ]
                            for _image in payload["images"]
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return len(text)

    def flush(self) -> None:
        return None


class _FakeStdout:
    def __init__(self, process: _FakeProcess) -> None:
        self.process = process

    def readline(self) -> str:
        if not self.process.output_lines:
            return ""
        return self.process.output_lines.pop(0)


class _FakeResult:
    def __init__(self, *, names, boxes) -> None:  # noqa: ANN001
        self.names = names
        self.boxes = boxes


class _FakeBox:
    def __init__(self, *, xyxy, cls, conf) -> None:  # noqa: ANN001
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf
