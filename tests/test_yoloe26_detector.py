from __future__ import annotations

import importlib

import pytest

from biominer.detection.legacy_yoloe_evidence import (
    DEFAULT_YOLOE26_PROMPTS,
    detections_from_yoloe_result,
    default_yoloe26_prompts,
    yoloe26_prompt_set_fingerprint,
    yoloe26_coarse_label,
)


def test_yoloe26_module_imports_without_ultralytics_runtime() -> None:
    module = importlib.import_module("biominer.detection.legacy_yoloe_evidence")

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


@pytest.mark.parametrize(
    "prompt",
    (
        "insect",
        "beetle",
        "fly",
        "mosquito",
        "bee",
        "wasp",
        "ant",
        "grasshopper",
        "cricket",
        "true bug",
        "cicada",
        "dragonfly",
        "damselfly",
        "mantis",
        "cockroach",
        "termite",
        "stick insect",
        "lacewing",
        "caddisfly",
        "mayfly",
        "stonefly",
        "earwig",
    ),
)
def test_yoloe26_broad_insect_prompts_remain_coarse_screening_evidence(prompt: str) -> None:
    assert yoloe26_coarse_label(prompt) == "insect_like"


def test_yoloe26_broad_insect_set_preserves_specific_lepidoptera_and_specimen_routes() -> None:
    assert yoloe26_coarse_label("butterfly") == "adult_butterfly"
    assert yoloe26_coarse_label("moth") == "moth_like"
    assert yoloe26_coarse_label("caterpillar") == "caterpillar"
    assert yoloe26_coarse_label("pinned insect specimen") == "pinned_specimen"


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


class _FakeResult:
    def __init__(self, *, names, boxes) -> None:  # noqa: ANN001
        self.names = names
        self.boxes = boxes


class _FakeBox:
    def __init__(self, *, xyxy, cls, conf) -> None:  # noqa: ANN001
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf
