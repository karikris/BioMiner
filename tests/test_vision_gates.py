from __future__ import annotations

import pytest

from biominer.vision.gates import BioClipGateMode, BioClipGatePolicy, bioclip_score_input_decision


@pytest.mark.parametrize("label", ["butterfly_like", "moth_like", "caterpillar", "pupa", "insect_like"])
def test_exclude_hard_negative_gate_scores_non_hard_negative_detections(label: str) -> None:
    decision = bioclip_score_input_decision(
        {"detection_status": "detected", "detector_label": label},
        BioClipGatePolicy(mode=BioClipGateMode.EXCLUDE_HARD_NEGATIVE),
    )

    assert decision.should_score is True
    assert decision.visual_input_kind == "detector_crop"
    assert decision.bioclip_gate_mode == "exclude_hard_negative"
    assert decision.bioclip_gate_decision == "score"
    assert decision.bioclip_gate_reason == "detected_non_hard_negative"


def test_exclude_hard_negative_gate_excludes_hard_negative_detection() -> None:
    decision = bioclip_score_input_decision(
        {"detection_status": "detected", "detector_label": "hard_negative"},
        BioClipGatePolicy(mode=BioClipGateMode.EXCLUDE_HARD_NEGATIVE),
    )

    assert decision.should_score is False
    assert decision.visual_input_kind is None
    assert decision.bioclip_gate_decision == "exclude"
    assert decision.bioclip_gate_reason == "hard_negative_detector_label"


def test_no_detection_becomes_whole_image_fallback_when_enabled() -> None:
    decision = bioclip_score_input_decision(
        {"detection_status": "no_detection", "detector_label": "no_detection"},
        BioClipGatePolicy(mode=BioClipGateMode.EXCLUDE_HARD_NEGATIVE, score_no_detection_whole_image=True),
    )

    assert decision.should_score is True
    assert decision.visual_input_kind == "whole_image"
    assert decision.bioclip_gate_decision == "score"
    assert decision.bioclip_gate_reason == "no_detection_whole_image_fallback"


@pytest.mark.parametrize("status", ["failed_image_load", "image_load_failed"])
def test_image_failures_are_excluded(status: str) -> None:
    decision = bioclip_score_input_decision(
        {"detection_status": status, "detector_label": "failed_image_load"},
        BioClipGatePolicy(mode=BioClipGateMode.EXCLUDE_HARD_NEGATIVE),
    )

    assert decision.should_score is False
    assert decision.bioclip_gate_decision == "exclude"
    assert decision.bioclip_gate_reason == "image_load_failed"


def test_legacy_butterfly_like_gate_keeps_old_filter() -> None:
    policy = BioClipGatePolicy.legacy_butterfly_like_only()

    assert bioclip_score_input_decision({"detection_status": "detected", "detector_label": "butterfly_like"}, policy).should_score
    moth = bioclip_score_input_decision({"detection_status": "detected", "detector_label": "moth_like"}, policy)

    assert moth.should_score is False
    assert moth.bioclip_gate_reason == "detector_label_not_eligible"
