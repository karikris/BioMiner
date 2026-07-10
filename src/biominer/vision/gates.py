from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


VisualInputKind = Literal["detector_crop", "detector_crop_segmentation", "whole_image"]
GateDecision = Literal["score", "exclude"]


class BioClipGateMode(StrEnum):
    BUTTERFLY_LIKE_ONLY = "butterfly_like_only"
    EXCLUDE_HARD_NEGATIVE = "exclude_hard_negative"


@dataclass(frozen=True)
class BioClipGatePolicy:
    mode: BioClipGateMode | str = BioClipGateMode.EXCLUDE_HARD_NEGATIVE
    eligible_detector_labels: tuple[str, ...] = ("butterfly_like",)
    detected_visual_input_kind: VisualInputKind = "detector_crop"
    score_no_detection_whole_image: bool = True

    @classmethod
    def legacy_butterfly_like_only(cls, *, eligible_detector_labels: tuple[str, ...] = ("butterfly_like",)) -> BioClipGatePolicy:
        return cls(
            mode=BioClipGateMode.BUTTERFLY_LIKE_ONLY,
            eligible_detector_labels=eligible_detector_labels,
            score_no_detection_whole_image=False,
        )

    @property
    def normalized_mode(self) -> BioClipGateMode:
        try:
            return self.mode if isinstance(self.mode, BioClipGateMode) else BioClipGateMode(str(self.mode))
        except ValueError as exc:
            valid = ", ".join(mode.value for mode in BioClipGateMode)
            raise ValueError(f"unsupported BioCLIP gate mode {self.mode!r}; expected one of: {valid}") from exc


@dataclass(frozen=True)
class ScoreInputDecision:
    should_score: bool
    visual_input_kind: VisualInputKind | None
    bioclip_gate_mode: str
    bioclip_gate_decision: GateDecision
    bioclip_gate_reason: str

    def as_row_fields(self) -> dict[str, str | None]:
        return {
            "visual_input_kind": self.visual_input_kind,
            "bioclip_gate_mode": self.bioclip_gate_mode,
            "bioclip_gate_decision": self.bioclip_gate_decision,
            "bioclip_gate_reason": self.bioclip_gate_reason,
        }


def bioclip_score_input_decision(row: dict[str, Any], policy: BioClipGatePolicy | None = None) -> ScoreInputDecision:
    active = policy or BioClipGatePolicy()
    mode = active.normalized_mode
    status = str(row.get("detection_status") or "").strip()
    label = str(row.get("detector_label") or "").strip()

    if status in {"failed_image_load", "image_load_failed"}:
        return _exclude(mode, "image_load_failed")
    if status == "no_detection":
        if active.score_no_detection_whole_image:
            return _score(mode, "whole_image", "no_detection_whole_image_fallback")
        return _exclude(mode, "no_detection_fallback_disabled")
    if status != "detected":
        return _exclude(mode, f"detection_status_not_scoreable:{status or 'missing'}")
    if not label:
        return _exclude(mode, "missing_detector_label")
    if label == "hard_negative":
        return _exclude(mode, "hard_negative_detector_label")
    if mode == BioClipGateMode.BUTTERFLY_LIKE_ONLY and label not in set(active.eligible_detector_labels):
        return _exclude(mode, "detector_label_not_eligible")
    return _score(mode, active.detected_visual_input_kind, "detected_non_hard_negative")


def _score(mode: BioClipGateMode, visual_input_kind: VisualInputKind, reason: str) -> ScoreInputDecision:
    return ScoreInputDecision(
        should_score=True,
        visual_input_kind=visual_input_kind,
        bioclip_gate_mode=mode.value,
        bioclip_gate_decision="score",
        bioclip_gate_reason=reason,
    )


def _exclude(mode: BioClipGateMode, reason: str) -> ScoreInputDecision:
    return ScoreInputDecision(
        should_score=False,
        visual_input_kind=None,
        bioclip_gate_mode=mode.value,
        bioclip_gate_decision="exclude",
        bioclip_gate_reason=reason,
    )


__all__ = [
    "BioClipGateMode",
    "BioClipGatePolicy",
    "ScoreInputDecision",
    "bioclip_score_input_decision",
]
