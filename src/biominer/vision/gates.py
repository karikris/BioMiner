from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, Literal


VisualInputKind = Literal["detector_crop", "detector_crop_segmentation", "whole_image"]
GateDecision = Literal["score", "review", "exclude"]

SUPPORTED_COMPARISON_ROUTES = frozenset(
    {"adult_field", "larval", "pinned_specimen"}
)
COMPARISON_ROUTE_BY_DETECTION_ROUTE = {
    "adult_butterfly_field": "adult_field",
    "caterpillar_field": "larval",
    "pinned_specimen": "pinned_specimen",
}
_ROUTING_POLICY_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}\Z")


class BioClipGateMode(StrEnum):
    ROUTED_VISUAL_DOMAIN = "routed_visual_domain"
    BUTTERFLY_LIKE_ONLY = "butterfly_like_only"
    EXCLUDE_HARD_NEGATIVE = "exclude_hard_negative"


@dataclass(frozen=True)
class BioClipGatePolicy:
    mode: BioClipGateMode | str = BioClipGateMode.ROUTED_VISUAL_DOMAIN
    supported_comparison_routes: tuple[str, ...] = ("adult_field",)
    eligible_detector_labels: tuple[str, ...] = ("butterfly_like",)
    detected_visual_input_kind: VisualInputKind = "detector_crop"
    score_no_detection_whole_image: bool = False

    def __post_init__(self) -> None:
        routes = tuple(
            dict.fromkeys(
                str(route).strip() for route in self.supported_comparison_routes
            )
        )
        invalid = sorted(set(routes) - SUPPORTED_COMPARISON_ROUTES)
        if invalid:
            raise ValueError(
                "unsupported BioCLIP comparison route(s): " + ", ".join(invalid)
            )
        object.__setattr__(self, "supported_comparison_routes", routes)

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
    detection_route: str | None
    routing_action: str | None
    bioclip_route: str | None
    routing_priority: str | None
    routing_reason: str | None
    routing_policy_version: str | None
    routing_policy_fingerprint: str | None

    def as_row_fields(self) -> dict[str, str | None]:
        return {
            "visual_input_kind": self.visual_input_kind,
            "bioclip_gate_mode": self.bioclip_gate_mode,
            "bioclip_gate_decision": self.bioclip_gate_decision,
            "bioclip_gate_reason": self.bioclip_gate_reason,
            "detection_route": self.detection_route,
            "routing_action": self.routing_action,
            "bioclip_route": self.bioclip_route,
            "routing_priority": self.routing_priority,
            "routing_reason": self.routing_reason,
            "routing_policy_version": self.routing_policy_version,
            "routing_policy_fingerprint": self.routing_policy_fingerprint,
        }


def bioclip_score_input_decision(row: dict[str, Any], policy: BioClipGatePolicy | None = None) -> ScoreInputDecision:
    active = policy or BioClipGatePolicy()
    mode = active.normalized_mode
    status = str(row.get("detection_status") or "").strip()
    label = str(row.get("detector_label") or "").strip()
    routing = _routing_fields(row)

    if mode == BioClipGateMode.ROUTED_VISUAL_DOMAIN:
        return _routed_decision(
            status=status,
            policy=active,
            routing=routing,
        )

    if status in {"failed_image_load", "image_load_failed"}:
        return _exclude(mode, "image_load_failed", routing=routing)
    if status == "no_detection":
        if active.score_no_detection_whole_image:
            return _score(
                mode,
                "whole_image",
                "no_detection_whole_image_fallback",
                routing=routing,
            )
        return _exclude(mode, "no_detection_fallback_disabled", routing=routing)
    if status != "detected":
        return _exclude(
            mode,
            f"detection_status_not_scoreable:{status or 'missing'}",
            routing=routing,
        )
    if not label:
        return _exclude(mode, "missing_detector_label", routing=routing)
    if label == "hard_negative":
        return _exclude(mode, "hard_negative_detector_label", routing=routing)
    if mode == BioClipGateMode.BUTTERFLY_LIKE_ONLY and label not in set(active.eligible_detector_labels):
        return _exclude(mode, "detector_label_not_eligible", routing=routing)
    return _score(
        mode,
        active.detected_visual_input_kind,
        "detected_non_hard_negative",
        routing=routing,
    )


def _routed_decision(
    *,
    status: str,
    policy: BioClipGatePolicy,
    routing: dict[str, str | None],
) -> ScoreInputDecision:
    if status in {"failed_image_load", "image_load_failed"}:
        return _exclude(policy.normalized_mode, "image_load_failed", routing=routing)
    if status == "no_detection":
        return _exclude(
            policy.normalized_mode,
            "routed_no_detection_not_scoreable",
            routing=routing,
        )
    if status != "detected":
        return _exclude(
            policy.normalized_mode,
            f"detection_status_not_scoreable:{status or 'missing'}",
            routing=routing,
        )

    action = routing["routing_action"]
    comparison_route = routing["bioclip_route"]
    if not action:
        return _exclude(
            policy.normalized_mode,
            "missing_routing_action",
            routing=routing,
        )
    if action in {"score", "review"}:
        identity_error = _routing_identity_error(routing)
        if identity_error is not None:
            return _exclude(
                policy.normalized_mode,
                identity_error,
                routing=routing,
            )
        if not routing["detection_route"]:
            return _exclude(
                policy.normalized_mode,
                "missing_detection_route",
                routing=routing,
            )
        if not comparison_route:
            return _exclude(
                policy.normalized_mode,
                "missing_bioclip_route",
                routing=routing,
            )
    if action == "review":
        if routing["routing_priority"] != "low":
            return _exclude(
                policy.normalized_mode,
                "review_routing_priority_not_low",
                routing=routing,
            )
        detection_route = routing["detection_route"]
        if detection_route not in {
            "adult_butterfly_field",
            "ambiguous_visual_domain",
        }:
            return _exclude(
                policy.normalized_mode,
                f"review_detection_route_not_supported:{detection_route}",
                routing=routing,
            )
        if comparison_route != "adult_field":
            return _exclude(
                policy.normalized_mode,
                "review_detection_comparison_route_mismatch:"
                f"{detection_route}:{comparison_route}",
                routing=routing,
            )
        return _review(
            policy.normalized_mode,
            "routed_for_review",
            routing=routing,
        )
    if action == "exclude":
        return _exclude(
            policy.normalized_mode,
            "routing_action_exclude",
            routing=routing,
        )
    if action != "score":
        return _exclude(
            policy.normalized_mode,
            f"unsupported_routing_action:{action}",
            routing=routing,
        )
    if comparison_route not in policy.supported_comparison_routes:
        return _exclude(
            policy.normalized_mode,
            f"unsupported_comparison_route:{comparison_route}",
            routing=routing,
        )
    detection_route = routing["detection_route"]
    expected_comparison_route = COMPARISON_ROUTE_BY_DETECTION_ROUTE.get(
        str(detection_route)
    )
    if expected_comparison_route is None:
        return _exclude(
            policy.normalized_mode,
            f"unsupported_detection_route:{detection_route}",
            routing=routing,
        )
    if comparison_route != expected_comparison_route:
        return _exclude(
            policy.normalized_mode,
            "detection_comparison_route_mismatch:"
            f"{detection_route}:{comparison_route}",
            routing=routing,
        )
    return _score(
        policy.normalized_mode,
        policy.detected_visual_input_kind,
        "routed_supported_comparison",
        routing=routing,
    )


def _score(
    mode: BioClipGateMode,
    visual_input_kind: VisualInputKind,
    reason: str,
    *,
    routing: dict[str, str | None],
) -> ScoreInputDecision:
    return ScoreInputDecision(
        should_score=True,
        visual_input_kind=visual_input_kind,
        bioclip_gate_mode=mode.value,
        bioclip_gate_decision="score",
        bioclip_gate_reason=reason,
        **routing,
    )


def _review(
    mode: BioClipGateMode,
    reason: str,
    *,
    routing: dict[str, str | None],
) -> ScoreInputDecision:
    return ScoreInputDecision(
        should_score=False,
        visual_input_kind=None,
        bioclip_gate_mode=mode.value,
        bioclip_gate_decision="review",
        bioclip_gate_reason=reason,
        **routing,
    )


def _exclude(
    mode: BioClipGateMode,
    reason: str,
    *,
    routing: dict[str, str | None],
) -> ScoreInputDecision:
    return ScoreInputDecision(
        should_score=False,
        visual_input_kind=None,
        bioclip_gate_mode=mode.value,
        bioclip_gate_decision="exclude",
        bioclip_gate_reason=reason,
        **routing,
    )


def _routing_fields(row: dict[str, Any]) -> dict[str, str | None]:
    return {
        name: _optional_string(row.get(name))
        for name in (
            "detection_route",
            "routing_action",
            "bioclip_route",
            "routing_priority",
            "routing_reason",
            "routing_policy_version",
            "routing_policy_fingerprint",
        )
    }


def _routing_identity_error(
    routing: dict[str, str | None],
) -> str | None:
    if not routing["routing_policy_version"]:
        return "missing_routing_policy_version"
    fingerprint = routing["routing_policy_fingerprint"]
    if not fingerprint:
        return "missing_routing_policy_fingerprint"
    if _ROUTING_POLICY_FINGERPRINT.fullmatch(fingerprint) is None:
        return "invalid_routing_policy_fingerprint"
    return None


def _optional_string(value: object) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


__all__ = [
    "BioClipGateMode",
    "BioClipGatePolicy",
    "COMPARISON_ROUTE_BY_DETECTION_ROUTE",
    "ScoreInputDecision",
    "SUPPORTED_COMPARISON_ROUTES",
    "bioclip_score_input_decision",
]
