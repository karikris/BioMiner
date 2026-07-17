from __future__ import annotations

from dataclasses import dataclass
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
BIOCLIP_GATE_MODE = "routed_visual_domain"


@dataclass(frozen=True)
class BioClipGatePolicy:
    supported_comparison_routes: tuple[str, ...] = ("adult_field",)
    detected_visual_input_kind: VisualInputKind = "detector_crop"

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
    status = str(row.get("detection_status") or "").strip()
    routing = _routing_fields(row)
    return _routed_decision(
        status=status,
        policy=active,
        routing=routing,
    )


def _routed_decision(
    *,
    status: str,
    policy: BioClipGatePolicy,
    routing: dict[str, str | None],
) -> ScoreInputDecision:
    if status in {"failed_image_load", "image_load_failed"}:
        return _exclude("image_load_failed", routing=routing)
    if status == "no_detection":
        return _exclude(
            "routed_no_detection_not_scoreable",
            routing=routing,
        )
    if status != "detected":
        return _exclude(
            f"detection_status_not_scoreable:{status or 'missing'}",
            routing=routing,
        )

    action = routing["routing_action"]
    comparison_route = routing["bioclip_route"]
    if not action:
        return _exclude(
            "missing_routing_action",
            routing=routing,
        )
    if action in {"score", "review"}:
        identity_error = _routing_identity_error(routing)
        if identity_error is not None:
            return _exclude(
                identity_error,
                routing=routing,
            )
        if not routing["detection_route"]:
            return _exclude(
                "missing_detection_route",
                routing=routing,
            )
        if not comparison_route:
            return _exclude(
                "missing_bioclip_route",
                routing=routing,
            )
    if action == "review":
        if routing["routing_priority"] != "low":
            return _exclude(
                "review_routing_priority_not_low",
                routing=routing,
            )
        detection_route = routing["detection_route"]
        if detection_route not in {
            "adult_butterfly_field",
            "ambiguous_visual_domain",
        }:
            return _exclude(
                f"review_detection_route_not_supported:{detection_route}",
                routing=routing,
            )
        if comparison_route != "adult_field":
            return _exclude(
                "review_detection_comparison_route_mismatch:"
                f"{detection_route}:{comparison_route}",
                routing=routing,
            )
        return _review(
            "routed_for_review",
            routing=routing,
        )
    if action == "exclude":
        return _exclude(
            "routing_action_exclude",
            routing=routing,
        )
    if action != "score":
        return _exclude(
            f"unsupported_routing_action:{action}",
            routing=routing,
        )
    if comparison_route not in policy.supported_comparison_routes:
        return _exclude(
            f"unsupported_comparison_route:{comparison_route}",
            routing=routing,
        )
    detection_route = routing["detection_route"]
    expected_comparison_route = COMPARISON_ROUTE_BY_DETECTION_ROUTE.get(
        str(detection_route)
    )
    if expected_comparison_route is None:
        return _exclude(
            f"unsupported_detection_route:{detection_route}",
            routing=routing,
        )
    if comparison_route != expected_comparison_route:
        return _exclude(
            "detection_comparison_route_mismatch:"
            f"{detection_route}:{comparison_route}",
            routing=routing,
        )
    return _score(
        policy.detected_visual_input_kind,
        "routed_supported_comparison",
        routing=routing,
    )


def _score(
    visual_input_kind: VisualInputKind,
    reason: str,
    *,
    routing: dict[str, str | None],
) -> ScoreInputDecision:
    return ScoreInputDecision(
        should_score=True,
        visual_input_kind=visual_input_kind,
        bioclip_gate_mode=BIOCLIP_GATE_MODE,
        bioclip_gate_decision="score",
        bioclip_gate_reason=reason,
        **routing,
    )


def _review(
    reason: str,
    *,
    routing: dict[str, str | None],
) -> ScoreInputDecision:
    return ScoreInputDecision(
        should_score=False,
        visual_input_kind=None,
        bioclip_gate_mode=BIOCLIP_GATE_MODE,
        bioclip_gate_decision="review",
        bioclip_gate_reason=reason,
        **routing,
    )


def _exclude(
    reason: str,
    *,
    routing: dict[str, str | None],
) -> ScoreInputDecision:
    return ScoreInputDecision(
        should_score=False,
        visual_input_kind=None,
        bioclip_gate_mode=BIOCLIP_GATE_MODE,
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
    "BIOCLIP_GATE_MODE",
    "BioClipGatePolicy",
    "COMPARISON_ROUTE_BY_DETECTION_ROUTE",
    "ScoreInputDecision",
    "SUPPORTED_COMPARISON_ROUTES",
    "bioclip_score_input_decision",
]
