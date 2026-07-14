from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from collections.abc import Mapping


DETECTION_ROUTES: tuple[str, ...] = (
    "adult_butterfly_field",
    "caterpillar_field",
    "pupa_or_chrysalis",
    "pinned_specimen",
    "possible_moth_or_other_insect",
    "artwork_logo_tattoo_or_other_artifact",
    "no_relevant_organism",
    "ambiguous_visual_domain",
)
BIOCLIP_ROUTES: tuple[str, ...] = ("adult_field", "larval", "pinned_specimen")
ROUTING_ACTIONS: tuple[str, ...] = ("score", "review", "exclude")
ROUTING_PRIORITIES: tuple[str, ...] = ("standard", "low", "none")

_DEFINITE_ADULT_PROMPTS = frozenset(
    {
        "adult butterfly",
        "butterfly",
        "live adult butterfly",
    }
)
_POSSIBLE_ADULT_PROMPTS = frozenset(
    {
        "butterfly wing",
        "lepidoptera",
        "possible adult butterfly",
    }
)
_LARVAL_PROMPTS = frozenset({"butterfly caterpillar", "caterpillar", "larva"})
_PUPA_PROMPTS = frozenset({"butterfly chrysalis", "chrysalis", "pupa"})
_PINNED_SPECIMEN_PROMPTS = frozenset(
    {
        "butterfly specimen",
        "pinned butterfly",
        "pinned butterfly specimen",
    }
)
_OTHER_INSECT_PROMPTS = frozenset({"moth", "other insect"})
_AMBIGUOUS_INSECT_PROMPTS = frozenset({"insect", "insect-like organism"})
_ARTIFACT_PROMPTS = frozenset(
    {
        "artwork",
        "butterfly artwork",
        "butterfly logo",
        "butterfly tattoo",
        "drawing",
        "logo",
        "museum label",
        "painting",
        "pattern",
        "sign",
        "tattoo",
        "text",
        "textile",
    }
)
_NO_ORGANISM_PROMPTS = frozenset(
    {
        "flower",
        "hand",
        "leaf",
        "no relevant organism",
        "person",
    }
)


@dataclass(frozen=True)
class DetectionRoutingPolicy:
    version: str = "detection-routing-policy-v1"
    possible_adult_route_enabled: bool = True
    possible_adult_route_threshold: float = 0.35
    ambiguous_insect_review_enabled: bool = True
    ambiguous_insect_review_threshold: float = 0.35
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        version = str(self.version).strip()
        if not version:
            raise ValueError("version must be non-empty")
        object.__setattr__(self, "version", version)
        _require_bool(self.possible_adult_route_enabled, "possible_adult_route_enabled")
        _require_bool(self.ambiguous_insect_review_enabled, "ambiguous_insect_review_enabled")
        _require_probability(self.possible_adult_route_threshold, "possible_adult_route_threshold")
        _require_probability(self.ambiguous_insect_review_threshold, "ambiguous_insect_review_threshold")
        object.__setattr__(self, "possible_adult_route_threshold", float(self.possible_adult_route_threshold))
        object.__setattr__(self, "ambiguous_insect_review_threshold", float(self.ambiguous_insect_review_threshold))
        object.__setattr__(self, "fingerprint", _policy_fingerprint(self))


@dataclass(frozen=True)
class DetectionRouteDecision:
    detection_route: str
    routing_action: str
    bioclip_route: str | None
    routing_priority: str
    routing_reason: str
    routing_policy_version: str
    routing_policy_fingerprint: str

    def __post_init__(self) -> None:
        if self.detection_route not in DETECTION_ROUTES:
            raise ValueError(f"unsupported detection route: {self.detection_route!r}")
        if self.routing_action not in ROUTING_ACTIONS:
            raise ValueError(f"unsupported routing action: {self.routing_action!r}")
        if self.bioclip_route is not None and self.bioclip_route not in BIOCLIP_ROUTES:
            raise ValueError(f"unsupported BioCLIP route: {self.bioclip_route!r}")
        if self.routing_priority not in ROUTING_PRIORITIES:
            raise ValueError(f"unsupported routing priority: {self.routing_priority!r}")
        if not str(self.routing_reason).strip():
            raise ValueError("routing_reason must be non-empty")
        if not str(self.routing_policy_version).strip():
            raise ValueError("routing_policy_version must be non-empty")
        if not _is_sha256(self.routing_policy_fingerprint):
            raise ValueError("routing_policy_fingerprint must be a sha256 digest")
        if not _valid_route_combination(self):
            raise ValueError(
                "invalid detection route/action/BioCLIP route/priority combination: "
                f"{self.detection_route}/{self.routing_action}/{self.bioclip_route}/{self.routing_priority}"
            )

    @property
    def requires_manual_review(self) -> bool:
        return self.routing_action == "review"

    def as_row_fields(self) -> dict[str, object]:
        return {
            "detection_route": self.detection_route,
            "routing_action": self.routing_action,
            "bioclip_route": self.bioclip_route,
            "routing_priority": self.routing_priority,
            "routing_reason": self.routing_reason,
            "routing_policy_version": self.routing_policy_version,
            "routing_policy_fingerprint": self.routing_policy_fingerprint,
        }


def route_detection(
    row: Mapping[str, object],
    policy: DetectionRoutingPolicy | None = None,
) -> DetectionRouteDecision:
    active = policy or DEFAULT_DETECTION_ROUTING_POLICY
    status = _normalise_token(row.get("detection_status"))
    if status == "no_detection":
        return _excluded(active, "no_relevant_organism", "detector_reported_no_detection")
    if status != "detected":
        return _excluded(active, "ambiguous_visual_domain", "detection_status_not_routable")

    raw_score = row.get("detector_score")
    score = _detector_score(raw_score)
    if raw_score is not None and score is None:
        return _excluded(active, "ambiguous_visual_domain", "invalid_detector_score")

    raw_prompt = row.get("detector_prompt")
    if "detector_prompt" in row and raw_prompt is not None:
        return _route_prompt(_normalise_prompt(raw_prompt), score=score, policy=active)
    return _route_legacy_label(_normalise_token(row.get("detector_label")), score=score, policy=active)


def _route_prompt(
    prompt: str,
    *,
    score: float | None,
    policy: DetectionRoutingPolicy,
) -> DetectionRouteDecision:
    if prompt in _DEFINITE_ADULT_PROMPTS:
        return _score(policy, "adult_butterfly_field", "adult_field", "definite_adult_prompt")
    if prompt in _POSSIBLE_ADULT_PROMPTS:
        return _possible_adult_decision(score=score, policy=policy, reason_prefix="possible_adult_prompt")
    if prompt in _LARVAL_PROMPTS:
        return _score(policy, "caterpillar_field", "larval", "larval_prompt")
    if prompt in _PUPA_PROMPTS:
        return _excluded(policy, "pupa_or_chrysalis", "pupa_prompt_no_bioclip_route")
    if prompt in _PINNED_SPECIMEN_PROMPTS:
        return _score(policy, "pinned_specimen", "pinned_specimen", "pinned_specimen_prompt")
    if prompt in _OTHER_INSECT_PROMPTS:
        return _excluded(policy, "possible_moth_or_other_insect", "other_insect_prompt")
    if prompt in _AMBIGUOUS_INSECT_PROMPTS:
        return _ambiguous_insect_decision(score=score, policy=policy, reason_prefix="ambiguous_insect_review")
    if prompt in _ARTIFACT_PROMPTS:
        return _excluded(policy, "artwork_logo_tattoo_or_other_artifact", "artifact_prompt")
    if prompt in _NO_ORGANISM_PROMPTS:
        return _excluded(policy, "no_relevant_organism", "no_relevant_organism_prompt")
    return _excluded(policy, "ambiguous_visual_domain", "unknown_detector_prompt")


def _route_legacy_label(
    label: str,
    *,
    score: float | None,
    policy: DetectionRoutingPolicy,
) -> DetectionRouteDecision:
    reason = f"legacy_detector_label_fallback:{label or 'missing'}"
    if label in {"adult_butterfly", "butterfly_like"}:
        return _score(policy, "adult_butterfly_field", "adult_field", reason)
    if label == "possible_adult_butterfly":
        return _possible_adult_decision(score=score, policy=policy, reason_prefix=reason)
    if label == "caterpillar":
        return _score(policy, "caterpillar_field", "larval", reason)
    if label == "pupa":
        return _excluded(policy, "pupa_or_chrysalis", reason)
    if label == "pinned_specimen":
        return _score(policy, "pinned_specimen", "pinned_specimen", reason)
    if label == "moth_like":
        return _excluded(policy, "possible_moth_or_other_insect", reason)
    if label == "insect_like":
        return _ambiguous_insect_decision(score=score, policy=policy, reason_prefix=reason)
    if label in {"artifact", "hard_negative"}:
        return _excluded(policy, "artwork_logo_tattoo_or_other_artifact", reason)
    if label in {"no_organism", "no_relevant_organism"}:
        return _excluded(policy, "no_relevant_organism", reason)
    return _excluded(policy, "ambiguous_visual_domain", "unknown_detector_label_fallback")


def _possible_adult_decision(
    *,
    score: float | None,
    policy: DetectionRoutingPolicy,
    reason_prefix: str,
) -> DetectionRouteDecision:
    if not policy.possible_adult_route_enabled:
        reason = (
            "possible_adult_route_disabled"
            if reason_prefix == "possible_adult_prompt"
            else f"{reason_prefix}:route_disabled"
        )
        return _excluded(policy, "ambiguous_visual_domain", reason)
    if score is None:
        return _excluded(policy, "ambiguous_visual_domain", "invalid_detector_score")
    if score < policy.possible_adult_route_threshold:
        reason = (
            "possible_adult_route_below_threshold"
            if reason_prefix == "possible_adult_prompt"
            else f"{reason_prefix}:below_threshold"
        )
        return _excluded(policy, "ambiguous_visual_domain", reason)
    reason = (
        "possible_adult_prompt_above_threshold"
        if reason_prefix == "possible_adult_prompt"
        else reason_prefix
    )
    return _score(policy, "adult_butterfly_field", "adult_field", reason)


def _ambiguous_insect_decision(
    *,
    score: float | None,
    policy: DetectionRoutingPolicy,
    reason_prefix: str,
) -> DetectionRouteDecision:
    if not policy.ambiguous_insect_review_enabled:
        reason = (
            "ambiguous_insect_review_disabled"
            if reason_prefix == "ambiguous_insect_review"
            else f"{reason_prefix}:review_disabled"
        )
        return _excluded(policy, "ambiguous_visual_domain", reason)
    if score is None:
        return _excluded(policy, "ambiguous_visual_domain", "invalid_detector_score")
    if score < policy.ambiguous_insect_review_threshold:
        reason = (
            "ambiguous_insect_review_below_threshold"
            if reason_prefix == "ambiguous_insect_review"
            else f"{reason_prefix}:below_threshold"
        )
        return _excluded(policy, "ambiguous_visual_domain", reason)
    reason = (
        "ambiguous_insect_review_above_threshold"
        if reason_prefix == "ambiguous_insect_review"
        else reason_prefix
    )
    return _review(policy, "ambiguous_visual_domain", "adult_field", reason)


def _score(
    policy: DetectionRoutingPolicy,
    detection_route: str,
    bioclip_route: str,
    reason: str,
) -> DetectionRouteDecision:
    return _decision(
        policy,
        detection_route=detection_route,
        action="score",
        bioclip_route=bioclip_route,
        priority="standard",
        reason=reason,
    )


def _review(
    policy: DetectionRoutingPolicy,
    detection_route: str,
    bioclip_route: str,
    reason: str,
) -> DetectionRouteDecision:
    return _decision(
        policy,
        detection_route=detection_route,
        action="review",
        bioclip_route=bioclip_route,
        priority="low",
        reason=reason,
    )


def _excluded(
    policy: DetectionRoutingPolicy,
    detection_route: str,
    reason: str,
) -> DetectionRouteDecision:
    return _decision(
        policy,
        detection_route=detection_route,
        action="exclude",
        bioclip_route=None,
        priority="none",
        reason=reason,
    )


def _decision(
    policy: DetectionRoutingPolicy,
    *,
    detection_route: str,
    action: str,
    bioclip_route: str | None,
    priority: str,
    reason: str,
) -> DetectionRouteDecision:
    return DetectionRouteDecision(
        detection_route=detection_route,
        routing_action=action,
        bioclip_route=bioclip_route,
        routing_priority=priority,
        routing_reason=reason,
        routing_policy_version=policy.version,
        routing_policy_fingerprint=policy.fingerprint,
    )


def _valid_route_combination(decision: DetectionRouteDecision) -> bool:
    key = (
        decision.detection_route,
        decision.routing_action,
        decision.bioclip_route,
        decision.routing_priority,
    )
    return key in {
        ("adult_butterfly_field", "score", "adult_field", "standard"),
        ("adult_butterfly_field", "review", "adult_field", "low"),
        ("caterpillar_field", "score", "larval", "standard"),
        ("pinned_specimen", "score", "pinned_specimen", "standard"),
        ("ambiguous_visual_domain", "review", "adult_field", "low"),
        ("ambiguous_visual_domain", "exclude", None, "none"),
        ("pupa_or_chrysalis", "exclude", None, "none"),
        ("possible_moth_or_other_insect", "exclude", None, "none"),
        ("artwork_logo_tattoo_or_other_artifact", "exclude", None, "none"),
        ("no_relevant_organism", "exclude", None, "none"),
    }


def _policy_fingerprint(policy: DetectionRoutingPolicy) -> str:
    payload = {
        "version": policy.version,
        "possible_adult_route_enabled": policy.possible_adult_route_enabled,
        "possible_adult_route_threshold": policy.possible_adult_route_threshold,
        "ambiguous_insect_review_enabled": policy.ambiguous_insect_review_enabled,
        "ambiguous_insect_review_threshold": policy.ambiguous_insect_review_threshold,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalise_prompt(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _normalise_token(value: object) -> str:
    return "_".join(str(value or "").strip().casefold().split())


def _detector_score(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def _require_bool(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_probability(value: object, field_name: str) -> None:
    score = _detector_score(value)
    if score is None:
        raise ValueError(f"{field_name} must be a finite number between 0.0 and 1.0")


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    if not text.startswith("sha256:") or len(text) != 71:
        return False
    return all(character in "0123456789abcdef" for character in text[7:])


DEFAULT_DETECTION_ROUTING_POLICY = DetectionRoutingPolicy()


__all__ = [
    "BIOCLIP_ROUTES",
    "DEFAULT_DETECTION_ROUTING_POLICY",
    "DETECTION_ROUTES",
    "ROUTING_ACTIONS",
    "ROUTING_PRIORITIES",
    "DetectionRouteDecision",
    "DetectionRoutingPolicy",
    "route_detection",
]
