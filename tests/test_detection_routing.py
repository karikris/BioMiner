from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from biominer.detection.routing import (
    BIOCLIP_ROUTES,
    DETECTION_ROUTES,
    ROUTING_ACTIONS,
    ROUTING_PRIORITIES,
    DetectionRouteDecision,
    DetectionRoutingPolicy,
    route_detection,
)


def _row(
    prompt: str | None = "butterfly",
    *,
    label: str = "butterfly_like",
    score: object = 0.9,
    status: str = "detected",
    include_prompt: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "detection_status": status,
        "detector_label": label,
        "detector_score": score,
    }
    if include_prompt:
        row["detector_prompt"] = prompt
    return row


def test_routing_contract_has_only_the_required_closed_values() -> None:
    assert set(DETECTION_ROUTES) == {
        "adult_butterfly_field",
        "caterpillar_field",
        "pupa_or_chrysalis",
        "pinned_specimen",
        "possible_moth_or_other_insect",
        "artwork_logo_tattoo_or_other_artifact",
        "no_relevant_organism",
        "ambiguous_visual_domain",
    }
    assert set(BIOCLIP_ROUTES) == {"adult_field", "larval", "pinned_specimen"}
    assert set(ROUTING_ACTIONS) == {"score", "review", "exclude"}
    assert set(ROUTING_PRIORITIES) == {"standard", "low", "none"}


def test_routing_policy_fingerprint_is_canonical_and_covers_every_policy_field() -> None:
    policy = DetectionRoutingPolicy()
    payload = {
        "ambiguous_insect_review_enabled": True,
        "ambiguous_insect_review_threshold": 0.35,
        "possible_adult_route_enabled": True,
        "possible_adult_route_threshold": 0.35,
        "version": "detection-routing-policy-v1",
    }
    expected = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert policy.fingerprint == expected
    for field_name, replacement in (
        ("version", "detection-routing-policy-v2"),
        ("possible_adult_route_enabled", False),
        ("possible_adult_route_threshold", 0.36),
        ("ambiguous_insect_review_enabled", False),
        ("ambiguous_insect_review_threshold", 0.36),
    ):
        assert replace(policy, **{field_name: replacement}).fingerprint != policy.fingerprint


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"version": ""}, "version"),
        ({"possible_adult_route_enabled": 1}, "possible_adult_route_enabled"),
        ({"ambiguous_insect_review_enabled": "yes"}, "ambiguous_insect_review_enabled"),
        ({"possible_adult_route_threshold": -0.01}, "possible_adult_route_threshold"),
        ({"ambiguous_insect_review_threshold": 1.01}, "ambiguous_insect_review_threshold"),
        ({"possible_adult_route_threshold": float("nan")}, "possible_adult_route_threshold"),
    ],
)
def test_routing_policy_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DetectionRoutingPolicy(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("prompt", "label", "route", "action", "bioclip_route", "priority", "reason"),
    [
        (
            "butterfly",
            "butterfly_like",
            "adult_butterfly_field",
            "score",
            "adult_field",
            "standard",
            "definite_adult_prompt",
        ),
        (
            "caterpillar",
            "caterpillar",
            "caterpillar_field",
            "score",
            "larval",
            "standard",
            "larval_prompt",
        ),
        (
            "chrysalis",
            "pupa",
            "pupa_or_chrysalis",
            "exclude",
            None,
            "none",
            "pupa_prompt_no_bioclip_route",
        ),
        (
            "pinned butterfly specimen",
            "butterfly_like",
            "pinned_specimen",
            "score",
            "pinned_specimen",
            "standard",
            "pinned_specimen_prompt",
        ),
        (
            "moth",
            "moth_like",
            "possible_moth_or_other_insect",
            "exclude",
            None,
            "none",
            "other_insect_prompt",
        ),
        (
            "drawing",
            "hard_negative",
            "artwork_logo_tattoo_or_other_artifact",
            "exclude",
            None,
            "none",
            "artifact_prompt",
        ),
        (
            "flower",
            "hard_negative",
            "no_relevant_organism",
            "exclude",
            None,
            "none",
            "no_relevant_organism_prompt",
        ),
    ],
)
def test_known_prompts_route_to_separate_comparison_domains(
    prompt: str,
    label: str,
    route: str,
    action: str,
    bioclip_route: str | None,
    priority: str,
    reason: str,
) -> None:
    decision = route_detection(_row(prompt, label=label))

    assert (
        decision.detection_route,
        decision.routing_action,
        decision.bioclip_route,
        decision.routing_priority,
        decision.routing_reason,
    ) == (route, action, bioclip_route, priority, reason)


@pytest.mark.parametrize("prompt", ["butterfly wing", "lepidoptera", "possible adult butterfly"])
def test_possible_adult_threshold_is_inclusive_and_routes_only_when_enabled(prompt: str) -> None:
    policy = DetectionRoutingPolicy(possible_adult_route_threshold=0.35)

    accepted = route_detection(_row(prompt, score=0.35), policy)
    below = route_detection(_row(prompt, score=0.349999), policy)
    disabled = route_detection(
        _row(prompt, score=0.99),
        replace(policy, possible_adult_route_enabled=False),
    )

    assert (
        accepted.detection_route,
        accepted.routing_action,
        accepted.bioclip_route,
        accepted.routing_priority,
        accepted.routing_reason,
    ) == (
        "adult_butterfly_field",
        "score",
        "adult_field",
        "standard",
        "possible_adult_prompt_above_threshold",
    )
    assert (below.detection_route, below.routing_action, below.bioclip_route) == (
        "ambiguous_visual_domain",
        "exclude",
        None,
    )
    assert below.routing_reason == "possible_adult_route_below_threshold"
    assert disabled.routing_reason == "possible_adult_route_disabled"


def test_ambiguous_insect_review_threshold_is_inclusive_and_configurable() -> None:
    policy = DetectionRoutingPolicy(ambiguous_insect_review_threshold=0.4)

    accepted = route_detection(_row("insect", label="insect_like", score=0.4), policy)
    below = route_detection(_row("insect", label="insect_like", score=0.399999), policy)
    disabled = route_detection(
        _row("insect", label="insect_like", score=0.99),
        replace(policy, ambiguous_insect_review_enabled=False),
    )

    assert (
        accepted.detection_route,
        accepted.routing_action,
        accepted.bioclip_route,
        accepted.routing_priority,
        accepted.routing_reason,
    ) == (
        "ambiguous_visual_domain",
        "review",
        "adult_field",
        "low",
        "ambiguous_insect_review_above_threshold",
    )
    assert below.routing_action == "exclude"
    assert below.routing_reason == "ambiguous_insect_review_below_threshold"
    assert disabled.routing_action == "exclude"
    assert disabled.routing_reason == "ambiguous_insect_review_disabled"


def test_unknown_or_empty_raw_prompt_fails_closed_even_when_legacy_label_looks_positive() -> None:
    for prompt in ("dragonfly", "", "   "):
        decision = route_detection(_row(prompt, label="butterfly_like"))
        assert (decision.detection_route, decision.routing_action, decision.bioclip_route) == (
            "ambiguous_visual_domain",
            "exclude",
            None,
        )
        assert decision.routing_reason == "unknown_detector_prompt"


def test_invalid_detector_score_fails_closed() -> None:
    for score in (None, "not-a-score", float("nan"), -0.1, 1.1):
        decision = route_detection(
            _row("butterfly wing", label="possible_adult_butterfly", score=score)
        )
        assert decision.detection_route == "ambiguous_visual_domain"
        assert decision.routing_action == "exclude"
        assert decision.routing_reason == "invalid_detector_score"

    invalid_definite = route_detection(_row("butterfly", score="not-a-score"))
    assert invalid_definite.routing_action == "exclude"
    assert invalid_definite.routing_reason == "invalid_detector_score"


def test_definite_routes_do_not_require_a_score() -> None:
    prompted = route_detection(_row("butterfly", score=None))

    assert prompted.routing_action == "score"
    assert prompted.bioclip_route == "adult_field"


def test_detected_rows_without_detector_prompt_are_excluded() -> None:
    decision = route_detection(
        {
            "detection_status": "detected",
            "detector_label": "butterfly_like",
            "detector_score": 0.9,
        }
    )

    assert (decision.detection_route, decision.routing_action, decision.bioclip_route) == (
        "ambiguous_visual_domain",
        "exclude",
        None,
    )
    assert decision.routing_reason == "missing_detector_prompt"


def test_no_detection_is_not_conflated_with_detector_failure() -> None:
    no_detection = route_detection(_row(None, label="no_detection", score=0.0, status="no_detection"))
    failed = route_detection(_row(None, label="", score=0.0, status="failed"))

    assert no_detection.detection_route == "no_relevant_organism"
    assert no_detection.routing_reason == "detector_reported_no_detection"
    assert no_detection.routing_action == "exclude"
    assert failed.detection_route == "ambiguous_visual_domain"
    assert failed.routing_reason == "detection_status_not_routable"
    assert failed.routing_action == "exclude"


def test_route_decision_exports_exact_persisted_fields_and_review_property() -> None:
    policy = DetectionRoutingPolicy()
    decision = route_detection(_row("insect", label="insect_like"), policy)

    assert decision.requires_manual_review is True
    assert decision.as_row_fields() == {
        "detection_route": "ambiguous_visual_domain",
        "routing_action": "review",
        "bioclip_route": "adult_field",
        "routing_priority": "low",
        "routing_reason": "ambiguous_insect_review_above_threshold",
        "routing_policy_version": policy.version,
        "routing_policy_fingerprint": policy.fingerprint,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"detection_route": "unknown"},
        {"routing_action": "maybe"},
        {"bioclip_route": "pupal"},
        {"routing_priority": "urgent"},
        {
            "detection_route": "caterpillar_field",
            "routing_action": "score",
            "bioclip_route": "adult_field",
        },
        {
            "detection_route": "pinned_specimen",
            "routing_action": "score",
            "bioclip_route": "adult_field",
        },
        {
            "detection_route": "no_relevant_organism",
            "routing_action": "review",
            "bioclip_route": None,
        },
    ],
)
def test_route_decision_rejects_unknown_or_cross_domain_combinations(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "detection_route": "adult_butterfly_field",
        "routing_action": "score",
        "bioclip_route": "adult_field",
        "routing_priority": "standard",
        "routing_reason": "test",
        "routing_policy_version": "test-v1",
        "routing_policy_fingerprint": "sha256:" + "a" * 64,
    }
    values.update(kwargs)
    if values["routing_action"] == "review":
        values["routing_priority"] = "low"

    with pytest.raises(ValueError):
        DetectionRouteDecision(**values)  # type: ignore[arg-type]
