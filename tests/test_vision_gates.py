from __future__ import annotations

import pytest

from biominer.vision.gates import BioClipGatePolicy, bioclip_score_input_decision


_ROUTING_POLICY_FINGERPRINT = "sha256:" + "a" * 64


def _routed_row(
    *,
    detection_status: str = "detected",
    routing_action: str = "score",
    bioclip_route: str = "adult_field",
    routing_priority: str = "standard",
) -> dict[str, object]:
    return {
        "detection_status": detection_status,
        "detector_label": "butterfly_like",
        "detection_route": "adult_butterfly_field",
        "routing_action": routing_action,
        "bioclip_route": bioclip_route,
        "routing_priority": routing_priority,
        "routing_reason": "test routing decision",
        "routing_policy_version": "detection-routing-v1",
        "routing_policy_fingerprint": _ROUTING_POLICY_FINGERPRINT,
    }


def test_routed_gate_is_default_and_scores_supported_adult_field_route() -> None:
    decision = bioclip_score_input_decision(_routed_row())

    assert decision.should_score is True
    assert decision.visual_input_kind == "detector_crop"
    assert decision.bioclip_gate_mode == "routed_visual_domain"
    assert decision.bioclip_gate_decision == "score"
    assert decision.bioclip_gate_reason == "routed_supported_comparison"
    assert decision.detection_route == "adult_butterfly_field"
    assert decision.routing_action == "score"
    assert decision.bioclip_route == "adult_field"
    assert decision.routing_priority == "standard"
    assert decision.routing_reason == "test routing decision"
    assert decision.routing_policy_version == "detection-routing-v1"
    assert decision.routing_policy_fingerprint == _ROUTING_POLICY_FINGERPRINT


@pytest.mark.parametrize(
    ("detection_route", "route"),
    [
        ("adult_butterfly_field", "adult_field"),
        ("caterpillar_field", "larval"),
        ("pinned_specimen", "pinned_specimen"),
    ],
)
def test_routed_gate_scores_only_explicitly_supported_comparison_routes(
    detection_route: str,
    route: str,
) -> None:
    decision = bioclip_score_input_decision(
        {
            **_routed_row(bioclip_route=route),
            "detection_route": detection_route,
        },
        BioClipGatePolicy(supported_comparison_routes=(
            "adult_field",
            "larval",
            "pinned_specimen",
        )),
    )

    assert decision.should_score is True
    assert decision.bioclip_route == route


def test_routed_gate_excludes_corrupt_detection_and_comparison_route_pair() -> None:
    decision = bioclip_score_input_decision(
        _routed_row(bioclip_route="larval"),
        BioClipGatePolicy(supported_comparison_routes=("adult_field", "larval")),
    )

    assert decision.should_score is False
    assert decision.bioclip_gate_decision == "exclude"
    assert decision.bioclip_gate_reason == (
        "detection_comparison_route_mismatch:adult_butterfly_field:larval"
    )


def test_routed_gate_excludes_unsupported_pupa_comparison() -> None:
    decision = bioclip_score_input_decision(
        _routed_row(bioclip_route="pupal"),
        BioClipGatePolicy(supported_comparison_routes=(
            "adult_field",
            "larval",
            "pinned_specimen",
        )),
    )

    assert decision.should_score is False
    assert decision.visual_input_kind is None
    assert decision.bioclip_gate_decision == "exclude"
    assert decision.bioclip_gate_reason == "unsupported_comparison_route:pupal"
    assert decision.bioclip_route == "pupal"


def test_routed_review_is_low_priority_and_unscored() -> None:
    decision = bioclip_score_input_decision(
        _routed_row(
            routing_action="review",
            bioclip_route="adult_field",
            routing_priority="low",
        )
        | {"detection_route": "ambiguous_visual_domain"}
    )

    assert decision.should_score is False
    assert decision.visual_input_kind is None
    assert decision.bioclip_gate_decision == "review"
    assert decision.bioclip_gate_reason == "routed_for_review"
    assert decision.routing_priority == "low"


@pytest.mark.parametrize(
    ("detection_route", "bioclip_route", "expected_reason"),
    [
        (
            "caterpillar_field",
            "larval",
            "review_detection_route_not_supported:caterpillar_field",
        ),
        (
            "pinned_specimen",
            "pinned_specimen",
            "review_detection_route_not_supported:pinned_specimen",
        ),
        (
            "ambiguous_visual_domain",
            "larval",
            "review_detection_comparison_route_mismatch:ambiguous_visual_domain:larval",
        ),
    ],
)
def test_routed_review_rejects_cross_domain_route_metadata(
    detection_route: str,
    bioclip_route: str,
    expected_reason: str,
) -> None:
    decision = bioclip_score_input_decision(
        {
            **_routed_row(
                routing_action="review",
                bioclip_route=bioclip_route,
                routing_priority="low",
            ),
            "detection_route": detection_route,
        }
    )

    assert decision.should_score is False
    assert decision.bioclip_gate_decision == "exclude"
    assert decision.bioclip_gate_reason == expected_reason


def test_routed_no_detection_is_not_scoreable() -> None:
    decision = bioclip_score_input_decision(
        _routed_row(
            detection_status="no_detection",
            routing_action="exclude",
            bioclip_route="none",
        )
    )

    assert decision.should_score is False
    assert decision.visual_input_kind is None
    assert decision.bioclip_gate_decision == "exclude"
    assert decision.bioclip_gate_reason == "routed_no_detection_not_scoreable"


def test_routed_gate_fails_closed_when_route_metadata_is_missing() -> None:
    decision = bioclip_score_input_decision(
        {"detection_status": "detected", "detector_label": "butterfly_like"}
    )

    assert decision.should_score is False
    assert decision.bioclip_gate_decision == "exclude"
    assert decision.bioclip_gate_reason == "missing_routing_action"


@pytest.mark.parametrize(
    ("missing_field", "expected_reason"),
    [
        ("routing_policy_version", "missing_routing_policy_version"),
        ("routing_policy_fingerprint", "missing_routing_policy_fingerprint"),
    ],
)
def test_routed_score_requires_complete_policy_identity(
    missing_field: str,
    expected_reason: str,
) -> None:
    row = _routed_row()
    row[missing_field] = None

    decision = bioclip_score_input_decision(row)

    assert decision.should_score is False
    assert decision.bioclip_gate_reason == expected_reason


def test_routed_score_rejects_malformed_policy_fingerprint() -> None:
    decision = bioclip_score_input_decision(
        _routed_row() | {"routing_policy_fingerprint": "sha256:not-a-digest"}
    )

    assert decision.should_score is False
    assert decision.bioclip_gate_reason == "invalid_routing_policy_fingerprint"
