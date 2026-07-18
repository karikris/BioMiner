"""Tests for the immutable dynamic reference-pool policy."""

from __future__ import annotations

from dataclasses import replace
import json
import re

import pytest

from biominer.bioclip.dynamic_pool_policy import (
    DYNAMIC_REFERENCE_POOL_GEOGRAPHIC_FALLBACK_ORDER,
    DYNAMIC_REFERENCE_POOL_POLICY_SCHEMA_VERSION,
    DYNAMIC_REFERENCE_POOL_STAGE_ORDER,
    DynamicReferencePoolPolicy,
    default_dynamic_reference_pool_policy,
)


def test_default_dynamic_pool_policy_covers_required_planning_controls() -> None:
    policy = default_dynamic_reference_pool_policy()
    payload = policy.to_dict()

    assert policy.schema_version == DYNAMIC_REFERENCE_POOL_POLICY_SCHEMA_VERSION
    assert tuple(stage for stage, _limit in policy.stage_member_limits) == (
        DYNAMIC_REFERENCE_POOL_STAGE_ORDER
    )
    assert [limit for _stage, limit in policy.stage_member_limits] == [96, 144, 192]
    assert policy.minimum_global_per_candidate <= policy.maximum_global_per_candidate
    assert policy.minimum_local_per_candidate <= policy.maximum_local_per_candidate
    assert policy.maximum_total_reference_members == 192
    assert policy.class_balance_mode == "round_robin_equal_quota"
    assert policy.maximum_class_count_difference == 1
    assert (
        policy.geographic_fallback_order
        == DYNAMIC_REFERENCE_POOL_GEOGRAPHIC_FALLBACK_ORDER
    )
    assert policy.maximum_members_per_burst_group == 1
    assert policy.maximum_members_per_observer == 2
    assert policy.maximum_members_per_locality == 2
    assert policy.uncertainty_expansion_increment == 2
    assert policy.always_include_target_candidate is True
    assert policy.always_include_safety_union_candidates is True
    assert policy.family_hard_pruning_allowed is False
    assert policy.geography_hard_pruning_allowed is False
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", policy.fingerprint)
    assert payload["policy_fingerprint"] == policy.fingerprint
    assert {
        "stage_member_limits",
        "minimum_global_per_candidate",
        "maximum_local_per_candidate",
        "maximum_total_reference_members",
        "class_balance_mode",
        "geographic_fallback_order",
        "maximum_members_per_observer",
        "uncertainty_margin_threshold",
        "always_include_target_candidate",
        "selection_seed",
        "policy_fingerprint",
    } <= set(payload)


def test_dynamic_pool_policy_json_round_trip_preserves_identity() -> None:
    policy = default_dynamic_reference_pool_policy()
    encoded = json.loads(json.dumps(policy.to_dict(), sort_keys=True))

    restored = DynamicReferencePoolPolicy.from_mapping(encoded)

    assert restored == policy
    assert restored.fingerprint == policy.fingerprint
    assert restored.to_dict() == policy.to_dict()


def test_every_policy_change_changes_fingerprint() -> None:
    policy = default_dynamic_reference_pool_policy()

    variants = (
        replace(policy, selection_seed=policy.selection_seed + 1),
        replace(policy, uncertainty_margin_threshold=0.04),
        replace(policy, maximum_members_per_observer=1),
        replace(policy, maximum_class_count_difference=0),
    )

    assert all(item.fingerprint != policy.fingerprint for item in variants)
    assert len({item.fingerprint for item in variants}) == len(variants)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"minimum_global_per_candidate": 9}, "global minimum"),
        ({"minimum_local_per_candidate": 7}, "local minimum"),
        (
            {
                "stage_member_limits": (
                    ("initial", 96),
                    ("uncertainty_expansion", 80),
                    ("selective_rescore", 192),
                )
            },
            "nondecreasing",
        ),
        ({"maximum_total_reference_members": 10}, "total pool budget"),
        (
            {"geographic_fallback_order": tuple(reversed(DYNAMIC_REFERENCE_POOL_GEOGRAPHIC_FALLBACK_ORDER))},
            "canonical safe sequence",
        ),
        ({"maximum_members_per_burst_group": 2}, "at most one"),
        ({"minimum_global_countries_per_candidate": 9}, "country minimum"),
        ({"allow_no_geo_global_fallback": False}, "no-geo global fallback"),
        ({"always_include_target_candidate": False}, "configured target"),
        ({"always_include_safety_union_candidates": False}, "safety union"),
        ({"family_hard_pruning_allowed": True}, "family evidence cannot hard-prune"),
        (
            {"geography_hard_pruning_allowed": True},
            "geography evidence cannot hard-prune",
        ),
        ({"uncertainty_margin_threshold": 2.1}, r"within \[0, 2\]"),
    ],
)
def test_dynamic_pool_policy_rejects_unsafe_or_inconsistent_configuration(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(default_dynamic_reference_pool_policy(), **changes)


def test_dynamic_pool_policy_rejects_tampered_or_unknown_serialization() -> None:
    payload = default_dynamic_reference_pool_policy().to_dict()
    payload["policy_fingerprint"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        DynamicReferencePoolPolicy.from_mapping(payload)

    payload = default_dynamic_reference_pool_policy().to_dict()
    payload["legacy_fallback"] = True
    with pytest.raises(ValueError, match="fields do not match"):
        DynamicReferencePoolPolicy.from_mapping(payload)
