from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from biominer.references.admission import (
    DEFAULT_REFERENCE_ADMISSION_MODE,
    REFERENCE_ADMISSION_POLICY_SCHEMA_VERSION,
    ReferenceAdmissionPolicy,
    default_reference_admission_policy,
)


def test_default_adaptive_policy_is_explicit_and_fingerprinted() -> None:
    policy = default_reference_admission_policy()

    assert policy.schema_version == REFERENCE_ADMISSION_POLICY_SCHEMA_VERSION
    assert policy.mode == DEFAULT_REFERENCE_ADMISSION_MODE
    assert policy.allowed_provider_sources == ("gbif",)
    assert policy.allowed_unreviewed_routes == ("adult_field",)
    assert policy.require_yoloe_route is True
    assert policy.require_canonical_media is True
    assert policy.require_statistical_audit is True
    assert policy.fingerprint.startswith("sha256:")
    assert len(policy.fingerprint) == 71
    payload = policy.to_dict()
    assert payload["policy_fingerprint"] == policy.fingerprint
    assert set(payload) == {
        "schema_version",
        "policy_version",
        "mode",
        "allowed_provider_sources",
        "allowed_unreviewed_routes",
        "accepted_taxon_reconciliation_statuses",
        "accepted_licence_policy_statuses",
        "minimum_decoded_width",
        "minimum_decoded_height",
        "minimum_subject_area_ratio",
        "require_yoloe_route",
        "require_canonical_media",
        "maximum_images_per_observation",
        "maximum_images_per_observer_before_reuse",
        "permit_research_only_licence",
        "require_statistical_audit",
        "audit_policy_version",
        "policy_fingerprint",
    }


def test_policy_is_immutable_and_normalizes_unordered_semantics() -> None:
    policy = default_reference_admission_policy()
    reordered = replace(
        policy,
        accepted_taxon_reconciliation_statuses=(
            "accepted_name_synonym",
            "accepted_key_exact",
            "accepted_name_synonym",
        ),
    )

    assert reordered.accepted_taxon_reconciliation_statuses == (
        "accepted_key_exact",
        "accepted_name_synonym",
    )
    assert reordered.fingerprint == policy.fingerprint
    with pytest.raises(FrozenInstanceError):
        policy.mode = "human_verified_strict"  # type: ignore[misc]


def test_mapping_round_trip_requires_every_field_and_matching_fingerprint() -> None:
    policy = default_reference_admission_policy()
    payload = policy.to_dict()

    assert ReferenceAdmissionPolicy.from_mapping(payload) == policy

    missing_mode = dict(payload)
    missing_mode.pop("mode")
    with pytest.raises(ValueError, match="missing=.*mode"):
        ReferenceAdmissionPolicy.from_mapping(missing_mode)

    tampered = dict(payload)
    tampered["minimum_decoded_width"] = 256
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ReferenceAdmissionPolicy.from_mapping(tampered)

    unknown = {**payload, "legacy_fast_start": True}
    with pytest.raises(ValueError, match="unknown=.*legacy_fast_start"):
        ReferenceAdmissionPolicy.from_mapping(unknown)


def test_adaptive_policy_fails_closed_on_weakened_core_gates() -> None:
    policy = default_reference_admission_policy()

    with pytest.raises(ValueError, match="only the GBIF"):
        replace(policy, allowed_provider_sources=("gbif", "inaturalist"))
    with pytest.raises(ValueError, match="explicit unreviewed route"):
        replace(policy, allowed_unreviewed_routes=())
    with pytest.raises(ValueError, match="positive subject-area"):
        replace(policy, minimum_subject_area_ratio=0)
    with pytest.raises(ValueError, match="YOLOE routing and canonical media"):
        replace(policy, require_yoloe_route=False)
    with pytest.raises(ValueError, match="statistical audit"):
        replace(policy, require_statistical_audit=False)


def test_strict_policy_cannot_silently_admit_unreviewed_routes() -> None:
    policy = default_reference_admission_policy()
    strict = replace(
        policy,
        policy_version="human-verified-strict-v1",
        mode="human_verified_strict",
        allowed_unreviewed_routes=(),
        minimum_subject_area_ratio=0,
        require_yoloe_route=False,
        require_canonical_media=False,
        require_statistical_audit=False,
        audit_policy_version="not-applicable-strict-v1",
    )

    assert strict.mode == "human_verified_strict"
    assert strict.allowed_unreviewed_routes == ()
    assert strict.fingerprint != policy.fingerprint
    with pytest.raises(ValueError, match="strict admission"):
        replace(strict, allowed_unreviewed_routes=("adult_field",))


def test_research_only_permission_and_status_must_agree() -> None:
    policy = default_reference_admission_policy()

    with pytest.raises(ValueError, match="must agree"):
        replace(policy, permit_research_only_licence=True)
    permitted = replace(
        policy,
        accepted_licence_policy_statuses=("allowed", "research_only"),
        permit_research_only_licence=True,
    )
    assert permitted.permit_research_only_licence is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minimum_decoded_width", 0, "positive integer"),
        ("minimum_decoded_height", True, "positive integer"),
        ("maximum_images_per_observation", 0, "positive integer"),
        ("maximum_images_per_observer_before_reuse", -1, "positive integer"),
        ("minimum_subject_area_ratio", 1.1, "in \\[0, 1\\]"),
        ("require_canonical_media", 1, "must be Boolean"),
    ],
)
def test_policy_rejects_invalid_threshold_and_boolean_values(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(default_reference_admission_policy(), **{field: value})
