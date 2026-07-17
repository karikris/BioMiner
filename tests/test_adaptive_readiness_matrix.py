from __future__ import annotations

from dataclasses import replace
import json

import polars as pl
import pytest

import biominer.references.readiness as readiness_module
from biominer.references.admission import default_reference_admission_policy
from biominer.references.readiness import (
    reference_readiness_allows_vision,
    reference_support_manifest_fingerprint,
    reference_support_manifest_schema,
    validate_reference_bank_readiness,
    validate_reference_support_manifest,
)
from biominer.references.support_admission import evaluate_support_admission
from test_reference_readiness import _build, _make_fixture, _policy, _sha
from test_support_admission import _evidence


def _provisional_payload() -> dict[str, object]:
    payload = json.loads(json.dumps(_build(_make_fixture()).readiness))
    policy = default_reference_admission_policy()
    payload.update(
        {
            "status": "ready_provisional",
            "permits_vision": True,
            "permits_reference_embedding": True,
            "permits_provisional_scoring": True,
            "permits_calibrated_scoring": False,
            "permits_scientific_release": False,
            "reference_admission_mode": policy.mode,
            "admission_policy_fingerprint": policy.fingerprint,
            "provisional_support_count": 2,
            "human_verified_support_count": 0,
            "statistical_audit_required": True,
        }
    )
    payload["counts"]["provisional_support_count"] = 2
    payload["counts"]["human_verified_support_count"] = 0
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check["check_id"] != "strict_support_only"
    ]
    _resign_bank_context(payload)
    return payload


def _resign_bank_context(payload: dict[str, object]) -> None:
    payload["bank_fingerprint"] = readiness_module.canonical_semantic_fingerprint(
        {
            "schema_version": payload["schema_version"],
            "reference_bank_version": payload["reference_bank_version"],
            "registry_version": payload["registry_version"],
            "target_accepted_taxon_key": payload["target_accepted_taxon_key"],
            "policy_fingerprint": payload["policy_fingerprint"],
            "reference_admission_mode": payload["reference_admission_mode"],
            "admission_policy_fingerprint": payload[
                "admission_policy_fingerprint"
            ],
            "model_input_fingerprint": payload["model_input_fingerprint"],
            "candidate_set_ids": payload["candidate_set_ids"],
            "candidate_set_fingerprints": payload[
                "candidate_set_fingerprints"
            ],
            "inputs": payload["inputs"],
        }
    )
    for check in payload["checks"]:
        check["evidence"]["reference_bank_fingerprint"] = payload[
            "bank_fingerprint"
        ]


def test_provisional_ready_permits_screening_but_not_release() -> None:
    payload = _provisional_payload()

    readiness_module._validate_readiness_payload(payload, published=False)  # noqa: SLF001
    assert payload["status"] == "ready_provisional"
    assert reference_readiness_allows_vision(payload)
    assert payload["permits_provisional_scoring"] is True
    assert payload["permits_calibrated_scoring"] is False
    assert payload["permits_scientific_release"] is False
    assert payload["statistical_audit_required"] is True


def test_strict_ready_retains_release_capability() -> None:
    result = _build(_make_fixture())

    validate_reference_bank_readiness(result)
    assert result.readiness["status"] == "ready"
    assert result.readiness["reference_admission_mode"] == "human_verified_strict"
    assert result.readiness["permits_scientific_release"] is True


def test_strict_blocked_and_target_support_shortfall_fail_closed() -> None:
    result = _build(_make_fixture(), policy=_policy(target_minimum=2))

    assert result.readiness["status"] == "blocked_missing_target_support"
    assert result.readiness["counts"]["target_minimum_shortfall_count"] == 1
    assert reference_readiness_allows_vision(result.readiness) is False


def test_provisional_blocked_has_no_scoring_capability() -> None:
    payload = _provisional_payload()
    payload["status"] = "blocked_missing_target_support"
    payload["permits_vision"] = False
    payload.update(readiness_module._readiness_capabilities(payload["status"]))  # noqa: SLF001
    payload["counts"]["target_minimum_shortfall_count"] = 1
    target = next(
        check for check in payload["checks"] if check["check_id"] == "target_adult_minimum"
    )
    target.update(status="failed", observed=0, required=1)

    readiness_module._validate_readiness_payload(payload, published=False)  # noqa: SLF001
    assert payload["permits_reference_embedding"] is False
    assert payload["permits_provisional_scoring"] is False


def test_stale_admission_policy_fingerprint_invalidates_permit() -> None:
    payload = _provisional_payload()
    payload["admission_policy_fingerprint"] = _sha("stale-policy")

    with pytest.raises(ValueError, match="fingerprint"):
        readiness_module._validate_readiness_payload(payload, published=False)  # noqa: SLF001


def test_admission_mode_change_invalidates_existing_readiness() -> None:
    result = _build(_make_fixture())
    strict_fingerprint = reference_support_manifest_fingerprint(
        result.support_manifest
    )
    adaptive = default_reference_admission_policy()
    rows = []
    for source in result.support_manifest.iter_rows(named=True):
        row = dict(source)
        row["reference_admission_mode"] = adaptive.mode
        row["reference_admission_policy_version"] = adaptive.policy_version
        row["reference_admission_policy_fingerprint"] = adaptive.fingerprint
        row["support_row_fingerprint"] = readiness_module._support_row_fingerprint(row)  # noqa: SLF001
        rows.append(row)
    changed = pl.DataFrame(
        rows,
        schema=reference_support_manifest_schema(),
        orient="row",
        strict=True,
    ).sort(readiness_module._SUPPORT_SORT)  # noqa: SLF001

    validate_reference_support_manifest(changed)
    assert reference_support_manifest_fingerprint(changed) != strict_fingerprint
    with pytest.raises(ValueError, match="support manifest fingerprint mismatch"):
        validate_reference_bank_readiness(replace(result, support_manifest=changed))


def test_adaptive_mode_without_audit_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="statistical audit"):
        replace(
            default_reference_admission_policy(),
            require_statistical_audit=False,
        )


def test_wrong_route_blocks_provisional_support() -> None:
    result = evaluate_support_admission(
        replace(_evidence(), route_compatible=False),
        default_reference_admission_policy(),
    )

    assert result.eligible is False
    assert "reference_route_incompatible" in result.reasons


def test_human_rejection_overrides_provisional_readiness_path() -> None:
    result = evaluate_support_admission(
        replace(
            _evidence(),
            human_rejected=True,
            human_rejection_reasons=("wrong species",),
        ),
        default_reference_admission_policy(),
    )

    assert result.eligible is False
    assert result.evidence_path == "none"
    assert "human_rejection_override" in result.reasons
