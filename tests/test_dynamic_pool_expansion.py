"""Tests for deterministic dynamic-pool expansion evidence."""

from __future__ import annotations

from dataclasses import replace
import json

import polars as pl
import pytest

from biominer.bioclip.dynamic_pool_expansion import (
    DYNAMIC_POOL_EXPANSION_SIGNALS,
    DynamicPoolExpansionSignalPolicy,
    build_dynamic_pool_expansion_evidence,
    default_dynamic_pool_expansion_signal_policy,
    validate_dynamic_pool_expansion_evidence,
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "run-20260718",
        "plan_id": f"dynamic-pool-plan:{'1' * 64}",
        "plan_fingerprint": _sha("2"),
        "candidate_scores_fingerprint": _sha("3"),
        "selection_policy_fingerprint": _sha("4"),
        "model_fingerprint": _sha("5"),
        "expansion_round": 0,
        "family_margin": 0.20,
        "species_margin": 0.20,
        "global_local_disagreement": 0.05,
        "prototype_method_disagreement": 0.05,
        "visual_input_disagreement": 0.05,
        "local_support_ratio": 1.0,
        "subject_area_ratio": 0.5,
        "known_competitor_margin": 0.20,
        "no_geo_global_fallback": False,
        "out_of_distribution_score": 0.1,
        "route_domain_compatible": True,
        "unavailable_signal_reasons": {},
    }
    row.update(changes)
    return row


def test_expansion_signal_policy_round_trips_and_is_fingerprinted() -> None:
    policy = default_dynamic_pool_expansion_signal_policy()
    payload = json.loads(json.dumps(policy.to_dict(), sort_keys=True))

    restored = DynamicPoolExpansionSignalPolicy.from_mapping(payload)

    assert restored == policy
    assert restored.fingerprint == policy.fingerprint
    assert replace(policy, species_margin_threshold=0.04).fingerprint != policy.fingerprint


def test_evidence_materializes_exact_triggered_and_observed_signals() -> None:
    frame = build_dynamic_pool_expansion_evidence(
        [
            _row(
                family_margin=0.05,
                species_margin=0.01,
                global_local_disagreement=0.25,
                local_support_ratio=0.25,
                subject_area_ratio=0.05,
                known_competitor_margin=0.02,
                no_geo_global_fallback=True,
                out_of_distribution_score=0.9,
                route_domain_compatible=False,
            )
        ]
    )

    validate_dynamic_pool_expansion_evidence(frame)
    row = frame.row(0, named=True)
    assert row["expansion_required"] is True
    assert row["triggered_signals"] == sorted(
        {
            "small_family_margin",
            "small_species_margin",
            "global_local_disagreement",
            "insufficient_local_support",
            "low_subject_area",
            "strong_known_competitor",
            "no_geo_global_fallback",
            "out_of_distribution",
            "route_domain_incompatible",
        }
    )
    assert row["observed_signals"] == sorted(DYNAMIC_POOL_EXPANSION_SIGNALS)
    assert row["unavailable_signals"] == []
    assert row["unavailable_signal_reasons"] == []


def test_unavailable_signals_do_not_trigger_expansion() -> None:
    frame = build_dynamic_pool_expansion_evidence(
        [
            _row(
                family_margin=None,
                species_margin=None,
                global_local_disagreement=None,
                unavailable_signal_reasons={
                    "small_family_margin": "family_evidence_unavailable",
                    "small_species_margin": "candidate_scores_unavailable",
                    "global_local_disagreement": "local_pool_unavailable",
                },
            )
        ]
    )

    row = frame.row(0, named=True)
    assert row["expansion_required"] is False
    assert row["triggered_signals"] == []
    assert row["unavailable_signals"] == [
        "global_local_disagreement",
        "small_family_margin",
        "small_species_margin",
    ]
    assert row["unavailable_signal_reasons"] == [
        "local_pool_unavailable",
        "family_evidence_unavailable",
        "candidate_scores_unavailable",
    ]


def test_evidence_is_input_order_independent_and_uniquely_grained() -> None:
    rows = [
        _row(),
        _row(
            plan_id=f"dynamic-pool-plan:{'6' * 64}",
            plan_fingerprint=_sha("7"),
        ),
    ]

    forward = build_dynamic_pool_expansion_evidence(rows)
    reverse = build_dynamic_pool_expansion_evidence(list(reversed(rows)))

    assert forward.equals(reverse)
    duplicate = pl.concat([forward, forward]).sort(
        "run_id", "plan_id", "expansion_round"
    )
    with pytest.raises(ValueError, match="grain is not unique"):
        validate_dynamic_pool_expansion_evidence(duplicate)


def test_evidence_requires_exact_reasons_for_every_unavailable_signal() -> None:
    with pytest.raises(ValueError, match="exactly match unavailable signals"):
        build_dynamic_pool_expansion_evidence([_row(family_margin=None)])

    with pytest.raises(ValueError, match="exactly match unavailable signals"):
        build_dynamic_pool_expansion_evidence(
            [
                _row(
                    unavailable_signal_reasons={
                        "small_species_margin": "unexpected_reason"
                    }
                )
            ]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family_margin", -0.01),
        ("species_margin", 2.01),
        ("local_support_ratio", 1.01),
        ("subject_area_ratio", -0.01),
        ("out_of_distribution_score", float("nan")),
    ],
)
def test_evidence_rejects_out_of_contract_signal_values(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        build_dynamic_pool_expansion_evidence([_row(**{field: value})])


def test_signal_policy_rejects_tampering_and_unbounded_thresholds() -> None:
    payload = default_dynamic_pool_expansion_signal_policy().to_dict()
    payload["policy_fingerprint"] = _sha("0")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        DynamicPoolExpansionSignalPolicy.from_mapping(payload)

    with pytest.raises(ValueError, match="minimum_subject_area_ratio"):
        replace(
            default_dynamic_pool_expansion_signal_policy(),
            minimum_subject_area_ratio=1.1,
        )
