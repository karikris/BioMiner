"""Tests for typed dynamic-pooling workflow settings."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from biominer.run.dynamic_pool_config import (
    DYNAMIC_POOLING_SETTINGS_FILE,
    DynamicPoolingSettings,
    load_dynamic_pooling_settings,
    write_dynamic_pooling_settings,
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def test_default_settings_are_safe_unselected_and_round_trip(tmp_path) -> None:
    settings = DynamicPoolingSettings()

    assert settings.candidate_strategy is None
    assert settings.fusion_method is None
    assert settings.selection_status == "unselected"
    assert settings.release_requires_human_review is True
    assert settings.representative_probability_sampling_required is True
    assert settings.selective_rerun_enabled is True
    assert settings.missing_geography_is_biological_absence is False
    assert settings.raw_scores_are_probabilities is False
    assert settings.fingerprint.startswith("sha256:")

    path = write_dynamic_pooling_settings(settings, tmp_path)
    assert path.name == DYNAMIC_POOLING_SETTINGS_FILE
    assert load_dynamic_pooling_settings(path) == settings
    assert json.loads(path.read_text(encoding="utf-8"))["settings_fingerprint"] == (
        settings.fingerprint
    )


def test_selected_strategy_and_fusion_require_bound_evidence() -> None:
    with pytest.raises(ValueError, match="requires evidence fingerprint"):
        DynamicPoolingSettings(candidate_strategy="parallel_family_geography_union")
    with pytest.raises(ValueError, match="requires evidence fingerprint"):
        DynamicPoolingSettings(fusion_method="unweighted_component_mean")

    partial = DynamicPoolingSettings(
        candidate_strategy="parallel_family_geography_union",
        candidate_strategy_selection_fingerprint=_sha("a"),
    )
    assert partial.selection_status == "partially_selected_with_bound_evidence"

    settings = DynamicPoolingSettings(
        candidate_strategy="parallel_family_geography_union",
        candidate_strategy_selection_fingerprint=_sha("a"),
        fusion_method="validation_fitted_linear",
        fusion_selection_fingerprint=_sha("b"),
    )
    assert settings.selection_status == "selected_with_bound_evidence"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("release_requires_human_review", False),
        ("representative_probability_sampling_required", False),
        ("selective_rerun_enabled", False),
        ("missing_geography_is_biological_absence", True),
        ("raw_scores_are_probabilities", True),
    ],
)
def test_settings_reject_unsafe_semantic_overrides(field: str, value: bool) -> None:
    with pytest.raises(ValueError, match="dynamic-pooling safety requires"):
        DynamicPoolingSettings(**{field: value})


def test_serialized_settings_reject_fingerprint_or_policy_tampering() -> None:
    settings = DynamicPoolingSettings()
    tampered = settings.to_dict()
    tampered["review_budget"] = 51
    with pytest.raises(ValueError, match="settings fingerprint mismatch"):
        DynamicPoolingSettings.from_mapping(tampered)

    tampered = settings.to_dict()
    pool = dict(tampered["reference_pool_policy"])
    pool["maximum_total_reference_members"] = 193
    tampered["reference_pool_policy"] = pool
    with pytest.raises(ValueError, match="policy fingerprint mismatch"):
        DynamicPoolingSettings.from_mapping(tampered)


def test_settings_fingerprint_changes_for_bounded_runtime_controls() -> None:
    first = DynamicPoolingSettings()
    second = replace(first, vector_score_batch_size=256)

    assert first.fingerprint != second.fingerprint
