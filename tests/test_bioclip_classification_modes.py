from __future__ import annotations

import pytest

from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    DEFAULT_RANK_BEAM_WIDTH,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_REPORT_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    POST_PILOT_PRODUCTION_CLASSIFICATION_MODE,
    SUPPORTED_CLASSIFICATION_MODES,
    TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
    TARGET_SCOPE_OBJECT_SCREENING,
    TARGET_FAMILY_REPORT_TOP_K,
    is_target_aware_classification,
    normalize_classification_mode,
)


def test_classification_mode_defaults_to_target_scope_object_screening() -> None:
    assert DEFAULT_CLASSIFICATION_MODE == TARGET_SCOPE_OBJECT_SCREENING
    assert normalize_classification_mode(None) == TARGET_SCOPE_OBJECT_SCREENING
    assert normalize_classification_mode("") == TARGET_SCOPE_OBJECT_SCREENING
    assert normalize_classification_mode("  ") == TARGET_SCOPE_OBJECT_SCREENING


def test_classification_mode_normalizes_aliases_and_separators() -> None:
    assert (
        normalize_classification_mode("target_scope_object_screening")
        == TARGET_SCOPE_OBJECT_SCREENING
    )
    assert (
        normalize_classification_mode("target-scope-object-screening")
        == TARGET_SCOPE_OBJECT_SCREENING
    )
    assert (
        normalize_classification_mode("hierarchical")
        == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    )
    assert (
        normalize_classification_mode("hierarchical-butterfly-classification")
        == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    )


def test_classification_mode_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="unsupported classification mode"):
        normalize_classification_mode("old_target_classifier")
    with pytest.raises(ValueError, match="unsupported classification mode"):
        normalize_classification_mode("build-week-prototype")


def test_explicit_hierarchical_mode_never_falls_back_to_target_screening() -> None:
    assert (
        normalize_classification_mode("hierarchical")
        == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    )


def test_target_aware_few_shot_mode_has_a_distinct_post_pilot_identity() -> None:
    assert (
        TARGET_AWARE_FEW_SHOT_CLASSIFICATION == "target_aware_few_shot_classification"
    )
    assert (
        POST_PILOT_PRODUCTION_CLASSIFICATION_MODE
        == TARGET_AWARE_FEW_SHOT_CLASSIFICATION
    )
    assert DEFAULT_CLASSIFICATION_MODE != TARGET_AWARE_FEW_SHOT_CLASSIFICATION
    assert TARGET_AWARE_FEW_SHOT_CLASSIFICATION in SUPPORTED_CLASSIFICATION_MODES
    assert normalize_classification_mode("target-aware-few-shot-classification") == (
        TARGET_AWARE_FEW_SHOT_CLASSIFICATION
    )
    assert (
        normalize_classification_mode("target_aware")
        == TARGET_AWARE_FEW_SHOT_CLASSIFICATION
    )
    assert is_target_aware_classification("target_aware_few_shot")


def test_default_visual_classification_widths_include_global_cascade_contract() -> None:
    assert GLOBAL_RANK_TOP_K_BEAM_STRATEGY == "global_rank_top_k"
    assert DEFAULT_RANK_BEAM_WIDTH == 3
    assert TARGET_FAMILY_REPORT_TOP_K == 3
    assert DEFAULT_SPECIES_FIRST_PASS_TOP_K == 20
    assert DEFAULT_SPECIES_RERANK_TOP_K == 5
    assert DEFAULT_SPECIES_REPORT_TOP_K == 3
