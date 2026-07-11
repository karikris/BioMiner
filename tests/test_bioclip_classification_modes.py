from __future__ import annotations

import pytest

from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    DEFAULT_FAMILY_TOP_K,
    DEFAULT_RANK_BEAM_WIDTH,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_REPORT_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    TARGET_SCOPE_OBJECT_SCREENING,
    normalize_classification_mode,
)


def test_classification_mode_defaults_to_target_scope_object_screening() -> None:
    assert DEFAULT_CLASSIFICATION_MODE == TARGET_SCOPE_OBJECT_SCREENING
    assert normalize_classification_mode(None) == TARGET_SCOPE_OBJECT_SCREENING
    assert normalize_classification_mode("") == TARGET_SCOPE_OBJECT_SCREENING
    assert normalize_classification_mode("  ") == TARGET_SCOPE_OBJECT_SCREENING


def test_classification_mode_normalizes_aliases_and_separators() -> None:
    assert normalize_classification_mode("target_scope_object_screening") == TARGET_SCOPE_OBJECT_SCREENING
    assert normalize_classification_mode("target-scope-object-screening") == TARGET_SCOPE_OBJECT_SCREENING
    assert normalize_classification_mode("target_screening") == TARGET_SCOPE_OBJECT_SCREENING
    assert normalize_classification_mode("hierarchical") == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    assert normalize_classification_mode("hierarchical-butterfly-classification") == HIERARCHICAL_BUTTERFLY_CLASSIFICATION


def test_classification_mode_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="unsupported classification mode"):
        normalize_classification_mode("old_target_classifier")


def test_explicit_hierarchical_mode_never_falls_back_to_target_screening() -> None:
    assert normalize_classification_mode("hierarchical") == HIERARCHICAL_BUTTERFLY_CLASSIFICATION


def test_default_visual_classification_widths_include_global_cascade_contract() -> None:
    assert GLOBAL_RANK_TOP_K_BEAM_STRATEGY == "global_rank_top_k"
    assert DEFAULT_RANK_BEAM_WIDTH == 3
    assert DEFAULT_FAMILY_TOP_K == 3
    assert DEFAULT_SPECIES_FIRST_PASS_TOP_K == 20
    assert DEFAULT_SPECIES_RERANK_TOP_K == 5
    assert DEFAULT_SPECIES_REPORT_TOP_K == 3
