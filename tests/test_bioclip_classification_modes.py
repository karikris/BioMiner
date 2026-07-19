from __future__ import annotations

import pytest

from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    SUPPORTED_CLASSIFICATION_MODES,
    TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
    is_target_aware_classification,
    normalize_classification_mode,
)


def test_classification_mode_defaults_to_target_aware_full_frame() -> None:
    assert DEFAULT_CLASSIFICATION_MODE == TARGET_AWARE_FEW_SHOT_CLASSIFICATION
    assert normalize_classification_mode(None) == TARGET_AWARE_FEW_SHOT_CLASSIFICATION
    assert normalize_classification_mode("") == TARGET_AWARE_FEW_SHOT_CLASSIFICATION
    assert normalize_classification_mode("  ") == TARGET_AWARE_FEW_SHOT_CLASSIFICATION


def test_target_aware_mode_aliases_share_one_production_identity() -> None:
    assert TARGET_AWARE_FEW_SHOT_CLASSIFICATION in SUPPORTED_CLASSIFICATION_MODES
    for value in (
        "target-aware-few-shot-classification",
        "target_aware",
        "target_aware_few_shot",
    ):
        assert normalize_classification_mode(value) == (
            TARGET_AWARE_FEW_SHOT_CLASSIFICATION
        )
        assert is_target_aware_classification(value)


@pytest.mark.parametrize("value", ["old_target_classifier", "build-week-prototype"])
def test_classification_mode_rejects_removed_workflows(value: str) -> None:
    with pytest.raises(ValueError, match="unsupported classification mode"):
        normalize_classification_mode(value)
