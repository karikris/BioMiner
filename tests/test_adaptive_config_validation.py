from __future__ import annotations

import pytest

from biominer.run import ProductionRunRequest, RunStage
from biominer.run.adaptive_config import (
    AdaptiveReferenceSettings,
    AdaptiveReferenceValidationContext,
    validate_adaptive_reference_settings,
)


def _validate(
    settings: AdaptiveReferenceSettings,
    **context: object,
) -> AdaptiveReferenceSettings:
    return validate_adaptive_reference_settings(
        settings,
        context=AdaptiveReferenceValidationContext(**context),
    )


def test_safe_adaptive_defaults_validate() -> None:
    settings = AdaptiveReferenceSettings()

    assert _validate(settings) is settings


def test_provisional_references_cannot_claim_strict_readiness() -> None:
    with pytest.raises(ValueError, match="provisional.*strict"):
        _validate(AdaptiveReferenceSettings(), strict_readiness_claim=True)


@pytest.mark.parametrize("split_use", ["calibration", "final_test"])
def test_unreviewed_references_cannot_enter_protected_splits(
    split_use: str,
) -> None:
    with pytest.raises(ValueError, match="unreviewed references"):
        _validate(
            AdaptiveReferenceSettings(),
            reference_split_uses=(split_use,),
        )


def test_final_flickr_export_cannot_disable_human_review() -> None:
    with pytest.raises(ValueError, match="final Flickr export requires human review"):
        _validate(
            AdaptiveReferenceSettings(
                flickr_release_requires_human_review=False,
            )
        )


def test_calibrated_probability_requires_calibrator() -> None:
    with pytest.raises(ValueError, match="valid calibrator"):
        _validate(
            AdaptiveReferenceSettings(
                initial_scoring_mode="calibrated_probability",
            )
        )


def test_adaptive_mode_requires_statistical_audit() -> None:
    with pytest.raises(ValueError, match="statistical reference audit"):
        _validate(
            AdaptiveReferenceSettings(statistical_reference_audit=False),
        )


def test_non_gbif_unreviewed_source_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported.*inat"):
        _validate(AdaptiveReferenceSettings(reference_source="inat"))


def test_strict_reviewed_mode_can_use_non_gbif_source_and_protected_splits() -> None:
    settings = AdaptiveReferenceSettings(
        reference_admission_mode="human_verified_strict",
        reference_source="inat",
        initial_scoring_mode="calibrated_probability",
    )

    assert _validate(
        settings,
        reference_split_uses=("calibration", "final_test"),
        calibrator_available=True,
    ) is settings


def test_production_request_uses_the_same_cross_field_validator() -> None:
    with pytest.raises(ValueError, match="provisional.*strict"):
        ProductionRunRequest(
            taxon="Papilio demoleus",
            strict_reference_readiness_claim=True,
        )
    with pytest.raises(ValueError, match="final Flickr export requires human review"):
        ProductionRunRequest(
            taxon="Papilio demoleus",
            flickr_release_requires_human_review=False,
            stages=(RunStage.FINAL_QUALITY_GATE,),
        )
