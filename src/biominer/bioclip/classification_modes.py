from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeGuard

from biominer.bioclip.cascade_contract import (
    DEFAULT_RANK_BEAM_WIDTH,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_REPORT_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
)


ClassificationMode = Literal[
    "target_scope_object_screening",
    "hierarchical_butterfly_classification",
    "target_aware_few_shot_classification",
]

TARGET_SCOPE_OBJECT_SCREENING: ClassificationMode = "target_scope_object_screening"
HIERARCHICAL_BUTTERFLY_CLASSIFICATION: ClassificationMode = (
    "hierarchical_butterfly_classification"
)
TARGET_AWARE_FEW_SHOT_CLASSIFICATION: ClassificationMode = (
    "target_aware_few_shot_classification"
)
DEFAULT_CLASSIFICATION_MODE: ClassificationMode = TARGET_SCOPE_OBJECT_SCREENING
# Promote this to the default only after the frozen pilot satisfies its acceptance policy.
POST_PILOT_PRODUCTION_CLASSIFICATION_MODE: ClassificationMode = (
    TARGET_AWARE_FEW_SHOT_CLASSIFICATION
)
TARGET_FAMILY_REPORT_TOP_K = 3

SUPPORTED_CLASSIFICATION_MODES: tuple[ClassificationMode, ...] = (
    TARGET_SCOPE_OBJECT_SCREENING,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
)

CLASSIFICATION_MODE_ALIASES: dict[str, ClassificationMode] = {
    "target_scope_object_screening": TARGET_SCOPE_OBJECT_SCREENING,
    "hierarchical": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    "hierarchical_butterfly": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    "hierarchical_butterfly_classification": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    "target_aware": TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
    "target_aware_few_shot": TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
    "target_aware_few_shot_classification": TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
}


@dataclass(frozen=True, slots=True)
class ClassificationModeContract:
    classification_mode: ClassificationMode
    deployment_status: Literal["diagnostic", "production"]
    target_always_scored: bool
    complete_regional_candidate_union_required: bool
    hierarchy_pruning_permitted: bool
    spatial_crop_permitted: bool
    visual_input: str
    silent_fallback_permitted: bool
    output_status: Literal["diagnostic", "production"]
    diagnostic_baselines: tuple[str, ...]


def classification_mode_contract(
    mode: str | None,
) -> ClassificationModeContract:
    normalized = normalize_classification_mode(mode)
    if normalized == TARGET_AWARE_FEW_SHOT_CLASSIFICATION:
        return ClassificationModeContract(
            classification_mode=normalized,
            deployment_status="production",
            target_always_scored=True,
            complete_regional_candidate_union_required=True,
            hierarchy_pruning_permitted=False,
            spatial_crop_permitted=False,
            visual_input="raw_full_image",
            silent_fallback_permitted=False,
            output_status="production",
            diagnostic_baselines=(
                TARGET_SCOPE_OBJECT_SCREENING,
                HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
                "B0",
            ),
        )
    return ClassificationModeContract(
        classification_mode=normalized,
        deployment_status="diagnostic",
        target_always_scored=normalized == TARGET_SCOPE_OBJECT_SCREENING,
        complete_regional_candidate_union_required=False,
        hierarchy_pruning_permitted=(
            normalized == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
        ),
        spatial_crop_permitted=True,
        visual_input="detector_crop",
        silent_fallback_permitted=False,
        output_status="diagnostic",
        diagnostic_baselines=(),
    )


def normalize_classification_mode(value: str | None) -> ClassificationMode:
    if value is None or not str(value).strip():
        return DEFAULT_CLASSIFICATION_MODE
    normalized = str(value).strip().casefold().replace("-", "_")
    try:
        return CLASSIFICATION_MODE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(SUPPORTED_CLASSIFICATION_MODES)
        aliases = ", ".join(sorted(CLASSIFICATION_MODE_ALIASES))
        raise ValueError(
            f"unsupported classification mode {value!r}; expected one of: {allowed}; aliases: {aliases}"
        ) from exc


def is_hierarchical_classification(
    mode: str | None,
) -> TypeGuard[Literal["hierarchical_butterfly_classification"]]:
    return normalize_classification_mode(mode) == HIERARCHICAL_BUTTERFLY_CLASSIFICATION


def is_target_aware_classification(
    mode: str | None,
) -> TypeGuard[Literal["target_aware_few_shot_classification"]]:
    return normalize_classification_mode(mode) == TARGET_AWARE_FEW_SHOT_CLASSIFICATION


__all__ = [
    "CLASSIFICATION_MODE_ALIASES",
    "ClassificationModeContract",
    "DEFAULT_CLASSIFICATION_MODE",
    "DEFAULT_RANK_BEAM_WIDTH",
    "DEFAULT_SPECIES_FIRST_PASS_TOP_K",
    "DEFAULT_SPECIES_REPORT_TOP_K",
    "DEFAULT_SPECIES_RERANK_TOP_K",
    "GLOBAL_RANK_TOP_K_BEAM_STRATEGY",
    "HIERARCHICAL_BUTTERFLY_CLASSIFICATION",
    "POST_PILOT_PRODUCTION_CLASSIFICATION_MODE",
    "SUPPORTED_CLASSIFICATION_MODES",
    "TARGET_AWARE_FEW_SHOT_CLASSIFICATION",
    "TARGET_SCOPE_OBJECT_SCREENING",
    "TARGET_FAMILY_REPORT_TOP_K",
    "ClassificationMode",
    "classification_mode_contract",
    "is_hierarchical_classification",
    "is_target_aware_classification",
    "normalize_classification_mode",
]
