from __future__ import annotations

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
]

TARGET_SCOPE_OBJECT_SCREENING: ClassificationMode = "target_scope_object_screening"
HIERARCHICAL_BUTTERFLY_CLASSIFICATION: ClassificationMode = "hierarchical_butterfly_classification"
DEFAULT_CLASSIFICATION_MODE: ClassificationMode = TARGET_SCOPE_OBJECT_SCREENING
DEFAULT_FAMILY_TOP_K = 3

SUPPORTED_CLASSIFICATION_MODES: tuple[ClassificationMode, ...] = (
    TARGET_SCOPE_OBJECT_SCREENING,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
)

CLASSIFICATION_MODE_ALIASES: dict[str, ClassificationMode] = {
    "target_screening": TARGET_SCOPE_OBJECT_SCREENING,
    "target_scope_screening": TARGET_SCOPE_OBJECT_SCREENING,
    "target_scope_object_screening": TARGET_SCOPE_OBJECT_SCREENING,
    "hierarchical": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    "hierarchical_butterfly": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    "hierarchical_butterfly_classification": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
}


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


def is_hierarchical_classification(mode: str | None) -> TypeGuard[Literal["hierarchical_butterfly_classification"]]:
    return normalize_classification_mode(mode) == HIERARCHICAL_BUTTERFLY_CLASSIFICATION


__all__ = [
    "CLASSIFICATION_MODE_ALIASES",
    "DEFAULT_CLASSIFICATION_MODE",
    "DEFAULT_FAMILY_TOP_K",
    "DEFAULT_RANK_BEAM_WIDTH",
    "DEFAULT_SPECIES_FIRST_PASS_TOP_K",
    "DEFAULT_SPECIES_REPORT_TOP_K",
    "DEFAULT_SPECIES_RERANK_TOP_K",
    "GLOBAL_RANK_TOP_K_BEAM_STRATEGY",
    "HIERARCHICAL_BUTTERFLY_CLASSIFICATION",
    "SUPPORTED_CLASSIFICATION_MODES",
    "TARGET_SCOPE_OBJECT_SCREENING",
    "ClassificationMode",
    "is_hierarchical_classification",
    "normalize_classification_mode",
]
