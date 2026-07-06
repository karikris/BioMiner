from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from biominer.registry.normalize import normalize_name_key


GENERIC_SINGLE_TOKENS = {
    "butterfly",
    "butterflies",
    "swallowtail",
    "swallowtails",
    "moth",
    "moths",
    "lime",
    "lemon",
    "common",
    "papillon",
    "papillons",
    "mariposa",
    "mariposas",
    "borboleta",
    "borboletas",
    "farfalla",
    "farfalle",
    "fjäril",
    "schmetterling",
    "vlinder",
    "vlinders",
    "蝴蝶",
}
GENERATED_TRANSLATION_SOURCES = {"mymemory", "translation", "libretranslate", "t5", "machine_translation"}
SAME_TAXON_LANGUAGE_SOURCES = {"wikimedia", "wikidata"}
MANUAL_REVIEW_STATES = {"reviewed", "curator_reviewed", "manual_reviewed", "query_approved"}
SCIENTIFIC_NAME_CLASSES = {"accepted_scientific", "canonical_scientific", "scientific_synonym", "scientific", "scientific_name", "synonym"}
COMMON_NAME_CLASSES = {"vernacular", "vernacular_alias", "common_name", "common_name_alias", "generated_translation"}
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class QueryEligibilityDecision:
    query_eligible: bool
    query_disabled_reason: str
    species_specificity_score: float


def assess_name_query_eligibility(row: dict[str, Any]) -> QueryEligibilityDecision:
    """Decide whether an enabled registry name is precise enough for Flickr search."""

    display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
    normalized = normalize_name_key(display_name)
    enabled = _boolish(row.get("enabled", True))
    disabled_reason = str(row.get("disabled_reason") or "")
    name_class = str(row.get("name_class") or "").casefold()
    trust_tier = str(row.get("trust_tier") or "").upper()
    source = _source_key(row.get("source"))
    tokens = _tokens(normalized)
    score = _species_specificity_score(tokens=tokens, name_class=name_class, trust_tier=trust_tier, source=source)

    if not enabled:
        return QueryEligibilityDecision(False, disabled_reason or "name_disabled", score)
    if not normalized:
        return QueryEligibilityDecision(False, "empty_name", 0.0)
    if name_class in SCIENTIFIC_NAME_CLASSES:
        return QueryEligibilityDecision(True, "", max(score, 0.95))
    if name_class not in COMMON_NAME_CLASSES:
        return QueryEligibilityDecision(False, "unsupported_name_class_for_query", score)
    generated_translation = _is_generated_translation(name_class=name_class, trust_tier=trust_tier, source=source)
    generated_translation_approved = _generated_translation_query_approved(row, source=source) if generated_translation else False
    if generated_translation and not generated_translation_approved:
        return QueryEligibilityDecision(False, "generated_translation_requires_review_or_corroboration", min(score, 0.45))
    if _is_generic_single_token(tokens):
        return QueryEligibilityDecision(False, "generic_single_token", min(score, 0.25))
    if _is_plural_group_name(tokens):
        return QueryEligibilityDecision(False, "plural_group_name", min(score, 0.3))
    if generated_translation_approved:
        score = max(score, 0.55)
    if score < 0.5:
        return QueryEligibilityDecision(False, "low_species_specificity", score)
    return QueryEligibilityDecision(True, "", score)


def _species_specificity_score(*, tokens: list[str], name_class: str, trust_tier: str, source: str) -> float:
    if name_class in SCIENTIFIC_NAME_CLASSES:
        return 1.0 if len(tokens) >= 2 else 0.75
    if not tokens:
        return 0.0
    score = 0.35
    if len(tokens) >= 2:
        score += 0.35
    if len(tokens) >= 3:
        score += 0.1
    if trust_tier in {"T1", "T2", "T3"}:
        score += 0.15
    elif trust_tier == "T4":
        score += 0.05
    if source in SAME_TAXON_LANGUAGE_SOURCES:
        score += 0.1
    if len(tokens) == 1 and tokens[0] in GENERIC_SINGLE_TOKENS:
        score -= 0.3
    return round(max(0.0, min(score, 1.0)), 3)


def _generated_translation_query_approved(row: dict[str, Any], *, source: str) -> bool:
    if source in SAME_TAXON_LANGUAGE_SOURCES:
        return True
    review_state = "_".join(str(row.get("review_state") or "").casefold().split())
    return review_state in MANUAL_REVIEW_STATES or _boolish(row.get("corroborated", False))


def _is_generated_translation(*, name_class: str, trust_tier: str, source: str) -> bool:
    return name_class == "generated_translation" or trust_tier == "T5" or source in GENERATED_TRANSLATION_SOURCES


def _is_generic_single_token(tokens: list[str]) -> bool:
    return len(tokens) == 1 and tokens[0] in GENERIC_SINGLE_TOKENS


def _is_plural_group_name(tokens: list[str]) -> bool:
    if len(tokens) != 1:
        return False
    token = tokens[0]
    return token.endswith("s") and token in GENERIC_SINGLE_TOKENS


def _tokens(normalized_name: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(normalized_name)]


def _source_key(value: object) -> str:
    return "_".join(str(value or "").casefold().split())


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "accepted", "enabled", "reviewed", "corroborated"}


__all__ = ["QueryEligibilityDecision", "assess_name_query_eligibility"]
