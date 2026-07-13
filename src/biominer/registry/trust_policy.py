from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TrustTier(StrEnum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"


TRUST_TIER_DEFINITIONS: dict[TrustTier, str] = {
    TrustTier.T1: "CoL XR accepted names, scientific synonyms, and CoL/GBIF vernaculars bound to accepted taxa",
    TrustTier.T2: "NCBI, Open Tree, ITIS/EOL, specialist authorities, and reviewed curated vernaculars",
    TrustTier.T3: "confidently linked Wikispecies, Wikidata, iNaturalist, BOLD, and corroborated checklist names",
    TrustTier.T4: "community names, homonyms, weak aliases, spelling variants, and unreviewed regional terms",
    TrustTier.T5: "machine-generated and dictionary-derived translations",
}

ACCEPTED_REVIEW_STATES = {"accepted", "reviewed", "enabled"}
COLLISION_FREE_VALUES = {"", "none", "no", "false", "clean", "no_collision", "unambiguous", "unique"}
COLLISION_VALUES = {"collision", "conflict", "ambiguous", "cross_taxon_collision"}
LOW_CONFIDENCE_VALUES = {"", "low", "weak"}
HIGH_CONFIDENCE_VALUES = {"high", "confident", "accepted"}

AUTHORITY_VERNACULAR_SOURCES = {
    "catalogue_of_life",
    "col",
    "eol",
    "gbif",
    "itis",
    "tmd",
    "tmd_de",
}
T1_SOURCES = {"catalogue_of_life", "col", "col_xr", "gbif"}
T2_SOURCES = {
    "ncbi",
    "open_tree",
    "opentree",
    "itis",
    "eol",
    "specialist_authority",
    "reviewed_curated",
    "tmd",
    "tmd_de",
}
T3_SOURCES = {
    "wikispecies",
    "wikidata",
    "inaturalist",
    "inat",
    "bold",
    "corroborated_checklist",
}
COMMUNITY_SOURCES = {"inaturalist", "inat", "flickr", "community"}
GENERATED_SOURCES = {"dictionary", "generated", "libretranslate", "machine_translation", "translation"}


@dataclass(frozen=True)
class TrustDecision:
    trust_tier: TrustTier
    enabled: bool
    disabled_reason: str = ""


def normalize_trust_tier(value: str | TrustTier | None) -> TrustTier:
    if isinstance(value, TrustTier):
        return value
    normalized = str(value or "").strip().upper()
    if not normalized:
        return TrustTier.T4
    try:
        return TrustTier(normalized)
    except ValueError as exc:
        allowed = ", ".join(tier.value for tier in TrustTier)
        raise ValueError(f"unknown trust tier {value!r}; expected one of: {allowed}") from exc


def source_default_trust_tier(source: str | None, name_class: str | None) -> TrustTier:
    source_key = _norm(source)
    class_key = _norm(name_class)
    if class_key in {"accepted_scientific", "scientific", "scientific_name", "scientific_synonym", "synonym"}:
        return TrustTier.T1
    if source_key in T1_SOURCES:
        return TrustTier.T1
    if source_key in T2_SOURCES or source_key in AUTHORITY_VERNACULAR_SOURCES:
        return TrustTier.T2
    if source_key in T3_SOURCES:
        return TrustTier.T3
    if source_key in GENERATED_SOURCES or "translation" in source_key or "generated" in class_key:
        return TrustTier.T5
    if source_key in COMMUNITY_SOURCES or class_key in {"vernacular_alias", "common_name_alias"}:
        return TrustTier.T4
    return TrustTier.T4


def should_enable_name_by_default(
    trust_tier: str | TrustTier | None,
    confidence: str | None = None,
    collision_status: str | None = None,
    *,
    review_state: str | None = None,
    external_taxon_link_confident: bool = False,
    corroborated: bool = False,
) -> bool:
    tier = normalize_trust_tier(trust_tier)
    review = _norm(review_state)
    if review in ACCEPTED_REVIEW_STATES:
        return True
    if tier in {TrustTier.T1, TrustTier.T2}:
        return True
    if _has_collision(collision_status):
        return False
    confidence_key = _norm(confidence)
    if tier == TrustTier.T3:
        return external_taxon_link_confident and confidence_key in HIGH_CONFIDENCE_VALUES
    if tier == TrustTier.T4:
        return confidence_key not in LOW_CONFIDENCE_VALUES
    if tier == TrustTier.T5:
        # Keep the assertion in names.parquet for audit and deduplication.
        # Query eligibility separately blocks unreviewed generated terms.
        return True
    return False


def disabled_reason_for_candidate(
    trust_tier: str | TrustTier | None,
    confidence: str | None = None,
    collision_status: str | None = None,
    *,
    review_state: str | None = None,
    external_taxon_link_confident: bool = False,
    corroborated: bool = False,
) -> str:
    if should_enable_name_by_default(
        trust_tier,
        confidence,
        collision_status,
        review_state=review_state,
        external_taxon_link_confident=external_taxon_link_confident,
        corroborated=corroborated,
    ):
        return ""
    tier = normalize_trust_tier(trust_tier)
    if _has_collision(collision_status):
        return "name_collision_requires_review"
    if tier == TrustTier.T3:
        return "wikidata_name_requires_confident_taxon_link"
    if tier == TrustTier.T4:
        return "weak_or_community_name_requires_review"
    return "name_requires_review"


def decide_name_trust(
    *,
    source: str | None,
    name_class: str | None,
    trust_tier: str | TrustTier | None = None,
    confidence: str | None = None,
    collision_status: str | None = None,
    review_state: str | None = None,
    external_taxon_link_confident: bool = False,
    corroborated: bool = False,
) -> TrustDecision:
    tier = normalize_trust_tier(trust_tier) if trust_tier else source_default_trust_tier(source, name_class)
    enabled = should_enable_name_by_default(
        tier,
        confidence,
        collision_status,
        review_state=review_state,
        external_taxon_link_confident=external_taxon_link_confident,
        corroborated=corroborated,
    )
    return TrustDecision(
        trust_tier=tier,
        enabled=enabled,
        disabled_reason="" if enabled else disabled_reason_for_candidate(
            tier,
            confidence,
            collision_status,
            review_state=review_state,
            external_taxon_link_confident=external_taxon_link_confident,
            corroborated=corroborated,
        ),
    )


def _has_collision(value: str | None) -> bool:
    normalized = _norm(value)
    if normalized in COLLISION_FREE_VALUES:
        return False
    return normalized in COLLISION_VALUES or bool(normalized)


def _norm(value: object) -> str:
    return "_".join(str(value or "").strip().casefold().split())


__all__ = [
    "TRUST_TIER_DEFINITIONS",
    "TrustDecision",
    "TrustTier",
    "decide_name_trust",
    "disabled_reason_for_candidate",
    "normalize_trust_tier",
    "should_enable_name_by_default",
    "source_default_trust_tier",
]
