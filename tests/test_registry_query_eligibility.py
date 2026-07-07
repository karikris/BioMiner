from __future__ import annotations

from biominer.registry.query_eligibility import assess_name_query_eligibility


def _name_row(display_name: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "display_name": display_name,
        "verbatim_name": display_name,
        "enabled": True,
        "name_class": "vernacular_alias",
        "source": "fixture",
        "trust_tier": "T3",
        "precision_tier": "medium",
        "confidence": "high",
        "review_state": "accepted",
    }
    row.update(overrides)
    return row


def test_specific_one_word_terms_with_thirteen_or_more_letters_are_species_queries() -> None:
    for term in ("Schmetterling", "Schwalbenschwanz"):
        decision = assess_name_query_eligibility(_name_row(term))

        assert decision.query_eligible is True
        assert decision.query_disabled_reason == ""


def test_one_word_terms_under_thirteen_letters_stay_blocked() -> None:
    for term in ("Butterfly", "borboleta", "farfalla", "papillon", "mariposa", "metalmark", "zwaluwstaart", "svalstjärt"):
        decision = assess_name_query_eligibility(_name_row(term))

        assert decision.query_eligible is False
        assert decision.query_disabled_reason == "generic_single_token"


def test_plural_group_terms_are_not_species_queries_even_when_long() -> None:
    decision = assess_name_query_eligibility(_name_row("swallowtails", trust_tier="T2"))

    assert decision.query_eligible is False
    assert decision.query_disabled_reason == "plural_group_name"


def test_short_multilingual_group_words_are_not_species_queries() -> None:
    for term in ("vlinder", "fjäril", "蝴蝶", "鳳蝶", "凤蝶"):
        decision = assess_name_query_eligibility(_name_row(term, trust_tier="T2"))

        assert decision.query_eligible is False
        assert decision.query_disabled_reason == "generic_single_token"


def test_generic_family_group_phrases_are_not_species_queries() -> None:
    for term in ("Swallowtail Butterfly", "Skipper Butterfly", "White Butterflies", "Metalmark Butterfly"):
        decision = assess_name_query_eligibility(_name_row(term, trust_tier="T2"))

        assert decision.query_eligible is False
        assert decision.query_disabled_reason == "generic_group_phrase"


def test_short_unapproved_single_token_common_nouns_are_not_species_queries() -> None:
    for term in ("Orange", "Blue", "Skipper", "Cabbage", "Queen", "Admiral", "Comma", "Brown", "Grey", "Monarch"):
        decision = assess_name_query_eligibility(_name_row(term, trust_tier="T2", precision_tier="medium"))

        assert decision.query_eligible is False
        assert decision.query_disabled_reason == "generic_single_token"


def test_query_approved_generic_single_token_can_remain_queryable() -> None:
    decision = assess_name_query_eligibility(
        _name_row(
            "Queen",
            trust_tier="T2",
            precision_tier="high",
            review_state="query_approved",
        )
    )

    assert decision.query_eligible is True
    assert decision.query_disabled_reason == ""


def test_query_approved_single_token_common_name_can_remain_queryable() -> None:
    decision = assess_name_query_eligibility(
        _name_row(
            "Monarch",
            trust_tier="T2",
            precision_tier="high",
            review_state="query_approved",
        )
    )

    assert decision.query_eligible is True
    assert decision.query_disabled_reason == ""


def test_scientific_monomials_are_too_broad_for_species_queries() -> None:
    for term in ("Papilio", "Papilionidae", "Papilionoidea"):
        decision = assess_name_query_eligibility(
            _name_row(
                term,
                name_class="accepted_scientific",
                source="GBIF",
                trust_tier="T1",
                precision_tier="high",
            )
        )

        assert decision.query_eligible is False
        assert decision.query_disabled_reason == "broad_scientific_name"


def test_binomial_scientific_names_remain_species_queries() -> None:
    decision = assess_name_query_eligibility(
        _name_row(
            "Papilio demoleus",
            name_class="accepted_scientific",
            source="GBIF",
            trust_tier="T1",
            precision_tier="high",
        )
    )

    assert decision.query_eligible is True
    assert decision.query_disabled_reason == ""


def test_wikimedia_alias_requires_same_taxon_binding_for_queries() -> None:
    decision = assess_name_query_eligibility(
        _name_row(
            "Zitronenfalter",
            source="Wikimedia",
            trust_tier="T3",
            confidence="high",
            review_state="accepted",
        )
    )

    assert decision.query_eligible is False
    assert decision.query_disabled_reason == "source_binding_required"


def test_wikimedia_alias_with_same_taxon_binding_can_be_queryable() -> None:
    decision = assess_name_query_eligibility(
        _name_row(
            "Zitronenfalter",
            source="Wikimedia",
            source_taxon_id="Q123",
            lineage_check="accepted_taxon_key",
            trust_tier="T3",
            confidence="high",
            review_state="accepted",
        )
    )

    assert decision.query_eligible is True
    assert decision.query_disabled_reason == ""


def test_specific_single_token_reviewed_translation_can_remain_queryable() -> None:
    decision = assess_name_query_eligibility(
        _name_row(
            "Limettenfalter",
            name_class="generated_translation",
            source="MyMemory",
            trust_tier="T5",
            review_state="reviewed",
        )
    )

    assert decision.query_eligible is True
    assert decision.query_disabled_reason == ""


def test_curated_hindi_and_kannada_source_backed_names_can_be_queryable() -> None:
    rows = [
        _name_row(
            "नींबू तितली",
            name_class="vernacular",
            source="Bharat Ki Titliya",
            trust_tier="T2",
            language="hin",
            region="IN",
        ),
        _name_row(
            "ನಿಂಬೆ ಚಿಟ್ಟೆ",
            name_class="vernacular",
            source="Karnataka Chitte",
            trust_tier="T2",
            language="kan",
            region="IN-KA",
        ),
    ]

    decisions = [assess_name_query_eligibility(row) for row in rows]

    assert [decision.query_eligible for decision in decisions] == [True, True]
    assert [decision.query_disabled_reason for decision in decisions] == ["", ""]


def test_generated_hindi_and_kannada_translations_are_not_queryable_by_default() -> None:
    rows = [
        _name_row(
            "नींबू तितली",
            name_class="generated_translation",
            source="MyMemory",
            trust_tier="T5",
            language="hin",
            review_state="candidate",
        ),
        _name_row(
            "ನಿಂಬೆ ಚಿಟ್ಟೆ",
            name_class="generated_translation",
            source="MyMemory",
            trust_tier="T5",
            language="kan",
            review_state="candidate",
        ),
    ]

    decisions = [assess_name_query_eligibility(row) for row in rows]

    assert [decision.query_eligible for decision in decisions] == [False, False]
    assert [decision.query_disabled_reason for decision in decisions] == [
        "generated_translation_requires_review_or_corroboration",
        "generated_translation_requires_review_or_corroboration",
    ]


def test_t4_names_require_review_or_corroboration_for_queries() -> None:
    candidate = assess_name_query_eligibility(
        _name_row(
            "Community Lime Butterfly",
            source="iNaturalist",
            trust_tier="T4",
            review_state="candidate",
            corroborated=False,
        )
    )
    reviewed = assess_name_query_eligibility(
        _name_row(
            "Reviewed Lime Butterfly",
            source="iNaturalist",
            trust_tier="T4",
            review_state="reviewed",
            corroborated=False,
        )
    )
    corroborated = assess_name_query_eligibility(
        _name_row(
            "Corroborated Lime Butterfly",
            source="iNaturalist",
            trust_tier="T4",
            review_state="candidate",
            corroborated=True,
        )
    )

    assert candidate.query_eligible is False
    assert candidate.query_disabled_reason == "weak_or_community_name_requires_review_or_corroboration"
    assert reviewed.query_eligible is True
    assert reviewed.query_disabled_reason == ""
    assert corroborated.query_eligible is True
    assert corroborated.query_disabled_reason == ""


def test_taxonomic_caution_regions_require_explicit_accepted_taxon_resolution() -> None:
    unbound = assess_name_query_eligibility(
        _name_row(
            "Caution Lime Butterfly",
            source="Cautionary Checklist",
            trust_tier="T2",
            region="Australia/New Guinea taxonomic caution",
        )
    )
    bound = assess_name_query_eligibility(
        _name_row(
            "Resolved Caution Lime Butterfly",
            source="Cautionary Checklist",
            trust_tier="T2",
            region="Australia/New Guinea taxonomic caution",
            source_taxon_id="gbif:100",
            lineage_check="accepted_taxon_key",
        )
    )

    assert unbound.query_eligible is False
    assert unbound.query_disabled_reason == "taxonomic_caution_region_requires_accepted_taxon_resolution"
    assert bound.query_eligible is True
    assert bound.query_disabled_reason == ""


def test_ambiguous_common_name_disabled_reason_blocks_queries_even_when_enabled() -> None:
    decision = assess_name_query_eligibility(
        _name_row(
            "Lime Butterfly",
            source="Butterflies of India",
            trust_tier="T2",
            disabled_reason="ambiguous_common_name",
        )
    )

    assert decision.query_eligible is False
    assert decision.query_disabled_reason == "ambiguous_common_name"
