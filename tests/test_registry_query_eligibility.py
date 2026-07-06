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


def test_one_word_terms_with_eight_or_more_letters_are_species_queries() -> None:
    for term in ("Butterfly", "Schmetterling", "borboleta", "farfalla", "papillon", "mariposa", "metalmark"):
        decision = assess_name_query_eligibility(_name_row(term))

        assert decision.query_eligible is True
        assert decision.query_disabled_reason == ""


def test_short_multilingual_generic_butterfly_words_are_not_species_queries() -> None:
    for term in ("vlinder", "fjäril", "蝴蝶"):
        decision = assess_name_query_eligibility(_name_row(term))

        assert decision.query_eligible is False
        assert decision.query_disabled_reason == "generic_single_token"


def test_one_word_group_terms_with_eight_or_more_letters_are_species_queries() -> None:
    for term in ("Schwalbenschwanz", "zwaluwstaart", "svalstjärt", "swallowtails"):
        decision = assess_name_query_eligibility(_name_row(term, trust_tier="T2"))

        assert decision.query_eligible is True
        assert decision.query_disabled_reason == ""


def test_short_multilingual_group_words_are_not_species_queries() -> None:
    for term in ("鳳蝶", "凤蝶"):
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
