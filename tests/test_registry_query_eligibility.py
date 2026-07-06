from __future__ import annotations

from biominer.registry.query_eligibility import assess_name_query_eligibility


def _name_row(display_name: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "display_name": display_name,
        "verbatim_name": display_name,
        "enabled": True,
        "name_class": "vernacular_alias",
        "source": "Wikimedia",
        "trust_tier": "T3",
        "precision_tier": "medium",
        "confidence": "high",
        "review_state": "accepted",
    }
    row.update(overrides)
    return row


def test_multilingual_generic_butterfly_words_are_not_species_queries() -> None:
    for term in ("Schmetterling", "borboleta", "farfalla", "vlinder", "fjäril", "蝴蝶"):
        decision = assess_name_query_eligibility(_name_row(term))

        assert decision.query_eligible is False
        assert decision.query_disabled_reason == "generic_single_token"


def test_multilingual_generic_group_words_are_not_species_queries() -> None:
    for term in ("Schwalbenschwanz", "zwaluwstaart", "svalstjärt"):
        decision = assess_name_query_eligibility(_name_row(term, trust_tier="T2"))

        assert decision.query_eligible is False
        assert decision.query_disabled_reason == "generic_single_token"


def test_generic_family_group_phrases_are_not_species_queries() -> None:
    for term in ("Swallowtail Butterfly", "Skipper Butterfly", "White Butterflies", "Metalmark Butterfly"):
        decision = assess_name_query_eligibility(_name_row(term, trust_tier="T2"))

        assert decision.query_eligible is False
        assert decision.query_disabled_reason == "generic_group_phrase"


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
