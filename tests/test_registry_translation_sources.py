from __future__ import annotations

from biominer.registry.translation_sources import generated_translation_candidate, translation_candidates_frame


def test_generated_translation_candidates_default_to_enabled_t5() -> None:
    frame = translation_candidates_frame(
        [
            generated_translation_candidate(
                source="LibreTranslate",
                source_language="en",
                target_language="de",
                source_name="lime butterfly",
                translated_name="Limettenfalter",
                accepted_taxon_key="gbif:100",
            )
        ]
    )

    row = frame.to_dicts()[0]
    assert row["trust_tier"] == "T5"
    assert row["enabled"] is True
    assert row["disabled_reason"] == ""
    assert row["review_state"] == "accepted"
    assert row["corroborated"] is False


def test_dictionary_translation_candidates_default_to_enabled_t5() -> None:
    frame = translation_candidates_frame(
        [
            generated_translation_candidate(
                source="local_dictionary",
                source_language="en",
                target_language="fr",
                source_name="lime butterfly",
                translated_name="papillon du citronnier",
                accepted_taxon_key="gbif:100",
                source_kind="dictionary",
            )
        ]
    )

    row = frame.to_dicts()[0]
    assert row["trust_tier"] == "T5"
    assert row["enabled"] is True
    assert row["disabled_reason"] == ""


def test_reviewed_or_corroborated_t5_translation_candidates_can_enable() -> None:
    frame = translation_candidates_frame(
        [
            {
                "source": "Translation",
                "source_record_id": "translation:reviewed",
                "source_language": "en",
                "target_language": "es",
                "source_name": "lime butterfly",
                "translated_name": "mariposa de la lima",
                "accepted_taxon_key": "gbif:100",
                "review_state": "accepted",
                "confidence": "low",
            },
            {
                "source": "Translation",
                "source_record_id": "translation:corroborated",
                "source_language": "en",
                "target_language": "pt",
                "source_name": "lime butterfly",
                "translated_name": "borboleta da lima",
                "accepted_taxon_key": "gbif:100",
                "review_state": "candidate",
                "corroborated": True,
            },
        ]
    ).sort("source_record_id")

    rows = frame.to_dicts()
    assert [row["enabled"] for row in rows] == [True, True]
    assert [row["disabled_reason"] for row in rows] == ["", ""]
    assert [row["trust_tier"] for row in rows] == ["T5", "T5"]
    assert rows[0]["corroborated"] is True
    assert rows[1]["review_state"] == "accepted"
