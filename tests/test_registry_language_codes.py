from __future__ import annotations

import json
from pathlib import Path

from biominer.registry.normalize import parse_language_tag


def test_parse_language_tag_preserves_bcp47_region_and_script() -> None:
    portuguese_brazil = parse_language_tag("pt-BR")
    chinese_traditional = parse_language_tag("zh-Hant")

    assert portuguese_brazil.language == "por"
    assert portuguese_brazil.api_language_code == "pt"
    assert portuguese_brazil.script == "Latn"
    assert portuguese_brazil.region == "BR"
    assert portuguese_brazil.bcp47 == "pt-BR"
    assert chinese_traditional.language == "zho"
    assert chinese_traditional.api_language_code == "zh"
    assert chinese_traditional.script == "Hant"
    assert chinese_traditional.region == ""
    assert chinese_traditional.bcp47 == "zh-Hant"


def test_checked_in_language_metadata_covers_regional_and_scripted_locales() -> None:
    table = Path("src/biominer/registry/language_code_metadata.json")

    assert table.exists()
    text = table.read_text(encoding="utf-8")
    assert '"bcp47": "pt-BR"' in text
    assert '"bcp47": "zh-Hant"' in text


def test_checked_in_language_metadata_covers_configured_translation_locales() -> None:
    target_locales = json.loads(Path("config/name_translation_target_locales.json").read_text(encoding="utf-8"))
    metadata = json.loads(Path("src/biominer/registry/language_code_metadata.json").read_text(encoding="utf-8"))

    rows_by_locale = {str(row["bcp47"]).casefold(): row for row in metadata}
    missing = [locale for locale in target_locales if str(locale).casefold() not in rows_by_locale]

    assert missing == []
    for locale in target_locales:
        row = rows_by_locale[str(locale).casefold()]
        assert row["bcp47"] == locale
        assert row["language"]
        assert row["api_language_code"]
        assert "script" in row
        assert "region" in row
        assert row["provider_support"]
