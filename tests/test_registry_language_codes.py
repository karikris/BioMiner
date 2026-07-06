from __future__ import annotations

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
