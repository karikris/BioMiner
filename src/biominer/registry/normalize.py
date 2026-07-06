from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
import json
import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION_SPACING = re.compile(r"\s*([,;:/()\-])\s*")
_LANGUAGE_ALIASES = {
    "arabic": "ara",
    "ar": "ara",
    "chinese": "zho",
    "cn": "zho",
    "cs": "ces",
    "czech": "ces",
    "da": "dan",
    "danish": "dan",
    "de": "deu",
    "deutsch": "deu",
    "dutch": "nld",
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "es": "spa",
    "fi": "fin",
    "finnish": "fin",
    "fr": "fra",
    "fre": "fra",
    "french": "fra",
    "ger": "deu",
    "german": "deu",
    "it": "ita",
    "italian": "ita",
    "ja": "jpn",
    "japanese": "jpn",
    "ko": "kor",
    "korean": "kor",
    "nl": "nld",
    "no": "nor",
    "norwegian": "nor",
    "pl": "pol",
    "polish": "pol",
    "pt": "por",
    "portuguese": "por",
    "ru": "rus",
    "russian": "rus",
    "spanish": "spa",
    "sv": "swe",
    "swedish": "swe",
    "zh": "zho",
}


@dataclass(frozen=True)
class LanguageTag:
    language: str
    api_language_code: str
    script: str
    region: str
    bcp47: str


def normalize_name_key(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    text = _WHITESPACE.sub(" ", text)
    text = _PUNCTUATION_SPACING.sub(r"\1", text)
    return text.casefold()


def normalize_language_code(value: object) -> str:
    return parse_language_tag(value).language


def parse_language_tag(value: object) -> LanguageTag:
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    if not text:
        return LanguageTag(language="", api_language_code="", script="", region="", bcp47="")
    canonical = _canonicalize_bcp47(text)
    metadata = _language_metadata().get(canonical.casefold())
    if metadata:
        return LanguageTag(
            language=str(metadata.get("language") or ""),
            api_language_code=str(metadata.get("api_language_code") or ""),
            script=str(metadata.get("script") or ""),
            region=str(metadata.get("region") or ""),
            bcp47=str(metadata.get("bcp47") or canonical),
        )
    parts = canonical.split("-")
    primary = parts[0].casefold()
    script = ""
    region = ""
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            script = part.title()
        elif (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            region = part.upper()
    language = _LANGUAGE_ALIASES.get(primary, primary)
    base = _language_metadata().get(language) or _language_metadata().get(primary)
    api_language_code = str((base or {}).get("api_language_code") or primary)
    bcp47 = "-".join(part for part in (api_language_code, script, region) if part)
    return LanguageTag(language=language, api_language_code=api_language_code, script=script, region=region, bcp47=bcp47)


def language_script_region(value: object) -> tuple[str, str, str]:
    tag = parse_language_tag(value)
    return tag.language, tag.script, tag.region


def _canonicalize_bcp47(value: str) -> str:
    parts = [part for part in value.replace("_", "-").split("-") if part]
    if not parts:
        return ""
    canonical = [parts[0].casefold()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            canonical.append(part.title())
        elif (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            canonical.append(part.upper())
        else:
            canonical.append(part.casefold())
    return "-".join(canonical)


@cache
def _language_metadata() -> dict[str, dict[str, object]]:
    with resources.files("biominer.registry").joinpath("language_code_metadata.json").open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    metadata: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bcp47 = str(row.get("bcp47") or "").casefold()
        language = str(row.get("language") or "").casefold()
        api_language_code = str(row.get("api_language_code") or "").casefold()
        if bcp47:
            metadata[bcp47] = row
        if language:
            metadata.setdefault(language, row)
        if api_language_code:
            metadata.setdefault(api_language_code, row)
    return metadata
