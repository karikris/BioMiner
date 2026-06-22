from __future__ import annotations

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


def normalize_name_key(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    text = _WHITESPACE.sub(" ", text)
    text = _PUNCTUATION_SPACING.sub(r"\1", text)
    return text.casefold()


def normalize_language_code(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    if not text:
        return ""
    token = text.replace("_", "-").split("-", 1)[0].strip().casefold()
    return _LANGUAGE_ALIASES.get(token, token)
