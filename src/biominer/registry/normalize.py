from __future__ import annotations

import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION_SPACING = re.compile(r"\s*([,;:/()\-])\s*")


def normalize_name_key(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    text = _WHITESPACE.sub(" ", text)
    text = _PUNCTUATION_SPACING.sub(r"\1", text)
    return text.casefold()
