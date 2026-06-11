from __future__ import annotations

import re
from collections.abc import Iterable

from biominer.common.text import normalize_text


SCIENTIFIC_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+ [a-z][a-z-]+\b")


def scientific_names_in_text(text: object) -> list[str]:
    return sorted(set(SCIENTIFIC_NAME_PATTERN.findall(str(text or ""))))


def species_text_matches(species_name: object, text_values: Iterable[object]) -> bool:
    species = normalize_text(species_name)
    if not species:
        return False
    return any(species in normalize_text(value) for value in text_values if value)
