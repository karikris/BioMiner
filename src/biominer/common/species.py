from __future__ import annotations

from collections.abc import Iterable

from biominer.common.text import normalize_text


def species_text_matches(species_name: object, text_values: Iterable[object]) -> bool:
    species = normalize_text(species_name)
    if not species:
        return False
    return any(species in normalize_text(value) for value in text_values if value)
