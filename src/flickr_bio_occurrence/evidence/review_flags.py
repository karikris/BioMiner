from __future__ import annotations

from collections.abc import Iterable


HUMAN_VERIFICATION_TERMS = (
    "identified by",
    "confirmed by",
    "verified by",
    "determined by",
    "det.",
    "ID by",
    "expert",
    "museum",
    "collection",
    "specimen",
    "species:",
    "taxon:",
)

MUSEUM_TERMS = ("museum",)
ARTWORK_TERMS = ("artwork", "illustration", "drawing", "painting", "plate")
SPECIMEN_TERMS = ("specimen", "pinned", "voucher")
TATTOO_TERMS = ("tattoo",)
COLLECTION_TERMS = ("collection", "collected", "collector")
CAPTIVE_TERMS = ("captive", "captivity", "zoo", "butterfly house", "butterfly farm", "enclosure")
NON_TARGET_ORDER_TERMS = ("moth", "lepidoptera larva", "caterpillar", "larva", "pupa", "chrysalis", "egg")


def detected_terms(text: str, terms: Iterable[str]) -> list[str]:
    normalized = text.casefold()
    return [term for term in terms if term.casefold() in normalized]


def contains_any_term(text: str, terms: Iterable[str]) -> bool:
    return bool(detected_terms(text, terms))
