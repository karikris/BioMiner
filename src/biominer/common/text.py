from __future__ import annotations


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def text_contains_any(value: object, terms: list[str] | tuple[str, ...] | set[str]) -> bool:
    text = normalize_text(value)
    return any(normalize_text(term) in text for term in terms)
