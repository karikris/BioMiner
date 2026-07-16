from __future__ import annotations


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
