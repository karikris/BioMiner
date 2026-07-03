from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Literal

import polars as pl


TRANSLATION_CANDIDATES_FILE = "translation_candidates.parquet"
TranslationSourceKind = Literal["generated", "dictionary"]


@dataclass(frozen=True)
class TranslationCandidate:
    source: str
    source_record_id: str
    source_language: str
    target_language: str
    source_name: str
    translated_name: str
    accepted_taxon_key: str
    trust_tier: str = "T5"
    enabled: bool = False
    disabled_reason: str = "generated_translation_requires_review"
    review_state: str = "candidate"
    confidence: str = "low"
    precision_tier: str = "low"

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["candidate_id"] = translation_candidate_id(self)
        return row


def generated_translation_candidate(
    *,
    source: str,
    source_language: str,
    target_language: str,
    source_name: str,
    translated_name: str,
    accepted_taxon_key: str,
    source_record_id: str | None = None,
    source_kind: TranslationSourceKind = "generated",
) -> TranslationCandidate:
    disabled_reason = (
        "dictionary_translation_requires_corroboration"
        if source_kind == "dictionary"
        else "generated_translation_requires_review"
    )
    record_id = source_record_id or translation_candidate_id_parts(
        source,
        accepted_taxon_key,
        source_language,
        target_language,
        source_name,
        translated_name,
    )
    return TranslationCandidate(
        source=source,
        source_record_id=record_id,
        source_language=source_language,
        target_language=target_language,
        source_name=source_name,
        translated_name=translated_name,
        accepted_taxon_key=accepted_taxon_key,
        disabled_reason=disabled_reason,
    )


def translation_candidates_frame(candidates: Iterable[TranslationCandidate | dict[str, object]]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        row = candidate.to_row() if isinstance(candidate, TranslationCandidate) else dict(candidate)
        normalized = _normalize_candidate_row(row)
        rows.append(normalized)
    if not rows:
        return pl.DataFrame([], schema=translation_candidate_schema())
    return pl.DataFrame(rows, schema=translation_candidate_schema()).unique(subset=["candidate_id"], keep="first")


def write_translation_candidates(candidates: Iterable[TranslationCandidate | dict[str, object]], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / TRANSLATION_CANDIDATES_FILE
    translation_candidates_frame(candidates).write_parquet(path)
    return path


def translation_candidate_schema() -> dict[str, pl.DataType]:
    return {
        "candidate_id": pl.String,
        "source": pl.String,
        "source_record_id": pl.String,
        "source_language": pl.String,
        "target_language": pl.String,
        "source_name": pl.String,
        "translated_name": pl.String,
        "accepted_taxon_key": pl.String,
        "trust_tier": pl.String,
        "enabled": pl.Boolean,
        "disabled_reason": pl.String,
        "review_state": pl.String,
        "confidence": pl.String,
        "precision_tier": pl.String,
    }


def translation_candidate_id(candidate: TranslationCandidate) -> str:
    return translation_candidate_id_parts(
        candidate.source,
        candidate.accepted_taxon_key,
        candidate.source_language,
        candidate.target_language,
        candidate.source_name,
        candidate.translated_name,
    )


def translation_candidate_id_parts(*parts: object) -> str:
    payload = "\u241f".join(_clean_text(part).casefold() for part in parts)
    return "translation:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _normalize_candidate_row(row: dict[str, object]) -> dict[str, object]:
    normalized = {
        "source": _clean_text(row.get("source")),
        "source_record_id": _clean_text(row.get("source_record_id")),
        "source_language": _clean_text(row.get("source_language") or "und"),
        "target_language": _clean_text(row.get("target_language") or "und"),
        "source_name": _clean_text(row.get("source_name")),
        "translated_name": _clean_text(row.get("translated_name")),
        "accepted_taxon_key": _clean_text(row.get("accepted_taxon_key")),
        "trust_tier": "T5",
        "enabled": bool(row.get("enabled", False)) and str(row.get("review_state") or "").casefold() in {"accepted", "reviewed"},
        "disabled_reason": _clean_text(row.get("disabled_reason") or "generated_translation_requires_review"),
        "review_state": _clean_text(row.get("review_state") or "candidate"),
        "confidence": _clean_text(row.get("confidence") or "low"),
        "precision_tier": _clean_text(row.get("precision_tier") or "low"),
    }
    if not normalized["enabled"] and not normalized["disabled_reason"]:
        normalized["disabled_reason"] = "generated_translation_requires_review"
    normalized["candidate_id"] = _clean_text(row.get("candidate_id")) or translation_candidate_id_parts(
        normalized["source"],
        normalized["accepted_taxon_key"],
        normalized["source_language"],
        normalized["target_language"],
        normalized["source_name"],
        normalized["translated_name"],
    )
    return normalized


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


__all__ = [
    "TRANSLATION_CANDIDATES_FILE",
    "TranslationCandidate",
    "generated_translation_candidate",
    "translation_candidate_schema",
    "translation_candidates_frame",
    "translation_candidate_id",
    "write_translation_candidates",
]
