from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from biominer.registry.normalize import normalize_name_key
from biominer.species.context import CommonName, SpeciesContext, SpeciesSearchTerm
from biominer.storage.parquet import write_parquet


COMPILER_VERSION = "species-query-compiler-v1"
BROAD_ANCHOR_TERMS = ("butterfly", "butterflies", "lepidoptera", "swallowtail", "caterpillar", "chrysalis", "pupa", "egg")


@dataclass(frozen=True)
class SpeciesQueryCompileResult:
    output_path: Path
    rows: int


def compile_species_flickr_queries(context: SpeciesContext, *, include_broad_anchored_terms: bool = True) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for term in _context_terms(context, include_broad_anchored_terms=include_broad_anchored_terms):
        for field in ("tags", "text"):
            rows.append(_query_row(context, term=term, search_field=field))
    if not rows:
        return pl.DataFrame([], schema=_query_schema())
    return pl.DataFrame(rows, schema=_query_schema()).unique("query_definition_id").sort(
        ["search_priority", "normalized_match_key", "query_definition_id"]
    )


def write_species_flickr_queries(
    context: SpeciesContext,
    output_path: str | Path,
    *,
    include_broad_anchored_terms: bool = True,
) -> SpeciesQueryCompileResult:
    frame = compile_species_flickr_queries(context, include_broad_anchored_terms=include_broad_anchored_terms)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(frame, output)
    return SpeciesQueryCompileResult(output_path=output, rows=frame.height)


def _context_terms(context: SpeciesContext, *, include_broad_anchored_terms: bool) -> tuple[SpeciesSearchTerm, ...]:
    terms: list[SpeciesSearchTerm] = [
        SpeciesSearchTerm(
            term=context.scientific_name,
            language="la",
            term_class="accepted_scientific",
            source="registry",
            source_record_id=context.accepted_taxon_key,
            trust_tier="T1",
            precision_tier="high",
            confidence="high",
        )
    ]
    terms.extend(
        SpeciesSearchTerm(
            term=synonym,
            language="la",
            term_class="scientific_synonym",
            source="registry",
            source_record_id=context.accepted_taxon_key,
            trust_tier="T1",
            precision_tier="high",
            confidence="high",
        )
        for synonym in context.synonyms
    )
    terms.extend(_common_name_term(name) for name in context.common_names)
    for term in context.search_terms:
        if not term.enabled:
            continue
        if _is_duplicate_term(term.term, terms):
            continue
        terms.append(_anchor_if_broad(context, term) if include_broad_anchored_terms else term)
    return _dedupe_terms(terms)


def _common_name_term(name: CommonName) -> SpeciesSearchTerm:
    return SpeciesSearchTerm(
        term=name.name,
        language=name.language,
        term_class="vernacular",
        source=name.source,
        source_record_id=name.source_record_id,
        trust_tier=name.trust_tier,
        precision_tier="high",
        confidence=name.confidence,
        region=name.region,
        bbox=name.bbox,
        review_state=name.review_state,
    )


def _anchor_if_broad(context: SpeciesContext, term: SpeciesSearchTerm) -> SpeciesSearchTerm:
    term_class = term.term_class.casefold()
    precision = str(term.precision_tier or "").casefold()
    confidence = str(term.confidence or "").casefold()
    broad = term_class.startswith("broad") or term.term.casefold() in BROAD_ANCHOR_TERMS or precision == "low" or confidence == "broad"
    if not broad:
        return term
    normalized = term.term.casefold()
    if context.scientific_name.casefold() in normalized:
        return term
    return SpeciesSearchTerm(
        **{
            **term.__dict__,
            "term": f"{context.scientific_name} {term.term}",
            "term_class": term.term_class or "broad_anchored",
        }
    )


def _query_row(context: SpeciesContext, *, term: SpeciesSearchTerm, search_field: str) -> dict[str, Any]:
    normalized = normalize_name_key(term.term)
    priority = _priority(term.term_class, search_field)
    return {
        "query_definition_id": _stable_id(
            "species-flickr-query",
            context.registry_version,
            context.accepted_taxon_key,
            normalized,
            term.language,
            term.region,
            term.bbox,
            term.term_class,
            term.source,
            term.source_record_id,
            search_field,
        ),
        "registry_schema_version": "species-context-v1",
        "compiler_version": COMPILER_VERSION,
        "registry_version": context.registry_version,
        "accepted_taxon_key": context.accepted_taxon_key,
        "accepted_scientific_name": context.scientific_name,
        "accepted_rank": "SPECIES",
        "family_key": context.family_key,
        "family": context.family,
        "genus_key": context.genus_key,
        "genus": context.genus,
        "species_key": context.species_key,
        "species": context.canonical_name,
        "name_id": term.source_record_id or "",
        "source_term": term.term,
        "normalized_query_term": term.term,
        "normalized_match_key": normalized,
        "language": term.language,
        "script": "",
        "region": term.region or "",
        "bbox": term.bbox or "",
        "name_class": term.term_class,
        "source": term.source or "",
        "trust_tier": term.trust_tier or "",
        "confidence": term.confidence or "",
        "precision_tier": term.precision_tier or "",
        "search_field": search_field,
        "search_priority": priority,
        "enabled": bool(term.enabled),
        "disabled_reason": "",
    }


def _priority(term_class: str, search_field: str) -> int:
    offset = 0 if search_field == "tags" else 40
    value = term_class.casefold()
    if value in {"accepted_scientific", "canonical_scientific"}:
        return 10 + offset
    if value == "scientific_synonym":
        return 15 + offset
    if value in {"vernacular", "vernacular_alias", "common_name", "regional_common_name"}:
        return 20 + offset
    if value.startswith("translation"):
        return 100 + offset
    if value.startswith("broad"):
        return 80 + offset
    return 90 + offset


def _dedupe_terms(terms: Iterable[SpeciesSearchTerm]) -> tuple[SpeciesSearchTerm, ...]:
    seen: set[tuple[str, str, str | None, str | None]] = set()
    output: list[SpeciesSearchTerm] = []
    for term in terms:
        key = (normalize_name_key(term.term), term.language, term.region, term.bbox)
        if key in seen:
            continue
        seen.add(key)
        output.append(term)
    return tuple(output)


def _is_duplicate_term(value: str, terms: list[SpeciesSearchTerm]) -> bool:
    key = normalize_name_key(value)
    return any(normalize_name_key(term.term) == key for term in terms)


def _stable_id(*parts: object) -> str:
    payload = json.dumps([str(part or "") for part in parts], ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _query_schema() -> dict[str, pl.DataType]:
    return {
        "query_definition_id": pl.String,
        "registry_schema_version": pl.String,
        "compiler_version": pl.String,
        "registry_version": pl.String,
        "accepted_taxon_key": pl.String,
        "accepted_scientific_name": pl.String,
        "accepted_rank": pl.String,
        "family_key": pl.String,
        "family": pl.String,
        "genus_key": pl.String,
        "genus": pl.String,
        "species_key": pl.String,
        "species": pl.String,
        "name_id": pl.String,
        "source_term": pl.String,
        "normalized_query_term": pl.String,
        "normalized_match_key": pl.String,
        "language": pl.String,
        "script": pl.String,
        "region": pl.String,
        "bbox": pl.String,
        "name_class": pl.String,
        "source": pl.String,
        "trust_tier": pl.String,
        "confidence": pl.String,
        "precision_tier": pl.String,
        "search_field": pl.String,
        "search_priority": pl.Int64,
        "enabled": pl.Boolean,
        "disabled_reason": pl.String,
    }
