from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


TARGET_SPECIES = "Papilio demoleus"
DEFAULT_SPECIES_CANDIDATE_LIMIT = 2_000


@dataclass(frozen=True)
class SpeciesCandidate:
    scientific_name: str
    canonical_name: str
    rank: str
    family: str | None
    genus: str | None
    source: str | None
    source_taxon_id: str | None
    is_target_species: bool

    @property
    def label(self) -> str:
        return f"a photo of {self.scientific_name}"


def load_species_candidates(
    path: str | Path,
    *,
    limit: int = DEFAULT_SPECIES_CANDIDATE_LIMIT,
    target_species: str = TARGET_SPECIES,
) -> list[SpeciesCandidate]:
    source = Path(path)
    frame = _read_candidate_frame(source)
    candidates = [_candidate_from_row(row, target_species=target_species) for row in frame.to_dicts()]
    deduped = _dedupe_candidates(candidates)
    ordered = sorted(
        deduped,
        key=lambda item: (
            not item.is_target_species,
            str(item.family or ""),
            str(item.genus or ""),
            item.scientific_name.casefold(),
        ),
    )
    if not any(candidate.scientific_name == target_species for candidate in ordered):
        ordered.insert(
            0,
            SpeciesCandidate(
                scientific_name=target_species,
                canonical_name=target_species,
                rank="species",
                family="Papilionidae",
                genus="Papilio",
                source="pinned_target",
                source_taxon_id=None,
                is_target_species=True,
            ),
        )
    return ordered[:limit]


def species_labels(candidates: list[SpeciesCandidate]) -> list[str]:
    return [candidate.label for candidate in candidates]


def label_to_scientific_name(candidates: list[SpeciesCandidate]) -> dict[str, str]:
    return {candidate.label: candidate.scientific_name for candidate in candidates}


def _read_candidate_frame(path: Path) -> pl.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.casefold() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.casefold() == ".tsv" else ","
        return pl.read_csv(path, separator=separator)
    raise ValueError(f"Unsupported species candidate file type: {path}")


def _candidate_from_row(row: dict[str, Any], *, target_species: str) -> SpeciesCandidate:
    scientific_name = _first_text(
        row,
        "scientific_name",
        "scientificName",
        "accepted_scientific_name",
        "accepted_taxon",
        "canonical_name",
        "species",
    )
    if not scientific_name:
        raise ValueError("Species candidate row is missing a scientific name")
    canonical_name = _first_text(row, "canonical_name", "canonicalName") or scientific_name
    rank = (_first_text(row, "rank", "taxon_rank", "taxonRank") or "species").casefold()
    genus = _first_text(row, "genus") or scientific_name.split(" ", 1)[0]
    return SpeciesCandidate(
        scientific_name=scientific_name,
        canonical_name=canonical_name,
        rank=rank,
        family=_first_text(row, "family"),
        genus=genus,
        source=_first_text(row, "source"),
        source_taxon_id=_first_text(row, "source_taxon_id", "taxon_id", "taxonID"),
        is_target_species=_normalize(scientific_name) == _normalize(target_species),
    )


def _dedupe_candidates(candidates: list[SpeciesCandidate]) -> list[SpeciesCandidate]:
    best: dict[str, SpeciesCandidate] = {}
    for candidate in candidates:
        if candidate.rank != "species":
            continue
        key = _normalize(candidate.scientific_name)
        existing = best.get(key)
        if existing is None or (candidate.is_target_species and not existing.is_target_species):
            best[key] = candidate
    return list(best.values())


def _first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
