from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import polars as pl

from biominer.registry.normalize import normalize_name_key
from biominer.species.context import SpeciesContext
from biominer.species.registry_refresh import resolve_species_context


InputRank = Literal["auto", "family", "genus", "species"]
AcceptedRank = Literal["family", "genus", "species"]

INPUT_RANKS: tuple[str, ...] = ("auto", "family", "genus", "species")
ACCEPTED_RANKS: tuple[str, ...] = ("family", "genus", "species")


@dataclass(frozen=True)
class TaxonScope:
    """Resolved family/genus/species scope for a production BioMiner run."""

    input_name: str
    input_rank: InputRank
    accepted_taxon_key: str
    accepted_scientific_name: str
    accepted_rank: AcceptedRank
    registry_version: str
    species_contexts: tuple[SpeciesContext, ...]

    def __post_init__(self) -> None:
        input_name = _clean_required(self.input_name, "input_name")
        accepted_taxon_key = _clean_required(self.accepted_taxon_key, "accepted_taxon_key")
        accepted_scientific_name = _clean_required(self.accepted_scientific_name, "accepted_scientific_name")
        registry_version = _clean_required(self.registry_version, "registry_version")
        input_rank = _normalize_rank(self.input_rank, INPUT_RANKS, "input_rank")
        accepted_rank = _normalize_rank(self.accepted_rank, ACCEPTED_RANKS, "accepted_rank")
        species_contexts = tuple(self.species_contexts)
        if not species_contexts:
            raise ValueError("species_contexts must contain at least one SpeciesContext")
        for context in species_contexts:
            if not isinstance(context, SpeciesContext):
                raise TypeError("species_contexts must contain SpeciesContext instances")
        object.__setattr__(self, "input_name", input_name)
        object.__setattr__(self, "input_rank", input_rank)
        object.__setattr__(self, "accepted_taxon_key", accepted_taxon_key)
        object.__setattr__(self, "accepted_scientific_name", accepted_scientific_name)
        object.__setattr__(self, "accepted_rank", accepted_rank)
        object.__setattr__(self, "registry_version", registry_version)
        object.__setattr__(self, "species_contexts", species_contexts)

    @property
    def species_count(self) -> int:
        return len(self.species_contexts)

    @property
    def is_multi_species(self) -> bool:
        return self.species_count > 1

    @property
    def species_names(self) -> tuple[str, ...]:
        return tuple(context.scientific_name for context in self.species_contexts)

    @classmethod
    def from_species_context(
        cls,
        context: SpeciesContext,
        *,
        input_name: str | None = None,
        input_rank: InputRank = "species",
    ) -> TaxonScope:
        return cls(
            input_name=input_name or context.scientific_name,
            input_rank=input_rank,
            accepted_taxon_key=context.accepted_taxon_key,
            accepted_scientific_name=context.scientific_name,
            accepted_rank="species",
            registry_version=context.registry_version,
            species_contexts=(context,),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_name": self.input_name,
            "input_rank": self.input_rank,
            "accepted_taxon_key": self.accepted_taxon_key,
            "accepted_scientific_name": self.accepted_scientific_name,
            "accepted_rank": self.accepted_rank,
            "registry_version": self.registry_version,
            "species_contexts": [context.to_dict() for context in self.species_contexts],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaxonScope:
        return cls(
            input_name=str(payload["input_name"]),
            input_rank=_normalize_rank(str(payload.get("input_rank") or "auto"), INPUT_RANKS, "input_rank"),
            accepted_taxon_key=str(payload["accepted_taxon_key"]),
            accepted_scientific_name=str(payload["accepted_scientific_name"]),
            accepted_rank=_normalize_rank(str(payload["accepted_rank"]), ACCEPTED_RANKS, "accepted_rank"),
            registry_version=str(payload["registry_version"]),
            species_contexts=tuple(SpeciesContext.from_dict(item) for item in payload.get("species_contexts", ())),
        )


def resolve_taxon_scope_from_registry(
    *,
    registry_dir: str | Path,
    input_name: str,
    input_rank: InputRank = "auto",
) -> TaxonScope:
    """Resolve a family/genus/species production scope from registry Parquet files."""

    registry = Path(registry_dir)
    rank = _normalize_rank(input_rank, INPUT_RANKS, "input_rank")
    taxa = _read_taxa(registry)
    taxon = _resolve_taxon_row(taxa, input_name=input_name, input_rank=rank)
    accepted_rank = _accepted_rank_from_row(taxon)
    species_rows = _species_rows_for_scope(taxa, taxon, accepted_rank=accepted_rank)
    if not species_rows:
        raise ValueError(f"no species found under {accepted_rank}: {taxon['scientific_name']}")
    species_contexts = tuple(
        resolve_species_context(registry_dir=registry, accepted_taxon_key=str(row["accepted_taxon_key"]))
        for row in species_rows
    )
    registry_version = _registry_version(registry) or species_contexts[0].registry_version
    return TaxonScope(
        input_name=input_name,
        input_rank=rank,
        accepted_taxon_key=str(taxon["accepted_taxon_key"]),
        accepted_scientific_name=str(taxon["scientific_name"]),
        accepted_rank=accepted_rank,
        registry_version=registry_version,
        species_contexts=species_contexts,
    )


def _clean_required(value: object, field_name: str) -> str:
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def _normalize_rank(value: object, allowed: tuple[str, ...], field_name: str) -> Any:
    normalized = str(value or "").strip().casefold()
    if normalized not in allowed:
        joined = ", ".join(allowed)
        raise ValueError(f"{field_name} must be one of: {joined}")
    return normalized


def _read_taxa(registry: Path) -> pl.DataFrame:
    path = registry / "taxa.parquet"
    if not path.exists():
        raise FileNotFoundError(f"registry taxa parquet not found: {path}")
    taxa = pl.read_parquet(path)
    if taxa.is_empty():
        raise ValueError("registry taxa.parquet is empty")
    return taxa


def _registry_version(registry: Path) -> str:
    manifest_path = registry / "manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        return str(json.loads(manifest_path.read_text(encoding="utf-8")).get("registry_version") or "")
    except json.JSONDecodeError:
        return ""


def _resolve_taxon_row(taxa: pl.DataFrame, *, input_name: str, input_rank: str) -> dict[str, Any]:
    candidates = _matching_taxa(taxa, input_name=input_name)
    if input_rank != "auto":
        candidates = candidates.filter(pl.col("rank").str.to_lowercase() == input_rank)
    else:
        candidates = candidates.filter(pl.col("rank").str.to_lowercase().is_in(["family", "genus", "species"]))
    rows = candidates.sort(_taxon_sort_columns(candidates)).to_dicts() if not candidates.is_empty() else []
    if not rows:
        rank_label = "family/genus/species" if input_rank == "auto" else input_rank
        raise ValueError(f"{rank_label} not found in registry: {input_name}")
    distinct = {(str(row.get("accepted_taxon_key") or ""), str(row.get("rank") or "").upper()) for row in rows}
    if len(distinct) > 1:
        matches = ", ".join(f"{row.get('rank')}:{row.get('scientific_name')}[{row.get('accepted_taxon_key')}]" for row in rows)
        raise ValueError(f"ambiguous taxon match for {input_name}: {matches}")
    return rows[0]


def _matching_taxa(taxa: pl.DataFrame, *, input_name: str) -> pl.DataFrame:
    cleaned = " ".join(str(input_name or "").split())
    if not cleaned:
        raise ValueError("input_name is required")
    if "accepted_taxon_key" not in taxa.columns or "scientific_name" not in taxa.columns or "rank" not in taxa.columns:
        raise ValueError("taxa.parquet must include accepted_taxon_key, scientific_name, and rank columns")
    name_key = normalize_name_key(cleaned)
    return taxa.filter(
        (pl.col("accepted_taxon_key") == cleaned)
        | (pl.col("scientific_name").map_elements(normalize_name_key, return_dtype=pl.String) == name_key)
    )


def _accepted_rank_from_row(row: dict[str, Any]) -> AcceptedRank:
    return _normalize_rank(str(row.get("rank") or ""), ACCEPTED_RANKS, "accepted_rank")


def _species_rows_for_scope(taxa: pl.DataFrame, taxon: dict[str, Any], *, accepted_rank: AcceptedRank) -> list[dict[str, Any]]:
    species = taxa.filter(pl.col("rank").str.to_uppercase() == "SPECIES")
    key = str(taxon.get("accepted_taxon_key") or "")
    if accepted_rank == "species":
        species = species.filter(pl.col("accepted_taxon_key") == key)
    elif accepted_rank == "genus":
        species = species.filter(pl.col("genus_key") == key)
    elif accepted_rank == "family":
        species = species.filter(pl.col("family_key") == key)
    return species.sort(_taxon_sort_columns(species)).to_dicts() if not species.is_empty() else []


def _taxon_sort_columns(frame: pl.DataFrame) -> list[str]:
    return [column for column in ("family", "genus", "scientific_name", "accepted_taxon_key") if column in frame.columns]
