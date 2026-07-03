from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from biominer.species.context import SpeciesContext


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
