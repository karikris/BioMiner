from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciesSeed:
    accepted_scientific_name: str
    canonical_name: str
    genus: str
    specific_epithet: str
    vernacular_names: list[str]
    synonyms: list[str]
    search_terms: list[str]
    taxon_rank: str = "species"
    gbif_taxon_key: str | None = None
    ala_taxon_id: str | None = None
    inat_taxon_id: str | None = None
    known_regions: list[str] | None = None
    sensitive_species_flag: bool = False


def get_seed_species(name: str) -> SpeciesSeed:
    if name.lower() != "papilio demoleus":
        raise KeyError(f"no seed species configured for {name}")
    terms = ["Papilio demoleus", "lime butterfly", "chequered swallowtail", "citrus swallowtail", "swallowtail"]
    return SpeciesSeed(
        accepted_scientific_name="Papilio demoleus",
        canonical_name="Papilio demoleus",
        genus="Papilio",
        specific_epithet="demoleus",
        vernacular_names=["lime butterfly", "chequered swallowtail", "citrus swallowtail"],
        synonyms=[],
        search_terms=terms,
        known_regions=["Asia", "Australia", "Pacific"],
    )
