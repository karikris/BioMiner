from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.species.context import SpeciesContext


PROMPT_VARIANT_VERSION = "object-bioclip-prompts-v1"


@dataclass(frozen=True)
class CandidateTaxon:
    scientific_name: str
    accepted_taxon_key: str | None = None
    rank: str = "species"
    family: str | None = None
    genus: str | None = None
    common_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateSet:
    candidate_set_id: str
    registry_version: str
    target_accepted_taxon_key: str
    target_scientific_name: str
    family_candidates: tuple[CandidateTaxon, ...]
    genus_candidates: tuple[CandidateTaxon, ...]
    species_candidates: tuple[CandidateTaxon, ...]
    prompt_variant_version: str
    geospatial_scope: str | None
    source_evidence: tuple[str, ...]

    def prompt_labels(self, rank: str) -> tuple[str, ...]:
        if rank == "family":
            candidates = self.family_candidates
        elif rank == "genus":
            candidates = self.genus_candidates
        else:
            candidates = self.species_candidates
        labels: list[str] = []
        for candidate in candidates:
            labels.append(candidate.scientific_name)
            labels.append(f"a photo of {candidate.scientific_name}")
            labels.extend(candidate.common_names)
        return _unique(labels)


def build_candidate_set(
    context: SpeciesContext,
    *,
    species_candidate_path: str | Path | None = None,
    records: list[dict[str, Any]] | None = None,
    geospatial_scope: str | None = None,
) -> CandidateSet:
    target = _target_candidate(context)
    source_evidence = ["species_context"]
    candidate_rows = _candidate_rows(species_candidate_path) if species_candidate_path else []
    if candidate_rows:
        source_evidence.append(str(species_candidate_path))
    species = _species_candidates(context, candidate_rows)
    species, query_provenance_added = _add_query_provenance_candidates(species, records or [])
    if query_provenance_added:
        source_evidence.append("query_provenance")
    if not any(_norm(candidate.scientific_name) == _norm(context.scientific_name) for candidate in species):
        species.insert(0, target)
    species = _dedupe_taxa(species)
    genus = tuple(candidate for candidate in species if _norm(candidate.genus) == _norm(context.genus))
    family = tuple(candidate for candidate in species if _norm(candidate.family) == _norm(context.family))
    candidate_set_id = _candidate_set_id(context=context, species=species, geospatial_scope=geospatial_scope)
    return CandidateSet(
        candidate_set_id=candidate_set_id,
        registry_version=context.registry_version,
        target_accepted_taxon_key=context.accepted_taxon_key,
        target_scientific_name=context.scientific_name,
        family_candidates=family or (target,),
        genus_candidates=genus or (target,),
        species_candidates=tuple(species) or (target,),
        prompt_variant_version=PROMPT_VARIANT_VERSION,
        geospatial_scope=geospatial_scope,
        source_evidence=tuple(source_evidence),
    )


def _target_candidate(context: SpeciesContext) -> CandidateTaxon:
    return CandidateTaxon(
        scientific_name=context.scientific_name,
        accepted_taxon_key=context.accepted_taxon_key,
        rank="species",
        family=context.family,
        genus=context.genus,
        common_names=tuple(name.name for name in context.common_names),
    )


def _candidate_rows(path: str | Path) -> list[dict[str, Any]]:
    return pl.read_parquet(path).to_dicts()


def _species_candidates(context: SpeciesContext, rows: list[dict[str, Any]]) -> list[CandidateTaxon]:
    target = _target_candidate(context)
    candidates: list[CandidateTaxon] = [target]
    for row in rows:
        scientific_name = _first_text(row, "scientific_name", "accepted_scientific_name", "canonical_name", "species")
        if not scientific_name:
            continue
        family = _first_text(row, "family")
        genus = _first_text(row, "genus") or scientific_name.split(" ", 1)[0]
        same_genus = _norm(genus) == _norm(context.genus)
        same_family = _norm(family) == _norm(context.family)
        is_target = _norm(scientific_name) == _norm(context.scientific_name)
        if not (is_target or same_genus or same_family):
            continue
        candidates.append(
            CandidateTaxon(
                scientific_name=scientific_name,
                accepted_taxon_key=_first_text(row, "accepted_taxon_key", "source_taxon_id", "taxon_id", "taxonID"),
                rank=(_first_text(row, "rank", "taxon_rank") or "species").casefold(),
                family=family,
                genus=genus,
                common_names=_split_names(_first_text(row, "common_names", "vernacular_names")),
            )
        )
    return candidates


def _add_query_provenance_candidates(
    candidates: list[CandidateTaxon],
    records: list[dict[str, Any]],
) -> tuple[list[CandidateTaxon], bool]:
    by_key = {str(candidate.accepted_taxon_key or ""): candidate for candidate in candidates if candidate.accepted_taxon_key}
    added = False
    for record in records:
        keys = record.get("discovery_species_keys") or record.get("discovery_accepted_taxon_keys") or ()
        names = record.get("scientific_names_detected") or ()
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(names, str):
            names = [names]
        for key, name in zip(keys, names, strict=False):
            if str(key) in by_key or not name:
                continue
            candidates.append(CandidateTaxon(scientific_name=str(name), accepted_taxon_key=str(key), rank="species"))
            by_key[str(key)] = candidates[-1]
            added = True
    return candidates, added


def _candidate_set_id(*, context: SpeciesContext, species: list[CandidateTaxon], geospatial_scope: str | None) -> str:
    payload = {
        "registry_version": context.registry_version,
        "target": context.accepted_taxon_key,
        "species": [(candidate.accepted_taxon_key, candidate.scientific_name) for candidate in species],
        "prompt_variant_version": PROMPT_VARIANT_VERSION,
        "geospatial_scope": geospatial_scope,
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _dedupe_taxa(candidates: list[CandidateTaxon]) -> list[CandidateTaxon]:
    seen: set[str] = set()
    output: list[CandidateTaxon] = []
    for candidate in candidates:
        key = _norm(candidate.accepted_taxon_key or candidate.scientific_name)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def _first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def _split_names(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return _unique(value.replace(";", "|").split("|"))


def _unique(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(str(value or "").split())
        key = _norm(cleaned)
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return tuple(output)


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
