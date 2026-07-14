from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from biominer.bioclip.prompt_templates import (
    BUTTERFLY_ROOT_ACCEPTED_TAXON_KEY,
    BUTTERFLY_ROOT_SCIENTIFIC_NAME,
    AcceptedTaxonPromptContext,
    PromptNameEvidence,
    PromptVariant,
    TaxonomicPathNode,
    build_species_prompt_variants,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.species.context import SpeciesContext


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
    common_names: tuple[str, ...] = ()
    prompt_name_evidence: tuple[PromptNameEvidence, ...] = ()
    taxonomy_version: str | None = None
    taxonomy_fingerprint: str | None = None
    family_taxon_key: str | None = None
    genus_taxon_key: str | None = None
    taxonomic_status: str = "accepted"

    @property
    def label(self) -> str:
        return f"a photo of {self.scientific_name}"


def load_species_candidates(
    path: str | Path,
    *,
    limit: int = DEFAULT_SPECIES_CANDIDATE_LIMIT,
    target_species: str | None = None,
    allow_unregistered_target: bool = False,
    target_context: SpeciesContext | None = None,
) -> list[SpeciesCandidate]:
    source = Path(path)
    frame = _read_candidate_frame(source)
    candidates = [
        _candidate_from_row(row, target_species=target_species)
        for row in frame.to_dicts()
    ]
    deduped = _dedupe_candidates(candidates)
    ordered = sorted(
        deduped,
        key=lambda item: (
            bool(target_species) and not item.is_target_species,
            str(item.family or ""),
            str(item.genus or ""),
            item.scientific_name.casefold(),
        ),
    )
    if target_species and not any(
        _normalize(candidate.scientific_name) == _normalize(target_species)
        for candidate in ordered
    ):
        if not allow_unregistered_target or target_context is None:
            raise ValueError(
                f"target species is absent from candidates: {target_species}; "
                "provide registry candidates or use allow_unregistered_target with a SpeciesContext"
            )
        ordered.insert(0, _candidate_from_context(target_context))
    return ordered[:limit]


def species_labels(candidates: list[SpeciesCandidate]) -> list[str]:
    return [candidate.label for candidate in candidates]


def label_to_scientific_name(candidates: list[SpeciesCandidate]) -> dict[str, str]:
    return {candidate.label: candidate.scientific_name for candidate in candidates}


def taxon_metadata_by_scientific_name(
    candidates: list[SpeciesCandidate],
) -> dict[str, dict[str, str | None]]:
    return {
        candidate.scientific_name: {
            "genus": candidate.genus,
            "family": candidate.family,
        }
        for candidate in candidates
    }


def species_prompt_variants(
    candidates: list[SpeciesCandidate],
    *,
    route: str = "adult_field",
    life_stage: str | None = None,
) -> list[PromptVariant]:
    variants: list[PromptVariant] = []
    for candidate in candidates:
        variants.extend(
            build_species_prompt_variants(
                context=_candidate_prompt_context(candidate),
                route=route,
                life_stage=life_stage,
                vernacular_names=candidate.prompt_name_evidence,
            )
        )
    return variants


def _read_candidate_frame(path: Path) -> pl.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.casefold() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.casefold() == ".tsv" else ","
        return pl.read_csv(path, separator=separator)
    raise ValueError(f"Unsupported species candidate file type: {path}")


def _candidate_from_row(
    row: dict[str, Any], *, target_species: str | None
) -> SpeciesCandidate:
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
    canonical_name = (
        _first_text(row, "canonical_name", "canonicalName") or scientific_name
    )
    rank = (_first_text(row, "rank", "taxon_rank", "taxonRank") or "species").casefold()
    genus = _first_text(row, "genus") or scientific_name.split(" ", 1)[0]
    common_names = _split_common_names(
        _first_text(
            row,
            "common_names",
            "commonNames",
            "vernacular_names",
            "vernacularNames",
        )
    )
    accepted_taxon_key = _first_text(
        row,
        "accepted_taxon_key",
        "gbif_species_key",
        "source_taxon_id",
        "taxon_id",
        "taxonID",
    )
    return SpeciesCandidate(
        scientific_name=scientific_name,
        canonical_name=canonical_name,
        rank=rank,
        family=_first_text(row, "family"),
        genus=genus,
        source=_first_text(row, "source"),
        source_taxon_id=accepted_taxon_key,
        is_target_species=bool(
            target_species and _normalize(scientific_name) == _normalize(target_species)
        ),
        common_names=common_names,
        prompt_name_evidence=_prompt_name_evidence_from_row(
            row,
            common_names=common_names,
            accepted_taxon_key=accepted_taxon_key,
        ),
        taxonomy_version=_first_text(
            row,
            "registry_version",
            "taxonomy_version",
            "source_version",
        ),
        taxonomy_fingerprint=_first_text(
            row,
            "taxonomy_fingerprint",
            "registry_fingerprint",
        ),
        family_taxon_key=_first_text(row, "family_key", "family_taxon_key"),
        genus_taxon_key=_first_text(row, "genus_key", "genus_taxon_key"),
        taxonomic_status=(
            _first_text(row, "taxonomic_status", "status") or "accepted"
        ).casefold(),
    )


def _candidate_from_context(context: SpeciesContext) -> SpeciesCandidate:
    return SpeciesCandidate(
        scientific_name=context.scientific_name,
        canonical_name=context.canonical_name,
        rank="species",
        family=context.family,
        genus=context.genus,
        source="species_context",
        source_taxon_id=context.accepted_taxon_key,
        is_target_species=True,
        common_names=tuple(name.name for name in context.common_names),
        prompt_name_evidence=tuple(
            PromptNameEvidence(
                display_name=name.name,
                name_class="vernacular",
                trust_tier=name.trust_tier or "unrated",
                source=name.source or "species_context",
                source_record_id=(
                    name.source_record_id
                    or f"species-context:{context.accepted_taxon_key}:{_normalize(name.name)}"
                ),
                language=name.language,
                review_state=name.review_state or "",
            )
            for name in context.common_names
        ),
        taxonomy_version=context.registry_version,
        taxonomy_fingerprint=canonical_semantic_fingerprint(
            {
                "registry_version": context.registry_version,
                "accepted_taxon_key": context.accepted_taxon_key,
                "scientific_name": context.scientific_name,
                "family_key": context.family_key,
                "family": context.family,
                "genus_key": context.genus_key,
                "genus": context.genus,
            }
        ),
        family_taxon_key=context.family_key,
        genus_taxon_key=context.genus_key,
        taxonomic_status="accepted",
    )


def _candidate_prompt_context(
    candidate: SpeciesCandidate,
) -> AcceptedTaxonPromptContext:
    if not candidate.source_taxon_id:
        raise ValueError(
            f"Species candidate lacks accepted taxon identity: {candidate.scientific_name}"
        )
    if not candidate.family or not candidate.genus:
        raise ValueError(
            f"Species candidate lacks family/genus taxonomy: {candidate.scientific_name}"
        )
    if candidate.taxonomic_status.casefold() != "accepted":
        raise ValueError(
            f"Species candidate is not accepted taxonomy: {candidate.scientific_name}"
        )
    taxonomy_source = candidate.source or "species_candidate_table"
    taxonomy_version = candidate.taxonomy_version or "unversioned"
    fingerprint = candidate.taxonomy_fingerprint or canonical_semantic_fingerprint(
        {
            "source": taxonomy_source,
            "version": taxonomy_version,
            "accepted_taxon_key": candidate.source_taxon_id,
            "scientific_name": candidate.scientific_name,
            "family": candidate.family,
            "family_taxon_key": candidate.family_taxon_key,
            "genus": candidate.genus,
            "genus_taxon_key": candidate.genus_taxon_key,
        }
    )
    return AcceptedTaxonPromptContext(
        accepted_taxon_key=candidate.source_taxon_id,
        scientific_name=candidate.scientific_name,
        genus=candidate.genus,
        family=candidate.family,
        taxonomic_path=(
            TaxonomicPathNode(
                rank="SUPERFAMILY",
                scientific_name=BUTTERFLY_ROOT_SCIENTIFIC_NAME,
                accepted_taxon_key=BUTTERFLY_ROOT_ACCEPTED_TAXON_KEY,
            ),
            TaxonomicPathNode(
                rank="FAMILY",
                scientific_name=candidate.family,
                accepted_taxon_key=candidate.family_taxon_key,
            ),
            TaxonomicPathNode(
                rank="GENUS",
                scientific_name=candidate.genus,
                accepted_taxon_key=candidate.genus_taxon_key,
            ),
            TaxonomicPathNode(
                rank="SPECIES",
                scientific_name=candidate.scientific_name,
                accepted_taxon_key=candidate.source_taxon_id,
            ),
        ),
        taxonomy_source=taxonomy_source,
        taxonomy_version=taxonomy_version,
        taxonomy_fingerprint=fingerprint,
        taxonomic_status="ACCEPTED",
    )


def _prompt_name_evidence_from_row(
    row: dict[str, Any],
    *,
    common_names: tuple[str, ...],
    accepted_taxon_key: str | None,
) -> tuple[PromptNameEvidence, ...]:
    trust_tier = _first_text(
        row,
        "common_name_trust_tier",
        "vernacular_trust_tier",
    )
    source = _first_text(row, "common_name_source", "vernacular_source")
    if not common_names or not trust_tier or not source:
        return ()
    name_class = _first_text(row, "common_name_class", "vernacular_name_class") or (
        "vernacular"
    )
    language = _first_text(row, "common_name_language", "vernacular_language") or "und"
    review_state = (
        _first_text(
            row,
            "common_name_review_state",
            "vernacular_review_state",
        )
        or ""
    )
    record_prefix = (
        _first_text(
            row,
            "common_name_source_record_id",
            "vernacular_source_record_id",
        )
        or f"{source}:{accepted_taxon_key or 'unknown'}"
    )
    weak_homonym = _boolean_value(
        row.get("common_name_weak_homonym", row.get("vernacular_weak_homonym", False))
    )
    return tuple(
        PromptNameEvidence(
            display_name=name,
            name_class=name_class,
            trust_tier=trust_tier,
            source=source,
            source_record_id=f"{record_prefix}:{index}",
            language=language,
            review_state=review_state,
            weak_homonym=weak_homonym,
        )
        for index, name in enumerate(common_names)
    )


def _dedupe_candidates(candidates: list[SpeciesCandidate]) -> list[SpeciesCandidate]:
    best: dict[str, SpeciesCandidate] = {}
    for candidate in candidates:
        if candidate.rank != "species":
            continue
        key = _normalize(candidate.scientific_name)
        existing = best.get(key)
        if existing is None or (
            candidate.is_target_species and not existing.is_target_species
        ):
            best[key] = candidate
    return list(best.values())


def _first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def _split_common_names(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    names: list[str] = []
    for part in value.replace(";", "|").split("|"):
        cleaned = " ".join(part.strip().split())
        if cleaned and cleaned not in names:
            names.append(cleaned)
    return tuple(names)


def _boolean_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"", "0", "false", "no"}:
        return False
    if normalized in {"1", "true", "yes"}:
        return True
    raise ValueError(f"invalid boolean value: {value!r}")


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
