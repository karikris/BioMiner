from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Any, Mapping, Sequence

from biominer.bioclip.bioclip import DEFAULT_TRIAGE_LABELS
from biominer.bioclip.prompt_templates import PromptVariant
from biominer.bioclip.species_candidates import SpeciesCandidate, species_prompt_variants
from biominer.common.species import species_text_matches


BUTTERFLY_FAMILIES: tuple[str, ...] = (
    "Hesperiidae",
    "Papilionidae",
    "Pieridae",
    "Lycaenidae",
    "Riodinidae",
    "Nymphalidae",
    "Hedylidae",
)


class CandidateMode(StrEnum):
    TRIAGE = "triage"
    FAMILY = "family"
    GENUS = "genus"
    SPECIES = "species"
    HYBRID = "hybrid"
    RESCUE_FULL_SPECIES = "rescue_full_species"


class CandidateStrategy(StrEnum):
    ALL = "all"
    METADATA = "metadata"
    GEO = "geo"
    HIERARCHICAL = "hierarchical"
    FAMILY_TOPK = "family_topk"
    GENUS_TOPK = "genus_topk"
    RESCUE = "rescue"


@dataclass(frozen=True)
class CandidateSet:
    mode: CandidateMode
    strategy: CandidateStrategy
    label_sets: Mapping[str, tuple[str, ...]]
    species_candidates: tuple[SpeciesCandidate, ...] = ()
    species_prompt_variants: tuple[PromptVariant, ...] = ()
    family_candidates: tuple[str, ...] = ()
    genus_candidates_by_family: Mapping[str, tuple[str, ...]] | None = None
    provenance: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        return candidate_set_signature(self.label_sets)

    @property
    def label_count(self) -> int:
        return sum(len(labels) for labels in self.label_sets.values())


def parse_candidate_mode(value: str | CandidateMode) -> CandidateMode:
    if isinstance(value, CandidateMode):
        return value
    try:
        return CandidateMode(str(value).strip().casefold())
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in CandidateMode)
        raise ValueError(f"Unsupported candidate mode {value!r}; expected one of: {allowed}") from exc


def parse_candidate_strategy(value: str | CandidateStrategy) -> CandidateStrategy:
    if isinstance(value, CandidateStrategy):
        return value
    try:
        return CandidateStrategy(str(value).strip().casefold())
    except ValueError as exc:
        allowed = ", ".join(strategy.value for strategy in CandidateStrategy)
        raise ValueError(f"Unsupported candidate strategy {value!r}; expected one of: {allowed}") from exc


def candidate_set_signature(label_sets: Mapping[str, Sequence[str]]) -> str:
    payload = {
        str(name): list(labels)
        for name, labels in sorted(label_sets.items(), key=lambda item: str(item[0]))
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def build_candidate_set(
    record: Mapping[str, Any],
    *,
    species_candidates: Sequence[SpeciesCandidate],
    mode: str | CandidateMode = CandidateMode.HYBRID,
    strategy: str | CandidateStrategy = CandidateStrategy.ALL,
    candidate_limit: int | None = None,
) -> CandidateSet:
    parsed_mode = parse_candidate_mode(mode)
    parsed_strategy = parse_candidate_strategy(strategy)
    selected_species = tuple(_select_species_candidates(record, species_candidates, parsed_strategy, candidate_limit))
    label_sets: dict[str, tuple[str, ...]] = {}
    families: tuple[str, ...] = ()
    genera_by_family: Mapping[str, tuple[str, ...]] | None = None
    variants: tuple[PromptVariant, ...] = ()
    provenance: list[str] = []

    if parsed_mode in {CandidateMode.TRIAGE, CandidateMode.HYBRID, CandidateMode.RESCUE_FULL_SPECIES}:
        label_sets["triage"] = tuple(DEFAULT_TRIAGE_LABELS)
        provenance.append("triage_labels")
    if parsed_mode in {CandidateMode.FAMILY, CandidateMode.HYBRID}:
        families = family_candidates(selected_species)
        label_sets["family"] = family_labels(families)
        provenance.append("family_from_species_candidates")
    if parsed_mode in {CandidateMode.GENUS, CandidateMode.HYBRID}:
        families = families or retained_families_from_record(record) or family_candidates(selected_species)
        genera_by_family = genus_candidates_by_family(selected_species, families=families)
        genus_labels_flat = tuple(
            label
            for family in sorted(genera_by_family)
            for label in genus_labels(genera_by_family[family])
        )
        label_sets["genus"] = genus_labels_flat
        provenance.append("genus_from_family_gate")
    if parsed_mode in {CandidateMode.SPECIES, CandidateMode.HYBRID, CandidateMode.RESCUE_FULL_SPECIES}:
        if parsed_mode == CandidateMode.RESCUE_FULL_SPECIES and parsed_strategy != CandidateStrategy.RESCUE:
            selected_species = tuple(_limit_candidates(species_candidates, candidate_limit))
            provenance.append("rescue_full_species")
        variants = tuple(species_prompt_variants(list(selected_species)))
        label_sets["species"] = tuple(variant.label for variant in variants)
        provenance.append(f"species_{parsed_strategy.value}")

    return CandidateSet(
        mode=parsed_mode,
        strategy=parsed_strategy,
        label_sets=label_sets,
        species_candidates=selected_species,
        species_prompt_variants=variants,
        family_candidates=families,
        genus_candidates_by_family=genera_by_family,
        provenance=tuple(provenance),
    )


def family_candidates(candidates: Sequence[SpeciesCandidate]) -> tuple[str, ...]:
    families = {candidate.family for candidate in candidates if candidate.family}
    ordered = [family for family in BUTTERFLY_FAMILIES if family in families]
    ordered.extend(sorted(family for family in families if family not in BUTTERFLY_FAMILIES))
    return tuple(ordered)


def family_labels(families: Sequence[str] = BUTTERFLY_FAMILIES) -> tuple[str, ...]:
    return tuple(f"a photo of a {family} butterfly" for family in families)


def retained_families_from_record(record: Mapping[str, Any], *, default_top_k: int = 3) -> tuple[str, ...]:
    rows = _coerce_topk_rows(record.get("family_topk_json"))
    if rows:
        return tuple(str(row["label"]).replace("a photo of a ", "").replace(" butterfly", "") for row in rows[:default_top_k])
    top1 = record.get("family_top1")
    return (str(top1),) if top1 else ()


def genus_candidates_by_family(
    candidates: Sequence[SpeciesCandidate],
    *,
    families: Sequence[str],
    per_family_limit: int = 8,
) -> dict[str, tuple[str, ...]]:
    family_set = {str(family) for family in families}
    grouped: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        if candidate.family in family_set and candidate.genus:
            grouped[str(candidate.family)].add(candidate.genus)
    return {
        family: tuple(sorted(genera)[:per_family_limit])
        for family, genera in sorted(grouped.items())
    }


def genus_labels(genera: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"a photo of a {genus} butterfly" for genus in genera)


def _select_species_candidates(
    record: Mapping[str, Any],
    candidates: Sequence[SpeciesCandidate],
    strategy: CandidateStrategy,
    candidate_limit: int | None,
) -> list[SpeciesCandidate]:
    if strategy == CandidateStrategy.METADATA:
        selected = [candidate for candidate in candidates if _metadata_matches_candidate(record, candidate)]
        return selected or list(_limit_candidates(candidates, candidate_limit))
    if strategy == CandidateStrategy.GEO:
        selected = _species_from_record_names(record, candidates, ("geo_candidate_species_json", "geo_species_candidates_json"))
        return selected or list(_limit_candidates(candidates, candidate_limit))
    if strategy in {CandidateStrategy.HIERARCHICAL, CandidateStrategy.GENUS_TOPK}:
        genera = _retained_genera_from_record(record)
        if genera:
            selected = [candidate for candidate in candidates if candidate.genus in genera]
            return selected[:candidate_limit] if candidate_limit else selected
    if strategy == CandidateStrategy.FAMILY_TOPK:
        families = retained_families_from_record(record)
        if families:
            family_set = set(families)
            selected = [candidate for candidate in candidates if candidate.family in family_set]
            return selected[:candidate_limit] if candidate_limit else selected
    return list(_limit_candidates(candidates, candidate_limit))


def _limit_candidates(candidates: Sequence[SpeciesCandidate], limit: int | None) -> Sequence[SpeciesCandidate]:
    return candidates[:limit] if limit and limit > 0 else candidates


def _metadata_matches_candidate(record: Mapping[str, Any], candidate: SpeciesCandidate) -> bool:
    text_values = [
        record.get("title"),
        record.get("description"),
        record.get("tags"),
        record.get("machine_tags"),
        record.get("raw_title"),
        record.get("raw_description"),
        record.get("raw_tags"),
    ]
    if species_text_matches(candidate.scientific_name, text_values):
        return True
    return any(species_text_matches(common_name, text_values) for common_name in candidate.common_names)


def _species_from_record_names(
    record: Mapping[str, Any],
    candidates: Sequence[SpeciesCandidate],
    field_names: Sequence[str],
) -> list[SpeciesCandidate]:
    names: set[str] = set()
    for field_name in field_names:
        for row in _coerce_topk_rows(record.get(field_name)):
            name = row.get("scientific_name") or row.get("species") or row.get("species_name")
            if name:
                names.add(_normalize(name))
    if not names:
        return []
    return [candidate for candidate in candidates if _normalize(candidate.scientific_name) in names]


def _retained_genera_from_record(record: Mapping[str, Any]) -> set[str]:
    rows = _coerce_topk_rows(record.get("genus_topk_json"))
    names = {str(row.get("genus") or row.get("label") or "").replace("a photo of a ", "").replace(" butterfly", "") for row in rows}
    return {name for name in names if name}


def _coerce_topk_rows(value: object) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        value = decoded
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
