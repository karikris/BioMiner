from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Any, Mapping, Sequence

import polars as pl

from biominer.bioclip.bioclip import DEFAULT_TRIAGE_LABELS
from biominer.bioclip.prompt_templates import PromptVariant
from biominer.bioclip.species_candidates import SpeciesCandidate, species_prompt_variants
from biominer.common.species import species_text_matches
from biominer.geo.grid import candidate_set_for_point


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
    species_candidate_sources_json: str = "[]"
    geo_candidate_cell_id: str | None = None
    geo_candidate_grid_level: str | None = None
    geo_candidate_fallback_level: str | None = None

    @property
    def signature(self) -> str:
        return _candidate_signature(self)

    @property
    def label_count(self) -> int:
        return sum(len(labels) for labels in self.label_sets.values())


@dataclass(frozen=True)
class _CandidateSelection:
    species_candidates: tuple[SpeciesCandidate, ...]
    provenance: tuple[str, ...]
    species_candidate_sources_json: str = "[]"
    geo_candidate_cell_id: str | None = None
    geo_candidate_grid_level: str | None = None
    geo_candidate_fallback_level: str | None = None


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
    geo_species_index: pl.DataFrame | None = None,
    geo_grid_level: str = "G4_5deg",
    geo_min_species_per_cell: int = 5,
    geo_include_neighbours: bool = False,
) -> CandidateSet:
    parsed_mode = parse_candidate_mode(mode)
    parsed_strategy = parse_candidate_strategy(strategy)
    selection = _select_species_candidates(
        record,
        species_candidates,
        parsed_strategy,
        candidate_limit,
        geo_species_index=geo_species_index,
        geo_grid_level=geo_grid_level,
        geo_min_species_per_cell=geo_min_species_per_cell,
        geo_include_neighbours=geo_include_neighbours,
    )
    selected_species = selection.species_candidates
    label_sets: dict[str, tuple[str, ...]] = {}
    families: tuple[str, ...] = ()
    genera_by_family: Mapping[str, tuple[str, ...]] | None = None
    variants: tuple[PromptVariant, ...] = ()
    provenance: list[str] = list(selection.provenance)

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
        species_candidate_sources_json=selection.species_candidate_sources_json,
        geo_candidate_cell_id=selection.geo_candidate_cell_id,
        geo_candidate_grid_level=selection.geo_candidate_grid_level,
        geo_candidate_fallback_level=selection.geo_candidate_fallback_level,
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
    *,
    geo_species_index: pl.DataFrame | None,
    geo_grid_level: str,
    geo_min_species_per_cell: int,
    geo_include_neighbours: bool,
) -> _CandidateSelection:
    geo = _geo_candidate_selection(
        record,
        candidates,
        geo_species_index=geo_species_index,
        geo_grid_level=geo_grid_level,
        geo_min_species_per_cell=geo_min_species_per_cell,
        geo_include_neighbours=geo_include_neighbours,
        candidate_limit=candidate_limit,
    )
    if strategy == CandidateStrategy.METADATA:
        selected = [candidate for candidate in candidates if _metadata_matches_candidate(record, candidate)]
        if selected:
            return _selection(selected, "metadata_match", candidate_limit, source_rows=[{"source": "metadata_match", "species_count": len(selected)}])
        return _selection(candidates, "metadata_fallback_global", candidate_limit, source_rows=[{"source": "global_fallback"}])
    if strategy == CandidateStrategy.GEO:
        if geo is not None and geo.species_candidates:
            return geo
        selected = _species_from_record_names(record, candidates, ("geo_candidate_species_json", "geo_species_candidates_json"))
        if selected:
            return _selection(selected, "record_geo_candidates", candidate_limit, source_rows=[{"source": "record_geo_candidates", "species_count": len(selected)}])
        if geo is not None:
            return _selection(
                candidates,
                "geo_empty_global_rescue",
                candidate_limit,
                source_rows=_json_rows(geo.species_candidate_sources_json) + [{"source": "global_rescue"}],
                geo_candidate_cell_id=geo.geo_candidate_cell_id,
                geo_candidate_grid_level=geo.geo_candidate_grid_level,
                geo_candidate_fallback_level=geo.geo_candidate_fallback_level,
            )
        return _selection(candidates, "geo_unavailable_global_rescue", candidate_limit, source_rows=[{"source": "global_rescue"}])
    if strategy == CandidateStrategy.HIERARCHICAL:
        base = list(geo.species_candidates) if geo is not None and geo.species_candidates else list(candidates)
        metadata_rescue = [candidate for candidate in candidates if _metadata_matches_candidate(record, candidate)]
        families = retained_families_from_record(record)
        genera = _retained_genera_from_record(record)
        if genera:
            gated = [candidate for candidate in base if candidate.genus in genera]
            gate_source = "genus_gate"
        elif families:
            family_set = set(families)
            gated = [candidate for candidate in base if candidate.family in family_set]
            gate_source = "family_gate"
        else:
            gated = base
            gate_source = "geo_base" if geo is not None and geo.species_candidates else "global_base"
        selected = _dedupe_species([*gated, *metadata_rescue])
        source_rows = _json_rows(geo.species_candidate_sources_json) if geo is not None else [{"source": "global_candidates"}]
        source_rows.append({"source": gate_source, "species_count": len(gated)})
        if metadata_rescue:
            source_rows.append({"source": "metadata_rescue", "species_count": len(metadata_rescue)})
        if not selected and geo is not None and geo.species_candidates:
            selected = list(geo.species_candidates)
            source_rows.append({"source": "geo_only_fallback", "species_count": len(selected)})
        return _selection(
            selected or candidates,
            "hierarchical",
            candidate_limit,
            source_rows=source_rows,
            geo_candidate_cell_id=geo.geo_candidate_cell_id if geo is not None else None,
            geo_candidate_grid_level=geo.geo_candidate_grid_level if geo is not None else None,
            geo_candidate_fallback_level=geo.geo_candidate_fallback_level if geo is not None else None,
        )
    if strategy == CandidateStrategy.GENUS_TOPK:
        genera = _retained_genera_from_record(record)
        if genera:
            selected = [candidate for candidate in candidates if candidate.genus in genera]
            return _selection(selected, "genus_topk", candidate_limit, source_rows=[{"source": "genus_topk", "species_count": len(selected)}])
    if strategy == CandidateStrategy.FAMILY_TOPK:
        families = retained_families_from_record(record)
        if families:
            family_set = set(families)
            selected = [candidate for candidate in candidates if candidate.family in family_set]
            return _selection(selected, "family_topk", candidate_limit, source_rows=[{"source": "family_topk", "species_count": len(selected)}])
    return _selection(candidates, "all_candidates", candidate_limit, source_rows=[{"source": "all_candidates"}])


def _selection(
    candidates: Sequence[SpeciesCandidate],
    provenance: str,
    candidate_limit: int | None,
    *,
    source_rows: Sequence[Mapping[str, object]],
    geo_candidate_cell_id: str | None = None,
    geo_candidate_grid_level: str | None = None,
    geo_candidate_fallback_level: str | None = None,
) -> _CandidateSelection:
    selected = tuple(_limit_candidates(_dedupe_species(candidates), candidate_limit))
    return _CandidateSelection(
        species_candidates=selected,
        provenance=(provenance,),
        species_candidate_sources_json=_json_dumps(source_rows),
        geo_candidate_cell_id=geo_candidate_cell_id,
        geo_candidate_grid_level=geo_candidate_grid_level,
        geo_candidate_fallback_level=geo_candidate_fallback_level,
    )


def _geo_candidate_selection(
    record: Mapping[str, Any],
    candidates: Sequence[SpeciesCandidate],
    *,
    geo_species_index: pl.DataFrame | None,
    geo_grid_level: str,
    geo_min_species_per_cell: int,
    geo_include_neighbours: bool,
    candidate_limit: int | None,
) -> _CandidateSelection | None:
    if geo_species_index is None:
        return None
    latitude = _optional_float(record.get("latitude", record.get("decimalLatitude")))
    longitude = _optional_float(record.get("longitude", record.get("decimalLongitude")))
    if latitude is None or longitude is None:
        return None
    lookup = candidate_set_for_point(
        geo_species_index,
        latitude=latitude,
        longitude=longitude,
        preferred_grid_level=geo_grid_level,
        min_species_per_cell=geo_min_species_per_cell,
        include_neighbours=geo_include_neighbours,
    )
    selected = _species_from_geo_frame(lookup.candidates, candidates)
    source_rows = [
        {
            "source": "gbif_geo",
            "requested_grid_level": lookup.requested_grid_level,
            "grid_level": lookup.selected_grid_level,
            "geocell_id": lookup.geocell_id,
            "fallback_reason": lookup.fallback_reason,
            "index_rows": lookup.candidates.height,
            "species_count": len(selected),
        }
    ]
    return _selection(
        selected,
        "gbif_geo_candidates",
        candidate_limit,
        source_rows=source_rows,
        geo_candidate_cell_id=lookup.geocell_id,
        geo_candidate_grid_level=lookup.selected_grid_level,
        geo_candidate_fallback_level=lookup.fallback_reason,
    )


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


def _species_from_geo_frame(geo_frame: pl.DataFrame, candidates: Sequence[SpeciesCandidate]) -> list[SpeciesCandidate]:
    if geo_frame.is_empty():
        return []
    rows = geo_frame.to_dicts()
    names = {_normalize(row.get("scientific_name")) for row in rows if row.get("scientific_name")}
    keys = {
        _normalize_source_key(row.get("species_key") or row.get("speciesKey") or row.get("taxonKey"))
        for row in rows
        if row.get("species_key") or row.get("speciesKey") or row.get("taxonKey")
    }
    selected: list[SpeciesCandidate] = []
    for candidate in candidates:
        if _normalize(candidate.scientific_name) in names:
            selected.append(candidate)
            continue
        source_key = _normalize_source_key(candidate.source_taxon_id)
        if source_key and source_key in keys:
            selected.append(candidate)
    return selected


def _dedupe_species(candidates: Sequence[SpeciesCandidate]) -> list[SpeciesCandidate]:
    deduped: dict[str, SpeciesCandidate] = {}
    for candidate in candidates:
        key = _normalize(candidate.scientific_name)
        if key and key not in deduped:
            deduped[key] = candidate
    return list(deduped.values())


def _candidate_signature(candidate_set: CandidateSet) -> str:
    payload = {
        "label_sets": {
            str(name): list(labels)
            for name, labels in sorted(candidate_set.label_sets.items(), key=lambda item: str(item[0]))
        },
        "provenance": list(candidate_set.provenance),
        "species_candidate_sources_json": candidate_set.species_candidate_sources_json,
        "geo_candidate_cell_id": candidate_set.geo_candidate_cell_id,
        "geo_candidate_grid_level": candidate_set.geo_candidate_grid_level,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _json_dumps(rows: Sequence[Mapping[str, object]]) -> str:
    return json.dumps([dict(row) for row in rows], ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _json_rows(value: str) -> list[dict[str, object]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [row for row in decoded if isinstance(row, dict)] if isinstance(decoded, list) else []


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _normalize_source_key(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).removeprefix("gbif:")


def _normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
