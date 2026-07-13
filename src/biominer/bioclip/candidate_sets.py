from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.filter.extractor import SCIENTIFIC_NAME_PATTERN
from biominer.species.context import SpeciesContext


PROMPT_VARIANT_VERSION = "object-bioclip-prompts-v1"
CANDIDATE_SET_CONTRACT_VERSION = "object-bioclip-candidates-v2"
REGIONAL_CANDIDATE_SCHEMA_VERSION = "regional-candidate-species-v1.0.0"
_REGIONAL_CANDIDATE_FIELDS = {
    "schema_version",
    "candidate_set_id",
    "target_accepted_taxon_key",
    "geo_cluster_id",
    "candidate_accepted_taxon_key",
    "candidate_reason",
    "target_candidate",
    "candidate_priority",
    "source_versions",
    "candidate_set_fingerprint",
}


@dataclass(frozen=True)
class CandidateTaxon:
    scientific_name: str
    accepted_taxon_key: str | None = None
    rank: str = "species"
    family: str | None = None
    genus: str | None = None
    common_names: tuple[str, ...] = ()
    candidate_reasons: tuple[str, ...] = ()
    source_versions: tuple[str, ...] = ()
    target_candidate: bool = False
    candidate_priority: int | None = None

    def __post_init__(self) -> None:
        if self.candidate_priority is not None:
            if isinstance(self.candidate_priority, bool) or not isinstance(
                self.candidate_priority, int
            ):
                raise TypeError("candidate_priority must be an integer or null")
            if self.candidate_priority < 0:
                raise ValueError("candidate_priority must be nonnegative")


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
    candidate_contract_version: str = CANDIDATE_SET_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.candidate_contract_version != CANDIDATE_SET_CONTRACT_VERSION:
            raise ValueError(
                "unsupported candidate contract version: "
                f"{self.candidate_contract_version}"
            )

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
    geo_prior_table: pl.DataFrame | None = None,
    allow_single_target_fixture: bool = False,
) -> CandidateSet:
    target = _target_candidate(context)
    source_evidence = ["species_context"]
    if geospatial_scope:
        source_evidence.append(f"geospatial_scope:{geospatial_scope}")
    candidate_rows = _candidate_rows(species_candidate_path) if species_candidate_path else []
    if candidate_rows:
        source_evidence.append(str(species_candidate_path))
    candidate_rows, regional_candidate_set_id, regional_source_versions = (
        _select_regional_candidate_rows(
            candidate_rows,
            context=context,
            geospatial_scope=geospatial_scope,
        )
    )
    if regional_candidate_set_id:
        source_evidence.append(f"regional_candidate_set:{regional_candidate_set_id}")
        source_evidence.extend(
            f"regional_candidate_source:{version}" for version in regional_source_versions
        )
    geo_prior_rows = geo_prior_table.to_dicts() if geo_prior_table is not None and not geo_prior_table.is_empty() else []
    if geo_prior_rows:
        source_evidence.append("geospatial_prior_table")
    species = _species_candidates(context, candidate_rows)
    if regional_candidate_set_id is None:
        species, _ = _add_geospatial_prior_candidates(species, context, geo_prior_rows)
        species, query_provenance_added = _add_query_provenance_candidates(
            species,
            records or [],
            candidate_lookup=_candidate_lookup(candidate_rows),
            group_lookup=_candidate_group_lookup(candidate_rows),
        )
        if query_provenance_added:
            source_evidence.append("query_provenance")
        species, metadata_text_added = _add_metadata_text_candidates(species, records or [])
        if metadata_text_added:
            source_evidence.append("metadata_text")
        species, comment_added = _add_comment_candidates(species, records or [])
        if comment_added:
            source_evidence.append("comments")
    if not any(_norm(candidate.scientific_name) == _norm(context.scientific_name) for candidate in species):
        species.insert(0, target)
    species = _dedupe_taxa(species)
    if len(species) <= 1 and not allow_single_target_fixture:
        raise ValueError(
            "species candidate set requires registry-derived same-genus/same-family candidates; "
            "pass allow_single_target_fixture=True only for explicit tests"
        )
    if len(species) <= 1:
        source_evidence.append("single_target_fixture")
    genus = tuple(
        candidate
        for candidate in species
        if candidate.genus
        and (
            regional_candidate_set_id is not None
            or _norm(candidate.genus) == _norm(context.genus)
        )
    )
    family = tuple(candidate for candidate in species if candidate.family)
    candidate_set_id = regional_candidate_set_id or _candidate_set_id(
        context=context,
        species=species,
        geospatial_scope=geospatial_scope,
    )
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


def build_candidate_set_for_taxon_scope(
    taxon_scope: Any,
    *,
    target_context: SpeciesContext | None = None,
    species_candidate_path: str | Path | None = None,
    records: list[dict[str, Any]] | None = None,
    geospatial_scope: str | None = None,
    geo_prior_table: pl.DataFrame | None = None,
    allow_single_target_fixture: bool = False,
) -> CandidateSet:
    contexts = tuple(getattr(taxon_scope, "species_contexts", ()) or ())
    if not contexts:
        raise ValueError("taxon_scope must include at least one SpeciesContext")
    for context in contexts:
        if not isinstance(context, SpeciesContext):
            raise TypeError("taxon_scope.species_contexts must contain SpeciesContext instances")
    scope_rank = str(getattr(taxon_scope, "accepted_rank", "species") or "species").casefold()
    if scope_rank not in {"family", "genus", "species"}:
        raise ValueError("taxon_scope.accepted_rank must be family, genus, or species")
    target = target_context or contexts[0]
    if target not in contexts and not any(_norm(context.accepted_taxon_key) == _norm(target.accepted_taxon_key) for context in contexts):
        raise ValueError("target_context must belong to taxon_scope.species_contexts")

    candidate_rows = _candidate_rows(species_candidate_path) if species_candidate_path else []
    geo_prior_rows = geo_prior_table.to_dicts() if geo_prior_table is not None and not geo_prior_table.is_empty() else []
    source_evidence = [f"taxon_scope:{scope_rank}"]
    if geospatial_scope:
        source_evidence.append(f"geospatial_scope:{geospatial_scope}")
    if candidate_rows:
        source_evidence.append(str(species_candidate_path))
    if geo_prior_rows:
        source_evidence.append("geospatial_prior_table")

    if scope_rank in {"family", "genus"}:
        species = [_target_candidate(context) for context in contexts]
        species.extend(_scope_candidate_rows(taxon_scope, rows=candidate_rows))
    else:
        species = _species_candidates(target, candidate_rows)
        if len(_dedupe_taxa(species)) <= 1 and not allow_single_target_fixture:
            raise ValueError(
                "species candidate set requires registry-derived same-genus/same-family candidates; "
                "pass allow_single_target_fixture=True only for explicit tests"
            )

    species = _dedupe_taxa(species)
    species, _ = _add_geospatial_prior_candidates(species, target, geo_prior_rows)
    species, query_provenance_added = _add_query_provenance_candidates(
        species,
        records or [],
        candidate_lookup=_candidate_lookup(candidate_rows),
        group_lookup=_candidate_group_lookup(candidate_rows),
    )
    if query_provenance_added:
        source_evidence.append("query_provenance")
    species, metadata_text_added = _add_metadata_text_candidates(species, records or [])
    if metadata_text_added:
        source_evidence.append("metadata_text")
    species, comment_added = _add_comment_candidates(species, records or [])
    if comment_added:
        source_evidence.append("comments")
    species = _dedupe_taxa(species)
    genus = tuple(candidate for candidate in species if candidate.genus)
    family = tuple(candidate for candidate in species if candidate.family)
    return CandidateSet(
        candidate_set_id=_candidate_set_id_for_scope(
            registry_version=str(getattr(taxon_scope, "registry_version", target.registry_version) or target.registry_version),
            target_accepted_taxon_key=str(getattr(taxon_scope, "accepted_taxon_key", target.accepted_taxon_key) or target.accepted_taxon_key),
            target_scientific_name=str(getattr(taxon_scope, "accepted_scientific_name", target.scientific_name) or target.scientific_name),
            species=species,
            geospatial_scope=geospatial_scope,
        ),
        registry_version=str(getattr(taxon_scope, "registry_version", target.registry_version) or target.registry_version),
        target_accepted_taxon_key=str(getattr(taxon_scope, "accepted_taxon_key", target.accepted_taxon_key) or target.accepted_taxon_key),
        target_scientific_name=str(getattr(taxon_scope, "accepted_scientific_name", target.scientific_name) or target.scientific_name),
        family_candidates=family,
        genus_candidates=genus,
        species_candidates=tuple(species),
        prompt_variant_version=PROMPT_VARIANT_VERSION,
        geospatial_scope=geospatial_scope,
        source_evidence=tuple(_unique(source_evidence)),
    )


def _target_candidate(context: SpeciesContext) -> CandidateTaxon:
    return CandidateTaxon(
        scientific_name=context.scientific_name,
        accepted_taxon_key=context.accepted_taxon_key,
        rank="species",
        family=context.family,
        genus=context.genus,
        common_names=tuple(name.name for name in context.common_names),
        candidate_reasons=("target",),
        source_versions=(f"registry:{context.registry_version}",),
        target_candidate=True,
        candidate_priority=0,
    )


def _candidate_rows(path: str | Path) -> list[dict[str, Any]]:
    return pl.read_parquet(path).to_dicts()


def _select_regional_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    context: SpeciesContext,
    geospatial_scope: str | None,
) -> tuple[list[dict[str, Any]], str | None, tuple[str, ...]]:
    if not rows:
        return rows, None, ()
    if not _is_regional_candidate_rows(rows):
        if any(
            "candidate_accepted_taxon_key" in row
            or str(row.get("schema_version") or "").startswith(
                "regional-candidate-species-"
            )
            for row in rows
        ):
            raise ValueError("regional candidate rows do not satisfy the versioned contract")
        return rows, None, ()
    if any(row.get("schema_version") != REGIONAL_CANDIDATE_SCHEMA_VERSION for row in rows):
        raise ValueError("regional candidate rows have an unsupported schema version")

    target_rows = [
        row
        for row in rows
        if _norm(row.get("target_accepted_taxon_key"))
        == _norm(context.accepted_taxon_key)
    ]
    if not target_rows:
        raise ValueError("regional candidate artifact has no set for the target taxon")
    cluster_ids = sorted({_first_text(row, "geo_cluster_id") or "" for row in target_rows})
    requested_scope = str(geospatial_scope or "").strip()
    if requested_scope in cluster_ids:
        target_rows = [
            row for row in target_rows if _first_text(row, "geo_cluster_id") == requested_scope
        ]
    elif len(cluster_ids) != 1:
        raise ValueError(
            "regional candidate artifact contains multiple geographic clusters; "
            "geospatial_scope must select one geo_cluster_id"
        )

    candidate_set_ids = {
        _first_text(row, "candidate_set_id") or "" for row in target_rows
    }
    if len(candidate_set_ids) != 1 or "" in candidate_set_ids:
        raise ValueError("regional candidate selection must resolve to one candidate set")
    candidate_set_id = next(iter(candidate_set_ids))
    target_candidates = [row for row in target_rows if row.get("target_candidate") is True]
    if len(target_candidates) != 1:
        raise ValueError("regional candidate set must contain exactly one target candidate")
    target_candidate_key = _first_text(
        target_candidates[0], "candidate_accepted_taxon_key"
    )
    if _norm(target_candidate_key) != _norm(context.accepted_taxon_key):
        raise ValueError("regional target candidate does not match the target taxon")
    priorities = sorted(_required_nonnegative_int(row.get("candidate_priority")) for row in target_rows)
    if priorities != list(range(len(target_rows))):
        raise ValueError("regional candidate priorities must be contiguous")
    if len({_first_text(row, "candidate_accepted_taxon_key") for row in target_rows}) != len(
        target_rows
    ):
        raise ValueError("regional candidate set contains duplicate species")
    if len({_first_text(row, "candidate_set_fingerprint") for row in target_rows}) != 1:
        raise ValueError("regional candidate set contains conflicting fingerprints")
    source_versions = _unique(
        version
        for row in target_rows
        for version in _split_names(row.get("source_versions"))
    )
    ordered = sorted(
        target_rows,
        key=lambda row: (
            _required_nonnegative_int(row.get("candidate_priority")),
            _first_text(row, "candidate_accepted_taxon_key") or "",
        ),
    )
    return ordered, candidate_set_id, source_versions


def _is_regional_candidate_rows(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(_REGIONAL_CANDIDATE_FIELDS <= set(row) for row in rows)


def _species_candidates(context: SpeciesContext, rows: list[dict[str, Any]]) -> list[CandidateTaxon]:
    if _is_regional_candidate_rows(rows):
        candidates = [candidate for row in rows if (candidate := _candidate_from_row(row))]
        if not any(candidate.target_candidate for candidate in candidates):
            raise ValueError("regional candidate set has no target candidate")
        return _dedupe_taxa(candidates)
    target = _target_candidate(context)
    candidates: list[CandidateTaxon] = [target]
    for row in rows:
        candidate = _candidate_from_row(row)
        if candidate is None:
            continue
        if not _candidate_in_context_scope(candidate, context):
            continue
        candidates.append(candidate)
    return candidates


def _scope_candidate_rows(taxon_scope: Any, *, rows: list[dict[str, Any]]) -> list[CandidateTaxon]:
    scope_rank = str(getattr(taxon_scope, "accepted_rank", "") or "").casefold()
    scope_key = _norm(getattr(taxon_scope, "accepted_taxon_key", ""))
    scope_name = _norm(getattr(taxon_scope, "accepted_scientific_name", ""))
    candidates: list[CandidateTaxon] = []
    for row in rows:
        candidate = _candidate_from_row(row)
        if candidate is None:
            continue
        if scope_rank == "family":
            row_key = _norm(_first_text(row, "family_key"))
            row_name = _norm(candidate.family)
        elif scope_rank == "genus":
            row_key = _norm(_first_text(row, "genus_key"))
            row_name = _norm(candidate.genus)
        else:
            continue
        if (scope_key and row_key == scope_key) or (scope_name and row_name == scope_name):
            candidates.append(candidate)
    return _dedupe_taxa(candidates)


def _add_geospatial_prior_candidates(
    candidates: list[CandidateTaxon],
    context: SpeciesContext,
    rows: list[dict[str, Any]],
) -> tuple[list[CandidateTaxon], bool]:
    by_name = {_norm(candidate.scientific_name) for candidate in candidates}
    by_key = {str(candidate.accepted_taxon_key or "") for candidate in candidates if candidate.accepted_taxon_key}
    added = False
    for row in rows:
        candidate = _candidate_from_row(row)
        if candidate is None or not _candidate_in_context_scope(candidate, context):
            continue
        name_key = _norm(candidate.scientific_name)
        taxon_key = str(candidate.accepted_taxon_key or "")
        if name_key in by_name or (taxon_key and taxon_key in by_key):
            continue
        candidates.append(candidate)
        by_name.add(name_key)
        if taxon_key:
            by_key.add(taxon_key)
        added = True
    return candidates, added


def _candidate_in_context_scope(candidate: CandidateTaxon, context: SpeciesContext) -> bool:
    same_genus = _norm(candidate.genus) == _norm(context.genus)
    same_family = _norm(candidate.family) == _norm(context.family)
    is_target = _norm(candidate.scientific_name) == _norm(context.scientific_name)
    is_target_key = bool(candidate.accepted_taxon_key) and _norm(candidate.accepted_taxon_key) in {
        _norm(context.accepted_taxon_key),
        _norm(context.species_key),
    }
    return is_target or is_target_key or same_genus or same_family


def _add_query_provenance_candidates(
    candidates: list[CandidateTaxon],
    records: list[dict[str, Any]],
    *,
    candidate_lookup: dict[str, CandidateTaxon],
    group_lookup: dict[str, tuple[CandidateTaxon, ...]],
) -> tuple[list[CandidateTaxon], bool]:
    by_key = {str(candidate.accepted_taxon_key or ""): candidate for candidate in candidates if candidate.accepted_taxon_key}
    by_name = {_norm(candidate.scientific_name) for candidate in candidates}
    added = False
    for record in records:
        keys = _query_provenance_keys(record)
        names = record.get("scientific_names_detected") or ()
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(names, str):
            names = [names]
        for key in keys:
            normalized_key = str(key or "")
            if not normalized_key or normalized_key in by_key:
                continue
            candidate = candidate_lookup.get(normalized_key)
            if candidate is None:
                continue
            candidates.append(candidate)
            by_key[normalized_key] = candidate
            by_name.add(_norm(candidate.scientific_name))
            added = True
        for key in _query_provenance_group_keys(record):
            for candidate in group_lookup.get(key, ()):
                name_key = _norm(candidate.scientific_name)
                taxon_key = str(candidate.accepted_taxon_key or "")
                if name_key in by_name or (taxon_key and taxon_key in by_key):
                    continue
                candidates.append(candidate)
                by_name.add(name_key)
                if taxon_key:
                    by_key[taxon_key] = candidate
                added = True
        for key, name in zip(keys, names, strict=False):
            if str(key) in by_key or not name:
                continue
            candidates.append(CandidateTaxon(scientific_name=str(name), accepted_taxon_key=str(key), rank="species"))
            by_key[str(key)] = candidates[-1]
            by_name.add(_norm(str(name)))
            added = True
    return candidates, added


def _candidate_lookup(rows: list[dict[str, Any]]) -> dict[str, CandidateTaxon]:
    lookup: dict[str, CandidateTaxon] = {}
    for row in rows:
        candidate = _candidate_from_row(row)
        if candidate is None:
            continue
        for key_name in ("accepted_taxon_key", "source_taxon_id", "taxon_id", "taxonID", "species_key"):
            key = _first_text(row, key_name)
            if key:
                lookup.setdefault(key, candidate)
    return lookup


def _candidate_group_lookup(rows: list[dict[str, Any]]) -> dict[str, tuple[CandidateTaxon, ...]]:
    grouped: dict[str, list[CandidateTaxon]] = {}
    for row in rows:
        candidate = _candidate_from_row(row)
        if candidate is None:
            continue
        for key_name in ("family_key", "genus_key"):
            key = _first_text(row, key_name)
            if key:
                grouped.setdefault(key, []).append(candidate)
    return {key: tuple(_dedupe_taxa(candidates)) for key, candidates in grouped.items()}


def _candidate_from_row(row: dict[str, Any]) -> CandidateTaxon | None:
    rank = (_first_text(row, "rank", "taxon_rank") or "species").casefold()
    if rank != "species":
        return None
    scientific_name = _first_text(row, "scientific_name", "accepted_scientific_name", "canonical_name", "species")
    if not scientific_name:
        return None
    genus = _first_text(row, "genus") or scientific_name.split(" ", 1)[0]
    return CandidateTaxon(
        scientific_name=scientific_name,
        accepted_taxon_key=_first_text(
            row,
            "candidate_accepted_taxon_key",
            "accepted_taxon_key",
            "source_taxon_id",
            "taxon_id",
            "taxonID",
            "species_key",
        ),
        rank=rank,
        family=_first_text(row, "family"),
        genus=genus,
        common_names=_split_names(_first_value(row, "common_names", "vernacular_names")),
        candidate_reasons=_split_names(_first_value(row, "candidate_reason")),
        source_versions=_split_names(_first_value(row, "source_versions")),
        target_candidate=bool(row.get("target_candidate") is True),
        candidate_priority=_optional_nonnegative_int(row.get("candidate_priority")),
    )


def _query_provenance_keys(record: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in ("discovery_species_keys", "discovery_accepted_taxon_keys"):
        value = record.get(field) or ()
        values = [value] if isinstance(value, str) else value
        keys.extend(str(item) for item in values if item)
    return keys


def _query_provenance_group_keys(record: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in ("discovery_family_keys", "discovery_genus_keys"):
        value = record.get(field) or ()
        values = [value] if isinstance(value, str) else value
        keys.extend(str(item) for item in values if item)
    return keys


def _add_metadata_text_candidates(
    candidates: list[CandidateTaxon],
    records: list[dict[str, Any]],
) -> tuple[list[CandidateTaxon], bool]:
    by_name = {_norm(candidate.scientific_name) for candidate in candidates}
    added = False
    for record in records:
        names = record.get("scientific_names_detected") or ()
        if isinstance(names, str):
            names = [names]
        for name in names:
            cleaned = " ".join(str(name or "").split())
            key = _norm(cleaned)
            if not cleaned or key in by_name:
                continue
            candidates.append(
                CandidateTaxon(
                    scientific_name=cleaned,
                    accepted_taxon_key=None,
                    rank="species",
                    genus=cleaned.split(" ", 1)[0],
                )
            )
            by_name.add(key)
            added = True
    return candidates, added


def _add_comment_candidates(
    candidates: list[CandidateTaxon],
    records: list[dict[str, Any]],
) -> tuple[list[CandidateTaxon], bool]:
    by_name = {_norm(candidate.scientific_name) for candidate in candidates}
    added = False
    for record in records:
        for name in _comment_candidate_names(record):
            cleaned = " ".join(str(name or "").split())
            key = _norm(cleaned)
            if not cleaned or key in by_name:
                continue
            candidates.append(
                CandidateTaxon(
                    scientific_name=cleaned,
                    accepted_taxon_key=None,
                    rank="species",
                    genus=cleaned.split(" ", 1)[0],
                )
            )
            by_name.add(key)
            added = True
    return candidates, added


def _comment_candidate_names(record: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for field in ("comment_species_candidate", "species_name_from_comments"):
        value = record.get(field) or ()
        items = [value] if isinstance(value, str) else value
        values.extend(str(item) for item in items if item)
    comments_text = str(record.get("comments_text") or "")
    if comments_text:
        values.extend(SCIENTIFIC_NAME_PATTERN.findall(comments_text))
    return _unique(values)


def _candidate_set_id(*, context: SpeciesContext, species: list[CandidateTaxon], geospatial_scope: str | None) -> str:
    payload = {
        "candidate_contract_version": CANDIDATE_SET_CONTRACT_VERSION,
        "registry_version": context.registry_version,
        "target": context.accepted_taxon_key,
        "species": [
            {
                "accepted_taxon_key": candidate.accepted_taxon_key,
                "scientific_name": candidate.scientific_name,
                "rank": candidate.rank,
                "family": candidate.family,
                "genus": candidate.genus,
                "common_names": candidate.common_names,
                "candidate_reasons": candidate.candidate_reasons,
                "source_versions": candidate.source_versions,
                "target_candidate": candidate.target_candidate,
                "candidate_priority": candidate.candidate_priority,
            }
            for candidate in species
        ],
        "prompt_variant_version": PROMPT_VARIANT_VERSION,
        "geospatial_scope": geospatial_scope,
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _candidate_set_id_for_scope(
    *,
    registry_version: str,
    target_accepted_taxon_key: str,
    target_scientific_name: str,
    species: list[CandidateTaxon],
    geospatial_scope: str | None,
) -> str:
    payload = {
        "candidate_contract_version": CANDIDATE_SET_CONTRACT_VERSION,
        "registry_version": registry_version,
        "target": target_accepted_taxon_key,
        "target_scientific_name": target_scientific_name,
        "species": [
            {
                "accepted_taxon_key": candidate.accepted_taxon_key,
                "scientific_name": candidate.scientific_name,
                "rank": candidate.rank,
                "family": candidate.family,
                "genus": candidate.genus,
                "common_names": candidate.common_names,
                "candidate_reasons": candidate.candidate_reasons,
                "source_versions": candidate.source_versions,
                "target_candidate": candidate.target_candidate,
                "candidate_priority": candidate.candidate_priority,
            }
            for candidate in species
        ],
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


def _first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    return _required_nonnegative_int(value)


def _required_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("candidate_priority must be an integer")
    if value < 0:
        raise ValueError("candidate_priority must be nonnegative")
    return value


def _split_names(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, list | tuple):
        return _unique(value)
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
