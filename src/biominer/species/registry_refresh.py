from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.registry.normalize import normalize_name_key
from biominer.species.context import CommonName, RegionHint, SpeciesContext, SpeciesSearchTerm
from biominer.storage.parquet import write_parquet


def resolve_species_context(
    *,
    registry_dir: str | Path,
    scientific_name: str | None = None,
    accepted_taxon_key: str | None = None,
) -> SpeciesContext:
    registry = Path(registry_dir)
    taxa = pl.read_parquet(registry / "taxa.parquet")
    names = _read_optional_parquet(registry / "names.parquet")
    snapshots = _read_optional_parquet(registry / "source_snapshots.parquet")
    manifest = _read_manifest(registry / "manifest.json")

    species = _find_species_row(taxa, scientific_name=scientific_name, accepted_taxon_key=accepted_taxon_key)
    key = str(species["accepted_taxon_key"])
    registry_version = str(manifest.get("registry_version") or _first_value(names, "registry_version") or "")
    species_names = _species_names(names, key)

    common_names = tuple(_common_name_from_row(row) for row in species_names if _is_common_name(row))
    synonyms = tuple(
        _unique_texts(
            [
                str(row.get("display_name") or row.get("verbatim_name") or "")
                for row in species_names
                if str(row.get("name_class") or "") == "scientific_synonym"
            ]
        )
    )
    search_terms = tuple(_search_term_from_row(row) for row in species_names if bool(row.get("enabled", True)))
    regions = tuple(
        RegionHint(
            region=str(row.get("region") or ""),
            bbox=str(row.get("bbox") or "") or None,
            source=str(row.get("source") or "") or None,
            confidence=str(row.get("confidence") or "") or None,
        )
        for row in species_names
        if row.get("region")
    )
    source_versions = {
        str(row.get("source") or ""): str(row.get("source_version") or "")
        for row in snapshots.to_dicts()
        if row.get("source")
    }

    return SpeciesContext(
        scientific_name=str(species["scientific_name"]),
        accepted_taxon_key=key,
        canonical_name=str(species.get("species") or species.get("scientific_name") or ""),
        family=str(species.get("family") or ""),
        genus=str(species.get("genus") or ""),
        family_key=str(species.get("family_key") or ""),
        genus_key=str(species.get("genus_key") or ""),
        species_key=str(species.get("species_key") or key),
        registry_version=registry_version,
        synonyms=synonyms,
        common_names=common_names,
        search_terms=search_terms,
        regions=regions,
        source_versions=source_versions,
    )


def write_species_registry_outputs(
    *,
    context: SpeciesContext,
    registry_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    registry = Path(registry_dir)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    key = context.accepted_taxon_key
    names = _species_names(_read_optional_parquet(registry / "names.parquet"), key)
    evidence = _species_names(_read_optional_parquet(registry / "name_evidence.parquet"), key)
    context_path = context.write_json(output / "species_context.json")
    names_path = output / "species_names.parquet"
    evidence_path = output / "species_name_evidence.parquet"
    write_parquet(pl.DataFrame(names) if names else pl.DataFrame(), names_path)
    write_parquet(pl.DataFrame(evidence) if evidence else pl.DataFrame(), evidence_path)
    report = {
        "scientific_name": context.scientific_name,
        "accepted_taxon_key": context.accepted_taxon_key,
        "registry_version": context.registry_version,
        "status": "resolved_from_registry",
        "refreshed": False,
        "species_name_rows": len(names),
        "species_name_evidence_rows": len(evidence),
        "written_at": datetime.now(UTC).isoformat(),
        "outputs": {
            "species_context": str(context_path),
            "species_names": str(names_path),
            "species_name_evidence": str(evidence_path),
        },
    }
    report_path = output / "species_registry_refresh_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {**report, "report": str(report_path)}


def _find_species_row(
    taxa: pl.DataFrame,
    *,
    scientific_name: str | None,
    accepted_taxon_key: str | None,
) -> dict[str, Any]:
    if taxa.is_empty():
        raise ValueError("registry taxa.parquet is empty")
    frame = taxa
    if accepted_taxon_key:
        frame = frame.filter(pl.col("accepted_taxon_key") == accepted_taxon_key)
    elif scientific_name:
        target = normalize_name_key(scientific_name)
        frame = frame.filter(pl.col("scientific_name").map_elements(normalize_name_key, return_dtype=pl.String) == target)
    else:
        raise ValueError("scientific_name or accepted_taxon_key is required")
    frame = frame.filter(pl.col("rank").str.to_uppercase() == "SPECIES") if "rank" in frame.columns else frame
    if frame.is_empty():
        label = accepted_taxon_key or scientific_name
        raise ValueError(f"species not found in registry: {label}")
    return frame.sort("accepted_taxon_key").to_dicts()[0]


def _species_names(frame: pl.DataFrame, accepted_taxon_key: str) -> list[dict[str, Any]]:
    if frame.is_empty() or "accepted_taxon_key" not in frame.columns:
        return []
    return frame.filter(pl.col("accepted_taxon_key") == accepted_taxon_key).sort(
        [column for column in ("name_class", "display_name", "source", "source_record_id") if column in frame.columns]
    ).to_dicts()


def _common_name_from_row(row: dict[str, Any]) -> CommonName:
    return CommonName(
        name=str(row.get("display_name") or row.get("verbatim_name") or ""),
        language=str(row.get("language") or "und"),
        script=str(row.get("script") or "") or None,
        region=str(row.get("region") or "") or None,
        bbox=str(row.get("bbox") or "") or None,
        source=str(row.get("source") or "") or None,
        source_record_id=str(row.get("source_record_id") or "") or None,
        trust_tier=str(row.get("trust_tier") or "") or None,
        confidence=str(row.get("confidence") or "") or None,
        review_state="accepted" if bool(row.get("enabled", True)) else "disabled",
    )


def _search_term_from_row(row: dict[str, Any]) -> SpeciesSearchTerm:
    return SpeciesSearchTerm(
        term=str(row.get("display_name") or row.get("verbatim_name") or ""),
        language=str(row.get("language") or "und"),
        term_class=str(row.get("name_class") or "unknown"),
        source=str(row.get("source") or "") or None,
        source_record_id=str(row.get("source_record_id") or "") or None,
        trust_tier=str(row.get("trust_tier") or "") or None,
        precision_tier=str(row.get("precision_tier") or "") or None,
        confidence=str(row.get("confidence") or "") or None,
        region=str(row.get("region") or "") or None,
        bbox=str(row.get("bbox") or "") or None,
        enabled=bool(row.get("enabled", True)),
        review_state="accepted" if bool(row.get("enabled", True)) else "disabled",
    )


def _is_common_name(row: dict[str, Any]) -> bool:
    return str(row.get("name_class") or "") in {"vernacular", "vernacular_alias", "common_name", "regional_common_name"}


def _read_optional_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_value(frame: pl.DataFrame, column: str) -> str | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    values = [value for value in frame[column].to_list() if value not in (None, "")]
    return str(values[0]) if values else None


def _unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(value.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output
