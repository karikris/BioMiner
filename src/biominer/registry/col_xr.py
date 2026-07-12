from __future__ import annotations

import csv
from datetime import UTC, datetime
import io
from pathlib import Path
import zipfile
from typing import Any

from biominer.registry.scope import load_scope
from biominer.registry.unified import COL_XR_DATASET_KEY, COL_XR_DOI, COL_XR_RELEASE


def extract_col_xr_snapshot(
    archive: str | Path,
    *,
    scope_path: str | Path = "config/butterfly_scope.json",
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Extract the configured butterfly scope from a CoL Darwin Core archive.

    CSV/TSV parsing is confined to this external Darwin Core boundary. The
    compiled registry remains Parquet-only.
    """

    scope = load_scope(scope_path)
    with zipfile.ZipFile(archive) as bundle:
        taxon_rows = _read_table(bundle, ("taxon.txt", "taxon.tsv", "taxon.csv"))
        vernacular_rows = _read_table(
            bundle,
            ("vernacularname.txt", "vernacularname.tsv", "vernacularname.csv", "vernacular.txt"),
            required=False,
        )
    included_families = set(scope.included_families)
    accepted_by_id = {
        _field(row, "taxonID", "ID"): row
        for row in taxon_rows
        if _accepted_status(row) and _field(row, "taxonID", "ID")
    }
    in_scope_ids = {
        taxon_id
        for taxon_id, row in accepted_by_id.items()
        if _field(row, "family") in included_families
        or _field(row, "scientificName") == scope.root_scientific_name
        or _field(row, "scientificName") in included_families
    }
    changed = True
    while changed:
        changed = False
        for taxon_id in tuple(in_scope_ids):
            parent_id = _field(accepted_by_id.get(taxon_id, {}), "parentNameUsageID")
            if parent_id and parent_id in accepted_by_id and parent_id not in in_scope_ids:
                in_scope_ids.add(parent_id)
                changed = True

    taxa: list[dict[str, Any]] = []
    names: list[dict[str, Any]] = []
    for taxon_id in sorted(in_scope_ids):
        source = accepted_by_id[taxon_id]
        rank = _field(source, "taxonRank").upper()
        scientific_name = _field(source, "scientificName")
        key = f"col:{taxon_id}"
        parent_id = _field(source, "parentNameUsageID")
        family = _field(source, "family") or (scientific_name if rank == "FAMILY" else "")
        genus = _field(source, "genus") or (scientific_name if rank == "GENUS" else "")
        taxa.append(
            {
                "accepted_taxon_key": key,
                "scientific_name": scientific_name,
                "rank": rank,
                "parent_key": f"col:{parent_id}" if parent_id else "",
                "family_key": _rank_key(accepted_by_id, taxon_id, "FAMILY"),
                "family": family,
                "genus_key": _rank_key(accepted_by_id, taxon_id, "GENUS"),
                "genus": genus,
                "species_key": key if rank == "SPECIES" else "",
                "species": scientific_name if rank == "SPECIES" else "",
                "status": "ACCEPTED",
                "source_taxon_id": taxon_id,
                "scientific_name_authorship": _field(source, "scientificNameAuthorship"),
                "source_dataset_key": COL_XR_DATASET_KEY,
                "source_release": COL_XR_RELEASE,
            }
        )
        names.append(_name_row(key, scientific_name, source, "accepted_scientific", taxon_id))

    for synonym in taxon_rows:
        if _accepted_status(synonym):
            continue
        accepted_id = _field(synonym, "acceptedNameUsageID")
        if accepted_id not in in_scope_ids:
            continue
        scientific_name = _field(synonym, "scientificName")
        if scientific_name:
            names.append(
                _name_row(
                    f"col:{accepted_id}",
                    scientific_name,
                    synonym,
                    "scientific_synonym",
                    _field(synonym, "taxonID", "ID"),
                )
            )
    for vernacular in vernacular_rows:
        taxon_id = _field(vernacular, "taxonID")
        if taxon_id not in in_scope_ids:
            continue
        term = _field(vernacular, "vernacularName", "name")
        if not term:
            continue
        names.append(
            {
                "accepted_taxon_key": f"col:{taxon_id}",
                "display_name": term,
                "verbatim_name": term,
                "language": _field(vernacular, "language"),
                "countryCode": _field(vernacular, "countryCode"),
                "name_class": "vernacular",
                "source": "CoL XR",
                "source_record_id": _field(vernacular, "ID", "id") or f"{taxon_id}:{term}",
                "source_taxon_id": taxon_id,
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "review_state": "source_accepted",
                "enabled": True,
            }
        )
    return {
        "source": "CoL XR",
        "source_version": COL_XR_RELEASE,
        "source_dataset_key": COL_XR_DATASET_KEY,
        "doi": COL_XR_DOI,
        "retrieved_at": retrieved_at or datetime.now(UTC).isoformat(),
        "source_url": f"https://www.checklistbank.org/dataset/{COL_XR_DATASET_KEY}",
        "citation": f"Catalogue of Life {COL_XR_RELEASE}; DOI {COL_XR_DOI}",
        "taxa": taxa,
        "names": names,
    }


def _read_table(
    bundle: zipfile.ZipFile,
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> list[dict[str, str]]:
    candidates = tuple(name.casefold() for name in candidates)
    member = next(
        (name for name in bundle.namelist() if Path(name).name.casefold() in candidates),
        None,
    )
    if member is None:
        if required:
            raise FileNotFoundError("CoL XR Darwin Core archive has no Taxon table")
        return []
    raw = bundle.read(member).decode("utf-8-sig")
    delimiter = "\t" if "\t" in raw.partition("\n")[0] else ","
    return [dict(row) for row in csv.DictReader(io.StringIO(raw), delimiter=delimiter)]


def _field(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value).strip()
        suffix = "/" + name
        for key, candidate in row.items():
            if str(key).endswith(suffix) and candidate not in (None, ""):
                return str(candidate).strip()
    return ""


def _accepted_status(row: dict[str, Any]) -> bool:
    return _field(row, "taxonomicStatus").casefold() in {"accepted", "provisionally accepted", ""} and not _field(
        row, "acceptedNameUsageID"
    )


def _rank_key(rows: dict[str, dict[str, Any]], taxon_id: str, rank: str) -> str:
    current_id = taxon_id
    visited: set[str] = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        row = rows.get(current_id, {})
        if _field(row, "taxonRank").upper() == rank:
            return f"col:{current_id}"
        current_id = _field(row, "parentNameUsageID")
    return ""


def _name_row(
    accepted_key: str,
    display_name: str,
    source: dict[str, Any],
    name_class: str,
    record_id: str,
) -> dict[str, Any]:
    return {
        "accepted_taxon_key": accepted_key,
        "display_name": display_name,
        "verbatim_name": display_name,
        "language": "la",
        "name_class": name_class,
        "source": "CoL XR",
        "source_record_id": record_id,
        "source_taxon_id": _field(source, "taxonID", "ID"),
        "trust_tier": "T1",
        "precision_tier": "high",
        "confidence": "high",
        "review_state": "source_accepted",
        "enabled": True,
    }
