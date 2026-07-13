from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import tarfile
from typing import Any

import polars as pl

from biominer.registry.normalize import normalize_name_key
from biominer.storage.parquet import write_parquet


NCBI_TAXA_FILE = "ncbi_taxa.parquet"
NCBI_NAMES_FILE = "ncbi_names.parquet"
NCBI_SNAPSHOTS_FILE = "source_snapshots.parquet"
NCBI_TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"

_TAXA_SCHEMA = {
    "accepted_taxon_key": pl.String,
    "accepted_scientific_name": pl.String,
    "expected_family": pl.String,
    "ncbi_tax_id": pl.String,
    "ncbi_scientific_name": pl.String,
    "ncbi_genus": pl.String,
    "ncbi_family": pl.String,
    "ncbi_order": pl.String,
    "ncbi_class": pl.String,
    "ncbi_phylum": pl.String,
    "ncbi_kingdom": pl.String,
    "match_status": pl.String,
    "rejection_reason": pl.String,
}
_NAMES_SCHEMA = {
    "accepted_taxon_key": pl.String,
    "ncbi_tax_id": pl.String,
    "display_name": pl.String,
    "ncbi_name_class": pl.String,
    "name_class": pl.String,
    "language": pl.String,
    "trust_tier": pl.String,
}
_SNAPSHOT_SCHEMA = {
    "source": pl.String,
    "source_version": pl.String,
    "retrieved_at": pl.String,
    "source_path": pl.String,
    "source_url": pl.String,
}


def harvest_ncbi_taxonomy(
    archive: str | Path,
    registry_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Match accepted butterfly species to NCBI and write filtered Parquet."""

    archive_path = Path(archive)
    registry = Path(registry_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    taxa = pl.read_parquet(registry / "taxa.parquet").filter(pl.col("rank") == "SPECIES")
    target_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in taxa.select(["accepted_taxon_key", "scientific_name", "family"]).iter_rows(named=True):
        target_by_name.setdefault(normalize_name_key(row["scientific_name"]), []).append(dict(row))

    candidate_tax_ids: dict[str, set[str]] = {}
    with tarfile.open(archive_path, "r:gz") as bundle:
        for fields in _dmp_rows(bundle, "names.dmp"):
            if len(fields) < 4:
                continue
            tax_id, name, _unique_name, name_class = fields[:4]
            if name_class not in {"scientific name", "synonym", "equivalent name"}:
                continue
            key = normalize_name_key(name)
            if key in target_by_name:
                candidate_tax_ids.setdefault(tax_id, set()).add(key)

    lineage_by_tax_id: dict[str, dict[str, str]] = {}
    with tarfile.open(archive_path, "r:gz") as bundle:
        for fields in _dmp_rows(bundle, "rankedlineage.dmp"):
            if len(fields) < 10 or fields[0] not in candidate_tax_ids:
                continue
            lineage_by_tax_id[fields[0]] = {
                "scientific_name": fields[1],
                "species": fields[2],
                "genus": fields[3],
                "family": fields[4],
                "order": fields[5],
                "class": fields[6],
                "phylum": fields[7],
                "kingdom": fields[8] or fields[9],
            }

    taxon_rows: list[dict[str, Any]] = []
    accepted_by_tax_id: dict[str, str] = {}
    for tax_id, matched_keys in candidate_tax_ids.items():
        lineage = lineage_by_tax_id.get(tax_id, {})
        for matched_key in matched_keys:
            candidates = target_by_name.get(matched_key, [])
            for target in candidates:
                expected_family = str(target["family"])
                ncbi_family = str(lineage.get("family") or "")
                rejection = ""
                if not lineage:
                    rejection = "ncbi_missing_ranked_lineage"
                elif ncbi_family != expected_family:
                    rejection = "ncbi_family_conflict"
                accepted = not rejection
                if accepted:
                    existing = accepted_by_tax_id.get(tax_id)
                    if existing and existing != str(target["accepted_taxon_key"]):
                        rejection = "ncbi_tax_id_cross_taxon_collision"
                        accepted = False
                    else:
                        accepted_by_tax_id[tax_id] = str(target["accepted_taxon_key"])
                taxon_rows.append(
                    {
                        "accepted_taxon_key": str(target["accepted_taxon_key"]),
                        "accepted_scientific_name": str(target["scientific_name"]),
                        "expected_family": expected_family,
                        "ncbi_tax_id": tax_id,
                        "ncbi_scientific_name": str(lineage.get("scientific_name") or ""),
                        "ncbi_genus": str(lineage.get("genus") or ""),
                        "ncbi_family": ncbi_family,
                        "ncbi_order": str(lineage.get("order") or ""),
                        "ncbi_class": str(lineage.get("class") or ""),
                        "ncbi_phylum": str(lineage.get("phylum") or ""),
                        "ncbi_kingdom": str(lineage.get("kingdom") or ""),
                        "match_status": "accepted" if accepted else "rejected",
                        "rejection_reason": rejection,
                    }
                )

    name_rows: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as bundle:
        for fields in _dmp_rows(bundle, "names.dmp"):
            if len(fields) < 4:
                continue
            tax_id, name, _unique_name, ncbi_name_class = fields[:4]
            accepted_key = accepted_by_tax_id.get(tax_id)
            if not accepted_key or not name:
                continue
            mapped = _mapped_name_class(ncbi_name_class)
            if mapped is None:
                continue
            name_class, language = mapped
            name_rows.append(
                {
                    "accepted_taxon_key": accepted_key,
                    "ncbi_tax_id": tax_id,
                    "display_name": name,
                    "ncbi_name_class": ncbi_name_class,
                    "name_class": name_class,
                    "language": language,
                    "trust_tier": "T2",
                }
            )

    taxa_frame = pl.DataFrame(taxon_rows, schema=_TAXA_SCHEMA).unique(
        ["accepted_taxon_key", "ncbi_tax_id"], keep="first", maintain_order=True
    ).sort(["accepted_scientific_name", "ncbi_tax_id"])
    names_frame = pl.DataFrame(name_rows, schema=_NAMES_SCHEMA).unique(
        ["accepted_taxon_key", "display_name", "ncbi_name_class"],
        keep="first",
        maintain_order=True,
    ).sort(["accepted_taxon_key", "name_class", "display_name"])
    write_parquet(taxa_frame, output / NCBI_TAXA_FILE)
    write_parquet(names_frame, output / NCBI_NAMES_FILE)
    write_parquet(
        pl.DataFrame(
            [
                {
                    "source": "NCBI",
                    "source_version": "new_taxdump",
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "source_path": str(output),
                    "source_url": NCBI_TAXDUMP_URL,
                }
            ],
            schema=_SNAPSHOT_SCHEMA,
        ),
        output / NCBI_SNAPSHOTS_FILE,
    )
    return {
        "source_dir": str(output),
        "matched_taxa": taxa_frame.filter(pl.col("match_status") == "accepted").height,
        "rejected_taxa": taxa_frame.filter(pl.col("match_status") == "rejected").height,
        "name_rows": names_frame.height,
        "rejections": _count_map(taxa_frame.filter(pl.col("match_status") == "rejected"), "rejection_reason"),
    }


def _dmp_rows(bundle: tarfile.TarFile, member: str):
    extracted = bundle.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"NCBI taxonomy archive is missing {member}")
    for raw in extracted:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        fields = line.split("\t|\t")
        fields[-1] = fields[-1].removesuffix("\t|")
        yield [field.strip() for field in fields]


def _mapped_name_class(value: str) -> tuple[str, str] | None:
    if value == "scientific name":
        return "accepted_scientific", "la"
    if value in {
        "synonym",
        "equivalent name",
        "genbank synonym",
        "includes",
        "in-part",
        "misnomer",
        "misspelling",
    }:
        return "scientific_synonym", "la"
    if value in {"common name", "genbank common name"}:
        return "vernacular", "eng"
    return None


def _count_map(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty():
        return {}
    return {
        str(row[column]): int(row["len"])
        for row in frame.group_by(column).len().sort(column).iter_rows(named=True)
    }


__all__ = [
    "NCBI_NAMES_FILE",
    "NCBI_SNAPSHOTS_FILE",
    "NCBI_TAXA_FILE",
    "harvest_ncbi_taxonomy",
]
