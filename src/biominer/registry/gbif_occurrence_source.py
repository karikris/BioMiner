from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from biominer.registry.scope import ButterflyScope
from biominer.storage.parquet import iter_parquet_batches, write_parquet_batches


GBIF_SIMPLE_PARQUET_SOURCE_VERSION = "gbif-simple-parquet-download"
GBIF_USERNAME_ENV = "GBIF_USERNAME"
GBIF_PASSWORD_ENV = "GBIF_PASSWORD"
GBIF_CHECKLIST_KEY_ENV = "GBIF_CHECKLIST_KEY"
PAPILIONOIDEA_KEY_ENV = "PAPILIONOIDEA_KEY"
GBIF_DOWNLOAD_CHUNK_SIZE = 100_000
DEFAULT_OCCURRENCE_PARQUET_BATCH_SIZE = 50_000


@dataclass(frozen=True)
class DownloadMeta:
    source_url: str | None
    source_version: str
    checksum: str | None


@dataclass(frozen=True)
class ParsedOccurrenceSnapshot:
    taxa_by_key: dict[str, dict[str, Any]]
    names_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]]
    rows_scanned: int
    rows_in_scope: int
    observations: int
    dataset_key: str
    dataset_citation: str


def build_gbif_source_snapshot_from_occurrence_archive(
    archive: str | Path,
    scope: ButterflyScope,
    *,
    retrieved_at: str,
    source_version: str | None = None,
    source_url: str | None = None,
    source_parquet: str | Path | None = None,
    delete_download_after: bool = True,
    archive_batch_size: int = GBIF_DOWNLOAD_CHUNK_SIZE,
    progress_every: int = 50_000,
) -> dict[str, Any]:
    """Build a GBIF source snapshot from a downloaded SIMPLE_PARQUET archive."""

    _validate_runtime_values(archive_batch_size=archive_batch_size, progress_every=progress_every)

    archive_path = Path(archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"GBIF occurrence download archive not found: {archive_path}")

    if not source_parquet:
        source_parquet = archive_path.with_suffix(".parquet")

    download_meta = _build_download_meta(
        source_version=source_version,
        source_url=source_url,
        archive=archive_path,
    )

    parsed = _parse_occurrence_archive(
        archive=archive_path,
        scope=scope,
        progress_every=progress_every,
        archive_batch_size=archive_batch_size,
        output_parquet=Path(source_parquet),
    )

    taxa = list(parsed.taxa_by_key.values())
    names = list(parsed.names_by_key.values())
    if parsed.rows_in_scope == 0:
        raise ValueError("GBIF occurrence download contains no accepted in-scope records")

    if delete_download_after:
        archive_path.unlink(missing_ok=True)

    metrics = {
        "rows_scanned": parsed.rows_scanned,
        "rows_in_scope": parsed.rows_in_scope,
        "observations": parsed.observations,
        "accepted_species_count": len(
            {
                row["accepted_taxon_key"]
                for row in taxa
                if str(row.get("rank") or "").upper() == "SPECIES"
            }
        ),
        "accepted_genus_count": len(
            {
                row["accepted_taxon_key"]
                for row in taxa
                if str(row.get("rank") or "").upper() == "GENUS"
            }
        ),
        "accepted_family_count": len(
            {
                row["accepted_taxon_key"]
                for row in taxa
                if str(row.get("rank") or "").upper() == "FAMILY"
            }
        ),
        "gbif_calls": 0,
        "gbif_request_attempts": 0,
        "gbif_retries": 0,
    }

    return {
        "source": "GBIF",
        "source_version": download_meta.source_version,
        "source_url": download_meta.source_url,
        "source_dataset_key": parsed.dataset_key,
        "source_dataset_citation": parsed.dataset_citation,
        "source_checksum": download_meta.checksum,
        "retrieved_at": retrieved_at,
        "taxa": taxa,
        "names": names,
        "source_assertions": [],
        "metrics": metrics,
    }


def _parse_occurrence_archive(
    archive: Path,
    *,
    scope: ButterflyScope,
    progress_every: int,
    archive_batch_size: int,
    output_parquet: Path,
) -> ParsedOccurrenceSnapshot:
    included_family_keys = {_normalize_gbif_key(key) for key in scope.family_taxon_keys.values() if key}
    family_key_by_name = {
        str(name).casefold(): _normalize_gbif_key(value)
        for name, value in scope.family_taxon_keys.items()
        if value
    }
    included_families = {name.casefold() for name in scope.included_families}

    root_key = _normalize_gbif_key(scope.root_taxon_key)
    taxa_by_key: dict[str, dict[str, Any]] = {}
    names_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    seen = 0
    kept = 0
    dataset_key = ""
    dataset_name = ""
    observations = 0

    if root_key:
        taxa_by_key[root_key] = {
            "accepted_taxon_key": root_key,
            "scientific_name": scope.root_scientific_name,
            "rank": scope.root_rank,
            "parent_key": "",
            "family_key": "",
            "family": "",
            "genus_key": "",
            "genus": "",
            "species_key": "",
            "species": "",
            "taxonomic_status": "ACCEPTED",
            "source_taxon_id": root_key.removeprefix("gbif:"),
        }
        names_by_key[_name_id(root_key, "accepted_scientific", "la", "Latn", "GBIF")] = _accepted_scientific_name(
            accepted_taxon_key=root_key,
            accepted_name=scope.root_scientific_name,
            source_record_id=f"{root_key}:root",
            name_term=(scope.root_rank or "").casefold(),
        )

    def iter_output_frames() -> Iterator[pl.DataFrame]:
        row_buffer: list[dict[str, Any]] = []
        nonlocal seen, kept, observations, dataset_key, dataset_name

        for frame in _iter_occurrence_parquet_frames(archive, batch_size=archive_batch_size):
            seen += frame.height
            for row in frame.iter_rows(named=True):
                observed = _as_mapping(row)
                if _should_skip_record(observed, included_family_keys, included_families):
                    continue

                family_name = _field(observed, "family")
                family_key = _as_gbif_key(
                    _field(observed, "familyKey", "family_key")
                )
                if not family_key and family_name:
                    family_key = family_key_by_name.get(family_name.casefold(), "")
                if not family_key:
                    # If neither a family key nor an accepted key is available,
                    # this row cannot be reliably attached to a supported taxon.
                    continue

                species_key = _as_gbif_key(
                    _field(observed, "acceptedTaxonKey")
                    or _field(observed, "speciesKey")
                    or _field(observed, "species_key")
                    or _field(observed, "key")
                )
                if not species_key:
                    continue

                species_name = _field(
                    observed,
                    "species",
                    "scientificName",
                    "acceptedScientificName",
                    "name",
                )
                if not species_name:
                    continue

                taxonomic_status = _field(observed, "taxonomicStatus", "taxonomic_status")
                if taxonomic_status:
                    taxonomic_status = taxonomic_status.upper()
                if taxonomic_status and taxonomic_status != "ACCEPTED":
                    continue
                if not taxonomic_status:
                    taxonomic_status = "ACCEPTED"

                genus_key = _as_gbif_key(
                    _field(observed, "genusKey", "genus_key")
                )
                genus_name = _field(observed, "genus")

                kept += 1
                dataset_row_key = _field(observed, "datasetKey")
                row_dataset_key = _normalize_gbif_key(dataset_row_key)
                if dataset_row_key:
                    dataset_key = row_dataset_key

                dataset_row_name = _field(observed, "datasetName", "dataset_name", "datasetTitle")
                if dataset_row_name and not dataset_name:
                    dataset_name = dataset_row_name

                if family_key not in taxa_by_key:
                    taxa_by_key[family_key] = {
                        "accepted_taxon_key": family_key,
                        "scientific_name": family_name or _field(observed, "family", "familyName")
                        or "unknown",
                        "rank": "FAMILY",
                        "parent_key": root_key,
                        "family_key": family_key,
                        "family": family_name or "",
                        "genus_key": "",
                        "genus": "",
                        "species_key": "",
                        "species": "",
                        "taxonomic_status": taxonomic_status,
                        "source_taxon_id": family_key.removeprefix("gbif:"),
                    }
                    names_by_key[
                        _name_id(
                            family_key,
                            "accepted_scientific",
                            "la",
                            "Latn",
                            "GBIF",
                        )
                    ] = _accepted_scientific_name(
                        accepted_taxon_key=family_key,
                        accepted_name=(family_name or "unknown"),
                        source_record_id=f"{family_key}:family",
                        name_term="family",
                    )

                if genus_key:
                    if genus_key not in taxa_by_key:
                        taxa_by_key[genus_key] = {
                            "accepted_taxon_key": genus_key,
                            "scientific_name": genus_name or species_name,
                            "rank": "GENUS",
                            "parent_key": family_key,
                            "family_key": family_key,
                            "family": family_name,
                            "genus_key": genus_key,
                            "genus": genus_name or "",
                            "species_key": "",
                            "species": "",
                            "taxonomic_status": taxonomic_status,
                            "source_taxon_id": genus_key.removeprefix("gbif:"),
                        }
                        if genus_name:
                            names_by_key[
                                _name_id(
                                    genus_key,
                                    "accepted_scientific",
                                    "la",
                                    "Latn",
                                    "GBIF",
                                )
                            ] = _accepted_scientific_name(
                                accepted_taxon_key=genus_key,
                                accepted_name=genus_name,
                                source_record_id=f"{genus_key}:genus",
                                name_term="genus",
                            )

                if species_key not in taxa_by_key:
                    taxa_by_key[species_key] = {
                        "accepted_taxon_key": species_key,
                        "scientific_name": species_name,
                        "rank": "SPECIES",
                        "parent_key": genus_key or family_key,
                        "family_key": family_key,
                        "family": family_name,
                        "genus_key": genus_key,
                        "genus": genus_name,
                        "species_key": species_key,
                        "species": species_name,
                        "taxonomic_status": taxonomic_status,
                        "source_taxon_id": species_key.removeprefix("gbif:"),
                    }

                names_by_key[
                    _name_id(species_key, "accepted_scientific", "la", "Latn", "GBIF")
                ] = _accepted_scientific_name(
                    accepted_taxon_key=species_key,
                    accepted_name=species_name,
                    source_record_id=f"{species_key}:species",
                    name_term="species",
                )

                vernacular = _field(observed, "vernacularName", "vernacular_name")
                if vernacular:
                    vernacular_id = vernacular.encode("utf-8")
                    names_by_key[
                        _name_id(
                            species_key,
                            "vernacular",
                            _field(observed, "language", "vernacular_language") or "eng",
                            "",
                            "GBIF",
                        )
                    ] = _vernacular_name(
                        accepted_taxon_key=species_key,
                        display_name=vernacular,
                        language=_field(observed, "language", "vernacular_language") or "eng",
                        region=_field(observed, "countryCode", "country_code"),
                        source_record_id=f"{species_key}:vernacular:{hashlib.sha1(vernacular_id).hexdigest()}",
                    )

                row_buffer.append(
                    {
                        "accepted_taxon_key": species_key,
                        "accepted_scientific_name": species_name,
                        "taxon_rank": "SPECIES",
                        "family_key": family_key,
                        "family": family_name,
                        "genus_key": genus_key,
                        "genus": genus_name,
                        "species_key": species_key,
                        "dataset_key": row_dataset_key,
                        "dataset_name": dataset_name,
                        "country_code": _field(observed, "countryCode", "country_code"),
                        "vernacular_name": vernacular,
                        "language": _field(observed, "language", "vernacular_language"),
                        "record_count": 1,
                    }
                )
                observations += 1
                if len(row_buffer) >= DEFAULT_OCCURRENCE_PARQUET_BATCH_SIZE:
                    yield pl.DataFrame(row_buffer)
                    row_buffer.clear()
        if row_buffer:
            yield pl.DataFrame(row_buffer)

    schema = {
        "accepted_taxon_key": pl.String,
        "accepted_scientific_name": pl.String,
        "taxon_rank": pl.String,
        "family_key": pl.String,
        "family": pl.String,
        "genus_key": pl.String,
        "genus": pl.String,
        "species_key": pl.String,
        "dataset_key": pl.String,
        "dataset_name": pl.String,
        "country_code": pl.String,
        "vernacular_name": pl.String,
        "language": pl.String,
        "record_count": pl.UInt64,
    }
    write_parquet_batches(
        iter_output_frames(),
        output_parquet,
        schema=schema,
        overwrite=True,
    )

    return ParsedOccurrenceSnapshot(
        taxa_by_key=taxa_by_key,
        names_by_key=names_by_key,
        rows_scanned=seen,
        rows_in_scope=kept,
        observations=observations,
        dataset_key=dataset_key,
        dataset_citation=dataset_name,
    )


def _iter_occurrence_parquet_frames(
    archive: Path,
    *,
    batch_size: int,
) -> Iterator[pl.DataFrame]:
    with zipfile.ZipFile(archive) as bundle:
        members = _select_occurrence_members(bundle)
        if not members:
            raise ValueError("GBIF occurrence download archive contains no parquet files")
        for member in members:
            with bundle.open(member) as archive_file:
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
                    shutil.copyfileobj(archive_file, temp_file, length=16 * 1024 * 1024)
                    temp_path = Path(temp_file.name)
            try:
                for frame in iter_parquet_batches(temp_path, batch_size=batch_size):
                    yield frame
            finally:
                temp_path.unlink(missing_ok=True)


def _select_occurrence_members(bundle: zipfile.ZipFile) -> list[str]:
    members = []
    for member in bundle.namelist():
        name = member.lower()
        if name.endswith("/"):
            continue
        if name.endswith(".parquet") and "meta" not in name and "README" not in name:
            members.append(member)
    members.sort()
    return members


def _should_skip_record(
    row: Mapping[str, object],
    included_family_keys: set[str],
    included_families: set[str],
) -> bool:
    if not included_family_keys and not included_families:
        return False

    family_key = _normalize_gbif_key(_field(row, "familyKey", "family_key"))
    family_name = _field(row, "family").casefold()

    return not (family_key in included_family_keys or family_name in included_families)


def _name_id(accepted_taxon_key: str, name_class: str, language: str, script: str, source: str) -> tuple[str, str, str, str, str]:
    return (accepted_taxon_key, name_class, language, script, source)


def _accepted_scientific_name(*, accepted_taxon_key: str, accepted_name: str, source_record_id: str, name_term: str) -> dict[str, object]:
    return {
        "accepted_taxon_key": accepted_taxon_key,
        "verbatim_name": accepted_name,
        "display_name": accepted_name,
        "language": "la",
        "script": "Latn",
        "region": "",
        "bbox": "",
        "name_class": "accepted_scientific",
        "source": "GBIF",
        "source_record_id": source_record_id,
        "source_taxon_id": accepted_taxon_key.removeprefix("gbif:"),
        "trust_tier": "T1",
        "precision_tier": "high",
        "confidence": "high",
        "name_term": name_term,
        "enabled": True,
    }


def _vernacular_name(
    *,
    accepted_taxon_key: str,
    display_name: str,
    language: str,
    region: str,
    source_record_id: str,
) -> dict[str, object]:
    return {
        "accepted_taxon_key": accepted_taxon_key,
        "verbatim_name": display_name,
        "display_name": display_name,
        "language": language or "eng",
        "script": "",
        "region": region,
        "bbox": "",
        "name_class": "vernacular",
        "source": "GBIF",
        "source_record_id": source_record_id,
        "source_taxon_id": accepted_taxon_key.removeprefix("gbif:"),
        "trust_tier": "T2",
        "precision_tier": "medium",
        "confidence": "medium",
        "name_term": "vernacular",
        "enabled": True,
    }


def _build_download_meta(
    *,
    source_version: str | None,
    source_url: str | None,
    archive: Path,
) -> DownloadMeta:
    url = source_url
    if not url:
        url = f"file://{archive}"
    version = source_version or GBIF_SIMPLE_PARQUET_SOURCE_VERSION

    checksum = None
    if archive.exists():
        checksum = str(archive.stat().st_size)

    return DownloadMeta(
        source_url=url,
        source_version=version,
        checksum=checksum,
    )


def _as_mapping(row: Mapping[str, object] | dict[str, object]) -> dict[str, object]:
    return dict(row)


def _field(row: Mapping[str, object], *names: str) -> str:
    normalized = {str(key).casefold().replace("_", ""): value for key, value in row.items()}
    for name in names:
        key = name.casefold().replace("_", "")
        value = normalized.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_gbif_key(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return f"gbif:{text.removeprefix('gbif:')}"


def _as_gbif_key(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return f"gbif:{text.removeprefix('gbif:')}"


def _validate_runtime_values(*, archive_batch_size: int, progress_every: int) -> None:
    if archive_batch_size < 1:
        raise ValueError("archive_batch_size must be positive")
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")
