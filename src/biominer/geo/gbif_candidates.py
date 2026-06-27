from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Protocol

import httpx
import polars as pl

from biominer.geo.builder import build_geo_candidate_tables
from biominer.registry.gbif import GBIFClient
from biominer.registry.gbif_production import ProductionGBIFClient
from biominer.storage.parquet import write_parquet


GBIF_OCCURRENCE_FIELDS: tuple[str, ...] = (
    "key",
    "datasetKey",
    "basisOfRecord",
    "occurrenceID",
    "taxonKey",
    "speciesKey",
    "scientificName",
    "family",
    "genus",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "countryCode",
    "year",
    "eventDate",
    "issues",
)

DEFAULT_GBIF_OCCURRENCE_PAGE_SIZE = 300


class GBIFOccurrenceClient(Protocol):
    def occurrence_search(self, params: dict[str, object]) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class GeoSpeciesWorkItem:
    species_key: str
    scientific_name: str
    family: str | None
    genus: str | None


@dataclass(frozen=True)
class SpeciesIngestResult:
    species_key: str
    status: str
    occurrence_rows: int
    pages_written: int
    error: str | None = None


class GBIFGeoState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gbif_geo_occurrence_progress (
                species_key TEXT PRIMARY KEY,
                query_fingerprint TEXT NOT NULL,
                offset INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def begin_species(self, species_key: str, fingerprint: str) -> dict[str, Any]:
        now = _now()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT species_key, query_fingerprint, offset, status, attempts, last_error, updated_at
                FROM gbif_geo_occurrence_progress
                WHERE species_key = ?
                """,
                (species_key,),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO gbif_geo_occurrence_progress (
                        species_key, query_fingerprint, offset, status, attempts, last_error, updated_at
                    )
                    VALUES (?, ?, 0, 'pending', 0, NULL, ?)
                    """,
                    (species_key, fingerprint, now),
                )
                self._conn.commit()
                return {
                    "species_key": species_key,
                    "query_fingerprint": fingerprint,
                    "offset": 0,
                    "status": "pending",
                    "attempts": 0,
                    "last_error": None,
                    "updated_at": now,
                }
            payload = dict(row)
            if payload["query_fingerprint"] != fingerprint:
                raise ValueError(
                    f"GBIF geo resume fingerprint mismatch for species {species_key}: "
                    f"{payload['query_fingerprint']} != {fingerprint}"
                )
            return payload

    def mark_page_success(self, *, species_key: str, offset: int, rows_returned: int, complete: bool) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE gbif_geo_occurrence_progress
                SET offset = ?, status = ?, last_error = NULL, updated_at = ?
                WHERE species_key = ?
                """,
                (offset + rows_returned, "completed" if complete else "pending", _now(), species_key),
            )
            self._conn.commit()

    def mark_error(self, *, species_key: str, error: str, exhausted: bool) -> int:
        with self._lock:
            current = self._conn.execute(
                "SELECT attempts FROM gbif_geo_occurrence_progress WHERE species_key = ?",
                (species_key,),
            ).fetchone()
            attempts = int(current["attempts"]) + 1 if current is not None else 1
            self._conn.execute(
                """
                UPDATE gbif_geo_occurrence_progress
                SET attempts = ?, status = ?, last_error = ?, updated_at = ?
                WHERE species_key = ?
                """,
                (attempts, "failed" if exhausted else "retryable_error", error, _now(), species_key),
            )
            self._conn.commit()
            return attempts

    def summary(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM gbif_geo_occurrence_progress
                GROUP BY status
                ORDER BY status
                """
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def build_gbif_geo_candidates(
    *,
    taxa_path: str | Path,
    output_dir: str | Path,
    geo_version: str,
    state_db: str | Path,
    limit_species: int = 0,
    max_retries: int = 8,
    workers: int = 4,
    page_size: int = DEFAULT_GBIF_OCCURRENCE_PAGE_SIZE,
    client_factory: Callable[[], GBIFOccurrenceClient] | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parts_dir = output / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    work_items = load_geo_species_work_items(taxa_path, limit_species=limit_species)
    state = GBIFGeoState(state_db)
    factory = client_factory or (lambda: ProductionGBIFClient(max_retries=max_retries, max_connections=max(1, workers)))
    results: list[SpeciesIngestResult] = []
    try:
        if workers <= 1:
            for item in work_items:
                results.append(
                    _ingest_species_occurrences(
                        item,
                        state=state,
                        output_dir=parts_dir,
                        client_factory=factory,
                        max_retries=max_retries,
                        page_size=page_size,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gbif-geo") as executor:
                futures = [
                    executor.submit(
                        _ingest_species_occurrences,
                        item,
                        state=state,
                        output_dir=parts_dir,
                        client_factory=factory,
                        max_retries=max_retries,
                        page_size=page_size,
                    )
                    for item in work_items
                ]
                for future in as_completed(futures):
                    results.append(future.result())
        occurrences = _consolidated_occurrence_parts(parts_dir)
        outputs = build_geo_candidate_tables(
            occurrences,
            output_dir=output,
            geo_version=geo_version,
        )
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        manifest.update(
            {
                "source": "GBIF occurrence/search",
                "taxa": str(taxa_path),
                "state_db": str(state_db),
                "species_seen": len(work_items),
                "species_completed": sum(1 for row in results if row.status == "completed"),
                "species_failed": sum(1 for row in results if row.status == "failed"),
                "part_files": len(sorted(parts_dir.glob("*.parquet"))),
                "page_size": page_size,
                "workers": workers,
                "max_retries": max_retries,
                "state_counts": state.summary(),
            }
        )
        outputs["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "geo_version": geo_version,
            "output_dir": str(output),
            "state_db": str(state_db),
            "species_seen": len(work_items),
            "species_completed": manifest["species_completed"],
            "species_failed": manifest["species_failed"],
            "occurrence_rows": occurrences.height,
            "outputs": {name: str(path) for name, path in outputs.items()},
            "state_counts": manifest["state_counts"],
        }
    finally:
        state.close()


def load_geo_species_work_items(taxa_path: str | Path, *, limit_species: int = 0) -> list[GeoSpeciesWorkItem]:
    frame = pl.read_parquet(taxa_path)
    rows: list[GeoSpeciesWorkItem] = []
    for row in frame.to_dicts():
        if str(row.get("rank") or "").casefold() != "species":
            continue
        species_key = _bare_gbif_key(_first_text(row, "species_key", "speciesKey", "accepted_taxon_key", "taxonKey"))
        scientific_name = _first_text(row, "scientific_name", "scientificName", "accepted_scientific_name", "species")
        if not species_key or not scientific_name:
            continue
        rows.append(
            GeoSpeciesWorkItem(
                species_key=species_key,
                scientific_name=scientific_name,
                family=_first_text(row, "family"),
                genus=_first_text(row, "genus") or scientific_name.split(" ", 1)[0],
            )
        )
    deduped = {row.species_key: row for row in rows}
    ordered = sorted(deduped.values(), key=lambda row: (_species_key_sort(row.species_key), row.scientific_name.casefold()))
    return ordered[:limit_species] if limit_species and limit_species > 0 else ordered


def _ingest_species_occurrences(
    item: GeoSpeciesWorkItem,
    *,
    state: GBIFGeoState,
    output_dir: Path,
    client_factory: Callable[[], GBIFOccurrenceClient],
    max_retries: int,
    page_size: int,
) -> SpeciesIngestResult:
    base_params: dict[str, object] = {
        "speciesKey": item.species_key,
        "hasCoordinate": "true",
        "limit": page_size,
    }
    fingerprint = _query_fingerprint(base_params)
    progress = state.begin_species(item.species_key, fingerprint)
    if progress["status"] == "completed":
        return SpeciesIngestResult(species_key=item.species_key, status="completed", occurrence_rows=0, pages_written=0)

    occurrence_rows = 0
    pages_written = 0
    client = client_factory()
    try:
        while True:
            progress = state.begin_species(item.species_key, fingerprint)
            offset = int(progress["offset"])
            if progress["status"] == "completed":
                return SpeciesIngestResult(
                    species_key=item.species_key,
                    status="completed",
                    occurrence_rows=occurrence_rows,
                    pages_written=pages_written,
                )
            try:
                payload = client.occurrence_search({**base_params, "offset": offset})
            except Exception as exc:  # noqa: BLE001 - retryable API state belongs in SQLite.
                attempts = state.mark_error(
                    species_key=item.species_key,
                    error=str(exc),
                    exhausted=int(progress["attempts"]) + 1 >= max_retries,
                )
                if attempts >= max_retries or not _is_retryable_error(exc):
                    return SpeciesIngestResult(
                        species_key=item.species_key,
                        status="failed",
                        occurrence_rows=occurrence_rows,
                        pages_written=pages_written,
                        error=str(exc),
                    )
                continue
            raw_rows = _results(payload)
            normalized = _normalize_occurrence_rows(raw_rows, item)
            if normalized:
                part = output_dir / f"species_{item.species_key}_offset_{offset:09d}.parquet"
                write_parquet(_occurrence_part_frame(normalized), part)
                occurrence_rows += len(normalized)
                pages_written += 1
            complete = _is_final_page(payload, rows_returned=len(raw_rows), offset=offset, limit=page_size)
            state.mark_page_success(
                species_key=item.species_key,
                offset=offset,
                rows_returned=len(raw_rows),
                complete=complete,
            )
            if complete:
                return SpeciesIngestResult(
                    species_key=item.species_key,
                    status="completed",
                    occurrence_rows=occurrence_rows,
                    pages_written=pages_written,
                )
    finally:
        client.close()


def _normalize_occurrence_rows(rows: Sequence[dict[str, Any]], item: GeoSpeciesWorkItem) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        latitude = _float_value(row.get("decimalLatitude"))
        longitude = _float_value(row.get("decimalLongitude"))
        if latitude is None or longitude is None:
            continue
        occurrence_key = _first_text(row, "key") or _stable_occurrence_key(row, item)
        normalized.append(
            {
                "key": occurrence_key,
                "datasetKey": _first_text(row, "datasetKey"),
                "basisOfRecord": _first_text(row, "basisOfRecord"),
                "occurrenceID": _first_text(row, "occurrenceID"),
                "taxonKey": _first_text(row, "taxonKey") or item.species_key,
                "speciesKey": _first_text(row, "speciesKey") or item.species_key,
                "scientificName": _first_text(row, "scientificName") or item.scientific_name,
                "family": _first_text(row, "family") or item.family,
                "genus": _first_text(row, "genus") or item.genus,
                "decimalLatitude": latitude,
                "decimalLongitude": longitude,
                "coordinateUncertaintyInMeters": _float_value(row.get("coordinateUncertaintyInMeters")),
                "countryCode": _first_text(row, "countryCode"),
                "year": _int_value(row.get("year")),
                "eventDate": _first_text(row, "eventDate"),
                "issues": _issues_text(row.get("issues")),
            }
        )
    return normalized


def _occurrence_part_frame(rows: Sequence[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_occurrence_schema(), orient="row")


def _consolidated_occurrence_parts(parts_dir: Path) -> pl.DataFrame:
    paths = sorted(parts_dir.glob("*.parquet"))
    if not paths:
        return pl.DataFrame(schema=_occurrence_schema())
    frame = pl.read_parquet(paths)
    if frame.height and "key" in frame.columns:
        frame = frame.unique(subset=["key"], keep="first", maintain_order=True)
    sort_columns = [column for column in ("speciesKey", "key") if column in frame.columns]
    return frame.sort(sort_columns) if sort_columns and frame.height else frame


def _occurrence_schema() -> dict[str, pl.DataType]:
    return {
        "key": pl.Utf8,
        "datasetKey": pl.Utf8,
        "basisOfRecord": pl.Utf8,
        "occurrenceID": pl.Utf8,
        "taxonKey": pl.Utf8,
        "speciesKey": pl.Utf8,
        "scientificName": pl.Utf8,
        "family": pl.Utf8,
        "genus": pl.Utf8,
        "decimalLatitude": pl.Float64,
        "decimalLongitude": pl.Float64,
        "coordinateUncertaintyInMeters": pl.Float64,
        "countryCode": pl.Utf8,
        "year": pl.Int64,
        "eventDate": pl.Utf8,
        "issues": pl.Utf8,
    }


def _query_fingerprint(params: dict[str, object]) -> str:
    payload = json.dumps({"endpoint": "/occurrence/search", **params}, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _is_final_page(payload: dict[str, Any], *, rows_returned: int, offset: int, limit: int) -> bool:
    if payload.get("endOfRecords") is True:
        return True
    if rows_returned == 0:
        return True
    count = payload.get("count")
    if isinstance(count, int) and offset + rows_returned >= count:
        return True
    return rows_returned < limit


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return True


def _bare_gbif_key(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).removeprefix("gbif:")


def _first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def _float_value(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int_value(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _issues_text(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def _stable_occurrence_key(row: dict[str, Any], item: GeoSpeciesWorkItem) -> str:
    payload = json.dumps(
        {
            "speciesKey": item.species_key,
            "occurrenceID": row.get("occurrenceID"),
            "datasetKey": row.get("datasetKey"),
            "decimalLatitude": row.get("decimalLatitude"),
            "decimalLongitude": row.get("decimalLongitude"),
            "eventDate": row.get("eventDate"),
        },
        sort_keys=True,
        default=str,
    )
    return f"derived:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _species_key_sort(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _now() -> str:
    return datetime.now(UTC).isoformat()
