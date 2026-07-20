"""Australia-only, metadata-only Flickr discovery planning.

The module deliberately keeps GBIF occurrence support, query associations, and
Flickr request identity separate.  It does not download images or write S3.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import httpx
import polars as pl

from biominer.flickr_fetch.query_planner import FlickrQuery
from biominer.registry.normalize import normalize_name_key
from biominer.registry.unified import stable_identity


GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
AUSTRALIA_PLACE_QUERY = "Australia"
AUSTRALIA_WOE_ID = "23424748"
AUSTRALIA_FLICKR_BBOX = "112.0,-44.5,154.0,-10.0"
COMMON_NAME_CLASSES = {"vernacular", "vernacular_alias", "common_name", "common_name_alias"}
INDIGENOUS_MARKERS = ("indigenous", "aboriginal", "first nations", "first_nations")
GBIF_REQUEST_INTERVAL_SECONDS = 0.5
GBIF_MAX_ATTEMPTS = 8
GBIF_MAX_RETRY_DELAY_SECONDS = 3600


class GBIFRequestFailure(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds


def build_australia_presence(
    *,
    registry_dir: str | Path,
    state_db: str | Path,
    output_path: str | Path,
    workers: int = 1,
    request_interval_seconds: float = GBIF_REQUEST_INTERVAL_SECONDS,
    max_attempts: int = GBIF_MAX_ATTEMPTS,
    sleep: Any = time.sleep,
) -> pl.DataFrame:
    """Resume a durable, globally paced GBIF `country=AU` occurrence scan."""

    if workers != 1:
        raise ValueError("GBIF Australia presence discovery uses exactly one globally paced dispatcher")
    if request_interval_seconds < 0:
        raise ValueError("request_interval_seconds must not be negative")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    taxa = pl.read_parquet(Path(registry_dir) / "taxa.parquet")
    species = taxa.filter((pl.col("rank") == "SPECIES") & (pl.col("taxonomic_status") == "ACCEPTED")).select(
        "accepted_taxon_key", "scientific_name"
    ).unique("accepted_taxon_key").sort("accepted_taxon_key")
    state = Path(state_db)
    state.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(state) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS gbif_australia_presence (
            accepted_taxon_key TEXT PRIMARY KEY, scientific_name TEXT NOT NULL,
            occurrence_count INTEGER, status TEXT NOT NULL, error TEXT, retrieved_at TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,
            last_http_status INTEGER, last_error_class TEXT)"""
        )
        _migrate_presence_schema(conn)
        conn.executemany(
            "INSERT OR IGNORE INTO gbif_australia_presence(accepted_taxon_key, scientific_name, status) VALUES (?, ?, 'pending')",
            [(row["accepted_taxon_key"], row["scientific_name"]) for row in species.to_dicts()],
        )
        # A terminated worker leaves rows claimed. Its completed counts are
        # durable; every other result is safe to retry from this single dispatcher.
        conn.execute("UPDATE gbif_australia_presence SET status = 'pending' WHERE status = 'claimed'")
        conn.execute(
            """UPDATE gbif_australia_presence
               SET status = 'retry_wait', next_attempt_at = ?, last_error_class = COALESCE(last_error_class, 'recovered_retryable')
               WHERE status = 'failed' AND last_error_class IS NULL""",
            (datetime.now(UTC).isoformat(),),
        )
    with sqlite3.connect(state) as conn:
        while True:
            now = datetime.now(UTC)
            row = conn.execute(
                """SELECT accepted_taxon_key, scientific_name, attempt_count
                   FROM gbif_australia_presence
                   WHERE status = 'pending'
                      OR (status = 'retry_wait' AND next_attempt_at <= ?)
                   ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, next_attempt_at, accepted_taxon_key
                   LIMIT 1""",
                (now.isoformat(),),
            ).fetchone()
            if row is None:
                retry_at = conn.execute(
                    "SELECT MIN(next_attempt_at) FROM gbif_australia_presence WHERE status = 'retry_wait'"
                ).fetchone()[0]
                if not retry_at:
                    break
                delay = max(0.0, (datetime.fromisoformat(str(retry_at)) - now).total_seconds())
                if delay:
                    sleep(delay)
                continue
            key, scientific_name, attempts = str(row[0]), str(row[1]), int(row[2])
            conn.execute("UPDATE gbif_australia_presence SET status = 'claimed' WHERE accepted_taxon_key = ?", (key,))
            conn.commit()
            started = time.monotonic()
            try:
                count = _gbif_australia_count(scientific_name)
                conn.execute(
                    """UPDATE gbif_australia_presence
                       SET occurrence_count = ?, status = 'complete', error = NULL, retrieved_at = ?,
                           next_attempt_at = NULL, last_http_status = 200, last_error_class = NULL
                       WHERE accepted_taxon_key = ?""",
                    (count, datetime.now(UTC).isoformat(), key),
                )
            except GBIFRequestFailure as exc:
                _record_gbif_failure(conn, key=key, attempts=attempts + 1, error=exc, max_attempts=max_attempts)
            except Exception as exc:  # network/client failures are retryable operational failures.
                _record_gbif_failure(
                    conn,
                    key=key,
                    attempts=attempts + 1,
                    error=GBIFRequestFailure(str(exc), http_status=None),
                    max_attempts=max_attempts,
                )
            conn.commit()
            delay = request_interval_seconds - (time.monotonic() - started)
            if delay > 0:
                sleep(delay)
    with sqlite3.connect(state) as conn:
        rows = conn.execute(
            """SELECT accepted_taxon_key, scientific_name, COALESCE(occurrence_count, 0), status, error,
                      retrieved_at, attempt_count, next_attempt_at, last_http_status, last_error_class
               FROM gbif_australia_presence ORDER BY accepted_taxon_key"""
        ).fetchall()
    frame = pl.DataFrame(rows, schema=["accepted_taxon_key", "scientific_name", "gbif_au_occurrence_count", "status", "error", "retrieved_at", "attempt_count", "next_attempt_at", "last_http_status", "last_error_class"], orient="row")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)
    return frame


def compile_australia_query_plan(
    *, registry_dir: str | Path, presence: pl.DataFrame, output_dir: str | Path, place_id: str | None = None,
    woe_id: str | None = None, bbox: str | None = None, cutoff: str
) -> tuple[pl.DataFrame, pl.DataFrame, tuple[FlickrQuery, ...]]:
    """Compile globally de-duplicated terms in the required Australia-first order."""

    if not place_id and not woe_id and not bbox:
        raise ValueError("Australia Flickr plan requires a Flickr place_id, woe_id, or bbox")
    scope_identity = f"bbox:{bbox}" if bbox else (f"woe:{woe_id}" if woe_id else f"place:{place_id}")
    effective_place_id = None if bbox else place_id
    effective_woe_id = None if bbox else woe_id
    registry = Path(registry_dir)
    taxa = pl.read_parquet(registry / "taxa.parquet")
    names = pl.read_parquet(registry / "names.parquet")
    presence_map = {r["accepted_taxon_key"]: (int(r["gbif_au_occurrence_count"]), r["status"]) for r in presence.to_dicts()}
    taxon_rank = {r["accepted_taxon_key"]: r["rank"] for r in taxa.select("accepted_taxon_key", "rank").to_dicts()}
    species_keys = {key for key, rank in taxon_rank.items() if rank == "SPECIES"}
    local = {key for key in species_keys if presence_map.get(key, (0, "failed"))[0] > 0 and presence_map.get(key, (0, "failed"))[1] == "complete"}
    non_local = {key for key in species_keys if presence_map.get(key, (0, "failed"))[1] == "complete" and key not in local}
    eligible = names.filter(pl.col("enabled") & pl.col("query_eligible")).to_dicts()
    selected: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    for row in eligible:
        key = str(row.get("accepted_taxon_key") or "")
        rank = taxon_rank.get(key, "")
        name_class = str(row.get("name_class") or "")
        review = str(row.get("review_state") or "").casefold()
        region = str(row.get("region") or "").casefold()
        indigenous = any(marker in name_class.casefold() or marker in region for marker in INDIGENOUS_MARKERS)
        reviewed_indigenous = indigenous and ("review" in review or "curat" in review or "approv" in review)
        common = name_class in COMMON_NAME_CLASSES and not indigenous
        stage: int | None = None
        if key in local and common:
            stage = 0
        elif key in local and reviewed_indigenous:
            stage = 1
        elif key in local and name_class == "accepted_scientific":
            stage = 2
        elif key in non_local and common:
            stage = 3
        elif key in non_local and name_class == "accepted_scientific":
            stage = 4
        elif rank == "GENUS" and common:
            stage = 5
        elif rank == "FAMILY" and common:
            stage = 6
        elif rank == "SUPERFAMILY" and common:
            stage = 7
        elif rank == "ORDER" and common:
            stage = 8
        if stage is None:
            continue
        term = str(row.get("display_name") or row.get("verbatim_name") or "").strip()
        normalized = normalize_name_key(term)
        if not normalized:
            continue
        prepared = dict(row)
        prepared.update({"source_term": term, "normalized_match_key": normalized, "search_priority": stage, "australia_stage": stage})
        associations.append(prepared)
        selected.append(prepared)
    # One Flickr call per canonical term/field; preserve all taxon/name links in associations.
    chosen: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(selected, key=lambda r: (r["search_priority"], r["normalized_match_key"], r["source_term"], r.get("name_id", ""))):
        for field in ("tags", "text"):
            identity = (row["normalized_match_key"], field)
            if identity in seen:
                continue
            seen.add(identity)
            definition = dict(row)
            definition["search_field"] = field
            definition["logical_query_id"] = stable_identity("flickr-logical-query", row["normalized_match_key"], field, "australia", scope_identity)
            definition["query_definition_id"] = stable_identity("australia-flickr-query", definition["logical_query_id"], cutoff)
            chosen.append(definition)
    definitions = pl.DataFrame(chosen) if chosen else pl.DataFrame()
    association_frame = pl.DataFrame(associations) if associations else pl.DataFrame()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    definitions.write_parquet(output / "australia_flickr_query_definitions.parquet")
    association_frame.write_parquet(output / "australia_flickr_keyword_associations.parquet")
    queries = tuple(
        FlickrQuery(
            term=str(row["source_term"]), normalized_term=str(row["normalized_match_key"]), language=str(row.get("language") or "und"),
            search_field=str(row["search_field"]), lane="bbox_page", page=1, per_page=250, has_geo=1,
            place_id=effective_place_id, woe_id=effective_woe_id, bbox=bbox, region=f"australia-{scope_identity}", min_upload_date="2004-02-10", max_upload_date=cutoff,
            logical_query_id=str(row["logical_query_id"]), canonical_keyword_id=str(row.get("canonical_keyword_id") or stable_identity("canonical-keyword", row["normalized_match_key"])),
            keyword_id=str(row.get("keyword_id") or row.get("name_id") or "") or None,
            original_trust_tier=str(row.get("original_trust_tier") or row.get("trust_tier") or "T4"),
            effective_trust_tier=str(row.get("effective_trust_tier") or row.get("trust_tier") or "T4"),
            term_type=str(row.get("name_class") or ""), trust_tier=str(row.get("trust_tier") or "T4"),
            registry_version=str(row.get("registry_version") or ""), query_definition_id=str(row["query_definition_id"]),
            accepted_taxon_key=str(row.get("accepted_taxon_key") or ""), accepted_scientific_name=str(row.get("scientific_name") or ""),
            query_priority=int(row["search_priority"]),
        ) for row in chosen
    )
    manifest = {"scope": "Australia", "place_id": place_id, "woe_id": woe_id, "bbox": bbox, "cutoff": cutoff, "definitions": len(chosen), "associations": len(associations), "created_at": datetime.now(UTC).isoformat()}
    (output / "australia_flickr_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return definitions, association_frame, queries


def resolve_australia_place_id(*, api_key: str) -> str:
    payload = httpx.get("https://www.flickr.com/services/rest/", params={"method": "flickr.places.find", "api_key": api_key, "query": AUSTRALIA_PLACE_QUERY, "format": "json", "nojsoncallback": 1}, timeout=30).json()
    places = payload.get("places", {}).get("place", [])
    if not places:
        raise RuntimeError("Flickr did not return a place identifier for Australia")
    for place in places:
        if str(place.get("place_type") or "") == "12":
            return str(place["place_id"])
    return str(places[0]["place_id"])


def _gbif_australia_count(scientific_name: str) -> int:
    try:
        response = httpx.get(
            GBIF_OCCURRENCE_URL,
            params={"scientificName": scientific_name, "country": "AU", "limit": 0},
            headers={"User-Agent": "BioMiner/1.0 (+https://github.com/karikris/BioMiner)"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        retry_after = _retry_after_seconds(exc.response.headers.get("Retry-After")) if status == 429 else None
        raise GBIFRequestFailure(str(exc), http_status=status, retry_after_seconds=retry_after) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise GBIFRequestFailure(str(exc), http_status=None) from exc
    if not isinstance(payload, dict):
        raise GBIFRequestFailure("GBIF occurrence response must be a JSON object", http_status=None)
    try:
        return int(payload.get("count") or 0)
    except (TypeError, ValueError) as exc:
        raise GBIFRequestFailure("GBIF occurrence response count must be an integer", http_status=None) from exc


def _migrate_presence_schema(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(gbif_australia_presence)").fetchall()}
    for name, sql_type in (
        ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
        ("next_attempt_at", "TEXT"),
        ("last_http_status", "INTEGER"),
        ("last_error_class", "TEXT"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE gbif_australia_presence ADD COLUMN {name} {sql_type}")


def _record_gbif_failure(
    conn: sqlite3.Connection,
    *,
    key: str,
    attempts: int,
    error: GBIFRequestFailure,
    max_attempts: int,
) -> None:
    status = error.http_status
    retryable = status is None or status == 429 or status >= 500
    error_class = "retryable" if retryable else "terminal"
    if not retryable:
        next_status = "failed"
        next_attempt_at = None
    elif attempts >= max_attempts:
        next_status = "failed_exhausted"
        next_attempt_at = None
        error_class = "retry_exhausted"
    else:
        next_status = "retry_wait"
        delay = error.retry_after_seconds if error.retry_after_seconds is not None else min(
            GBIF_MAX_RETRY_DELAY_SECONDS,
            float(2 ** min(attempts, 12)),
        )
        next_attempt_at = datetime.fromtimestamp(datetime.now(UTC).timestamp() + delay, UTC).isoformat()
    conn.execute(
        """UPDATE gbif_australia_presence
           SET status = ?, error = ?, retrieved_at = ?, attempt_count = ?, next_attempt_at = ?,
               last_http_status = ?, last_error_class = ?
           WHERE accepted_taxon_key = ?""",
        (next_status, str(error)[:1000], datetime.now(UTC).isoformat(), attempts, next_attempt_at, status, error_class, key),
    )


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
