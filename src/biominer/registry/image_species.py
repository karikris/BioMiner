"""Collect source-bound species from an explicitly selected GBIF image database.

This is a fresh extraction, not a resumption of an older enriched registry. The
current user-selected database takes precedence over generic ground-zero prose;
its complete manifest chain must still reach the retained DWCA archive.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.registry.scope import load_scope


SCHEMA_VERSION = "biominer-image-species-collection/v3"
RAW_ARCHIVE_SHA256 = "3494944db9bca6917e0176e63852f8c233cdd7da95904c041c55ff8e8d31c6b9"
DEFAULT_SCOPE = Path(__file__).resolve().parents[3] / "config/butterfly_scope.json"
EventCallback = Callable[[dict[str, Any]], None]


def _event(logger: EventCallback | None, event: str, **values: Any) -> None:
    if logger is not None:
        logger({"event": event, **values})


def _hash(path: Path, logger: EventCallback | None = None) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    completed = 0
    _event(logger, "input_hash_started", path=str(path), total_bytes=total)
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024 * 1024):
            digest.update(chunk)
            completed += len(chunk)
            _event(logger, "input_hash_progress", path=str(path), bytes=completed, total_bytes=total)
    return digest.hexdigest()


def _digest(value: object) -> str:
    text = str(value or "")
    if not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", text):
        raise ValueError("invalid recorded SHA-256")
    return text.removeprefix("sha256:")


def _reference_path(value: object, manifest: Path) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve(strict=True)
    # The retained raw receipt uses repository-relative paths; later receipts
    # use absolute paths. New small manifests may use sibling-relative paths.
    for root in (manifest.parent, Path(__file__).resolve().parents[3]):
        if (root / candidate).exists():
            return (root / candidate).resolve(strict=True)
    raise FileNotFoundError(candidate)


def _raw_member_bindings(parent: dict[str, Any], raw: dict[str, Any], raw_path: Path) -> list[dict[str, Any]]:
    outputs = raw.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("raw receipt is missing member outputs")
    by_member: dict[str, dict[str, Any]] = {}
    for item in outputs:
        if not isinstance(item, dict) or item.get("member") in by_member:
            raise ValueError("raw receipt has invalid or duplicate member outputs")
        by_member[str(item.get("member"))] = item
    if set(by_member) != {"occurrence.txt", "multimedia.txt", "verbatim.txt"}:
        raise ValueError("raw receipt must identify the three retained DWCA members")
    artifacts = parent.get("input", {}).get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("join input is missing raw member artifacts")
    bindings = []
    for role in ("occurrence", "multimedia"):
        artifact = artifacts.get(role)
        member = by_member[role + ".txt"]
        if not isinstance(artifact, dict):
            raise ValueError(f"join input is missing raw {role} artifact")
        try:
            artifact_path = _reference_path(artifact["path"], raw_path)
            member_path = _reference_path(member["path"], raw_path)
            if artifact_path != member_path:
                raise ValueError("raw member path mismatch")
            for field in ("physical_bytes", "column_count"):
                if not isinstance(member[field], int) or member[field] <= 0 or artifact[field] != member[field]:
                    raise ValueError("raw member metadata mismatch")
            if member_path.stat().st_size != member["physical_bytes"]:
                raise ValueError("raw member physical byte count mismatch")
            if parent["counts"][role + "_rows"] != member["row_count"]:
                raise ValueError("raw member row count mismatch")
            digest = _digest(artifact["physical_sha256"])
        except KeyError as exc:
            raise ValueError("raw member binding is missing required metadata") from exc
        bindings.append({"role": role, "path": str(member_path),
                         "join_recorded_sha256": digest,
                         "raw_receipt_has_member_sha256": False,
                         "fresh_member_sha256_verified": False,
                         "path_bytes_columns_rows_match_raw_receipt": True})
    return bindings


def _manifest_chain(path: Path) -> list[dict[str, Any]]:
    """Verify actual manifest bytes and parent input/output joins, without raw scans."""
    chain: list[dict[str, Any]] = []
    visited: set[Path] = set()
    while True:
        path = path.resolve(strict=True)
        if path in visited or len(visited) >= 32:
            raise ValueError("cyclic or excessive input manifest chain")
        visited.add(path)
        contents = path.read_bytes()
        payload = json.loads(contents)
        chain.append({"path": str(path), "sha256": hashlib.sha256(contents).hexdigest(),
                      "schema_version": str(payload.get("schema_version", ""))})
        if payload.get("schema_version") == "biominer-dwca-members-parquet-v1":
            if _digest(payload.get("source", {}).get("archive_sha256")) != RAW_ARCHIVE_SHA256:
                raise ValueError("input lineage does not reach the retained raw GBIF archive")
            return chain
        source = payload.get("input", {}).get("source_manifest")
        if not isinstance(source, dict):
            raise ValueError("input manifest chain has no raw DWCA receipt")
        child = _reference_path(source["path"], path)
        child_bytes = child.read_bytes()
        if hashlib.sha256(child_bytes).hexdigest() != _digest(source.get("physical_sha256")):
            raise ValueError("input manifest checksum mismatch")
        child_payload = json.loads(child_bytes)
        output = child_payload.get("output")
        parent_input = payload["input"]
        if child_payload.get("schema_version") == "biominer-dwca-members-parquet-v1":
            if payload.get("schema_version") != "biominer-gbif-occurrence-multimedia-join/v1":
                raise ValueError("only the raw occurrence/multimedia join may bind DWCA members")
            chain[-1]["raw_member_bindings"] = _raw_member_bindings(payload, child_payload, child)
        else:
            if (not isinstance(output, dict) or parent_input.get("physical_sha256") is None
                    or output.get("physical_sha256") is None):
                raise ValueError("single-artifact manifest edge is missing required SHA-256 bindings")
            if _digest(parent_input["physical_sha256"]) != _digest(output.get("physical_sha256")):
                raise ValueError("input artifact checksum does not match source manifest output")
        path = child


def _write_frame(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_key_sql(column: str) -> str:
    """Keep source-owned alphanumeric identifiers without numeric coercion.

    Source-owned keys retain exact spelling, including digit-only identifiers
    and opaque uppercase identifiers such as ``2E1``. Digits do not establish
    legacy numeric API identity. Decimal or signed strings are never coerced.
    """
    value = f"TRIM(CAST({column} AS VARCHAR))"
    return (f"CASE WHEN REGEXP_MATCHES({value}, '^[A-Z0-9]+$') "
            f"AND NOT REGEXP_MATCHES({value}, '^0+$') "
            f"THEN {value} END")


def _source_key_order(value: str) -> tuple[int, int | str, str]:
    return (0, int(value), value) if value.isdigit() else (1, value, value)


def collect_image_species(
    database: Path, manifest: Path, output: Path, *, threads: int = 4,
    memory_limit: str = "2GB", logger: EventCallback | None = None,
    scope_path: Path = DEFAULT_SCOPE,
) -> dict[str, Any]:
    """Verify immutable input and publish deterministic species/variant Parquet files.

    Existing complete collections are reused only after input and artifact hashes
    match. Incomplete directories are never overwritten. Name/family conflicts
    remain visible and are ineligible for automatic enrichment queries.
    """
    if threads < 1:
        raise ValueError("threads must be positive")
    if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?(?:KB|MB|GB|TB)", memory_limit.upper()):
        raise ValueError("memory_limit must be a positive size such as 2GB")
    database, manifest, output = database.resolve(strict=True), manifest.resolve(strict=True), output.resolve()
    scope_path = scope_path.resolve(strict=True)
    scope_sha256 = _hash(scope_path)
    scope = load_scope(scope_path)
    family_keys = sorted({int(value) for value in scope.family_taxon_keys.values()})
    if len(family_keys) != 7:
        raise ValueError("image species collection requires the configured seven-family scope")
    chain = _manifest_chain(manifest)
    source = json.loads(manifest.read_text(encoding="utf-8"))
    db = source.get("database", {})
    if db.get("table") != "gbif_media" or db.get("image_url_column") != "media_identifier":
        raise ValueError("expected a GBIF media URL database manifest")
    recorded_database = Path(str(db.get("path", "")))
    if not recorded_database.is_absolute():
        recorded_database = manifest.parent / recorded_database
    if recorded_database.resolve() != database:
        raise ValueError("database path does not match input manifest")
    before = database.stat()
    if before.st_size != db.get("physical_bytes"):
        raise ValueError("database byte count mismatch")
    database_hash = _hash(database, logger)
    if database_hash != _digest(db.get("physical_sha256")):
        raise ValueError("database checksum mismatch")
    identity = {"schema_version": SCHEMA_VERSION, "database_sha256": database_hash,
                "manifest_chain": chain, "scope_sha256": scope_sha256,
                "family_keys": family_keys, "family_names": sorted(scope.included_families),
                "policy": "http-image-source-owned-alphanumeric-species-seven-family-names/v3",
                "taxon_key_contract": "source-owned GBIF DWCA identifiers; digit-only spelling is not proof of legacy numeric v1 API identity"}
    fingerprint = canonical_semantic_fingerprint(identity)

    def verify_inputs_unchanged() -> None:
        after = database.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("input database changed during collection")
        if _manifest_chain(manifest) != chain or _hash(scope_path) != scope_sha256:
            raise ValueError("input manifest chain or scope changed during collection")

    if output.exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("semantic_fingerprint") != fingerprint:
            raise ValueError("existing collection input/configuration mismatch")
        for item in existing["artifacts"]:
            if _hash(output / item["path"]) != item["sha256"]:
                raise ValueError("existing collection artifact checksum mismatch")
        verify_inputs_unchanged()
        _event(logger, "species_collection_reused", species_count=existing["species_count"], output=str(output))
        return existing

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid4().hex}.staging"
    staging.mkdir()
    try:
        with duckdb.connect(str(database), read_only=True) as connection:
            connection.execute(f"SET threads={int(threads)}")
            connection.execute(f"SET memory_limit='{memory_limit.upper()}'")
            connection.execute("SET temp_directory='" + str(staging / "spill").replace("'", "''") + "'")
            columns = {row[0] for row in connection.execute("DESCRIBE gbif_media").fetchall()}
            required = {"speciesKey", "species", "familyKey", "family", "genus", "media_identifier", "media_type", "media_format"}
            if not required <= columns:
                raise ValueError(f"missing source columns: {sorted(required - columns)}")
            family_names = ",".join("'" + name.replace("'", "''") + "'" for name in scope.included_families)
            species_key = _source_key_sql("speciesKey")
            family_key = _source_key_sql("familyKey")
            normalized = f"""
                SELECT {species_key} AS species_key,
                       NULLIF(TRIM(CAST(species AS VARCHAR)), '') AS scientific_name,
                       {family_key} AS family_key,
                       COALESCE(TRIM(CAST(family AS VARCHAR)), '') AS family,
                       COALESCE(TRIM(CAST(genus AS VARCHAR)), '') AS genus,
                       CASE
                         WHEN NOT COALESCE(REGEXP_MATCHES(TRIM(media_identifier), '^https?://[^/[:space:]]+[^[:space:]]*$'), FALSE) THEN 'missing_or_invalid_http_url'
                         WHEN NOT (LOWER(TRIM(COALESCE(media_type,''))) = 'stillimage'
                                   OR LOWER(TRIM(COALESCE(media_format,''))) LIKE 'image/%') THEN 'not_explicit_image'
                         WHEN ({family_key}) IS NULL OR TRIM(COALESCE(family,'')) NOT IN ({family_names}) THEN 'outside_butterfly_scope'
                         WHEN ({species_key}) IS NULL THEN 'missing_or_invalid_species_key'
                         WHEN NULLIF(TRIM(CAST(species AS VARCHAR)), '') IS NULL THEN 'missing_species_name'
                         ELSE 'eligible'
                       END AS eligibility
                FROM gbif_media
            """
            _event(logger, "species_collection_scan_started", database_rows=db.get("row_count"), threads=threads,
                   memory_limit=memory_limit)
            counts = dict(connection.execute(f"WITH normalized AS ({normalized}) SELECT eligibility, COUNT(*) FROM normalized GROUP BY eligibility").fetchall())
            if sum(counts.values()) != db.get("row_count"):
                raise ValueError("database row count differs from manifest")
            if db.get("row_count", 0) > 0 and counts.get("eligible", 0) == 0:
                _event(logger, "species_collection_source_format_incompatibility", exclusion_counts=counts)
                raise ValueError("nonempty source database has no eligible image species; source-format incompatibility")
            grouped = connection.execute(f"""WITH normalized AS ({normalized})
                SELECT species_key, scientific_name, family_key, family, genus, COUNT(*) AS image_rows
                FROM normalized WHERE eligibility='eligible'
                GROUP BY species_key, scientific_name, family_key, family, genus
                ORDER BY species_key, image_rows DESC, scientific_name, family_key, family, genus
            """).fetchall()
        verify_inputs_unchanged()
        species_rows: list[dict[str, Any]] = []
        variants: list[dict[str, Any]] = []
        by_key: dict[str, list[tuple[Any, ...]]] = {}
        for row in grouped:
            by_key.setdefault(str(row[0]), []).append(row)
        conflicts: list[dict[str, Any]] = []
        for ordinal, (key, rows) in enumerate(sorted(by_key.items(), key=lambda item: _source_key_order(item[0])), 1):
            names = {row[1] for row in rows}
            families = {(row[2], row[3]) for row in rows}
            expected_family = {str(value): name for name, value in scope.family_taxon_keys.items()}
            mismatch = any(str(row[2]) in expected_family and expected_family[str(row[2])] != row[3] for row in rows)
            status = "conflicting_source_names_or_families" if len(names) > 1 or len(families) > 1 or mismatch else "source_bound"
            representative = rows[0]
            species_rows.append({"species_ordinal": ordinal, "accepted_taxon_key": f"gbif:{key}",
                "species_key": str(key), "scientific_name": representative[1],
                "family_key": str(representative[2]), "source_family_key": str(representative[2]),
                "family": representative[3], "genus": representative[4],
                "image_rows": sum(int(row[5]) for row in rows), "name_variant_count": len(names),
                "query_eligible": status == "source_bound", "binding_status": status})
            for row in rows:
                variants.append({"species_ordinal": ordinal, "accepted_taxon_key": f"gbif:{key}",
                    "species_key": str(key), "scientific_name": row[1], "family_key": str(row[2]),
                    "source_family_key": str(row[2]), "family": row[3], "genus": row[4], "image_rows": int(row[5]),
                    "query_eligible": status == "source_bound", "binding_status": status})
            if status != "source_bound":
                conflicts.append({"species_key": str(key), "scientific_names": sorted(names),
                                  "family_bindings": [[str(k), n] for k, n in sorted(families)],
                                  "family_name_mismatch": mismatch})
        fields = [("species_ordinal", pa.int64()), ("accepted_taxon_key", pa.string()),
                  ("species_key", pa.string()), ("scientific_name", pa.string()),
                  ("family_key", pa.string()), ("source_family_key", pa.string()),
                  ("family", pa.string()), ("genus", pa.string()),
                  ("image_rows", pa.int64())]
        tail = [("query_eligible", pa.bool_()), ("binding_status", pa.string())]
        _write_frame(staging / "species.parquet", species_rows, pa.schema(fields + [("name_variant_count", pa.int64())] + tail))
        _write_frame(staging / "source_name_variants.parquet", variants, pa.schema(fields + tail))
        diagnostics = {"exclusion_counts": {k: int(v) for k, v in sorted(counts.items()) if k != "eligible"},
                       "eligible_image_rows": int(counts.get("eligible", 0)), "source_conflicts": conflicts,
                       "exclusion_policy": "exclusive first matching reason; source rows are not modified"}
        (staging / "exclusions.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.rmtree(staging / "spill", ignore_errors=True)
        artifacts = [{"path": name, "sha256": _hash(staging / name), "bytes": (staging / name).stat().st_size}
                     for name in ("species.parquet", "source_name_variants.parquet", "exclusions.json")]
        artifacts[0].update(row_count=len(species_rows), column_count=len(fields) + len(tail) + 1)
        artifacts[1].update(row_count=len(variants), column_count=len(fields) + len(tail))
        result = {"schema_version": SCHEMA_VERSION, "semantic_fingerprint": fingerprint,
                  "created_at": datetime.now(UTC).isoformat(), "producer_git_sha": _git_sha(),
                  "producer": {"python": sys.version.split()[0], "duckdb": duckdb.__version__, "pyarrow": pa.__version__},
                  "input": {"database": str(database), "database_sha256": database_hash,
                            "manifest": str(manifest), "row_count": int(db["row_count"])},
                  "identity": identity, "raw_archive_sha256": RAW_ARCHIVE_SHA256,
                  "taxon_key_contract": identity["taxon_key_contract"],
                  "evidence_maturity": "GBIF provider-asserted species/name evidence; not verified biological identity",
                  "scope_conflict_resolution": "explicit current image-database request; validated chain to retained raw DWCA; no older registry or enrichment reused",
                  "species_count": len(species_rows), "query_eligible_species_count": sum(row["query_eligible"] for row in species_rows),
                  "source_name_variant_rows": len(variants), "eligible_image_rows": int(counts.get("eligible", 0)),
                  "artifacts": artifacts, "manifest_written_last": True}
        (staging / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.rename(output)
        _event(logger, "species_collection_complete", species_count=len(species_rows),
               query_eligible_species_count=result["query_eligible_species_count"], output=str(output))
        return result
    except BaseException:
        # Preserve failed-run evidence; a later invocation must use a fresh root.
        _event(logger, "species_collection_failed", staging=str(staging))
        raise
