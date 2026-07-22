from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.assertions import assertion_table, build_assertion


TAXONOMIC_REPAIR_VERSION = "biominer-gbif-taxonomic-repair/v1"
TAXONOMIC_RULE_VERSION = "accepted-name-species-binomial/v1.0.0"
TAXONOMIC_REPAIR_SCHEMA = pa.schema(
    [
        ("repair_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("affected_media_rows", pa.int64()),
        ("source_taxon_rank", pa.string()),
        ("source_species", pa.string()),
        ("source_scientific_name", pa.string()),
        ("source_accepted_scientific_name", pa.string()),
        ("source_taxon_key", pa.string()),
        ("source_accepted_taxon_key", pa.string()),
        ("source_taxonomic_status", pa.string()),
        ("derived_species", pa.string()),
        ("derivation_status", pa.string()),
        ("derivation_reason", pa.string()),
        ("backbone_snapshot", pa.string()),
        ("reviewer_status", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class TaxonomicRepairResult:
    output_directory: Path
    repair_path: Path
    assertion_path: Path
    manifest: dict[str, object]


def accepted_species_binomial(value: object | None) -> str | None:
    """Return a conservative species binomial encoded by an accepted name."""

    text = _trimmed(value)
    if text is None:
        return None
    tokens = text.split()
    if len(tokens) < 2:
        return None
    genus, epithet = tokens[:2]
    if not re.fullmatch(r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'-]+", genus):
        return None
    if not re.fullmatch(r"[a-zà-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*", epithet):
        return None
    if epithet.casefold() in {"sp", "sp.", "spp", "spp.", "indet", "indet."}:
        return None
    return f"{genus} {epithet}"


def publish_species_rank_repairs(
    *,
    v3_parquet: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_candidate_media_rows: int,
    expected_candidate_occurrences: int,
    code_commit: str,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> TaxonomicRepairResult:
    """Publish only same-record, species-rank repairs backed by accepted names."""

    source = Path(v3_parquet).resolve()
    destination = Path(output_directory).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    if threads < 1:
        raise ValueError("threads must be positive")
    required = {
        "gbifID",
        "taxonRank",
        "species",
        "scientificName",
        "acceptedScientificName",
        "taxonKey",
        "acceptedTaxonKey",
        "taxonomicStatus",
    }
    missing = required - set(pq.ParquetFile(source).schema_arrow.names)
    if missing:
        raise ValueError(f"v3 lacks taxonomic repair fields: {sorted(missing)}")
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows = _candidate_rows(source, memory_limit=memory_limit, threads=threads)
    candidate_media_rows = sum(int(row["affected_media_rows"]) for row in rows)
    if candidate_media_rows != expected_candidate_media_rows:
        raise ValueError(
            f"species-rank candidate media mismatch: {candidate_media_rows}"
        )
    if len(rows) != expected_candidate_occurrences:
        raise ValueError(f"species-rank candidate occurrence mismatch: {len(rows)}")
    assertions = []
    published = []
    for row in rows:
        derived = accepted_species_binomial(row["source_accepted_scientific_name"])
        direct_evidence = all(
            _trimmed(row[name]) is not None
            for name in (
                "source_taxon_key",
                "source_accepted_taxon_key",
                "source_scientific_name",
                "source_accepted_scientific_name",
            )
        )
        status = "PASS" if derived is not None and direct_evidence else "UNKNOWN"
        reason = (
            "same_record_accepted_taxon_key_and_name"
            if status == "PASS"
            else "insufficient_direct_taxonomic_evidence"
        )
        source_row_id = _source_row_id(source_snapshot_id, str(row["gbifID"]))
        published_row = {
            "repair_version": TAXONOMIC_REPAIR_VERSION,
            "source_snapshot_id": source_snapshot_id,
            "source_row_id": source_row_id,
            **row,
            "derived_species": derived,
            "derivation_status": status,
            "derivation_reason": reason,
            "backbone_snapshot": source_snapshot_id,
            "reviewer_status": "NOT_REQUIRED" if status == "PASS" else "PENDING",
        }
        published.append(published_row)
        if status == "PASS":
            assertions.append(
                build_assertion(
                    source_snapshot_version=source_snapshot_id,
                    source_row_id=source_row_id,
                    gbif_id=str(row["gbifID"]),
                    target_field="derived_species",
                    original_value=row["source_species"],
                    derived_value=derived,
                    evidence_source="acceptedScientificName|acceptedTaxonKey",
                    source_url_or_record_identifier=f"gbifID:{row['gbifID']}",
                    retrieval_timestamp=generated_at,
                    derivation_method="same_record_accepted_name_binomial",
                    derivation_rule_version=TAXONOMIC_RULE_VERSION,
                    confidence_class="DETERMINISTIC_DERIVATION",
                    validation_status="PASS",
                    conflict_status="PASS",
                    reviewer_status="NOT_REQUIRED",
                )
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    repair_path = staging / "species_rank_repairs.parquet"
    assertion_path = staging / "taxonomic_assertions.parquet"
    try:
        pq.write_table(
            pa.Table.from_pylist(published, schema=TAXONOMIC_REPAIR_SCHEMA),
            repair_path,
            compression="zstd",
        )
        pq.write_table(assertion_table(assertions), assertion_path, compression="zstd")
        repaired_occurrences = len(assertions)
        repaired_media_rows = sum(
            int(row["affected_media_rows"])
            for row in published
            if row["derivation_status"] == "PASS"
        )
        counts = {
            "candidate_media_rows": candidate_media_rows,
            "candidate_occurrences": len(rows),
            "repaired_media_rows": repaired_media_rows,
            "repaired_occurrences": repaired_occurrences,
            "unresolved_occurrences": len(rows) - repaired_occurrences,
        }
        validation = {
            "candidate_media_rows_match": candidate_media_rows
            == expected_candidate_media_rows,
            "candidate_occurrences_match": len(rows) == expected_candidate_occurrences,
            "all_candidates_retained": pq.ParquetFile(repair_path).metadata.num_rows
            == len(rows),
            "one_assertion_per_repair": pq.ParquetFile(assertion_path).metadata.num_rows
            == repaired_occurrences,
            "original_species_unchanged": all(row["source_species"] is None for row in rows),
            "only_species_rank": all(
                str(row["source_taxon_rank"]).upper() == "SPECIES" for row in rows
            ),
            "direct_evidence_required": all(
                row["derivation_status"] != "PASS"
                or (
                    row["source_taxon_key"]
                    and row["source_accepted_taxon_key"]
                    and row["source_accepted_scientific_name"]
                )
                for row in published
            ),
        }
        if not all(validation.values()):
            raise ValueError(f"taxonomic repair validation failed: {validation}")
        artifacts = [_artifact(repair_path), _artifact(assertion_path)]
        manifest = {
            "schema_version": TAXONOMIC_REPAIR_VERSION,
            "rule_version": TAXONOMIC_RULE_VERSION,
            "generated_at": generated_at,
            "code_commit": code_commit,
            "source_snapshot_id": source_snapshot_id,
            "input": str(source),
            "counts": counts,
            "validation": validation,
            "artifacts": artifacts,
            "policy": {
                "source_fields_unchanged": True,
                "same_record_evidence_only": True,
                "higher_rank_population_forbidden": True,
                "cross_record_taxon_key_inference_forbidden": True,
            },
            "network_requests": 0,
            "manifest_policy": {"written_last": True},
        }
        _write_json(staging / "manifest.json", manifest)
        for artifact in artifacts:
            _verify(staging, artifact)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return TaxonomicRepairResult(
        output_directory=destination,
        repair_path=destination / repair_path.name,
        assertion_path=destination / assertion_path.name,
        manifest=manifest,
    )


def _candidate_rows(
    source: Path, *, memory_limit: str, threads: int
) -> list[dict[str, object]]:
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        cursor = connection.execute(
            f"""
            SELECT trim(cast(gbifID AS VARCHAR)) AS gbifID,
                   count(*)::BIGINT AS affected_media_rows,
                   min(taxonRank) AS source_taxon_rank,
                   min(species) AS source_species,
                   min(scientificName) AS source_scientific_name,
                   min(acceptedScientificName) AS source_accepted_scientific_name,
                   min(cast(taxonKey AS VARCHAR)) AS source_taxon_key,
                   min(cast(acceptedTaxonKey AS VARCHAR)) AS source_accepted_taxon_key,
                   min(taxonomicStatus) AS source_taxonomic_status
            FROM read_parquet({_literal(str(source))})
            WHERE species IS NULL AND upper(trim(taxonRank)) = 'SPECIES'
            GROUP BY gbifID
            ORDER BY gbifID
            """
        )
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()


def _source_row_id(snapshot: str, gbif_id: str) -> str:
    value = f"{snapshot}|occurrence.txt|gbifID={gbif_id}"
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _trimmed(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _artifact(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
    }


def _verify(root: Path, artifact: dict[str, object]) -> None:
    if _sha256(root / str(artifact["path"])) != artifact["sha256"]:
        raise ValueError(f"taxonomic artifact checksum mismatch: {artifact['path']}")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TAXONOMIC_REPAIR_SCHEMA",
    "TAXONOMIC_REPAIR_VERSION",
    "TAXONOMIC_RULE_VERSION",
    "TaxonomicRepairResult",
    "accepted_species_binomial",
    "publish_species_rank_repairs",
]
