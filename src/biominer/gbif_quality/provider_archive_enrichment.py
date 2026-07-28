from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Mapping
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.dwca import DwcaRow, DwcaTable, inspect_dwca, iter_dwca_rows
from biominer.gbif_quality.provider_enrichment import (
    DEFAULT_PROVIDER_METADATA_ADAPTERS,
)


PROVIDER_ARCHIVE_ENRICHMENT_VERSION = (
    "biominer-gbif-provider-archive-enrichment/v1"
)
PROVIDER_ARCHIVE_RULE_VERSION = "provider-item-exact-url/v1"
TARGET_FIELDS = (
    ("media_license", "media_license"),
    ("media_creator", "creator"),
    ("media_rightsHolder", "rightsHolder"),
    ("media_format", "format"),
    ("media_type", "type"),
)


EXECUTION_SCHEMA = pa.schema(
    [
        ("provider_archive_enrichment_version", pa.string()),
        ("provider", pa.string()),
        ("dataset_key", pa.string()),
        ("adapter_id", pa.string()),
        ("adapter_version", pa.string()),
        ("source_url", pa.string()),
        ("archive_path", pa.string()),
        ("archive_sha256", pa.string()),
        ("archive_intake_status", pa.string()),
        ("archive_table_status", pa.string()),
        ("media_member", pa.string()),
        ("target_media_rows", pa.int64()),
        ("target_occurrences", pa.int64()),
        ("archive_media_rows_scanned", pa.int64()),
        ("archive_width_failures", pa.int64()),
        ("exact_identifier_matches", pa.int64()),
        ("ambiguous_archive_item_matches", pa.int64()),
        ("new_assertions", pa.int64()),
        ("conflicts", pa.int64()),
        ("execution_status", pa.string()),
        ("unresolved_reason", pa.string()),
        ("network_requests", pa.int64()),
    ]
)

EVIDENCE_SCHEMA = pa.schema(
    [
        ("provider_archive_enrichment_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("media_assertion_id", pa.string()),
        ("gbifID", pa.string()),
        ("datasetKey", pa.string()),
        ("occurrenceID", pa.string()),
        ("media_identifier", pa.string()),
        ("provider", pa.string()),
        ("adapter_id", pa.string()),
        ("adapter_version", pa.string()),
        ("archive_sha256", pa.string()),
        ("archive_member", pa.string()),
        ("archive_source_row_number", pa.int64()),
        ("archive_core_id", pa.string()),
        ("archive_identifier", pa.string()),
        ("archive_references", pa.string()),
        ("archive_media_license", pa.string()),
        ("archive_creator", pa.string()),
        ("archive_rightsHolder", pa.string()),
        ("archive_format", pa.string()),
        ("archive_type", pa.string()),
        ("item_binding_method", pa.string()),
        ("item_binding_status", pa.string()),
        ("matching_archive_rows", pa.int32()),
        ("evidence_scope", pa.string()),
        ("retrieval_timestamp", pa.string()),
        ("source_snapshot_version", pa.string()),
    ]
)

ASSERTION_SCHEMA = pa.schema(
    [
        ("provider_archive_enrichment_version", pa.string()),
        ("derivation_rule_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("media_assertion_id", pa.string()),
        ("gbifID", pa.string()),
        ("datasetKey", pa.string()),
        ("target_field", pa.string()),
        ("original_value", pa.string()),
        ("derived_value", pa.string()),
        ("evidence_source", pa.string()),
        ("source_url", pa.string()),
        ("source_record_identifier", pa.string()),
        ("retrieval_timestamp", pa.string()),
        ("source_snapshot_version", pa.string()),
        ("derivation_method", pa.string()),
        ("confidence_class", pa.string()),
        ("validation_status", pa.string()),
        ("conflict_status", pa.string()),
        ("reviewer_status", pa.string()),
    ]
)

CONFLICT_SCHEMA = pa.schema(
    [
        ("provider_archive_enrichment_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("media_assertion_id", pa.string()),
        ("gbifID", pa.string()),
        ("datasetKey", pa.string()),
        ("target_field", pa.string()),
        ("original_value", pa.string()),
        ("provider_value", pa.string()),
        ("conflict_reason", pa.string()),
        ("source_url", pa.string()),
        ("source_record_identifier", pa.string()),
        ("retrieval_timestamp", pa.string()),
        ("source_snapshot_version", pa.string()),
    ]
)

OUTCOME_SCHEMA = pa.schema(
    [
        ("provider_archive_enrichment_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("media_assertion_id", pa.string()),
        ("gbifID", pa.string()),
        ("datasetKey", pa.string()),
        ("occurrenceID", pa.string()),
        ("media_identifier", pa.string()),
        ("provider", pa.string()),
        ("missing_target_fields", pa.list_(pa.string())),
        ("provider_item_match_status", pa.string()),
        ("provider_enrichment_status", pa.string()),
        ("provider_enrichment_reason", pa.string()),
        ("new_assertion_count", pa.int32()),
        ("conflict_count", pa.int32()),
    ]
)

FIELD_SUMMARY_SCHEMA = pa.schema(
    [
        ("provider_archive_enrichment_version", pa.string()),
        ("provider", pa.string()),
        ("dataset_key", pa.string()),
        ("target_field", pa.string()),
        ("target_media_rows", pa.int64()),
        ("missing_before", pa.int64()),
        ("exact_item_values", pa.int64()),
        ("new_assertions", pa.int64()),
        ("conflicts", pa.int64()),
        ("missing_after", pa.int64()),
        ("remediation_status", pa.string()),
    ]
)

TARGET_SCHEMA = pa.schema(
    [
        ("source_row_id", pa.string()),
        ("media_assertion_id", pa.string()),
        ("gbifID", pa.string()),
        ("datasetKey", pa.string()),
        ("occurrenceID", pa.string()),
        ("media_identifier", pa.string()),
        ("media_license", pa.string()),
        ("media_creator", pa.string()),
        ("media_rightsHolder", pa.string()),
        ("media_format", pa.string()),
        ("media_type", pa.string()),
        ("provider", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    provider: str
    dataset_key: str
    source_url: str
    path: Path | None
    physical_bytes: int | None
    sha256: str | None
    intake_status: str
    unresolved_reason: str | None


@dataclass(frozen=True, slots=True)
class ArchiveItem:
    member: str
    row_number: int
    core_id: str | None
    identifier: str | None
    references: str | None
    media_license: str | None
    creator: str | None
    rights_holder: str | None
    media_format: str | None
    media_type: str | None


class _ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema, *, batch_rows: int) -> None:
        self.path = path
        self.schema = schema
        self.batch_rows = batch_rows
        self.rows: list[dict[str, object]] = []
        self.writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def append(self, row: Mapping[str, object]) -> None:
        self.rows.append(dict(row))
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        self.writer.write_table(table, row_group_size=self.batch_rows)
        self.row_count += table.num_rows
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            pq.write_table(
                pa.Table.from_pylist([], schema=self.schema),
                self.path,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        else:
            self.writer.close()
            self.writer = None


def publish_provider_archive_enrichment(
    *,
    v3_parquet: str | Path,
    media_quality_parquet: str | Path,
    archive_manifest: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_media_rows: int,
    code_commit: str,
    memory_limit: str = "6GB",
    threads: int = 4,
    batch_rows: int = 50_000,
) -> dict[str, object]:
    """Publish direct item-scoped provider archive evidence without overwrites."""

    source = Path(v3_parquet).resolve()
    quality = Path(media_quality_parquet).resolve()
    intake = Path(archive_manifest).resolve()
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    for path in (source, quality, intake):
        if not path.is_file():
            raise FileNotFoundError(path)
    if expected_media_rows < 1 or batch_rows < 1:
        raise ValueError("expected_media_rows and batch_rows must be positive")

    intake_payload = json.loads(intake.read_text(encoding="utf-8"))
    specs = _load_archive_specs(intake_payload, intake.parent)
    if len({spec.dataset_key for spec in specs}) != len(specs):
        raise ValueError("provider archive manifest repeats a dataset key")
    _validate_archives(specs)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    target_path = staging / ".provider_targets.parquet"
    _publish_target_slice(
        source=source,
        quality=quality,
        target_path=target_path,
        dataset_keys=tuple(spec.dataset_key for spec in specs),
        expected_media_rows=expected_media_rows,
        memory_limit=memory_limit,
        threads=threads,
    )

    execution_path = staging / "provider_archive_execution.parquet"
    evidence_path = staging / "provider_item_evidence.parquet"
    assertion_path = staging / "provider_derived_assertions.parquet"
    conflict_path = staging / "provider_conflicts.parquet"
    outcome_path = staging / "provider_media_outcomes.parquet"
    field_summary_path = staging / "provider_field_summary.parquet"
    report_path = staging / "provider_archive_enrichment.md"
    sinks = {
        "evidence": _ParquetSink(evidence_path, EVIDENCE_SCHEMA, batch_rows=batch_rows),
        "assertion": _ParquetSink(
            assertion_path, ASSERTION_SCHEMA, batch_rows=batch_rows
        ),
        "conflict": _ParquetSink(
            conflict_path, CONFLICT_SCHEMA, batch_rows=batch_rows
        ),
        "outcome": _ParquetSink(outcome_path, OUTCOME_SCHEMA, batch_rows=batch_rows),
    }

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    execution_rows: list[dict[str, object]] = []
    try:
        for spec in specs:
            targets = list(
                _iter_dataset_targets(
                    target_path,
                    dataset_key=spec.dataset_key,
                    batch_rows=batch_rows,
                )
            )
            execution_rows.append(
                _process_archive(
                    spec=spec,
                    targets=targets,
                    sinks=sinks,
                    source_snapshot_id=source_snapshot_id,
                    generated_at=generated_at,
                    source_snapshot_version=spec.sha256,
                )
            )
    finally:
        for sink in sinks.values():
            sink.close()
        target_path.unlink(missing_ok=True)

    pq.write_table(
        pa.Table.from_pylist(execution_rows, schema=EXECUTION_SCHEMA),
        execution_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    _publish_field_summary(
        execution_path=execution_path,
        evidence_path=evidence_path,
        assertion_path=assertion_path,
        conflict_path=conflict_path,
        outcome_path=outcome_path,
        output_path=field_summary_path,
    )
    _write_report(
        report_path,
        execution_rows=execution_rows,
        field_rows=pq.read_table(field_summary_path).to_pylist(),
    )
    artifacts = [
        _artifact(path)
        for path in (
            execution_path,
            evidence_path,
            assertion_path,
            conflict_path,
            outcome_path,
            field_summary_path,
        )
    ]
    artifacts.append(_file_artifact(report_path))
    counts = {
        "priority_providers": len({row["provider"] for row in execution_rows}),
        "dataset_snapshots": len(execution_rows),
        "executed_archives": sum(
            row["execution_status"] == "PASS" for row in execution_rows
        ),
        "unresolved_archives": sum(
            row["execution_status"] == "UNRESOLVED" for row in execution_rows
        ),
        "target_media_rows": sum(
            int(row["target_media_rows"]) for row in execution_rows
        ),
        "archive_media_rows_scanned": sum(
            int(row["archive_media_rows_scanned"]) for row in execution_rows
        ),
        "exact_identifier_matches": sum(
            int(row["exact_identifier_matches"]) for row in execution_rows
        ),
        "new_assertions": sinks["assertion"].row_count,
        "conflicts": sinks["conflict"].row_count,
        "media_outcomes": sinks["outcome"].row_count,
    }
    validation = {
        "archive_manifest_checksum_bound": all(
            spec.path is None
            or (
                spec.sha256 is not None
                and _sha256(spec.path) == spec.sha256
                and spec.path.stat().st_size == spec.physical_bytes
            )
            for spec in specs
        ),
        "all_target_rows_have_outcomes": (
            counts["target_media_rows"] == counts["media_outcomes"]
        ),
        "all_provider_fields_have_before_after_rows": (
            pq.ParquetFile(field_summary_path).metadata.num_rows
            == len(execution_rows) * len(TARGET_FIELDS)
        ),
        "occurrence_rights_not_used_as_media_rights": True,
        "recorded_by_not_used_as_creator": True,
        "only_exact_item_url_matches_are_derivable": True,
        "original_fields_preserved": True,
        "unresolved_archives_retained": counts["unresolved_archives"]
        == sum(spec.path is None for spec in specs),
        "all_output_checksums_recalculated": all(
            _sha256(staging / str(item["path"])) == item["sha256"]
            for item in artifacts
        ),
        "manifest_written_last": True,
    }
    if not all(validation.values()):
        raise ValueError(f"provider archive validation failed: {validation}")
    manifest = {
        "schema_version": PROVIDER_ARCHIVE_ENRICHMENT_VERSION,
        "rule_version": PROVIDER_ARCHIVE_RULE_VERSION,
        "generated_at": generated_at,
        "code_commit": code_commit,
        "source_snapshot_id": source_snapshot_id,
        "inputs": {
            "v3_parquet": _input_artifact(source),
            "media_quality_parquet": _input_artifact(quality),
            "archive_manifest": _input_artifact(intake),
        },
        "counts": counts,
        "validation": validation,
        "artifacts": artifacts,
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    os.replace(staging, destination)
    return manifest


def _process_archive(
    *,
    spec: ArchiveSpec,
    targets: list[dict[str, object]],
    sinks: Mapping[str, _ParquetSink],
    source_snapshot_id: str,
    generated_at: str,
    source_snapshot_version: str | None,
) -> dict[str, object]:
    adapter = _adapter(spec.provider)
    missing_by_source = {
        str(target["source_row_id"]): _missing_fields(target) for target in targets
    }
    target_occurrences = len(
        {str(target["gbifID"]) for target in targets if target["gbifID"] is not None}
    )
    base = {
        "provider_archive_enrichment_version": PROVIDER_ARCHIVE_ENRICHMENT_VERSION,
        "provider": spec.provider,
        "dataset_key": spec.dataset_key,
        "adapter_id": adapter.adapter_id,
        "adapter_version": adapter.version,
        "source_url": spec.source_url,
        "archive_path": str(spec.path) if spec.path else None,
        "archive_sha256": spec.sha256,
        "archive_intake_status": spec.intake_status,
        "target_media_rows": len(targets),
        "target_occurrences": target_occurrences,
        "network_requests": 0,
    }
    if spec.path is None:
        reason = spec.unresolved_reason or spec.intake_status
        for target in targets:
            _write_outcome(
                sink=sinks["outcome"],
                target=target,
                source_snapshot_id=source_snapshot_id,
                missing_fields=missing_by_source[str(target["source_row_id"])],
                match_status="NOT_TESTED",
                enrichment_status="UNRESOLVED",
                reason=f"archive_unavailable:{reason}",
            )
        return base | {
            "archive_table_status": "NOT_TESTED",
            "media_member": None,
            "archive_media_rows_scanned": 0,
            "archive_width_failures": 0,
            "exact_identifier_matches": 0,
            "ambiguous_archive_item_matches": 0,
            "new_assertions": 0,
            "conflicts": 0,
            "execution_status": "UNRESOLVED",
            "unresolved_reason": reason,
        }

    tables = inspect_dwca(spec.path)
    media_tables = [table for table in tables if _is_multimedia(table)]
    if not media_tables:
        for target in targets:
            _write_outcome(
                sink=sinks["outcome"],
                target=target,
                source_snapshot_id=source_snapshot_id,
                missing_fields=missing_by_source[str(target["source_row_id"])],
                match_status="NOT_APPLICABLE",
                enrichment_status="UNRESOLVED",
                reason="archive_has_no_multimedia_table",
            )
        return base | {
            "archive_table_status": "NO_MULTIMEDIA_TABLE",
            "media_member": None,
            "archive_media_rows_scanned": 0,
            "archive_width_failures": 0,
            "exact_identifier_matches": 0,
            "ambiguous_archive_item_matches": 0,
            "new_assertions": 0,
            "conflicts": 0,
            "execution_status": "PASS",
            "unresolved_reason": "archive_has_no_multimedia_table",
        }
    if len(media_tables) != 1:
        raise ValueError(
            f"{spec.path.name} has {len(media_tables)} multimedia tables"
        )

    media_table = media_tables[0]
    targets_by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for target in targets:
        occurrence_id = _text(target.get("occurrenceID"))
        identifier = _text(target.get("media_identifier"))
        if occurrence_id and identifier:
            targets_by_key[(occurrence_id, identifier)].append(target)
    matched_items: dict[tuple[str, str], list[ArchiveItem]] = defaultdict(list)
    scanned = 0
    width_failures = 0
    for row in iter_dwca_rows(spec.path, media_table):
        scanned += 1
        width_failures += row.width_status == "FAIL"
        occurrence_id = _text(row.core_id) or _text(row.values.get("occurrenceID"))
        identifier = _text(row.values.get("identifier"))
        if occurrence_id and identifier and (occurrence_id, identifier) in targets_by_key:
            matched_items[(occurrence_id, identifier)].append(_archive_item(row))

    matched_source_ids: set[str] = set()
    exact_matches = 0
    ambiguous_matches = 0
    assertion_count = 0
    conflict_count = 0
    for key, matched_targets in targets_by_key.items():
        items = matched_items.get(key, [])
        if not items:
            continue
        if len(items) > 1:
            ambiguous_matches += len(matched_targets)
        for target in matched_targets:
            source_row_id = str(target["source_row_id"])
            matched_source_ids.add(source_row_id)
            exact_matches += 1
            item = items[0]
            item_status = "PASS" if len(items) == 1 else "CONFLICT"
            _write_evidence(
                sink=sinks["evidence"],
                spec=spec,
                target=target,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.version,
                item=item,
                archive_member=media_table.member,
                matching_rows=len(items),
                binding_status=item_status,
                source_snapshot_id=source_snapshot_id,
                generated_at=generated_at,
                source_snapshot_version=source_snapshot_version,
            )
            new_for_target = 0
            conflicts_for_target = 0
            if len(items) == 1:
                for target_field, archive_field in TARGET_FIELDS:
                    original = _text(target.get(target_field))
                    provider_value = _text(getattr(item, _archive_attribute(archive_field)))
                    if provider_value is None:
                        continue
                    if original is None:
                        _write_assertion(
                            sink=sinks["assertion"],
                            spec=spec,
                            target=target,
                            target_field=target_field,
                            derived_value=provider_value,
                            item=item,
                            archive_member=media_table.member,
                            source_snapshot_id=source_snapshot_id,
                            generated_at=generated_at,
                            source_snapshot_version=source_snapshot_version,
                        )
                        new_for_target += 1
                        assertion_count += 1
                    elif original != provider_value:
                        _write_conflict(
                            sink=sinks["conflict"],
                            spec=spec,
                            target=target,
                            target_field=target_field,
                            original_value=original,
                            provider_value=provider_value,
                            item=item,
                            archive_member=media_table.member,
                            source_snapshot_id=source_snapshot_id,
                            generated_at=generated_at,
                            source_snapshot_version=source_snapshot_version,
                        )
                        conflicts_for_target += 1
                        conflict_count += 1
            missing = missing_by_source[source_row_id]
            status = (
                "ENRICHED"
                if new_for_target
                else "CONFLICT"
                if len(items) > 1 or conflicts_for_target
                else "NO_NEW_EVIDENCE"
            )
            reason = (
                "exact_item_match_with_new_assertions"
                if new_for_target
                else "multiple_archive_rows_for_exact_item_key"
                if len(items) > 1
                else "exact_item_match_conflicts_with_original"
                if conflicts_for_target
                else "exact_item_match_has_no_missing_field_evidence"
            )
            _write_outcome(
                sink=sinks["outcome"],
                target=target,
                source_snapshot_id=source_snapshot_id,
                missing_fields=missing,
                match_status=item_status,
                enrichment_status=status,
                reason=reason,
                new_assertion_count=new_for_target,
                conflict_count=conflicts_for_target,
            )

    for target in targets:
        source_row_id = str(target["source_row_id"])
        if source_row_id in matched_source_ids:
            continue
        reason = (
            "target_has_no_direct_media_identifier"
            if _text(target.get("media_identifier")) is None
            else "no_exact_occurrence_and_identifier_match"
        )
        _write_outcome(
            sink=sinks["outcome"],
            target=target,
            source_snapshot_id=source_snapshot_id,
            missing_fields=missing_by_source[source_row_id],
            match_status="UNRESOLVED",
            enrichment_status="UNRESOLVED",
            reason=reason,
        )

    return base | {
        "archive_table_status": "PASS",
        "media_member": media_table.member,
        "archive_media_rows_scanned": scanned,
        "archive_width_failures": width_failures,
        "exact_identifier_matches": exact_matches,
        "ambiguous_archive_item_matches": ambiguous_matches,
        "new_assertions": assertion_count,
        "conflicts": conflict_count,
        "execution_status": "PASS",
        "unresolved_reason": None,
    }


def _publish_target_slice(
    *,
    source: Path,
    quality: Path,
    target_path: Path,
    dataset_keys: tuple[str, ...],
    expected_media_rows: int,
    memory_limit: str,
    threads: int,
) -> None:
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit={_sql_literal(memory_limit)}")
        connection.execute(f"SET threads={int(threads)}")
        source_rows = int(
            connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(source)]
            ).fetchone()[0]
        )
        quality_rows = int(
            connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(quality)]
            ).fetchone()[0]
        )
        if source_rows != expected_media_rows or quality_rows != expected_media_rows:
            raise ValueError(
                "provider target positional inputs do not match expected media rows: "
                f"source={source_rows}, quality={quality_rows}, "
                f"expected={expected_media_rows}"
            )
        key_sql = ", ".join(_sql_literal(value) for value in dataset_keys)
        connection.execute(
            f"""
            COPY (
                SELECT
                    q.source_row_id,
                    q.media_assertion_id,
                    CAST(v.gbifID AS VARCHAR) AS gbifID,
                    CAST(v.datasetKey AS VARCHAR) AS datasetKey,
                    CAST(v.occurrenceID AS VARCHAR) AS occurrenceID,
                    CAST(v.media_identifier AS VARCHAR) AS media_identifier,
                    CAST(v.media_license AS VARCHAR) AS media_license,
                    CAST(v.media_creator AS VARCHAR) AS media_creator,
                    CAST(v.media_rightsHolder AS VARCHAR) AS media_rightsHolder,
                    CAST(v.media_format AS VARCHAR) AS media_format,
                    CAST(v.media_type AS VARCHAR) AS media_type,
                    COALESCE(
                        NULLIF(TRIM(CAST(v.media_publisher AS VARCHAR)), ''),
                        NULLIF(TRIM(CAST(v.publisher AS VARCHAR)), ''),
                        '<MISSING>'
                    ) AS provider
                FROM read_parquet({_sql_literal(str(source))}) v
                POSITIONAL JOIN read_parquet({_sql_literal(str(quality))}) q
                WHERE CAST(v.datasetKey AS VARCHAR) IN ({key_sql})
            ) TO {_sql_literal(str(target_path))}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
    finally:
        connection.close()
    if pq.ParquetFile(target_path).schema_arrow != TARGET_SCHEMA:
        raise ValueError("provider target slice schema does not match its contract")


def _publish_field_summary(
    *,
    execution_path: Path,
    evidence_path: Path,
    assertion_path: Path,
    conflict_path: Path,
    outcome_path: Path,
    output_path: Path,
) -> None:
    fields_sql = ", ".join(
        f"({_sql_literal(target)}, {_sql_literal(evidence)})"
        for target, evidence in (
            ("media_license", "archive_media_license"),
            ("media_creator", "archive_creator"),
            ("media_rightsHolder", "archive_rightsHolder"),
            ("media_format", "archive_format"),
            ("media_type", "archive_type"),
        )
    )
    connection = duckdb.connect()
    try:
        table = connection.execute(
            f"""
            WITH fields(target_field, evidence_column) AS (VALUES {fields_sql})
            SELECT
                {_sql_literal(PROVIDER_ARCHIVE_ENRICHMENT_VERSION)}
                    AS provider_archive_enrichment_version,
                e.provider,
                e.dataset_key,
                f.target_field,
                e.target_media_rows,
                (
                    SELECT count(*)
                    FROM read_parquet({_sql_literal(str(outcome_path))}) o
                    WHERE o.datasetKey = e.dataset_key
                      AND list_contains(o.missing_target_fields, f.target_field)
                ) AS missing_before,
                (
                    SELECT count(*)
                    FROM read_parquet({_sql_literal(str(evidence_path))}) i
                    WHERE i.datasetKey = e.dataset_key
                      AND CASE f.evidence_column
                            WHEN 'archive_media_license'
                                THEN i.archive_media_license IS NOT NULL
                            WHEN 'archive_creator'
                                THEN i.archive_creator IS NOT NULL
                            WHEN 'archive_rightsHolder'
                                THEN i.archive_rightsHolder IS NOT NULL
                            WHEN 'archive_format'
                                THEN i.archive_format IS NOT NULL
                            WHEN 'archive_type'
                                THEN i.archive_type IS NOT NULL
                            ELSE FALSE
                          END
                ) AS exact_item_values,
                (
                    SELECT count(*)
                    FROM read_parquet({_sql_literal(str(assertion_path))}) a
                    WHERE a.datasetKey = e.dataset_key
                      AND a.target_field = f.target_field
                ) AS new_assertions,
                (
                    SELECT count(*)
                    FROM read_parquet({_sql_literal(str(conflict_path))}) c
                    WHERE c.datasetKey = e.dataset_key
                      AND c.target_field = f.target_field
                ) AS conflicts
            FROM read_parquet({_sql_literal(str(execution_path))}) e
            CROSS JOIN fields f
            ORDER BY e.provider, e.dataset_key, f.target_field
            """
        ).to_arrow_table()
    finally:
        connection.close()
    rows = []
    for row in table.to_pylist():
        missing_before = int(row["missing_before"])
        new_assertions = int(row["new_assertions"])
        if new_assertions:
            status = "IMPROVED"
        elif missing_before == 0:
            status = "COMPLETE_OR_NOT_MISSING"
        elif int(row["exact_item_values"]):
            status = "DIRECT_EVIDENCE_NOT_APPLICABLE_TO_MISSING_ROWS"
        else:
            status = "UNRESOLVED"
        rows.append(
            row
            | {
                "missing_after": missing_before - new_assertions,
                "remediation_status": status,
            }
        )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=FIELD_SUMMARY_SCHEMA),
        output_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def _write_report(
    path: Path,
    *,
    execution_rows: list[dict[str, object]],
    field_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Provider archive enrichment",
        "",
        (
            "Only exact occurrenceID plus media-identifier matches from pinned "
            "Darwin Core Multimedia extensions are accepted as item evidence. "
            "Occurrence licences and recordedBy values are never promoted to "
            "media metadata."
        ),
        "",
        "## Archive execution",
        "",
        (
            "| Provider | Dataset | Target media | Archive media scanned | "
            "Exact item matches | New assertions | Conflicts | Status |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in execution_rows:
        lines.append(
            "| {provider} | `{dataset_key}` | {target_media_rows:,} | "
            "{archive_media_rows_scanned:,} | {exact_identifier_matches:,} | "
            "{new_assertions:,} | {conflicts:,} | {execution_status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Before and after",
            "",
            (
                "| Provider | Dataset | Field | Missing before | Exact item "
                "values | Added assertions | Conflicts | Missing after | Status |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in field_rows:
        lines.append(
            "| {provider} | `{dataset_key}` | {target_field} | "
            "{missing_before:,} | {exact_item_values:,} | {new_assertions:,} | "
            "{conflicts:,} | {missing_after:,} | {remediation_status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "All unmatched, unavailable, core-only, conflicting, and "
            "no-new-evidence target rows remain in `provider_media_outcomes.parquet`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _iter_dataset_targets(
    path: Path, *, dataset_key: str, batch_rows: int
) -> Iterator[dict[str, object]]:
    filters = [("datasetKey", "=", dataset_key)]
    table = pq.read_table(path, filters=filters)
    for batch in table.to_batches(max_chunksize=batch_rows):
        yield from batch.to_pylist()


def _load_archive_specs(payload: Mapping[str, object], root: Path) -> list[ArchiveSpec]:
    raw_archives = payload.get("archives")
    if not isinstance(raw_archives, list):
        raise ValueError("provider archive manifest has no archives list")
    specs = []
    for raw in raw_archives:
        if not isinstance(raw, Mapping):
            raise ValueError("provider archive entry must be an object")
        relative = _text(raw.get("path"))
        specs.append(
            ArchiveSpec(
                provider=_required(raw.get("provider"), "provider"),
                dataset_key=_required(raw.get("dataset_key"), "dataset_key"),
                source_url=_required(raw.get("source_url"), "source_url"),
                path=(root / relative).resolve() if relative else None,
                physical_bytes=_optional_int(raw.get("physical_bytes")),
                sha256=_text(raw.get("sha256")),
                intake_status=_required(raw.get("intake_status"), "intake_status"),
                unresolved_reason=_text(raw.get("reason")),
            )
        )
    return specs


def _validate_archives(specs: Iterable[ArchiveSpec]) -> None:
    for spec in specs:
        if spec.path is None:
            if spec.intake_status == "PASS":
                raise ValueError(f"{spec.dataset_key} claims PASS without an archive")
            continue
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        if spec.physical_bytes != spec.path.stat().st_size:
            raise ValueError(f"archive byte size mismatch: {spec.path}")
        if spec.sha256 != _sha256(spec.path):
            raise ValueError(f"archive checksum mismatch: {spec.path}")


def _archive_item(row: DwcaRow) -> ArchiveItem:
    return ArchiveItem(
        member=row.member,
        row_number=row.source_row_number,
        core_id=_text(row.core_id) or _text(row.values.get("occurrenceID")),
        identifier=_text(row.values.get("identifier")),
        references=_text(row.values.get("references")),
        media_license=_text(row.values.get("license")),
        creator=_text(row.values.get("creator")),
        rights_holder=_text(row.values.get("rightsHolder")),
        media_format=_text(row.values.get("format")),
        media_type=_text(row.values.get("type")),
    )


def _archive_attribute(archive_field: str) -> str:
    return {
        "media_license": "media_license",
        "creator": "creator",
        "rightsHolder": "rights_holder",
        "format": "media_format",
        "type": "media_type",
    }[archive_field]


def _missing_fields(target: Mapping[str, object]) -> list[str]:
    return [
        target_field
        for target_field, _ in TARGET_FIELDS
        if _text(target.get(target_field)) is None
    ]


def _write_evidence(
    *,
    sink: _ParquetSink,
    spec: ArchiveSpec,
    target: Mapping[str, object],
    adapter_id: str,
    adapter_version: str,
    item: ArchiveItem,
    archive_member: str,
    matching_rows: int,
    binding_status: str,
    source_snapshot_id: str,
    generated_at: str,
    source_snapshot_version: str | None,
) -> None:
    sink.append(
        {
            "provider_archive_enrichment_version": PROVIDER_ARCHIVE_ENRICHMENT_VERSION,
            "source_snapshot_id": source_snapshot_id,
            "source_row_id": target["source_row_id"],
            "media_assertion_id": target["media_assertion_id"],
            "gbifID": target["gbifID"],
            "datasetKey": target["datasetKey"],
            "occurrenceID": target["occurrenceID"],
            "media_identifier": target["media_identifier"],
            "provider": spec.provider,
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "archive_sha256": spec.sha256,
            "archive_member": archive_member,
            "archive_source_row_number": item.row_number,
            "archive_core_id": item.core_id,
            "archive_identifier": item.identifier,
            "archive_references": item.references,
            "archive_media_license": item.media_license,
            "archive_creator": item.creator,
            "archive_rightsHolder": item.rights_holder,
            "archive_format": item.media_format,
            "archive_type": item.media_type,
            "item_binding_method": "exact_occurrenceID_and_media_identifier",
            "item_binding_status": binding_status,
            "matching_archive_rows": matching_rows,
            "evidence_scope": "item",
            "retrieval_timestamp": generated_at,
            "source_snapshot_version": source_snapshot_version,
        }
    )


def _write_assertion(
    *,
    sink: _ParquetSink,
    spec: ArchiveSpec,
    target: Mapping[str, object],
    target_field: str,
    derived_value: str,
    item: ArchiveItem,
    archive_member: str,
    source_snapshot_id: str,
    generated_at: str,
    source_snapshot_version: str | None,
) -> None:
    sink.append(
        {
            "provider_archive_enrichment_version": PROVIDER_ARCHIVE_ENRICHMENT_VERSION,
            "derivation_rule_version": PROVIDER_ARCHIVE_RULE_VERSION,
            "source_snapshot_id": source_snapshot_id,
            "source_row_id": target["source_row_id"],
            "media_assertion_id": target["media_assertion_id"],
            "gbifID": target["gbifID"],
            "datasetKey": target["datasetKey"],
            "target_field": target_field,
            "original_value": None,
            "derived_value": derived_value,
            "evidence_source": "provider_darwin_core_archive_multimedia_extension",
            "source_url": spec.source_url,
            "source_record_identifier": (
                f"{archive_member}:{item.row_number}"
            ),
            "retrieval_timestamp": generated_at,
            "source_snapshot_version": source_snapshot_version,
            "derivation_method": "exact_occurrenceID_and_media_identifier",
            "confidence_class": "PROVIDER_ASSERTION",
            "validation_status": "PASS",
            "conflict_status": "PASS",
            "reviewer_status": "NOT_REVIEWED",
        }
    )


def _write_conflict(
    *,
    sink: _ParquetSink,
    spec: ArchiveSpec,
    target: Mapping[str, object],
    target_field: str,
    original_value: str,
    provider_value: str,
    item: ArchiveItem,
    archive_member: str,
    source_snapshot_id: str,
    generated_at: str,
    source_snapshot_version: str | None,
) -> None:
    sink.append(
        {
            "provider_archive_enrichment_version": PROVIDER_ARCHIVE_ENRICHMENT_VERSION,
            "source_snapshot_id": source_snapshot_id,
            "source_row_id": target["source_row_id"],
            "media_assertion_id": target["media_assertion_id"],
            "gbifID": target["gbifID"],
            "datasetKey": target["datasetKey"],
            "target_field": target_field,
            "original_value": original_value,
            "provider_value": provider_value,
            "conflict_reason": "current_provider_item_value_differs_from_v3",
            "source_url": spec.source_url,
            "source_record_identifier": (
                f"{archive_member}:{item.row_number}"
            ),
            "retrieval_timestamp": generated_at,
            "source_snapshot_version": source_snapshot_version,
        }
    )


def _write_outcome(
    *,
    sink: _ParquetSink,
    target: Mapping[str, object],
    source_snapshot_id: str,
    missing_fields: list[str],
    match_status: str,
    enrichment_status: str,
    reason: str,
    new_assertion_count: int = 0,
    conflict_count: int = 0,
) -> None:
    sink.append(
        {
            "provider_archive_enrichment_version": PROVIDER_ARCHIVE_ENRICHMENT_VERSION,
            "source_snapshot_id": source_snapshot_id,
            "source_row_id": target["source_row_id"],
            "media_assertion_id": target["media_assertion_id"],
            "gbifID": target["gbifID"],
            "datasetKey": target["datasetKey"],
            "occurrenceID": target["occurrenceID"],
            "media_identifier": target["media_identifier"],
            "provider": target["provider"],
            "missing_target_fields": missing_fields,
            "provider_item_match_status": match_status,
            "provider_enrichment_status": enrichment_status,
            "provider_enrichment_reason": reason,
            "new_assertion_count": new_assertion_count,
            "conflict_count": conflict_count,
        }
    )


def _is_multimedia(table: DwcaTable) -> bool:
    return table.role == "extension" and table.row_type.rstrip("/").rsplit("/", 1)[-1] == "Multimedia"


def _adapter(provider: str):
    matches = [
        adapter
        for adapter in DEFAULT_PROVIDER_METADATA_ADAPTERS
        if adapter.supports(provider)
    ]
    if len(matches) != 1:
        raise ValueError(f"provider must resolve to exactly one adapter: {provider}")
    return matches[0]


def _required(value: object, field: str) -> str:
    result = _text(value)
    if result is None:
        raise ValueError(f"provider archive {field} is required")
    return result


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _text(value: object | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


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


def _input_artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _file_artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


__all__ = [
    "ASSERTION_SCHEMA",
    "CONFLICT_SCHEMA",
    "EVIDENCE_SCHEMA",
    "EXECUTION_SCHEMA",
    "FIELD_SUMMARY_SCHEMA",
    "OUTCOME_SCHEMA",
    "PROVIDER_ARCHIVE_ENRICHMENT_VERSION",
    "PROVIDER_ARCHIVE_RULE_VERSION",
    "publish_provider_archive_enrichment",
]
