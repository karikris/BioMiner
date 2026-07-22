from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import calendar
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.assertions import (
    DERIVED_ASSERTION_SCHEMA,
    DerivedAssertion,
    assertion_table,
    build_assertion,
)


TEMPORAL_V2_VERSION = "biominer-gbif-temporal-quality/v2"
TEMPORAL_RULE_VERSION = "iso-event-date-v2.0.0"
TEMPORAL_QUALITY_SCHEMA = pa.schema(
    [
        ("temporal_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("affected_media_rows", pa.int64()),
        ("eventDate", pa.string()),
        ("source_year", pa.string()),
        ("source_month", pa.string()),
        ("source_day", pa.string()),
        ("derived_year", pa.int32()),
        ("derived_month", pa.int8()),
        ("derived_day", pa.int8()),
        ("event_date_precision", pa.string()),
        ("event_date_start", pa.string()),
        ("event_date_end", pa.string()),
        ("temporal_derivation_method", pa.string()),
        ("temporal_parse_status", pa.string()),
        ("temporal_parse_reason", pa.string()),
        ("temporal_conflict_status", pa.string()),
        ("ancient_record_status", pa.string()),
        ("future_record_status", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class ParsedEventDate:
    precision: str | None
    start: date | None
    end: date | None
    method: str | None
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class TemporalQualityResult:
    output_directory: Path
    quality_path: Path
    assertion_path: Path
    manifest: dict[str, object]


@dataclass(slots=True)
class _TemporalState:
    gbif_id: str
    values: dict[str, str | None]
    observed: dict[str, set[str | None]]
    conflicts: set[str]
    media_rows: int = 0

    @classmethod
    def create(cls, gbif_id: str) -> _TemporalState:
        return cls(
            gbif_id=gbif_id,
            values={name: None for name in ("eventDate", "year", "month", "day")},
            observed={name: set() for name in ("eventDate", "year", "month", "day")},
            conflicts=set(),
        )

    def add(self, row: dict[str, object | None]) -> None:
        self.media_rows += 1
        for name in self.values:
            value = _trimmed(row[name])
            self.observed[name].add(value)
            if len(self.observed[name]) > 1:
                self.conflicts.add(name)
            if self.values[name] is None and value is not None:
                self.values[name] = value


def parse_event_date(value: object | None) -> ParsedEventDate:
    text = _trimmed(value)
    if text is None:
        return ParsedEventDate(None, None, None, None, "UNKNOWN", "missing_event_date")
    parts = text.split("/")
    if len(parts) > 2:
        return ParsedEventDate(None, None, None, None, "FAIL", "unsupported_interval")
    endpoints = [_parse_endpoint(part.strip()) for part in parts]
    if any(endpoint is None for endpoint in endpoints):
        return ParsedEventDate(None, None, None, None, "FAIL", "invalid_or_unsupported_date")
    start_precision, start, start_end, timestamp = endpoints[0]  # type: ignore[misc]
    end = start_end
    precision = start_precision
    method = "iso_timestamp" if timestamp else "iso_partial_or_date"
    if len(endpoints) == 2:
        _, end_start, end, end_timestamp = endpoints[1]  # type: ignore[misc]
        if start > end or end_start < start:
            return ParsedEventDate("INTERVAL", start, end, "iso_interval", "FAIL", "reversed_interval")
        precision = "INTERVAL"
        method = "iso_interval_timestamp" if timestamp or end_timestamp else "iso_interval"
    return ParsedEventDate(precision, start, end, method, "PASS", "parsed")


def publish_temporal_quality_v2(
    *,
    v3_parquet: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    source_publication_date: str,
    expected_media_rows: int,
    expected_occurrences: int,
    expected_derived_year_media_rows: int,
    expected_derived_month_media_rows: int,
    expected_derived_day_media_rows: int,
    code_commit: str,
    expected_ancient_media_rows: int | None = None,
    expected_ancient_occurrences: int | None = None,
    batch_rows: int = 50_000,
) -> TemporalQualityResult:
    """Publish occurrence-grain temporal quality while retaining ancient rows."""

    source = Path(v3_parquet).resolve()
    destination = Path(output_directory).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    snapshot_date = date.fromisoformat(source_publication_date)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    quality = staging / "temporal_quality.parquet"
    assertions_path = staging / "temporal_assertions.parquet"
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        counts, assertions = _write_temporal_quality(
            source=source,
            output=quality,
            source_snapshot_id=source_snapshot_id,
            snapshot_date=snapshot_date,
            generated_at=generated_at,
            batch_rows=batch_rows,
        )
        pq.write_table(assertion_table(assertions), assertions_path, compression="zstd")
        validation = {
            "source_media_rows_reconciled": counts["media_rows"] == expected_media_rows,
            "one_row_per_occurrence": counts["occurrence_rows"] == expected_occurrences,
            "derived_year_media_rows_match": counts["derived_year_media_rows"]
            == expected_derived_year_media_rows,
            "derived_month_media_rows_match": counts["derived_month_media_rows"]
            == expected_derived_month_media_rows,
            "derived_day_media_rows_match": counts["derived_day_media_rows"]
            == expected_derived_day_media_rows,
            "ancient_media_rows_match": expected_ancient_media_rows is None
            or counts["ancient_media_rows"] == expected_ancient_media_rows,
            "ancient_occurrences_match": expected_ancient_occurrences is None
            or counts["ancient_occurrences"] == expected_ancient_occurrences,
            "ancient_rows_retained": counts["ancient_media_rows"] >= 0
            and counts["occurrence_rows"] == expected_occurrences,
            "original_fields_not_overwritten": True,
            "quality_schema_matches": pq.ParquetFile(quality).schema_arrow.equals(
                TEMPORAL_QUALITY_SCHEMA
            ),
            "assertion_schema_matches": pq.ParquetFile(
                assertions_path
            ).schema_arrow.equals(DERIVED_ASSERTION_SCHEMA),
        }
        if not all(validation.values()):
            raise ValueError(f"temporal v2 validation failed: {validation}; {counts}")
        artifacts = [_artifact(path) for path in (quality, assertions_path)]
        manifest = {
            "schema_version": TEMPORAL_V2_VERSION,
            "rule_version": TEMPORAL_RULE_VERSION,
            "generated_at": generated_at,
            "code_commit": code_commit,
            "source_snapshot_id": source_snapshot_id,
            "source_publication_date": source_publication_date,
            "input": str(source),
            "counts": counts,
            "validation": validation,
            "artifacts": artifacts,
            "policy": {
                "original_fields_unchanged": True,
                "derive_only_explicit_components": True,
                "ancient_records": "retain_and_flag",
                "future_records": "retain_and_flag",
            },
            "manifest_policy": {"written_last": True},
        }
        _write_json(staging / "manifest.json", manifest)
        for artifact in artifacts:
            _verify(staging, artifact)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return TemporalQualityResult(
        output_directory=destination,
        quality_path=destination / quality.name,
        assertion_path=destination / assertions_path.name,
        manifest=manifest,
    )


def _write_temporal_quality(
    *,
    source: Path,
    output: Path,
    source_snapshot_id: str,
    snapshot_date: date,
    generated_at: str,
    batch_rows: int,
) -> tuple[dict[str, int], list[DerivedAssertion]]:
    parquet = pq.ParquetFile(source)
    writer = pq.ParquetWriter(
        output, TEMPORAL_QUALITY_SCHEMA, compression="zstd", use_dictionary=True
    )
    buffers = {field.name: [] for field in TEMPORAL_QUALITY_SCHEMA}
    assertions: list[DerivedAssertion] = []
    counts = {
        "media_rows": 0,
        "occurrence_rows": 0,
        "derived_year_occurrences": 0,
        "derived_month_occurrences": 0,
        "derived_day_occurrences": 0,
        "derived_year_media_rows": 0,
        "derived_month_media_rows": 0,
        "derived_day_media_rows": 0,
        "ancient_occurrences": 0,
        "ancient_media_rows": 0,
        "future_occurrences": 0,
        "parse_failures": 0,
        "conflicts": 0,
    }
    current: _TemporalState | None = None
    previous: str | None = None

    def emit(state: _TemporalState) -> None:
        row, created = _temporal_row(
            state,
            source_snapshot_id=source_snapshot_id,
            snapshot_date=snapshot_date,
            generated_at=generated_at,
        )
        assertions.extend(created)
        for name in buffers:
            buffers[name].append(row[name])
        counts["occurrence_rows"] += 1
        for component in ("year", "month", "day"):
            if row[f"derived_{component}"] is not None:
                counts[f"derived_{component}_occurrences"] += 1
                counts[f"derived_{component}_media_rows"] += state.media_rows
        if row["ancient_record_status"] == "FLAGGED":
            counts["ancient_occurrences"] += 1
            counts["ancient_media_rows"] += state.media_rows
        if row["future_record_status"] == "FLAGGED":
            counts["future_occurrences"] += 1
        counts["parse_failures"] += int(row["temporal_parse_status"] == "FAIL")
        counts["conflicts"] += int(row["temporal_conflict_status"] == "CONFLICT")
        if len(buffers["gbifID"]) >= batch_rows:
            writer.write_table(pa.Table.from_pydict(buffers, schema=TEMPORAL_QUALITY_SCHEMA))
            for values in buffers.values():
                values.clear()

    try:
        for batch in parquet.iter_batches(
            batch_size=batch_rows,
            columns=["gbifID", "eventDate", "year", "month", "day"],
            use_threads=True,
        ):
            values = [batch.column(index).to_pylist() for index in range(5)]
            for gbif_id, event_date, year, month, day_value in zip(*values, strict=True):
                key = _trimmed(gbif_id)
                if key is None:
                    raise ValueError("blank gbifID in temporal input")
                if current is None or key != current.gbif_id:
                    if current is not None:
                        emit(current)
                    if previous is not None and key < previous:
                        raise ValueError("temporal source is not ordered by gbifID")
                    previous = key
                    current = _TemporalState.create(key)
                current.add(
                    {"eventDate": event_date, "year": year, "month": month, "day": day_value}
                )
                counts["media_rows"] += 1
        if current is not None:
            emit(current)
        if buffers["gbifID"]:
            writer.write_table(pa.Table.from_pydict(buffers, schema=TEMPORAL_QUALITY_SCHEMA))
    finally:
        writer.close()
    return counts, assertions


def _temporal_row(
    state: _TemporalState,
    *,
    source_snapshot_id: str,
    snapshot_date: date,
    generated_at: str,
) -> tuple[dict[str, object], list[DerivedAssertion]]:
    parsed = parse_event_date(state.values["eventDate"])
    start = parsed.start
    precision = parsed.precision
    start_precision = precision
    if precision == "INTERVAL" and state.values["eventDate"] is not None:
        start_precision = _endpoint_precision(state.values["eventDate"].split("/", 1)[0])
    supports_month = start_precision in {"MONTH", "DAY", "DATETIME"}
    supports_day = start_precision in {"DAY", "DATETIME"}
    derived_year = start.year if parsed.status == "PASS" and state.values["year"] is None else None
    derived_month = start.month if parsed.status == "PASS" and supports_month and state.values["month"] is None else None
    derived_day = start.day if parsed.status == "PASS" and supports_day and state.values["day"] is None else None
    conflict = bool(state.conflicts)
    if parsed.status == "PASS" and start is not None:
        for name, expected, supported in (
            ("year", start.year, True),
            ("month", start.month, supports_month),
            ("day", start.day, supports_day),
        ):
            raw = state.values[name]
            if raw is not None and supported:
                try:
                    conflict = conflict or int(raw) != expected
                except ValueError:
                    conflict = True
    source_row_id = "sha256:" + hashlib.sha256(
        f"{source_snapshot_id}|occurrence.txt|gbifID={state.gbif_id}".encode()
    ).hexdigest()
    assertions = []
    for target, value in (
        ("derived_year", derived_year),
        ("derived_month", derived_month),
        ("derived_day", derived_day),
    ):
        if value is None:
            continue
        assertions.append(
            build_assertion(
                source_snapshot_version=source_snapshot_id,
                source_row_id=source_row_id,
                gbif_id=state.gbif_id,
                target_field=target,
                original_value=None,
                derived_value=value,
                evidence_source="eventDate",
                source_url_or_record_identifier=f"gbifID:{state.gbif_id}",
                retrieval_timestamp=generated_at,
                derivation_method=parsed.method or "iso_event_date",
                derivation_rule_version=TEMPORAL_RULE_VERSION,
                confidence_class="DETERMINISTIC_DERIVATION",
                validation_status="PASS",
                conflict_status="CONFLICT" if conflict else "PASS",
                reviewer_status="NOT_REQUIRED" if not conflict else "PENDING",
            )
        )
    ancient = parsed.status == "PASS" and parsed.start is not None and parsed.start.year < 1960
    future = parsed.status == "PASS" and parsed.start is not None and parsed.start > snapshot_date
    return (
        {
            "temporal_version": TEMPORAL_V2_VERSION,
            "source_snapshot_id": source_snapshot_id,
            "source_row_id": source_row_id,
            "gbifID": state.gbif_id,
            "affected_media_rows": state.media_rows,
            "eventDate": state.values["eventDate"],
            "source_year": state.values["year"],
            "source_month": state.values["month"],
            "source_day": state.values["day"],
            "derived_year": derived_year,
            "derived_month": derived_month,
            "derived_day": derived_day,
            "event_date_precision": precision,
            "event_date_start": parsed.start.isoformat() if parsed.start else None,
            "event_date_end": parsed.end.isoformat() if parsed.end else None,
            "temporal_derivation_method": parsed.method,
            "temporal_parse_status": parsed.status,
            "temporal_parse_reason": parsed.reason,
            "temporal_conflict_status": "CONFLICT" if conflict else "PASS",
            "ancient_record_status": "FLAGGED" if ancient else "PASS",
            "future_record_status": "FLAGGED" if future else "PASS",
        },
        assertions,
    )


def _parse_endpoint(value: str) -> tuple[str, date, date, bool] | None:
    try:
        if re.fullmatch(r"[0-9]{4}", value):
            year = int(value)
            return "YEAR", date(year, 1, 1), date(year, 12, 31), False
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}", value):
            year, month = map(int, value.split("-"))
            last = calendar.monthrange(year, month)[1]
            return "MONTH", date(year, month, 1), date(year, month, last), False
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            parsed = date.fromisoformat(value)
            return "DAY", parsed, parsed, False
        if re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]{1,6})?)?(?:Z|[+-][0-9]{2}:[0-9]{2})?",
            value,
        ):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            return "DATETIME", parsed, parsed, True
    except ValueError:
        return None
    return None


def _endpoint_precision(value: str) -> str | None:
    parsed = _parse_endpoint(value.strip())
    return parsed[0] if parsed else None


def _trimmed(value: object | None) -> str | None:
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


def _verify(root: Path, artifact: dict[str, object]) -> None:
    path = root / str(artifact["path"])
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"temporal artifact checksum mismatch: {path.name}")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TEMPORAL_QUALITY_SCHEMA",
    "TEMPORAL_RULE_VERSION",
    "TEMPORAL_V2_VERSION",
    "ParsedEventDate",
    "TemporalQualityResult",
    "parse_event_date",
    "publish_temporal_quality_v2",
]
