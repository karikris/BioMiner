from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_final.bounded import seal_part, validate_part_receipt


SOURCE_SPINE_VERSION = "gbif-final-source-spine/v1"
CHECKPOINT_VERSION = "gbif-final-source-spine-checkpoint/v1"
MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_FILENAME = "checkpoint.json"

DERIVED_TEMPORAL_COLUMNS = (
    "derived_year",
    "derived_month",
    "derived_day",
    "temporal_derivation_method",
)
IDENTITY_COLUMNS = (
    "gbifID",
    "media_identifier",
    "media_references",
    "speciesKey",
    "species",
)
MEDIA_QUALITY_COLUMNS = (
    "source_row_id",
    "source_sort_position",
    "media_assertion_id",
    "gbifID",
)
TEMPORAL_AUDIT_COLUMNS = (
    "gbifID",
    "temporal_derivation_status",
    "source_media_rows",
)
SOURCE_SPINE_SCHEMA = pa.schema(
    [
        pa.field("source_ordinal", pa.int64(), nullable=False),
        pa.field("legacy_v3_ordinal", pa.int64(), nullable=False),
        pa.field("source_sort_position", pa.int64(), nullable=False),
        pa.field("source_row_id", pa.string(), nullable=False),
        pa.field("media_assertion_id", pa.string(), nullable=False),
        pa.field("gbifID", pa.string()),
        pa.field("media_identifier", pa.string()),
        pa.field("media_references", pa.string()),
        pa.field("speciesKey", pa.string()),
        pa.field("species", pa.string()),
    ]
)


def build_source_spine(
    *,
    temporal_parquet: str | Path,
    pre_temporal_parquet: str | Path,
    media_quality_parquet: str | Path,
    temporal_audit_parquet: str | Path,
    output_directory: str | Path,
    producer_git_sha: str,
    part_rows: int = 250_000,
    batch_rows: int = 100_000,
    verification_memory_limit: str = "4GB",
    verification_threads: int = 4,
) -> dict[str, Any]:
    """Publish a restartable positional identity spine for post-1960 rows.

    The temporal source preserves retained v3 order. This builder therefore
    maps each retained row to the corresponding stable quality identifiers by
    position, rather than joining non-unique URL fields.
    """

    if part_rows <= 0:
        raise ValueError("part_rows must be positive")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if verification_threads <= 0:
        raise ValueError("verification_threads must be positive")
    if not producer_git_sha.strip():
        raise ValueError("producer_git_sha must be non-empty")

    paths = {
        "temporal": Path(temporal_parquet).resolve(),
        "pre_temporal": Path(pre_temporal_parquet).resolve(),
        "media_quality": Path(media_quality_parquet).resolve(),
        "temporal_audit": Path(temporal_audit_parquet).resolve(),
    }
    files = {
        name: _open_parquet(path)
        for name, path in paths.items()
    }
    _validate_input_schemas(files)
    counts = {
        name: int(file.metadata.num_rows)
        for name, file in files.items()
    }
    if counts["pre_temporal"] != counts["media_quality"]:
        raise RuntimeError(
            "pre-temporal and media-quality row counts differ: "
            f"pre_temporal={counts['pre_temporal']}, "
            f"media_quality={counts['media_quality']}"
        )

    excluded_by_occurrence = _excluded_occurrences(paths["temporal_audit"])
    excluded_rows = sum(excluded_by_occurrence.values())
    if counts["pre_temporal"] != counts["temporal"] + excluded_rows:
        raise RuntimeError(
            "temporal scope does not reconcile with excluded pre-1960 rows: "
            f"pre_temporal={counts['pre_temporal']}, "
            f"temporal={counts['temporal']}, excluded={excluded_rows}"
        )

    input_inventory = {
        name: _input_inventory(paths[name], files[name])
        for name in paths
    }
    semantic_inputs = {
        name: {
            key: value
            for key, value in inventory.items()
            if key != "path"
        }
        for name, inventory in input_inventory.items()
    }
    semantic_config: dict[str, object] = {
        "schema_version": SOURCE_SPINE_VERSION,
        "producer_git_sha": producer_git_sha,
        "part_rows": part_rows,
        "inputs": semantic_inputs,
        "source_scope": {
            "baseline": "legacy_v3_rights_filtered",
            "baseline_rows": counts["pre_temporal"],
            "row_scope": "post_1960",
            "excluded_pre_1960_rows": excluded_rows,
            "post_1960_rows": counts["temporal"],
        },
    }
    run_fingerprint = canonical_semantic_fingerprint(semantic_config)
    dependencies = {
        "source_spine_version": SOURCE_SPINE_VERSION,
        "run_fingerprint": run_fingerprint,
        "producer_git_sha": producer_git_sha,
        "inputs": semantic_inputs,
    }

    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite published source spine: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging")
    _prepare_checkpoint(
        staging=staging,
        run_fingerprint=run_fingerprint,
        semantic_config=semantic_config,
        input_inventory=input_inventory,
    )
    parts_directory = staging / "parts"
    parts_directory.mkdir(exist_ok=True)

    temporal_cursor = _BatchCursor(
        files["temporal"].iter_batches(
            batch_size=batch_rows,
            columns=list(IDENTITY_COLUMNS),
            use_threads=True,
        ),
        label="temporal",
    )
    pre_cursor = _BatchCursor(
        files["pre_temporal"].iter_batches(
            batch_size=batch_rows,
            columns=list(IDENTITY_COLUMNS),
            use_threads=True,
        ),
        label="pre_temporal",
    )
    quality_cursor = _BatchCursor(
        files["media_quality"].iter_batches(
            batch_size=batch_rows,
            columns=list(MEDIA_QUALITY_COLUMNS),
            use_threads=True,
        ),
        label="media_quality",
    )

    buffer: list[pa.Table] = []
    buffered_rows = 0
    post_1960_ordinal = 0
    legacy_v3_ordinal = 0
    previous_source_sort_position: int | None = None
    excluded_seen: Counter[str] = Counter()
    receipt_paths: list[Path] = []

    def publish_buffer(*, force: bool = False) -> None:
        nonlocal buffer, buffered_rows
        if not buffer or (not force and buffered_rows < part_rows):
            return
        combined = pa.concat_tables(buffer)
        while combined.num_rows >= part_rows or (force and combined.num_rows):
            rows = (
                part_rows
                if combined.num_rows >= part_rows
                else combined.num_rows
            )
            table = combined.slice(0, rows)
            start = int(table["source_ordinal"][0].as_py())
            stop = start + table.num_rows
            part_index = start // part_rows
            part_path = parts_directory / f"part-{part_index:05d}.parquet"
            _seal_or_resume_part(
                table=table,
                path=part_path,
                start=start,
                stop=stop,
                dependencies=dependencies,
                row_group_size=min(part_rows, 100_000),
            )
            receipt_paths.append(
                part_path.with_suffix(".parquet.receipt.json")
            )
            combined = combined.slice(rows)
        buffer = [combined] if combined.num_rows else []
        buffered_rows = combined.num_rows

    while legacy_v3_ordinal < counts["pre_temporal"]:
        rows = min(
            batch_rows,
            counts["pre_temporal"] - legacy_v3_ordinal,
        )
        pre_batch = pre_cursor.take(rows)
        quality_batch = quality_cursor.take(rows)
        source_positions = _required_ints(
            quality_batch,
            "source_sort_position",
        )
        if source_positions:
            if (
                previous_source_sort_position is not None
                and source_positions[0] <= previous_source_sort_position
            ):
                raise RuntimeError(
                    "media-quality source_sort_position is not strictly increasing"
                )
            if any(
                right <= left
                for left, right in zip(source_positions, source_positions[1:])
            ):
                raise RuntimeError(
                    "media-quality source_sort_position is not strictly increasing"
                )
            previous_source_sort_position = source_positions[-1]

        pre_gbif_ids = _normalized_strings(pre_batch, "gbifID")
        quality_gbif_ids = _normalized_strings(quality_batch, "gbifID")
        if pre_gbif_ids != quality_gbif_ids:
            mismatch = _first_mismatch(pre_gbif_ids, quality_gbif_ids)
            raise RuntimeError(
                "pre-temporal and media-quality identities differ at "
                f"legacy_v3_ordinal={legacy_v3_ordinal + mismatch}"
            )
        _require_stable_identifiers(quality_batch, legacy_v3_ordinal)

        keep = []
        for gbif_id in pre_gbif_ids:
            excluded = gbif_id in excluded_by_occurrence
            keep.append(not excluded)
            if excluded:
                excluded_seen[gbif_id] += 1
        mask = pa.array(keep, type=pa.bool_())
        retained_rows = sum(keep)
        retained_pre = _filter_record_batch(pre_batch, mask)
        retained_quality = _filter_record_batch(quality_batch, mask)
        retained_temporal = temporal_cursor.take(retained_rows)
        _validate_temporal_identity(
            retained_pre,
            retained_temporal,
            post_1960_ordinal=post_1960_ordinal,
        )

        legacy_ordinals = [
            legacy_v3_ordinal + index
            for index, retain in enumerate(keep)
            if retain
        ]
        spine_table = pa.Table.from_arrays(
            [
                pa.array(
                    range(
                        post_1960_ordinal,
                        post_1960_ordinal + retained_rows,
                    ),
                    type=pa.int64(),
                ),
                pa.array(legacy_ordinals, type=pa.int64()),
                retained_quality.column(
                    retained_quality.schema.get_field_index(
                        "source_sort_position"
                    )
                ),
                retained_quality.column(
                    retained_quality.schema.get_field_index("source_row_id")
                ),
                retained_quality.column(
                    retained_quality.schema.get_field_index(
                        "media_assertion_id"
                    )
                ),
                *[
                    retained_temporal.column(
                        retained_temporal.schema.get_field_index(name)
                    )
                    for name in IDENTITY_COLUMNS
                ],
            ],
            schema=SOURCE_SPINE_SCHEMA,
        )
        buffer.append(spine_table)
        buffered_rows += spine_table.num_rows
        publish_buffer()
        post_1960_ordinal += retained_rows
        legacy_v3_ordinal += rows

    publish_buffer(force=True)
    pre_cursor.assert_exhausted()
    quality_cursor.assert_exhausted()
    temporal_cursor.assert_exhausted()
    if post_1960_ordinal != counts["temporal"]:
        raise RuntimeError(
            "source-spine output row count mismatch: "
            f"expected={counts['temporal']}, observed={post_1960_ordinal}"
        )
    if dict(excluded_seen) != excluded_by_occurrence:
        raise RuntimeError(
            "pre-1960 exclusion counts do not match the temporal audit: "
            f"expected={excluded_by_occurrence}, observed={dict(excluded_seen)}"
        )

    expected_parts = (
        counts["temporal"] + part_rows - 1
    ) // part_rows
    if len(receipt_paths) != expected_parts:
        raise RuntimeError(
            "source-spine part count mismatch: "
            f"expected={expected_parts}, observed={len(receipt_paths)}"
        )
    _reject_unexpected_parts(
        parts_directory=parts_directory,
        expected_receipts=receipt_paths,
    )
    part_evidence = _validate_parts(
        receipt_paths=receipt_paths,
        dependencies=dependencies,
        expected_rows=counts["temporal"],
    )
    uniqueness = _verify_spine_dataset(
        parts_directory=parts_directory,
        expected_rows=counts["temporal"],
        staging=staging,
        memory_limit=verification_memory_limit,
        threads=verification_threads,
    )

    manifest_path = staging / MANIFEST_FILENAME
    manifest = {
        "schema_version": SOURCE_SPINE_VERSION,
        "generated_at": _timestamp(),
        "producer_git_sha": producer_git_sha,
        "run_fingerprint": run_fingerprint,
        "input_inventory": input_inventory,
        "source_scope": semantic_config["source_scope"],
        "configuration": {
            "part_rows": part_rows,
            "batch_rows_is_non_semantic": True,
            "positional_identity_mapping": True,
            "network_execution": False,
        },
        "counts": {
            "pre_temporal_rows": counts["pre_temporal"],
            "excluded_pre_1960_rows": excluded_rows,
            "post_1960_rows": counts["temporal"],
            "parts": expected_parts,
        },
        "schema": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in SOURCE_SPINE_SCHEMA
        ],
        "schema_fingerprint": _schema_fingerprint(SOURCE_SPINE_SCHEMA),
        "part_evidence": part_evidence,
        "validation": {
            "input_parquet_reopened": True,
            "pre_temporal_and_media_quality_rows_align": True,
            "pre_1960_exclusions_reconcile": dict(excluded_seen)
            == excluded_by_occurrence,
            "temporal_identity_matches_retained_source": True,
            "source_ranges_contiguous": True,
            "stable_identifiers_complete": True,
            "source_row_ids_unique": uniqueness["source_row_ids_unique"],
            "media_assertion_ids_unique": uniqueness[
                "media_assertion_ids_unique"
            ],
            "source_ordinals_unique_and_complete": uniqueness[
                "source_ordinals_unique_and_complete"
            ],
            "all_parts_reopened_and_checksummed": True,
            "manifest_written_last": True,
        },
        "network_requests": 0,
        "manifest_policy": {
            "create_only": True,
            "checkpoint_resume": True,
            "manifest_written_last": True,
        },
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError(
            f"source-spine validation failed: {manifest['validation']}"
        )
    manifest["manifest_fingerprint"] = _manifest_fingerprint(manifest)
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        _validate_existing_manifest(
            existing=existing,
            expected=manifest,
            manifest_path=manifest_path,
            receipt_paths=receipt_paths,
        )
        manifest = existing
    else:
        _write_json_create_only(manifest_path, manifest)
    newest_part_mtime = max(
        path.stat().st_mtime_ns
        for path in [
            *receipt_paths,
            *(_part_path_from_receipt(path) for path in receipt_paths),
        ]
    )
    if manifest_path.stat().st_mtime_ns < newest_part_mtime:
        raise RuntimeError("source-spine manifest was not written last")
    with _publication_lock(destination):
        if destination.exists():
            raise FileExistsError(
                "source-spine publication appeared during the build: "
                f"{destination}"
            )
        os.rename(staging, destination)
    _fsync_directory(destination.parent)
    return _load_json(destination / MANIFEST_FILENAME)


class _BatchCursor:
    def __init__(
        self,
        batches: Iterator[pa.RecordBatch],
        *,
        label: str,
    ) -> None:
        self._batches = iter(batches)
        self._label = label
        self._batch: pa.RecordBatch | None = None
        self._offset = 0
        self._finished = False

    def take(self, rows: int) -> pa.RecordBatch:
        if rows < 0:
            raise ValueError("rows must be non-negative")
        if rows == 0:
            schema = self._batch.schema if self._batch is not None else None
            if schema is None:
                try:
                    self._batch = next(self._batches)
                    self._offset = 0
                    schema = self._batch.schema
                except StopIteration as error:
                    raise RuntimeError(
                        f"{self._label} ended before its schema was observed"
                    ) from error
            return pa.RecordBatch.from_arrays(
                [pa.array([], type=field.type) for field in schema],
                schema=schema,
            )
        pieces: list[pa.RecordBatch] = []
        remaining = rows
        while remaining:
            if self._batch is None or self._offset == self._batch.num_rows:
                try:
                    self._batch = next(self._batches)
                    self._offset = 0
                except StopIteration as error:
                    self._finished = True
                    raise RuntimeError(
                        f"{self._label} ended {remaining} rows early"
                    ) from error
            available = self._batch.num_rows - self._offset
            taken = min(available, remaining)
            pieces.append(self._batch.slice(self._offset, taken))
            self._offset += taken
            remaining -= taken
        if len(pieces) == 1:
            return pieces[0]
        return pa.Table.from_batches(pieces).combine_chunks().to_batches()[0]

    def assert_exhausted(self) -> None:
        if self._batch is not None and self._offset < self._batch.num_rows:
            raise RuntimeError(f"{self._label} contains unexpected extra rows")
        if self._finished:
            return
        try:
            extra = next(self._batches)
        except StopIteration:
            self._finished = True
            return
        if extra.num_rows:
            raise RuntimeError(f"{self._label} contains unexpected extra rows")
        self.assert_exhausted()


def _open_parquet(path: Path) -> pq.ParquetFile:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows <= 0:
            raise RuntimeError(f"source-spine input is empty: {path}")
        if parquet.metadata.num_row_groups <= 0:
            raise RuntimeError(
                f"source-spine input has no Parquet row groups: {path}"
            )
        if sum(
            parquet.metadata.row_group(index).num_rows
            for index in range(parquet.metadata.num_row_groups)
        ) != parquet.metadata.num_rows:
            raise RuntimeError(
                f"source-spine input row groups are incomplete: {path}"
            )
        return parquet
    except Exception as error:
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(f"cannot reopen source-spine input: {path}") from error


def _validate_input_schemas(files: Mapping[str, pq.ParquetFile]) -> None:
    pre_schema = files["pre_temporal"].schema_arrow
    temporal_schema = files["temporal"].schema_arrow
    if temporal_schema.names != [
        *pre_schema.names,
        *DERIVED_TEMPORAL_COLUMNS,
    ]:
        raise RuntimeError(
            "temporal schema is not the pre-temporal schema plus derived fields"
        )
    if pa.schema(list(temporal_schema)[: len(pre_schema)]) != pre_schema:
        raise RuntimeError("temporal source fields changed name or type")
    _require_columns(pre_schema, IDENTITY_COLUMNS, label="pre_temporal")
    _require_columns(
        files["media_quality"].schema_arrow,
        MEDIA_QUALITY_COLUMNS,
        label="media_quality",
    )
    _require_columns(
        files["temporal_audit"].schema_arrow,
        TEMPORAL_AUDIT_COLUMNS,
        label="temporal_audit",
    )


def _require_columns(
    schema: pa.Schema,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = [name for name in required if name not in schema.names]
    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def _excluded_occurrences(path: Path) -> dict[str, int]:
    table = pq.read_table(path, columns=list(TEMPORAL_AUDIT_COLUMNS))
    result: dict[str, int] = {}
    for row in table.to_pylist():
        if row["temporal_derivation_status"] != "excluded_pre_1960":
            continue
        gbif_id = _normalized_string(row["gbifID"])
        rows = row["source_media_rows"]
        if not gbif_id or not isinstance(rows, int) or rows <= 0:
            raise RuntimeError(
                "temporal audit contains an invalid pre-1960 exclusion"
            )
        if gbif_id in result:
            raise RuntimeError(
                f"temporal audit repeats excluded gbifID: {gbif_id}"
            )
        result[gbif_id] = rows
    if not result:
        raise RuntimeError("temporal audit has no pre-1960 exclusions")
    return result


def _input_inventory(
    path: Path,
    parquet: pq.ParquetFile,
) -> dict[str, object]:
    row_group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    return {
        "path": str(path),
        "physical_bytes": path.stat().st_size,
        "physical_sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
        "row_group_rows": row_group_rows,
        "schema_fingerprint": _schema_fingerprint(parquet.schema_arrow),
    }


def _prepare_checkpoint(
    *,
    staging: Path,
    run_fingerprint: str,
    semantic_config: Mapping[str, object],
    input_inventory: Mapping[str, object],
) -> None:
    checkpoint = staging / CHECKPOINT_FILENAME
    if staging.exists():
        if not checkpoint.is_file():
            raise RuntimeError(
                f"source-spine staging lacks checkpoint: {staging}"
            )
        payload = _load_json(checkpoint)
        if (
            payload.get("schema_version") != CHECKPOINT_VERSION
            or payload.get("run_fingerprint") != run_fingerprint
            or payload.get("semantic_config") != semantic_config
            or payload.get("checkpoint_fingerprint")
            != _checkpoint_fingerprint(payload)
        ):
            raise RuntimeError(
                f"stale source-spine checkpoint: {checkpoint}"
            )
        return
    staging.mkdir()
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_VERSION,
        "created_at": _timestamp(),
        "run_fingerprint": run_fingerprint,
        "semantic_config": semantic_config,
        "input_inventory": input_inventory,
        "checkpoint_policy": {
            "create_only": True,
            "sealed_parts_resumable": True,
        },
    }
    payload["checkpoint_fingerprint"] = _checkpoint_fingerprint(payload)
    _write_json_create_only(
        checkpoint,
        payload,
    )


def _normalized_strings(
    batch: pa.RecordBatch,
    name: str,
) -> list[str]:
    column = batch.column(batch.schema.get_field_index(name))
    return [_normalized_string(value) for value in column.to_pylist()]


def _normalized_string(value: object) -> str:
    return str(value or "").strip()


def _required_ints(
    batch: pa.RecordBatch,
    name: str,
) -> list[int]:
    values = batch.column(batch.schema.get_field_index(name)).to_pylist()
    if any(not isinstance(value, int) for value in values):
        raise RuntimeError(f"{name} contains a null or non-integer value")
    return [int(value) for value in values]


def _require_stable_identifiers(
    batch: pa.RecordBatch,
    legacy_v3_ordinal: int,
) -> None:
    for name in ("source_row_id", "media_assertion_id"):
        values = batch.column(batch.schema.get_field_index(name)).to_pylist()
        for index, value in enumerate(values):
            if not _normalized_string(value):
                raise RuntimeError(
                    f"{name} is blank at legacy_v3_ordinal="
                    f"{legacy_v3_ordinal + index}"
                )


def _filter_record_batch(
    batch: pa.RecordBatch,
    mask: pa.BooleanArray,
) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [pc.filter(column, mask) for column in batch.columns],
        schema=batch.schema,
    )


def _validate_temporal_identity(
    pre_batch: pa.RecordBatch,
    temporal_batch: pa.RecordBatch,
    *,
    post_1960_ordinal: int,
) -> None:
    if pre_batch.num_rows != temporal_batch.num_rows:
        raise RuntimeError(
            "temporal identity comparison received different row counts"
        )
    for name in IDENTITY_COLUMNS:
        left = pre_batch.column(pre_batch.schema.get_field_index(name))
        right = temporal_batch.column(
            temporal_batch.schema.get_field_index(name)
        )
        if left.equals(right):
            continue
        left_values = left.to_pylist()
        right_values = right.to_pylist()
        mismatch = _first_mismatch(left_values, right_values)
        raise RuntimeError(
            "temporal identity mismatch for "
            f"{name} at source_ordinal={post_1960_ordinal + mismatch}"
        )


def _first_mismatch(left: list[object], right: list[object]) -> int:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return min(len(left), len(right))


def _seal_or_resume_part(
    *,
    table: pa.Table,
    path: Path,
    start: int,
    stop: int,
    dependencies: Mapping[str, object],
    row_group_size: int,
) -> dict[str, Any]:
    receipt_path = path.with_suffix(".parquet.receipt.json")
    if path.exists() != receipt_path.exists():
        raise RuntimeError(
            f"source-spine part is only partially sealed: {path}"
        )
    if path.exists():
        receipt = validate_part_receipt(
            receipt_path,
            expected_dependencies=dependencies,
        )
        if (
            int(receipt["source_start_ordinal"]) != start
            or int(receipt["source_stop_ordinal"]) != stop
            or int(receipt["artifact"]["row_count"]) != table.num_rows
        ):
            raise RuntimeError(
                f"source-spine sealed part has stale range: {path}"
            )
        if pq.ParquetFile(path).schema_arrow != SOURCE_SPINE_SCHEMA:
            raise RuntimeError(
                f"source-spine sealed part has stale schema: {path}"
            )
        return receipt
    return seal_part(
        table=table,
        part_path=path,
        source_start_ordinal=start,
        source_stop_ordinal=stop,
        dependencies=dependencies,
        row_group_size=row_group_size,
    )


def _reject_unexpected_parts(
    *,
    parts_directory: Path,
    expected_receipts: list[Path],
) -> None:
    expected = {
        path.resolve()
        for receipt in expected_receipts
        for path in (receipt, _part_path_from_receipt(receipt))
    }
    observed = {
        path.resolve()
        for path in parts_directory.iterdir()
        if path.is_file()
    }
    extras = sorted(observed - expected)
    missing = sorted(expected - observed)
    if extras or missing:
        raise RuntimeError(
            "source-spine staging part inventory mismatch: "
            f"extras={[str(path) for path in extras]}, "
            f"missing={[str(path) for path in missing]}"
        )


def _validate_parts(
    *,
    receipt_paths: list[Path],
    dependencies: Mapping[str, object],
    expected_rows: int,
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    expected_start = 0
    for receipt_path in receipt_paths:
        receipt = validate_part_receipt(
            receipt_path,
            expected_dependencies=dependencies,
        )
        start = int(receipt["source_start_ordinal"])
        stop = int(receipt["source_stop_ordinal"])
        if start != expected_start:
            raise RuntimeError(
                "source-spine part ranges are not contiguous: "
                f"expected={expected_start}, observed={start}"
            )
        expected_start = stop
        evidence.append(
            {
                "receipt_path": receipt_path.relative_to(
                    receipt_path.parents[1]
                ).as_posix(),
                "receipt_sha256": _sha256(receipt_path),
                "part_path": (
                    Path("parts") / str(receipt["artifact"]["path"])
                ).as_posix(),
                "part_id": receipt["part_id"],
                "part_sha256": receipt["artifact"]["physical_sha256"],
                "physical_bytes": receipt["artifact"]["physical_bytes"],
                "row_count": receipt["artifact"]["row_count"],
                "row_group_count": receipt["artifact"]["row_group_count"],
                "source_start_ordinal": start,
                "source_stop_ordinal": stop,
            }
        )
    if expected_start != expected_rows:
        raise RuntimeError(
            "source-spine parts do not cover expected rows: "
            f"expected={expected_rows}, covered={expected_start}"
        )
    return evidence


def _verify_spine_dataset(
    *,
    parts_directory: Path,
    expected_rows: int,
    staging: Path,
    memory_limit: str,
    threads: int,
) -> dict[str, bool]:
    temporary = staging / ".verification_tmp"
    temporary.mkdir(exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={int(threads)}")
        connection.execute(
            "SET memory_limit=?",
            [memory_limit],
        )
        connection.execute(
            "SET temp_directory=?",
            [str(temporary)],
        )
        (
            rows,
            source_rows,
            media_rows,
            ordinals,
            minimum_ordinal,
            maximum_ordinal,
            ordinal_sum,
        ) = connection.execute(
            """
            SELECT
              count(*),
              count(DISTINCT source_row_id),
              count(DISTINCT media_assertion_id),
              count(DISTINCT source_ordinal),
              min(source_ordinal),
              max(source_ordinal),
              sum(source_ordinal)
            FROM read_parquet(?)
            """,
            [str(parts_directory / "*.parquet")],
        ).fetchone()
    finally:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
    expected_sum = expected_rows * (expected_rows - 1) // 2
    return {
        "source_row_ids_unique": int(rows) == int(source_rows) == expected_rows,
        "media_assertion_ids_unique": int(rows)
        == int(media_rows)
        == expected_rows,
        "source_ordinals_unique_and_complete": (
            int(rows) == int(ordinals) == expected_rows
            and int(minimum_ordinal) == 0
            and int(maximum_ordinal) == expected_rows - 1
            and int(ordinal_sum) == expected_sum
        ),
    }


def _validate_existing_manifest(
    *,
    existing: Mapping[str, object],
    expected: Mapping[str, object],
    manifest_path: Path,
    receipt_paths: list[Path],
) -> None:
    if existing.get("schema_version") != SOURCE_SPINE_VERSION:
        raise RuntimeError(
            f"stale source-spine manifest schema: {manifest_path}"
        )
    if existing.get("run_fingerprint") != expected.get("run_fingerprint"):
        raise RuntimeError(
            f"stale source-spine manifest run fingerprint: {manifest_path}"
        )
    if existing.get("manifest_fingerprint") != _manifest_fingerprint(existing):
        raise RuntimeError(
            f"source-spine manifest fingerprint mismatch: {manifest_path}"
        )
    existing_semantic = {
        key: value
        for key, value in existing.items()
        if key not in {"generated_at", "manifest_fingerprint"}
    }
    expected_semantic = {
        key: value
        for key, value in expected.items()
        if key not in {"generated_at", "manifest_fingerprint"}
    }
    if existing_semantic != expected_semantic:
        raise RuntimeError(
            f"source-spine manifest evidence is stale: {manifest_path}"
        )
    newest_part_mtime = max(
        path.stat().st_mtime_ns
        for path in [
            *receipt_paths,
            *(_part_path_from_receipt(path) for path in receipt_paths),
        ]
    )
    if manifest_path.stat().st_mtime_ns < newest_part_mtime:
        raise RuntimeError(
            f"source-spine manifest was not written last: {manifest_path}"
        )


def _schema_fingerprint(schema: pa.Schema) -> str:
    digest = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
    return f"sha256:{digest}"


def _part_path_from_receipt(receipt: Path) -> Path:
    suffix = ".receipt.json"
    raw = str(receipt)
    if not raw.endswith(suffix):
        raise RuntimeError(
            f"invalid bounded-part receipt filename: {receipt}"
        )
    return Path(raw[: -len(suffix)])


def _manifest_fingerprint(value: Mapping[str, object]) -> str:
    body = {
        key: item
        for key, item in value.items()
        if key not in {"manifest_fingerprint", "generated_at"}
    }
    return canonical_semantic_fingerprint(body)


def _checkpoint_fingerprint(value: Mapping[str, object]) -> str:
    body = {
        key: item
        for key, item in value.items()
        if key not in {"checkpoint_fingerprint", "created_at"}
    }
    return canonical_semantic_fingerprint(body)


@contextmanager
def _publication_lock(destination: Path) -> Iterator[None]:
    digest = hashlib.sha256(str(destination).encode("utf-8")).hexdigest()
    lock_path = destination.parent / f".source-spine-{digest}.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_create_only(
    path: Path,
    value: Mapping[str, object],
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "SOURCE_SPINE_SCHEMA",
    "SOURCE_SPINE_VERSION",
    "build_source_spine",
]
