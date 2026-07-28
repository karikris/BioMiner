from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_final.bounded import (
    ASSEMBLY_MANIFEST_VERSION,
    PART_RECEIPT_VERSION,
    assemble_parts,
    preflight_assembly,
    seal_part,
    validate_part_receipt,
)


def _table(ordinals: list[int], values: list[str]) -> pa.Table:
    return pa.table(
        {
            "source_ordinal": pa.array(ordinals, type=pa.int64()),
            "value": pa.array(values, type=pa.string()),
        }
    )


def _seal(
    root: Path,
    *,
    name: str,
    start: int,
    stop: int,
    values: list[str],
    dependencies: dict[str, object] | None = None,
) -> Path:
    path = root / f"{name}.parquet"
    seal_part(
        table=_table(list(range(start, stop)), values),
        part_path=path,
        source_start_ordinal=start,
        source_stop_ordinal=stop,
        dependencies=dependencies or {"source_sha256": "sha256:source"},
        row_group_size=1,
    )
    return path.with_suffix(".parquet.receipt.json")


def test_sealed_part_is_dependency_bound_and_create_only(tmp_path: Path) -> None:
    receipt_path = _seal(
        tmp_path,
        name="part-00000",
        start=0,
        stop=2,
        values=["a", "b"],
    )

    receipt = validate_part_receipt(
        receipt_path,
        expected_dependencies={"source_sha256": "sha256:source"},
    )
    assert receipt["schema_version"] == PART_RECEIPT_VERSION
    assert receipt["artifact"]["row_count"] == 2
    assert receipt["artifact"]["row_group_rows"] == [1, 1]
    assert receipt_path.stat().st_mtime_ns >= (
        tmp_path / "part-00000.parquet"
    ).stat().st_mtime_ns

    with pytest.raises(FileExistsError):
        _seal(
            tmp_path,
            name="part-00000",
            start=0,
            stop=2,
            values=["a", "b"],
        )
    with pytest.raises(RuntimeError, match="dependencies are stale"):
        validate_part_receipt(
            receipt_path,
            expected_dependencies={"source_sha256": "sha256:changed"},
        )

    other_receipt_path = _seal(
        tmp_path,
        name="part-content-change",
        start=0,
        stop=2,
        values=["different", "content"],
    )
    other_receipt = validate_part_receipt(other_receipt_path)
    assert other_receipt["part_id"] != receipt["part_id"]


def test_sealed_part_corruption_is_rejected(tmp_path: Path) -> None:
    receipt_path = _seal(
        tmp_path,
        name="part-00000",
        start=0,
        stop=1,
        values=["a"],
    )
    part = tmp_path / "part-00000.parquet"
    part.write_bytes(part.read_bytes()[:-8])

    with pytest.raises(Exception):
        validate_part_receipt(receipt_path)


def test_tampered_receipt_is_rejected(tmp_path: Path) -> None:
    receipt_path = _seal(
        tmp_path,
        name="part-00000",
        start=0,
        stop=1,
        values=["a"],
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact"]["row_count"] = 99
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(RuntimeError, match="receipt fingerprint mismatch"):
        validate_part_receipt(receipt_path)


def test_assembly_is_sequential_verified_and_manifest_last(tmp_path: Path) -> None:
    part_root = tmp_path / "parts"
    first = _seal(
        part_root,
        name="part-00000",
        start=0,
        stop=2,
        values=["a", "b"],
    )
    second = _seal(
        part_root,
        name="part-00001",
        start=2,
        stop=4,
        values=["c", "d"],
    )
    output = tmp_path / "final"

    preflight = preflight_assembly(
        part_receipts=[second, first],
        output_parent=tmp_path,
        expected_rows=4,
        minimum_headroom_bytes=0,
    )
    assert preflight["status"] == "PASS"
    assert preflight["part_count"] == 2

    absent_parent = tmp_path / "not-created-by-preflight"
    preflight_assembly(
        part_receipts=[first, second],
        output_parent=absent_parent,
        expected_rows=4,
        minimum_headroom_bytes=0,
    )
    assert not absent_parent.exists()

    manifest = assemble_parts(
        part_receipts=[second, first],
        output_directory=output,
        expected_rows=4,
        code_commit="deadbeef",
        source_scope={
            "row_scope": "post_1960",
            "rows": 4,
            "excluded_pre_1960_rows": 2,
        },
        row_group_size=1,
        minimum_headroom_bytes=0,
    )

    assert manifest["schema_version"] == ASSEMBLY_MANIFEST_VERSION
    assert manifest["counts"]["rows"] == 4
    assert manifest["validation"]["one_output_row_per_source_row"]
    assert manifest["validation"]["manifest_written_last"]
    table = pq.read_table(output / "gbif_media_final_enriched.parquet")
    assert table["source_ordinal"].to_pylist() == [0, 1, 2, 3]
    assert table["value"].to_pylist() == ["a", "b", "c", "d"]
    disk_manifest = json.loads((output / "manifest.json").read_text())
    assert disk_manifest == manifest
    assert (output / "manifest.json").stat().st_mtime_ns >= (
        output / "gbif_media_final_enriched.parquet"
    ).stat().st_mtime_ns

    with pytest.raises(FileExistsError):
        assemble_parts(
            part_receipts=[first, second],
            output_directory=output,
            expected_rows=4,
            code_commit="deadbeef",
            source_scope={"rows": 4},
            minimum_headroom_bytes=0,
        )


def test_assembly_refuses_gaps_overlaps_and_schema_drift(tmp_path: Path) -> None:
    first = _seal(
        tmp_path / "gap",
        name="part-00000",
        start=0,
        stop=1,
        values=["a"],
    )
    gap = _seal(
        tmp_path / "gap",
        name="part-00001",
        start=2,
        stop=3,
        values=["c"],
    )
    with pytest.raises(RuntimeError, match="not contiguous"):
        preflight_assembly(
            part_receipts=[first, gap],
            output_parent=tmp_path,
            expected_rows=3,
            minimum_headroom_bytes=0,
        )

    schema_root = tmp_path / "schema"
    normal = _seal(
        schema_root,
        name="part-00000",
        start=0,
        stop=1,
        values=["a"],
    )
    changed_part = schema_root / "part-00001.parquet"
    changed_table = pa.table(
        {
            "source_ordinal": pa.array([1], type=pa.int64()),
            "value": pa.array([1], type=pa.int64()),
        }
    )
    seal_part(
        table=changed_table,
        part_path=changed_part,
        source_start_ordinal=1,
        source_stop_ordinal=2,
        dependencies={"source_sha256": "sha256:source"},
    )
    with pytest.raises(RuntimeError, match="inconsistent schemas"):
        preflight_assembly(
            part_receipts=[
                normal,
                changed_part.with_suffix(".parquet.receipt.json"),
            ],
            output_parent=tmp_path,
            expected_rows=2,
            minimum_headroom_bytes=0,
        )
