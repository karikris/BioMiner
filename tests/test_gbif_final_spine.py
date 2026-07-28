from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import biominer.gbif_final.spine as spine_module
from biominer.gbif_final.spine import (
    SOURCE_SPINE_VERSION,
    build_source_spine,
    validate_source_spine,
)


def _write_parquet(
    path: Path,
    values: dict[str, list[object]],
    *,
    row_group_size: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(values),
        path,
        compression="zstd",
        row_group_size=row_group_size,
    )
    return path


def _fixture(root: Path) -> dict[str, Path]:
    pre_temporal = _write_parquet(
        root / "pre-temporal.parquet",
        {
            "gbifID": ["A", "X", "A", "B", "X", "C"],
            "media_identifier": [
                "https://img/a.jpg",
                "https://img/x1.jpg",
                "https://img/a.jpg",
                "https://img/b.jpg",
                "https://img/x2.jpg",
                None,
            ],
            "media_references": [
                "https://ref/a",
                "https://ref/x1",
                "https://ref/a",
                "https://ref/b",
                "https://ref/x2",
                "https://ref/c",
            ],
            "speciesKey": ["1", "9", "1", "2", "9", "3"],
            "species": ["Alpha", "Excluded", "Alpha", "Beta", "Excluded", "Gamma"],
        },
        row_group_size=2,
    )
    temporal = _write_parquet(
        root / "temporal.parquet",
        {
            "gbifID": ["A", "A", "B", "C"],
            "media_identifier": [
                "https://img/a.jpg",
                "https://img/a.jpg",
                "https://img/b.jpg",
                None,
            ],
            "media_references": [
                "https://ref/a",
                "https://ref/a",
                "https://ref/b",
                "https://ref/c",
            ],
            "speciesKey": ["1", "1", "2", "3"],
            "species": ["Alpha", "Alpha", "Beta", "Gamma"],
            "derived_year": [None, 2020, None, None],
            "derived_month": [None, 5, None, None],
            "derived_day": [None, 6, None, None],
            "temporal_derivation_method": [None, "event_date", None, None],
        },
        row_group_size=3,
    )
    media_quality = _write_parquet(
        root / "media-quality.parquet",
        {
            "source_row_id": [f"source-{index}" for index in range(6)],
            "source_sort_position": [10, 12, 15, 18, 22, 29],
            "media_assertion_id": [f"media-{index}" for index in range(6)],
            "gbifID": ["A", "X", "A", "B", "X", "C"],
        },
        row_group_size=4,
    )
    temporal_audit = _write_parquet(
        root / "temporal-audit.parquet",
        {
            "gbifID": ["A", "X"],
            "temporal_derivation_status": ["derived", "excluded_pre_1960"],
            "source_media_rows": [2, 2],
        },
        row_group_size=1,
    )
    return {
        "temporal_parquet": temporal,
        "pre_temporal_parquet": pre_temporal,
        "media_quality_parquet": media_quality,
        "temporal_audit_parquet": temporal_audit,
    }


def _build(root: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        **_fixture(root / "inputs"),
        "output_directory": root / "source-spine",
        "producer_git_sha": "deadbeef",
        "part_rows": 2,
        "batch_rows": 3,
    }
    values.update(overrides)
    return build_source_spine(**values)


def test_source_spine_preserves_duplicate_rows_without_join_inflation(
    tmp_path: Path,
) -> None:
    manifest = _build(tmp_path)

    assert manifest["schema_version"] == SOURCE_SPINE_VERSION
    assert manifest["counts"] == {
        "pre_temporal_rows": 6,
        "excluded_pre_1960_rows": 2,
        "post_1960_rows": 4,
        "parts": 2,
    }
    assert all(manifest["validation"].values())
    output = tmp_path / "source-spine"
    table = pq.read_table(
        sorted((output / "parts").glob("*.parquet"))
    )
    assert table["source_ordinal"].to_pylist() == [0, 1, 2, 3]
    assert table["legacy_v3_ordinal"].to_pylist() == [0, 2, 3, 5]
    assert table["source_sort_position"].to_pylist() == [10, 15, 18, 29]
    assert table["source_row_id"].to_pylist() == [
        "source-0",
        "source-2",
        "source-3",
        "source-5",
    ]
    assert table["media_assertion_id"].to_pylist() == [
        "media-0",
        "media-2",
        "media-3",
        "media-5",
    ]
    assert table["gbifID"].to_pylist() == ["A", "A", "B", "C"]
    assert table["media_identifier"].to_pylist()[:2] == [
        "https://img/a.jpg",
        "https://img/a.jpg",
    ]
    manifest_path = output / "manifest.json"
    newest_part_mtime = max(
        path.stat().st_mtime_ns for path in (output / "parts").iterdir()
    )
    assert manifest_path.stat().st_mtime_ns >= newest_part_mtime
    assert json.loads(manifest_path.read_text()) == manifest


def test_source_spine_rejects_temporal_identity_mismatch(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path / "inputs")
    temporal = pq.read_table(inputs["temporal_parquet"])
    columns = {
        name: temporal[name].to_pylist()
        for name in temporal.column_names
    }
    columns["media_references"][2] = "https://ref/wrong"
    _write_parquet(
        inputs["temporal_parquet"],
        columns,
        row_group_size=3,
    )

    with pytest.raises(RuntimeError, match="temporal identity mismatch"):
        build_source_spine(
            **inputs,
            output_directory=tmp_path / "source-spine",
            producer_git_sha="deadbeef",
            part_rows=2,
            batch_rows=3,
        )


def test_source_spine_resumes_only_checksum_matching_sealed_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(tmp_path / "inputs")
    output = tmp_path / "source-spine"
    actual_seal = spine_module.seal_part
    sealed = 0

    def stop_after_first_part(**kwargs: object) -> dict[str, object]:
        nonlocal sealed
        receipt = actual_seal(**kwargs)
        sealed += 1
        if sealed == 1:
            raise KeyboardInterrupt
        return receipt

    monkeypatch.setattr(spine_module, "seal_part", stop_after_first_part)
    with pytest.raises(KeyboardInterrupt):
        build_source_spine(
            **inputs,
            output_directory=output,
            producer_git_sha="deadbeef",
            part_rows=2,
            batch_rows=3,
        )

    staging = output.with_name(f".{output.name}.staging")
    first_part = staging / "parts" / "part-00000.parquet"
    first_receipt = first_part.with_suffix(".parquet.receipt.json")
    first_mtime = first_part.stat().st_mtime_ns
    assert first_receipt.is_file()

    monkeypatch.setattr(spine_module, "seal_part", actual_seal)
    manifest = build_source_spine(
        **inputs,
        output_directory=output,
        producer_git_sha="deadbeef",
        part_rows=2,
        batch_rows=1,
    )

    assert manifest["counts"]["parts"] == 2
    assert (output / "parts" / first_part.name).stat().st_mtime_ns == first_mtime
    assert not staging.exists()


def test_source_spine_refuses_stale_checkpoint(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path / "inputs")
    output = tmp_path / "source-spine"
    staging = output.with_name(f".{output.name}.staging")
    staging.mkdir()
    (staging / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": SOURCE_SPINE_VERSION,
                "run_fingerprint": "sha256:stale",
            }
        )
    )

    with pytest.raises(RuntimeError, match="stale source-spine checkpoint"):
        build_source_spine(
            **inputs,
            output_directory=output,
            producer_git_sha="deadbeef",
            part_rows=2,
            batch_rows=3,
        )


def test_published_source_spine_revalidates_exact_inputs(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path / "inputs")
    output = tmp_path / "source-spine"
    manifest = build_source_spine(
        **inputs,
        output_directory=output,
        producer_git_sha="deadbeef",
        part_rows=2,
        batch_rows=3,
    )

    validated = validate_source_spine(
        output,
        expected_inputs={
            "temporal": inputs["temporal_parquet"],
            "pre_temporal": inputs["pre_temporal_parquet"],
            "media_quality": inputs["media_quality_parquet"],
            "temporal_audit": inputs["temporal_audit_parquet"],
        },
        expected_producer_git_sha="deadbeef",
    )

    assert validated == manifest


def test_published_source_spine_rejects_changed_input(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path / "inputs")
    output = tmp_path / "source-spine"
    build_source_spine(
        **inputs,
        output_directory=output,
        producer_git_sha="deadbeef",
        part_rows=2,
        batch_rows=3,
    )
    temporal = pq.read_table(inputs["temporal_parquet"])
    changed = {
        name: temporal[name].to_pylist()
        for name in temporal.column_names
    }
    changed["derived_year"][0] = 2001
    _write_parquet(
        inputs["temporal_parquet"],
        changed,
        row_group_size=3,
    )

    with pytest.raises(
        RuntimeError,
        match="input inventory is stale: temporal",
    ):
        validate_source_spine(
            output,
            expected_inputs={
                "temporal": inputs["temporal_parquet"],
                "pre_temporal": inputs["pre_temporal_parquet"],
                "media_quality": inputs["media_quality_parquet"],
                "temporal_audit": inputs["temporal_audit_parquet"],
            },
        )
