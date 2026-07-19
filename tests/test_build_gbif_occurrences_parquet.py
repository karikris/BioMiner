from __future__ import annotations

import argparse
import zipfile

import polars as pl
import pytest

from scripts.build_gbif_occurrences_parquet import run


_COLUMNS = (
    "gbifID",
    "basisOfRecord",
    "datasetKey",
    "hasCoordinate",
    "decimalLatitude",
    "decimalLongitude",
)


def _write_dwca(path, rows: list[tuple[str, ...]]) -> None:
    payload = "\t".join(_COLUMNS) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("occurrence.txt", payload)


def _args(archive, output, manifest, *, expected_records: int) -> argparse.Namespace:
    return argparse.Namespace(
        archive=str(archive),
        citation="GBIF test citation",
        csv_block_size=32,
        doi="https://doi.org/10.15468/test",
        download_url="https://example.test/gbif.zip",
        expected_records=expected_records,
        force=False,
        manifest=str(manifest),
        output=str(output),
        progress_interval=1,
        source_snapshot_version="gbif-test-snapshot",
    )


def test_streaming_dwca_conversion_preserves_rows_and_adds_provenance(tmp_path) -> None:
    archive = tmp_path / "gbif.zip"
    output = tmp_path / "occurrences.parquet"
    manifest = tmp_path / "manifest.json"
    _write_dwca(
        archive,
        [
            ("100", "HUMAN_OBSERVATION", "dataset-a", "true", "-33.8", "151.2"),
            ("101", "PRESERVED_SPECIMEN", "dataset-b", "false", "", ""),
        ],
    )

    result = run(_args(archive, output, manifest, expected_records=2))

    frame = pl.read_parquet(output)
    assert frame.shape == (2, len(_COLUMNS) + 3)
    assert frame.get_column("key").to_list() == ["100", "101"]
    assert frame.get_column("source").to_list() == ["GBIF", "GBIF"]
    assert frame.get_column("sourceSnapshotVersion").to_list() == [
        "gbif-test-snapshot",
        "gbif-test-snapshot",
    ]
    assert result["output"]["row_count"] == 2
    assert result["output"]["physical_bytes"] == output.stat().st_size
    assert result["source"]["archive_bytes"] == archive.stat().st_size


def test_failed_row_count_does_not_replace_existing_parquet(tmp_path) -> None:
    archive = tmp_path / "gbif.zip"
    output = tmp_path / "occurrences.parquet"
    manifest = tmp_path / "manifest.json"
    _write_dwca(
        archive,
        [("100", "HUMAN_OBSERVATION", "dataset-a", "true", "-33.8", "151.2")],
    )
    pl.DataFrame({"sentinel": ["keep"]}).write_parquet(output)

    with pytest.raises(ValueError, match="does not match expected"):
        run(_args(archive, output, manifest, expected_records=2))

    assert pl.read_parquet(output).to_dicts() == [{"sentinel": "keep"}]
