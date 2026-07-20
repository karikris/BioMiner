from __future__ import annotations

import argparse
import json
import zipfile

import pyarrow.parquet as pq
import pytest

from scripts.build_dwca_members_parquet import DEFAULT_MEMBERS, run


def _write_dwca(path) -> None:
    payloads = {
        "occurrence.txt": "gbifID\tscientificName\n1\tPapilio demoleus\n2\t\n",
        "multimedia.txt": "id\tidentifier\n1\thttps://example.test/1.jpg\n",
        "verbatim.txt": "gbifID\teventDate\n1\t2026-07-19\n",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in payloads.items():
            archive.writestr(member, payload)


def _args(archive, output_dir, *, overwrite: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        archive=str(archive),
        output_dir=str(output_dir),
        members=DEFAULT_MEMBERS,
        manifest=None,
        csv_block_size=32,
        progress_interval=1,
        overwrite=overwrite,
    )


def test_streams_all_dwca_members_to_zstd_parquet_and_writes_manifest(tmp_path) -> None:
    archive = tmp_path / "download.zip"
    output_dir = tmp_path / "parquet"
    _write_dwca(archive)

    manifest = run(_args(archive, output_dir))

    assert [output["member"] for output in manifest["outputs"]] == list(DEFAULT_MEMBERS)
    assert pq.read_table(output_dir / "occurrence.parquet").to_pylist() == [
        {"gbifID": "1", "scientificName": "Papilio demoleus"},
        {"gbifID": "2", "scientificName": None},
    ]
    assert pq.read_table(output_dir / "multimedia.parquet").to_pylist() == [
        {"id": "1", "identifier": "https://example.test/1.jpg"}
    ]
    assert pq.read_table(output_dir / "verbatim.parquet").to_pylist() == [
        {"gbifID": "1", "eventDate": "2026-07-19"}
    ]
    saved_manifest = json.loads((output_dir / "dwca_parquet_manifest.json").read_text())
    assert saved_manifest["source"]["archive_sha256"] == manifest["source"]["archive_sha256"]


def test_does_not_replace_completed_member_without_overwrite(tmp_path) -> None:
    archive = tmp_path / "download.zip"
    output_dir = tmp_path / "parquet"
    _write_dwca(archive)
    run(_args(archive, output_dir))

    with pytest.raises(FileExistsError, match="use --overwrite"):
        run(_args(archive, output_dir))
