from __future__ import annotations

import argparse
from collections.abc import Iterator
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from urllib import request
from uuid import uuid4
from zipfile import ZipFile

import polars as pl

from biominer.storage.parquet import write_parquet_batches


_DEFAULT_URL = "https://api.gbif.org/v1/occurrence/download/request/0004170-260715120105164.zip"
_DEFAULT_DOI = "https://doi.org/10.15468/dl.7uut3k"
_DEFAULT_CITATION = (
    "GBIF.org (18 July 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.7uut3k"
)
_SCHEMA_VERSION = "biominer-gbif-occurrence-parquet-v1"
_ARCHIVE_MEMBER = "occurrence.txt"
_NULL_MARKERS = ["", "NULL", "null", "NA", "na", "N/A", "n/a"]
_REQUIRED_COLUMNS = (
    "basisOfRecord",
    "datasetKey",
    "hasCoordinate",
    "decimalLatitude",
    "decimalLongitude",
)
_DEFAULT_CSV_BLOCK_SIZE = 8 * 1024 * 1024
_DEFAULT_PROGRESS_INTERVAL = 1_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _fetch_archive(url: str, output: Path, *, force: bool) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        return output
    request.urlretrieve(url, output)
    return output


def _occurrence_columns(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        try:
            with archive.open(_ARCHIVE_MEMBER) as member:
                header = member.readline()
        except KeyError as error:
            raise ValueError(f"DWCA archive is missing {_ARCHIVE_MEMBER}") from error

    columns = header.decode("utf-8-sig").rstrip("\r\n").split("\t")
    if not columns or not all(columns) or len(set(columns)) != len(columns):
        raise ValueError(f"{_ARCHIVE_MEMBER} has an invalid column header")
    return columns


def _validate_columns(columns: list[str]) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"DWCA occurrence frame is missing required columns: {', '.join(missing)}")
    if "key" not in columns and "gbifID" not in columns:
        raise ValueError("DWCA occurrence frame is missing both gbifID and key")


def _validate_row_count(*, row_count: int, expected_records: int | None) -> None:
    if row_count == 0:
        raise ValueError("DWCA occurrence frame is empty")
    if expected_records is not None and row_count != expected_records:
        raise ValueError(
            f"DWCA occurrence row count {row_count} does not match expected {expected_records}"
        )


def _output_columns(columns: list[str]) -> list[str]:
    result = list(columns)
    for column in ("key", "source", "sourceSnapshotVersion"):
        if column not in result:
            result.append(column)
    return result


def _iter_occurrence_frames(
    path: Path,
    *,
    columns: list[str],
    source_snapshot_version: str,
    block_size: int,
    progress_interval: int,
    progress: dict[str, int],
) -> Iterator[pl.DataFrame]:
    """Read the DWCA member incrementally without materializing it in memory."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    if block_size <= 0:
        raise ValueError("csv block size must be positive")
    if progress_interval <= 0:
        raise ValueError("progress interval must be positive")

    column_types = {column: pa.string() for column in columns}
    with ZipFile(path) as archive:
        try:
            member = archive.open(_ARCHIVE_MEMBER)
        except KeyError as error:
            raise ValueError(f"DWCA archive is missing {_ARCHIVE_MEMBER}") from error
        with member:
            # The header was read separately to pin every field to string. This avoids
            # type drift across batches and mirrors the former all-string output.
            member.readline()
            reader = pacsv.open_csv(
                member,
                read_options=pacsv.ReadOptions(
                    block_size=block_size,
                    column_names=columns,
                    use_threads=True,
                ),
                parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                convert_options=pacsv.ConvertOptions(
                    column_types=column_types,
                    null_values=_NULL_MARKERS,
                    strings_can_be_null=True,
                ),
            )
            for batch in reader:
                table = pa.Table.from_batches([batch])
                if "key" not in columns:
                    table = table.append_column("key", table.column("gbifID"))
                if "source" not in columns:
                    table = table.append_column(
                        "source",
                        pa.array(["GBIF"] * table.num_rows, type=pa.string()),
                    )
                if "sourceSnapshotVersion" not in columns:
                    table = table.append_column(
                        "sourceSnapshotVersion",
                        pa.array(
                            [source_snapshot_version] * table.num_rows,
                            type=pa.string(),
                        ),
                    )
                progress["row_count"] += table.num_rows
                if progress["row_count"] // progress_interval > progress["reported_rows"] // progress_interval:
                    progress["reported_rows"] = progress["row_count"]
                    print(
                        json.dumps(
                            {
                                "event": "gbif_dwca_to_parquet_progress",
                                "rows_written": progress["row_count"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                yield pl.from_arrow(table)


def run(args: argparse.Namespace) -> dict[str, object]:
    archive_path = Path(args.archive)
    downloaded_at = datetime.now(UTC)
    _fetch_archive(args.download_url, archive_path, force=args.force)
    columns = _occurrence_columns(archive_path)
    _validate_columns(columns)
    output = Path(args.output)
    output_schema = {column: pl.String for column in _output_columns(columns)}
    staging = output.with_name(f".{output.name}.{uuid4().hex}.staging")
    progress = {"row_count": 0, "reported_rows": 0}
    try:
        write_parquet_batches(
            _iter_occurrence_frames(
                archive_path,
                columns=columns,
                source_snapshot_version=args.source_snapshot_version,
                block_size=args.csv_block_size,
                progress_interval=args.progress_interval,
                progress=progress,
            ),
            staging,
            compression="zstd",
            schema=output_schema,
            overwrite=False,
        )
        _validate_row_count(
            row_count=progress["row_count"],
            expected_records=args.expected_records,
        )
        staging.replace(output)
    finally:
        staging.unlink(missing_ok=True)

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": downloaded_at.isoformat().replace("+00:00", "Z"),
        "source": {
            "download_url": args.download_url,
            "doi": args.doi,
            "citation": args.citation,
            "archive_path": str(archive_path),
            "archive_sha256": _sha256_file(archive_path),
            "archive_bytes": archive_path.stat().st_size,
        },
        "output": {
            "path": str(output),
            "row_count": progress["row_count"],
            "column_count": len(output_schema),
            "physical_bytes": output.stat().st_size,
            "expected_row_count": args.expected_records,
            "source_snapshot_version": args.source_snapshot_version,
        },
    }
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build GBIF occurrence parquet from a DWCA zip"
    )
    parser.add_argument("--download-url", default=_DEFAULT_URL)
    parser.add_argument("--output", default="data/reference/gbif_occurrences.parquet")
    parser.add_argument("--archive", default="data/reference/gbif-occurrence-download.zip")
    parser.add_argument("--manifest", default="data/reference/gbif_occurrence_manifest.json")
    parser.add_argument("--doi", default=_DEFAULT_DOI)
    parser.add_argument("--citation", default=_DEFAULT_CITATION)
    parser.add_argument(
        "--source-snapshot-version",
        default="gbif-papilionoidea-australia-2026-07-18",
    )
    parser.add_argument("--expected-records", type=int, default=571_755)
    parser.add_argument("--csv-block-size", type=int, default=_DEFAULT_CSV_BLOCK_SIZE)
    parser.add_argument("--progress-interval", type=int, default=_DEFAULT_PROGRESS_INTERVAL)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = run(args)
    _write_json(Path(args.manifest), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
