"""Stream selected Darwin Core Archive members to Parquet.

This intentionally reads directly from the ZIP archive.  It does not require
extracting the 100+ GB text files first, and it keeps memory bounded by the CSV
block size.  The same Python command works on Linux/WSL2 and macOS.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
from typing import BinaryIO
from uuid import uuid4
from zipfile import ZipFile


DEFAULT_MEMBERS = ("occurrence.txt", "multimedia.txt", "verbatim.txt")
NULL_MARKERS = ["", "NULL", "null", "NA", "na", "N/A", "n/a"]
DEFAULT_CSV_BLOCK_SIZE = 32 * 1024 * 1024
DEFAULT_PROGRESS_INTERVAL = 1_000_000
SCHEMA_VERSION = "biominer-dwca-members-parquet-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_columns(archive: ZipFile, member_name: str) -> list[str]:
    try:
        with archive.open(member_name) as member:
            header = member.readline()
    except KeyError as error:
        raise ValueError(f"DWCA archive is missing {member_name}") from error
    columns = header.decode("utf-8-sig").rstrip("\r\n").split("\t")
    if not columns or not all(columns) or len(columns) != len(set(columns)):
        raise ValueError(f"{member_name} has an invalid column header")
    return columns


def _temporary_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.{uuid4().hex}.staging")


def _convert_member(
    archive_path: Path,
    member_name: str,
    output: Path,
    *,
    block_size: int,
    progress_interval: int,
    overwrite: bool,
) -> dict[str, object]:
    """Convert one ZIP member atomically, keeping every DWCA field as string."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")
    if block_size <= 0:
        raise ValueError("csv block size must be positive")
    if progress_interval <= 0:
        raise ValueError("progress interval must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _temporary_path(output)
    rows_written = 0
    writer: pq.ParquetWriter | None = None
    try:
        with ZipFile(archive_path) as archive:
            columns = _member_columns(archive, member_name)
            column_types = {column: pa.string() for column in columns}
            try:
                raw_member: BinaryIO = archive.open(member_name)
            except KeyError as error:
                raise ValueError(f"DWCA archive is missing {member_name}") from error
            with raw_member, staging.open("wb") as destination:
                # Consume the header ourselves so that pyarrow receives a stable,
                # explicit all-string schema for every record batch.
                raw_member.readline()
                reader = pacsv.open_csv(
                    raw_member,
                    read_options=pacsv.ReadOptions(
                        block_size=block_size,
                        column_names=columns,
                        use_threads=True,
                    ),
                    parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                    convert_options=pacsv.ConvertOptions(
                        column_types=column_types,
                        null_values=NULL_MARKERS,
                        strings_can_be_null=True,
                    ),
                )
                for batch in reader:
                    table = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                    writer.write_table(table)
                    rows_written += table.num_rows
                    if rows_written % progress_interval < table.num_rows:
                        print(
                            json.dumps(
                                {
                                    "event": "dwca_member_to_parquet_progress",
                                    "member": member_name,
                                    "rows_written": rows_written,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                if writer is None:
                    empty_schema = pa.schema([(column, pa.string()) for column in columns])
                    writer = pq.ParquetWriter(destination, empty_schema, compression="zstd")
                writer.close()
                writer = None
        staging.replace(output)
    finally:
        if writer is not None:
            writer.close()
        staging.unlink(missing_ok=True)

    return {
        "member": member_name,
        "path": str(output),
        "row_count": rows_written,
        "column_count": len(columns),
        "physical_bytes": output.stat().st_size,
        "compression": "zstd",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    archive = Path(args.archive)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    output_dir = Path(args.output_dir)
    members = tuple(args.members)
    unknown_members = sorted(set(members) - set(DEFAULT_MEMBERS))
    if unknown_members:
        raise ValueError(f"Unsupported DWCA members: {', '.join(unknown_members)}")
    if len(members) != len(set(members)):
        raise ValueError("DWCA members must not be repeated")

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    outputs = [
        _convert_member(
            archive,
            member,
            output_dir / f"{Path(member).stem}.parquet",
            block_size=args.csv_block_size,
            progress_interval=args.progress_interval,
            overwrite=args.overwrite,
        )
        for member in members
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "producer": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "source": {
            "archive_path": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": _sha256_file(archive),
        },
        "outputs": outputs,
    }
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "dwca_parquet_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream DWCA occurrence, multimedia, and verbatim members to Zstandard Parquet.",
        epilog=(
            "Example: python3 scripts/build_dwca_members_parquet.py "
            "--archive data/reference/download.zip --output-dir data/reference/dwca_parquet"
        ),
    )
    parser.add_argument("--archive", required=True, help="Path to the DWCA ZIP archive")
    parser.add_argument("--output-dir", required=True, help="Directory for *.parquet outputs")
    parser.add_argument(
        "--members",
        nargs="+",
        choices=DEFAULT_MEMBERS,
        default=DEFAULT_MEMBERS,
        help="DWCA members to convert (default: all three)",
    )
    parser.add_argument("--manifest", help="Output manifest path (default: <output-dir>/dwca_parquet_manifest.json)")
    parser.add_argument(
        "--csv-block-size",
        type=int,
        default=DEFAULT_CSV_BLOCK_SIZE,
        help="Maximum CSV parser block in bytes (default: 32 MiB)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Emit progress after this many rows (default: 1,000,000)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing parquet outputs")
    return parser


def main() -> int:
    args = _parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
