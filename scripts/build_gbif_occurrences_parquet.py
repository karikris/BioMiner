from __future__ import annotations

import argparse
from datetime import UTC, datetime
from io import BytesIO
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from urllib import request

import polars as pl

from biominer.storage.parquet import write_parquet


_DEFAULT_URL = "https://api.gbif.org/v1/occurrence/download/request/0004170-260715120105164.zip"
_DEFAULT_DOI = "https://doi.org/10.15468/dl.7uut3k"
_DEFAULT_CITATION = (
    "GBIF.org (18 July 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.7uut3k"
)
_SCHEMA_VERSION = "biominer-gbif-occurrence-parquet-v1"
_ARCHIVE_MEMBER = "occurrence.txt"
_NULL_MARKERS = ["", "NULL", "null", "NA", "na", "N/A", "n/a"]


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


def _read_occurrence_csv(payload: bytes, member_name: str) -> pl.DataFrame:
    # First try Polars fast-path. Some DWCA rows contain literal quotes in text fields,
    # so we avoid standard quote handling to keep the parser tolerant.
    reader_options: list[dict[str, object]] = [
        {
            "separator": "\t",
            "quote_char": "\x00",
            "null_values": _NULL_MARKERS,
            "ignore_errors": False,
            "truncate_ragged_lines": True,
        },
    ]
    for options in reader_options:
        with BytesIO(payload) as handle:
            try:
                return pl.read_csv(handle, **options)
            except Exception:
                continue
    try:
        import pyarrow.csv as pacsv

        with BytesIO(payload) as handle:
            table = pacsv.read_csv(
                handle,
                parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                convert_options=pacsv.ConvertOptions(
                    null_values=_NULL_MARKERS,
                    strings_can_be_null=True,
                ),
            )
        return pl.from_arrow(table)
    except Exception as error:
        raise ValueError(
            f"failed to parse {member_name} as a tab-delimited DWCA table"
        ) from error


def _read_occurrence_frame(path: Path) -> pl.DataFrame:
    with ZipFile(path) as archive:
        try:
            payload = archive.read(_ARCHIVE_MEMBER)
        except KeyError as error:
            raise ValueError(f"DWCA archive is missing {_ARCHIVE_MEMBER}") from error

    frame = _read_occurrence_csv(payload, _ARCHIVE_MEMBER)
    if "key" not in frame.columns:
        if "gbifID" in frame.columns:
            frame = frame.with_columns(pl.col("gbifID").alias("key"))
        else:
            raise ValueError("DWCA occurrence frame is missing both gbifID and key")
    return frame.with_columns([pl.col(name).cast(pl.Utf8, strict=False) for name in frame.columns])


def _validate_frame(
    frame: pl.DataFrame,
    *,
    expected_records: int | None,
) -> None:
    if frame.height == 0:
        raise ValueError("DWCA occurrence frame is empty")
    if expected_records is not None and frame.height != expected_records:
        raise ValueError(
            f"DWCA occurrence row count {frame.height} does not match expected {expected_records}"
        )
    required = (
        "basisOfRecord",
        "datasetKey",
        "hasCoordinate",
        "decimalLatitude",
        "decimalLongitude",
    )
    for field in required:
        if field not in frame.columns:
            raise ValueError(f"DWCA occurrence frame is missing required column {field}")


def run(args: argparse.Namespace) -> dict[str, object]:
    archive_path = Path(args.archive)
    downloaded_at = datetime.now(UTC)
    _fetch_archive(args.download_url, archive_path, force=args.force)
    frame = _read_occurrence_frame(archive_path).sort("key")
    _validate_frame(frame, expected_records=args.expected_records)
    if "source" not in frame.columns:
        frame = frame.with_columns(pl.lit("GBIF").alias("source"))
    if "sourceSnapshotVersion" not in frame.columns:
        frame = frame.with_columns(pl.lit(args.source_snapshot_version).alias("sourceSnapshotVersion"))

    output = Path(args.output)
    written = write_parquet(frame, output, compression="zstd")

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": downloaded_at.isoformat().replace("+00:00", "Z"),
        "source": {
            "download_url": args.download_url,
            "doi": _DEFAULT_DOI,
            "citation": _DEFAULT_CITATION,
            "archive_path": str(archive_path),
            "archive_sha256": _sha256_file(archive_path),
            "archive_bytes": archive_path.stat().st_size,
        },
        "output": {
            "path": str(output),
            "row_count": frame.height,
            "column_count": len(frame.columns),
            "physical_bytes": written.stat().st_size,
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
    parser.add_argument(
        "--source-snapshot-version",
        default="gbif-papilionoidea-australia-2026-07-18",
    )
    parser.add_argument("--expected-records", type=int, default=571_755)
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
