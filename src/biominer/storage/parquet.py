from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import polars as pl


CANONICAL_BUCKETED_RECORDS = "bucketed_records.parquet"
DEFAULT_PARQUET_COMPRESSION = "zstd"
BUCKET_VIEW_FILES = {
    "gold": "gold_records.parquet",
    "silver": "silver_records.parquet",
    "bronze": "bronze_records.parquet",
    "bin": "bin_records.parquet",
}


@dataclass(frozen=True)
class ParquetPartWrite:
    uri: str
    row_count: int
    byte_count: int | None
    compression: str | None


def write_parquet(
    frame: pl.DataFrame,
    path: str | Path,
    *,
    compression: str | None = DEFAULT_PARQUET_COMPRESSION,
    overwrite: bool = True,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and output.exists():
        raise FileExistsError(output)
    tmp = _temporary_output_path(output)
    try:
        _write_frame(frame, tmp, compression=compression)
        if not overwrite and output.exists():
            raise FileExistsError(output)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return output


def write_parquet_batches(
    batches: Iterable[pl.DataFrame],
    path: str | Path,
    *,
    compression: str | None = DEFAULT_PARQUET_COMPRESSION,
    overwrite: bool = True,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and output.exists():
        raise FileExistsError(output)
    tmp = _temporary_output_path(output)
    writer = None
    wrote_any = False
    try:
        with ExitStack() as stack:
            for frame in batches:
                if frame.is_empty():
                    continue
                table = frame.to_arrow()
                if writer is None:
                    import pyarrow.parquet as pq

                    stream = stack.enter_context(tmp.open("wb"))
                    writer = pq.ParquetWriter(stream, table.schema, compression=compression)
                writer.write_table(table)
                wrote_any = True
            if writer is not None:
                writer.close()
                writer = None
        if not wrote_any:
            _write_frame(pl.DataFrame(), tmp, compression=compression)
        if not overwrite and output.exists():
            raise FileExistsError(output)
        tmp.replace(output)
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
    return output


def write_parquet_part(
    frame: pl.DataFrame,
    path: str | Path,
    *,
    compression: str | None = DEFAULT_PARQUET_COMPRESSION,
    overwrite: bool = False,
) -> ParquetPartWrite:
    output = write_parquet(frame, path, compression=compression, overwrite=overwrite)
    return ParquetPartWrite(
        uri=str(output),
        row_count=frame.height,
        byte_count=output.stat().st_size if output.exists() else None,
        compression=compression,
    )


def write_bucket_views(frame: pl.DataFrame, output_dir: str | Path) -> dict[str, str]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for bucket, filename in BUCKET_VIEW_FILES.items():
        path = base / filename
        view = frame.filter(pl.col("occurrence_bin") == bucket) if "occurrence_bin" in frame.columns else pl.DataFrame()
        write_parquet(view, path)
        outputs[bucket] = str(path)
    return outputs


def _write_frame(frame: pl.DataFrame, path: Path, *, compression: str | None) -> None:
    if compression is None:
        frame.write_parquet(path)
        return
    frame.write_parquet(path, compression=compression)


def _temporary_output_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.{uuid4().hex}.tmp")
