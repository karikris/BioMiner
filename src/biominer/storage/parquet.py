from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import polars as pl


DEFAULT_PARQUET_COMPRESSION = "zstd"
DEFAULT_PARQUET_READ_BATCH_SIZE = 1_000


@dataclass(frozen=True)
class ParquetPartWrite:
    uri: str
    row_count: int
    byte_count: int | None
    compression: str | None


@dataclass(frozen=True)
class ParquetRowSource:
    """Reusable, bounded row iterator over a Parquet file."""

    path: str | Path
    batch_size: int = DEFAULT_PARQUET_READ_BATCH_SIZE

    def __iter__(self) -> Iterator[dict[str, object]]:
        for batch in iter_parquet_batches(self.path, batch_size=self.batch_size):
            yield from batch.iter_rows(named=True)


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
    schema: dict[str, pl.DataType] | None = None,
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
                if schema is not None:
                    frame = _frame_with_schema(frame, schema)
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
            _write_frame(pl.DataFrame(schema=schema or {}), tmp, compression=compression)
        if not overwrite and output.exists():
            raise FileExistsError(output)
        tmp.replace(output)
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - preserve the original write failure.
                pass
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


def iter_parquet_batches(
    path: str | Path,
    *,
    batch_size: int = DEFAULT_PARQUET_READ_BATCH_SIZE,
) -> Iterator[pl.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield pl.from_arrow(batch)
    finally:
        parquet_file.close()


def _write_frame(frame: pl.DataFrame, path: Path, *, compression: str | None) -> None:
    if compression is None:
        frame.write_parquet(path)
        return
    frame.write_parquet(path, compression=compression)


def _frame_with_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    missing = [name for name in schema if name not in frame.columns]
    if missing:
        frame = frame.with_columns([pl.lit(None, dtype=schema[name]).alias(name) for name in missing])
    return frame.select(list(schema)).cast(schema)


def _temporary_output_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.{uuid4().hex}.tmp")
