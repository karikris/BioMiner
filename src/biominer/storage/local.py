from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any
import json
from pathlib import Path

import polars as pl

from biominer.storage.parquet import (
    DEFAULT_PARQUET_COMPRESSION,
    DEFAULT_PARQUET_READ_BATCH_SIZE,
    ParquetPartWrite,
    iter_parquet_batches,
    write_parquet,
    write_parquet_batches,
    write_parquet_part,
)
from biominer.storage.uri import normalize_local_uri


class LocalStorageBackend:
    def __init__(self, *, prefix: str | Path = ".") -> None:
        self.prefix = str(prefix)

    def read_parquet(self, uri: str | Path) -> pl.DataFrame:
        return pl.read_parquet(normalize_local_uri(uri))

    def scan_parquet(self, uri: str | Path) -> pl.LazyFrame:
        return pl.scan_parquet(normalize_local_uri(uri))

    def iter_parquet_batches(
        self,
        uri: str | Path,
        *,
        batch_size: int = DEFAULT_PARQUET_READ_BATCH_SIZE,
    ) -> Iterator[pl.DataFrame]:
        yield from iter_parquet_batches(normalize_local_uri(uri), batch_size=batch_size)

    def write_parquet_shard(
        self,
        uri: str | Path,
        frame: pl.DataFrame,
        *,
        compression: str | None = DEFAULT_PARQUET_COMPRESSION,
        overwrite: bool = True,
    ) -> str:
        output = normalize_local_uri(uri)
        write_parquet(frame, output, compression=compression, overwrite=overwrite)
        return _preserve_uri_string(uri)

    def write_parquet_batches(
        self,
        uri: str | Path,
        batches: Iterable[pl.DataFrame],
        *,
        compression: str | None = DEFAULT_PARQUET_COMPRESSION,
        overwrite: bool = True,
    ) -> str:
        output = normalize_local_uri(uri)
        write_parquet_batches(batches, output, compression=compression, overwrite=overwrite)
        return _preserve_uri_string(uri)

    def write_parquet_part(
        self,
        uri: str | Path,
        frame: pl.DataFrame,
        *,
        compression: str | None = DEFAULT_PARQUET_COMPRESSION,
        overwrite: bool = False,
    ) -> ParquetPartWrite:
        output = normalize_local_uri(uri)
        result = write_parquet_part(frame, output, compression=compression, overwrite=overwrite)
        return ParquetPartWrite(
            uri=_preserve_uri_string(uri),
            row_count=result.row_count,
            byte_count=result.byte_count,
            compression=result.compression,
        )

    def list_shards(self, prefix: str | Path) -> list[str]:
        root = normalize_local_uri(prefix)
        if root.is_file() and root.suffix == ".parquet":
            return [str(root)]
        if not root.exists():
            return []
        return sorted(str(path) for path in root.rglob("*.parquet") if path.is_file())

    def write_json(self, uri: str | Path, payload: dict[str, Any]) -> str:
        output = normalize_local_uri(uri)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(output)
        return _preserve_uri_string(uri)

    def read_json(self, uri: str | Path) -> dict[str, Any]:
        payload = json.loads(normalize_local_uri(uri).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {uri}")
        return payload

    def write_text(self, uri: str | Path, text: str, *, encoding: str = "utf-8") -> str:
        output = normalize_local_uri(uri)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(text, encoding=encoding)
        tmp.replace(output)
        return _preserve_uri_string(uri)

    def read_text(self, uri: str | Path, *, encoding: str = "utf-8") -> str:
        return normalize_local_uri(uri).read_text(encoding=encoding)

    def delete(self, uri: str | Path) -> bool:
        path = normalize_local_uri(uri)
        if not path.exists():
            return False
        path.unlink()
        return True

    def exists(self, uri: str | Path) -> bool:
        return normalize_local_uri(uri).exists()


def _preserve_uri_string(uri: str | Path) -> str:
    return str(uri)
