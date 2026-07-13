from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import polars as pl

from biominer.storage.parquet import (
    DEFAULT_PARQUET_COMPRESSION,
    DEFAULT_PARQUET_READ_BATCH_SIZE,
    ParquetPartWrite,
)


@runtime_checkable
class CloudStorage(Protocol):
    def read_parquet(self, uri: str) -> pl.DataFrame: ...

    def scan_parquet(self, uri: str) -> pl.LazyFrame: ...

    def iter_parquet_batches(
        self,
        uri: str,
        *,
        batch_size: int = DEFAULT_PARQUET_READ_BATCH_SIZE,
    ) -> Iterator[pl.DataFrame]: ...

    def write_parquet_shard(
        self,
        uri: str,
        frame: pl.DataFrame,
        *,
        compression: str | None = DEFAULT_PARQUET_COMPRESSION,
        overwrite: bool = True,
    ) -> str: ...

    def write_parquet_batches(
        self,
        uri: str,
        batches: Iterable[pl.DataFrame],
        *,
        compression: str | None = DEFAULT_PARQUET_COMPRESSION,
        overwrite: bool = True,
    ) -> str: ...

    def write_parquet_part(
        self,
        uri: str,
        frame: pl.DataFrame,
        *,
        compression: str | None = DEFAULT_PARQUET_COMPRESSION,
        overwrite: bool = False,
    ) -> ParquetPartWrite: ...

    def list_shards(self, prefix: str) -> list[str]: ...

    def write_file(
        self,
        uri: str,
        source: str | Path,
        *,
        content_type: str | None = None,
        overwrite: bool = True,
    ) -> str: ...

    def materialize_file(
        self,
        uri: str,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> str: ...

    def write_json(self, uri: str, payload: dict[str, Any]) -> str: ...

    def read_json(self, uri: str) -> dict[str, Any]: ...

    def write_text(self, uri: str, text: str, *, encoding: str = "utf-8") -> str: ...

    def read_text(self, uri: str, *, encoding: str = "utf-8") -> str: ...

    def delete(self, uri: str) -> bool: ...

    def exists(self, uri: str) -> bool: ...

    def file_size(self, uri: str) -> int: ...

    def file_sha256(self, uri: str) -> str: ...
