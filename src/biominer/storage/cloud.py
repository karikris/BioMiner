from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class CloudStorage(Protocol):
    def read_parquet(self, uri: str) -> pl.DataFrame:
        ...

    def scan_parquet(self, uri: str) -> pl.LazyFrame:
        ...

    def write_parquet_shard(self, uri: str, frame: pl.DataFrame) -> str:
        ...

    def write_parquet_batches(self, uri: str, batches: Iterable[pl.DataFrame]) -> str:
        ...

    def list_shards(self, prefix: str) -> list[str]:
        ...

    def write_json(self, uri: str, payload: dict[str, Any]) -> str:
        ...

    def read_json(self, uri: str) -> dict[str, Any]:
        ...

    def delete(self, uri: str) -> bool:
        ...

    def exists(self, uri: str) -> bool:
        ...
