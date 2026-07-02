from __future__ import annotations

from typing import Any
import json
from pathlib import Path

import polars as pl

from biominer.storage.parquet import write_parquet
from biominer.storage.uri import normalize_local_uri


class LocalStorageBackend:
    def __init__(self, *, prefix: str | Path = ".") -> None:
        self.prefix = str(prefix)

    def read_parquet(self, uri: str | Path) -> pl.DataFrame:
        return pl.read_parquet(normalize_local_uri(uri))

    def scan_parquet(self, uri: str | Path) -> pl.LazyFrame:
        return pl.scan_parquet(normalize_local_uri(uri))

    def write_parquet_shard(self, uri: str | Path, frame: pl.DataFrame) -> str:
        output = normalize_local_uri(uri)
        write_parquet(frame, output)
        return _preserve_uri_string(uri)

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

    def exists(self, uri: str | Path) -> bool:
        return normalize_local_uri(uri).exists()


def _preserve_uri_string(uri: str | Path) -> str:
    return str(uri)
