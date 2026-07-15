from __future__ import annotations

from collections.abc import Iterable, Iterator
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import Any

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
from biominer.storage.content_address import (
    sha256_file,
    validate_content_addressed_uri,
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
        write_parquet_batches(
            batches, output, compression=compression, overwrite=overwrite
        )
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
        result = write_parquet_part(
            frame, output, compression=compression, overwrite=overwrite
        )
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

    def write_file(
        self,
        uri: str | Path,
        source: str | Path,
        *,
        content_type: str | None = None,
        overwrite: bool = True,
    ) -> str:
        output = normalize_local_uri(uri)
        source_path = normalize_local_uri(source)
        if not overwrite and output.exists():
            raise FileExistsError(uri)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with source_path.open("rb") as source_stream:
                with NamedTemporaryFile(
                    mode="wb",
                    dir=output.parent,
                    prefix=f".{output.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    copyfileobj(source_stream, temporary)
            if overwrite:
                temporary_path.replace(output)
            else:
                try:
                    os.link(temporary_path, output)
                except FileExistsError as exc:
                    raise FileExistsError(uri) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return _preserve_uri_string(uri)

    def materialize_file(
        self,
        uri: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> str:
        source = normalize_local_uri(uri)
        output = normalize_local_uri(destination)
        if not source.is_file():
            raise FileNotFoundError(uri)
        if not overwrite and output.exists():
            raise FileExistsError(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with source.open("rb") as source_stream:
                with NamedTemporaryFile(
                    mode="wb",
                    dir=output.parent,
                    prefix=f".{output.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    copyfileobj(source_stream, temporary)
            if overwrite:
                temporary_path.replace(output)
            else:
                try:
                    os.link(temporary_path, output)
                except FileExistsError as exc:
                    raise FileExistsError(destination) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return str(destination)

    def write_content_addressed_file(
        self,
        uri: str | Path,
        source: str | Path,
        *,
        expected_sha256: str,
        content_type: str | None = None,
    ) -> str:
        _ = content_type
        normalized = validate_content_addressed_uri(uri, expected_sha256)
        source_path = normalize_local_uri(source)
        if sha256_file(source_path) != normalized:
            raise ValueError("local source SHA-256 does not match content address")
        return self.write_file(uri, source_path, overwrite=True)

    def materialize_content_addressed_file(
        self,
        uri: str | Path,
        destination: str | Path,
        *,
        expected_sha256: str,
        overwrite: bool = False,
    ) -> str:
        normalized = validate_content_addressed_uri(uri, expected_sha256)
        source = normalize_local_uri(uri)
        output = normalize_local_uri(destination)
        if not source.is_file():
            raise FileNotFoundError(uri)
        if not overwrite and output.exists():
            raise FileExistsError(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with source.open("rb") as source_stream:
                with NamedTemporaryFile(
                    mode="wb",
                    dir=output.parent,
                    prefix=f".{output.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    copyfileobj(source_stream, temporary)
            if sha256_file(temporary_path) != normalized:
                raise OSError(
                    "materialized content-addressed file failed local SHA-256"
                )
            if overwrite:
                temporary_path.replace(output)
            else:
                try:
                    os.link(temporary_path, output)
                except FileExistsError as exc:
                    raise FileExistsError(destination) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return str(destination)

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

    def file_size(self, uri: str | Path) -> int:
        path = normalize_local_uri(uri)
        if not path.is_file():
            raise FileNotFoundError(uri)
        return path.stat().st_size

    def file_sha256(self, uri: str | Path) -> str:
        path = normalize_local_uri(uri)
        if not path.is_file():
            raise FileNotFoundError(uri)
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"


def _preserve_uri_string(uri: str | Path) -> str:
    return str(uri)
