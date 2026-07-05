from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from typing import Any
import json

import polars as pl

from biominer.storage.uri import is_s3_uri, join_uri


class S3StorageBackend:
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region: str = "auto",
    ) -> None:
        if not bucket:
            raise ValueError("bucket is required for S3-compatible storage")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.base_uri = join_uri(f"s3://{bucket}", self.prefix)

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return pl.read_parquet(self._open_input_file(uri))

    def scan_parquet(self, uri: str) -> pl.LazyFrame:
        return pl.scan_parquet(uri, storage_options=self._storage_options())

    def write_parquet_shard(self, uri: str, frame: pl.DataFrame) -> str:
        filesystem, path = self._filesystem_and_path(uri)
        with filesystem.open_output_stream(path) as stream:
            frame.write_parquet(stream)
        return uri

    def write_parquet_batches(self, uri: str, batches: Iterable[pl.DataFrame]) -> str:
        filesystem, path = self._filesystem_and_path(uri)
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

                        stream = stack.enter_context(filesystem.open_output_stream(path))
                        writer = pq.ParquetWriter(stream, table.schema, compression="zstd")
                    writer.write_table(table)
                    wrote_any = True
                if writer is not None:
                    writer.close()
                    writer = None
        finally:
            if writer is not None:
                writer.close()
        if not wrote_any:
            return self.write_parquet_shard(uri, pl.DataFrame())
        return uri

    def list_shards(self, prefix: str) -> list[str]:
        filesystem, path = self._filesystem_and_path(prefix)
        selector = self._pyarrow_fs().FileSelector(path, recursive=True)
        infos = filesystem.get_file_info(selector)
        return sorted(f"s3://{self.bucket}/{info.path}" for info in infos if info.is_file and info.path.endswith(".parquet"))

    def write_json(self, uri: str, payload: dict[str, Any]) -> str:
        filesystem, path = self._filesystem_and_path(uri)
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        with filesystem.open_output_stream(path) as stream:
            stream.write(encoded)
        return uri

    def read_json(self, uri: str) -> dict[str, Any]:
        filesystem, path = self._filesystem_and_path(uri)
        with filesystem.open_input_file(path) as stream:
            payload = json.loads(stream.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {uri}")
        return payload

    def delete(self, uri: str) -> bool:
        filesystem, path = self._filesystem_and_path(uri)
        if not self.exists(uri):
            return False
        filesystem.delete_file(path)
        return True

    def exists(self, uri: str) -> bool:
        filesystem, path = self._filesystem_and_path(uri)
        return filesystem.get_file_info(path).type != self._pyarrow_fs().FileType.NotFound

    def _filesystem_and_path(self, uri: str):
        bucket, key = _split_s3_uri(uri)
        if bucket != self.bucket:
            raise ValueError(f"S3 URI bucket {bucket!r} does not match configured bucket {self.bucket!r}")
        fs = self._pyarrow_fs().S3FileSystem(
            access_key=self.access_key_id,
            secret_key=self.secret_access_key,
            region=self.region,
            endpoint_override=self.endpoint_url,
        )
        return fs, f"{bucket}/{key}" if key else bucket

    def _open_input_file(self, uri: str):
        filesystem, path = self._filesystem_and_path(uri)
        return filesystem.open_input_file(path)

    def _storage_options(self) -> dict[str, str]:
        options: dict[str, str] = {}
        if self.endpoint_url:
            options["endpoint_url"] = self.endpoint_url
        if self.access_key_id:
            options["aws_access_key_id"] = self.access_key_id
        if self.secret_access_key:
            options["aws_secret_access_key"] = self.secret_access_key
        if self.region:
            options["region_name"] = self.region
        return options

    @staticmethod
    def _pyarrow_fs():
        try:
            import pyarrow.fs as pafs
        except ImportError as exc:  # pragma: no cover - pyarrow is a core dependency today.
            raise RuntimeError("pyarrow is required for S3-compatible storage") from exc
        return pafs


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not is_s3_uri(uri):
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    without_scheme = uri.removeprefix("s3://")
    bucket, _, key = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"missing bucket in S3 URI: {uri!r}")
    return bucket, key
