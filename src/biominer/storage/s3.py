from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from shutil import copyfileobj
import threading
from typing import Any
import json
from uuid import uuid4

import polars as pl

from biominer.storage.parquet import (
    DEFAULT_PARQUET_COMPRESSION,
    DEFAULT_PARQUET_READ_BATCH_SIZE,
    ParquetPartWrite,
)
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
        self._filesystem_instance = None
        self._filesystem_lock = threading.Lock()

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return pl.read_parquet(self._open_input_file(uri))

    def scan_parquet(self, uri: str) -> pl.LazyFrame:
        return pl.scan_parquet(uri, storage_options=self._storage_options())

    def iter_parquet_batches(
        self,
        uri: str,
        *,
        batch_size: int = DEFAULT_PARQUET_READ_BATCH_SIZE,
    ) -> Iterator[pl.DataFrame]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        import pyarrow.parquet as pq

        with self._open_input_file(uri) as stream:
            parquet_file = pq.ParquetFile(stream)
            try:
                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    yield pl.from_arrow(batch)
            finally:
                parquet_file.close()

    def write_parquet_shard(
        self,
        uri: str,
        frame: pl.DataFrame,
        *,
        compression: str | None = DEFAULT_PARQUET_COMPRESSION,
        overwrite: bool = True,
    ) -> str:
        filesystem, path = self._filesystem_and_path(uri)
        if not overwrite and self._path_exists(filesystem, path):
            raise FileExistsError(uri)
        staging_path = _staging_path(path)
        try:
            with filesystem.open_output_stream(staging_path) as stream:
                _write_frame(frame, stream, compression=compression)
            size, digest = _parquet_object_metrics(filesystem, staging_path)
            self._promote_staged_object(
                filesystem=filesystem,
                staging_path=staging_path,
                path=path,
                expected_size=size,
                expected_sha256=digest,
                overwrite=overwrite,
                uri=uri,
                staging_verified=True,
            )
        finally:
            self._delete_path_if_present(filesystem, staging_path)
        return uri

    def write_parquet_batches(
        self,
        uri: str,
        batches: Iterable[pl.DataFrame],
        *,
        compression: str | None = DEFAULT_PARQUET_COMPRESSION,
        overwrite: bool = True,
    ) -> str:
        filesystem, path = self._filesystem_and_path(uri)
        if not overwrite and self._path_exists(filesystem, path):
            raise FileExistsError(uri)
        staging_path = _staging_path(path)
        try:
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

                            stream = stack.enter_context(
                                filesystem.open_output_stream(staging_path)
                            )
                            writer = pq.ParquetWriter(
                                stream, table.schema, compression=compression
                            )
                        writer.write_table(table)
                        wrote_any = True
                    if writer is not None:
                        writer.close()
                        writer = None
            finally:
                if writer is not None:
                    writer.close()
            if not wrote_any:
                with filesystem.open_output_stream(staging_path) as stream:
                    _write_frame(pl.DataFrame(), stream, compression=compression)
            size, digest = _parquet_object_metrics(filesystem, staging_path)
            self._promote_staged_object(
                filesystem=filesystem,
                staging_path=staging_path,
                path=path,
                expected_size=size,
                expected_sha256=digest,
                overwrite=overwrite,
                uri=uri,
                staging_verified=True,
            )
        finally:
            self._delete_path_if_present(filesystem, staging_path)
        return uri

    def write_parquet_part(
        self,
        uri: str,
        frame: pl.DataFrame,
        *,
        compression: str | None = DEFAULT_PARQUET_COMPRESSION,
        overwrite: bool = False,
    ) -> ParquetPartWrite:
        self.write_parquet_shard(
            uri, frame, compression=compression, overwrite=overwrite
        )
        byte_count = None
        try:
            filesystem, path = self._filesystem_and_path(uri)
            size = getattr(filesystem.get_file_info(path), "size", None)
            byte_count = int(size) if size is not None and int(size) >= 0 else None
        except Exception:  # noqa: BLE001 - byte size is best-effort for remote stores.
            byte_count = None
        return ParquetPartWrite(
            uri=uri,
            row_count=frame.height,
            byte_count=byte_count,
            compression=compression,
        )

    def list_shards(self, prefix: str) -> list[str]:
        if str(prefix).endswith(".parquet"):
            return [str(prefix)] if self.exists(prefix) else []
        filesystem, path = self._filesystem_and_path(prefix)
        selector = self._pyarrow_fs().FileSelector(path, recursive=True)
        try:
            infos = filesystem.get_file_info(selector)
        except FileNotFoundError:
            return []
        return sorted(
            _file_info_s3_uri(self.bucket, info.path)
            for info in infos
            if info.is_file and info.path.endswith(".parquet")
        )

    def write_file(
        self,
        uri: str,
        source: str | Path,
        *,
        content_type: str | None = None,
        overwrite: bool = True,
    ) -> str:
        filesystem, path = self._filesystem_and_path(uri)
        if not overwrite and self._path_exists(filesystem, path):
            raise FileExistsError(uri)
        metadata = {"Content-Type": content_type} if content_type else None
        source_path = Path(source)
        source_size = source_path.stat().st_size
        with source_path.open("rb") as source_stream:
            source_sha256 = _stream_sha256(source_stream)
        staging_path = _staging_path(path)
        try:
            with source_path.open("rb") as source_stream:
                with filesystem.open_output_stream(
                    staging_path,
                    compression=None,
                    metadata=metadata,
                ) as output_stream:
                    copyfileobj(source_stream, output_stream)
            self._promote_staged_object(
                filesystem=filesystem,
                staging_path=staging_path,
                path=path,
                expected_size=source_size,
                expected_sha256=source_sha256,
                overwrite=overwrite,
                uri=uri,
            )
        finally:
            self._delete_path_if_present(filesystem, staging_path)
        return uri

    def write_json(self, uri: str, payload: dict[str, Any]) -> str:
        filesystem, path = self._filesystem_and_path(uri)
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        staging_path = _staging_path(path)
        try:
            with filesystem.open_output_stream(
                staging_path,
                compression=None,
                metadata={"Content-Type": "application/json"},
            ) as stream:
                stream.write(encoded)
            self._promote_staged_object(
                filesystem=filesystem,
                staging_path=staging_path,
                path=path,
                expected_size=len(encoded),
                expected_sha256="sha256:" + sha256(encoded).hexdigest(),
                overwrite=True,
                uri=uri,
            )
        finally:
            self._delete_path_if_present(filesystem, staging_path)
        return uri

    def read_json(self, uri: str) -> dict[str, Any]:
        filesystem, path = self._filesystem_and_path(uri)
        with filesystem.open_input_file(path) as stream:
            payload = json.loads(stream.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {uri}")
        return payload

    def write_text(self, uri: str, text: str, *, encoding: str = "utf-8") -> str:
        filesystem, path = self._filesystem_and_path(uri)
        encoded = text.encode(encoding)
        staging_path = _staging_path(path)
        try:
            with filesystem.open_output_stream(
                staging_path,
                compression=None,
                metadata={"Content-Type": f"text/plain; charset={encoding}"},
            ) as stream:
                stream.write(encoded)
            self._promote_staged_object(
                filesystem=filesystem,
                staging_path=staging_path,
                path=path,
                expected_size=len(encoded),
                expected_sha256="sha256:" + sha256(encoded).hexdigest(),
                overwrite=True,
                uri=uri,
            )
        finally:
            self._delete_path_if_present(filesystem, staging_path)
        return uri

    def read_text(self, uri: str, *, encoding: str = "utf-8") -> str:
        filesystem, path = self._filesystem_and_path(uri)
        with filesystem.open_input_file(path) as stream:
            return stream.read().decode(encoding)

    def delete(self, uri: str) -> bool:
        filesystem, path = self._filesystem_and_path(uri)
        if not self.exists(uri):
            return False
        filesystem.delete_file(path)
        return True

    def exists(self, uri: str) -> bool:
        filesystem, path = self._filesystem_and_path(uri)
        return (
            filesystem.get_file_info(path).type != self._pyarrow_fs().FileType.NotFound
        )

    def file_size(self, uri: str) -> int:
        filesystem, path = self._filesystem_and_path(uri)
        info = filesystem.get_file_info(path)
        if info.type == self._pyarrow_fs().FileType.NotFound:
            raise FileNotFoundError(uri)
        size = int(info.size)
        if size < 0:
            raise OSError(f"object size is unavailable for {uri}")
        return size

    def file_sha256(self, uri: str) -> str:
        with self._open_input_file(uri) as stream:
            return _stream_sha256(stream)

    def _promote_staged_object(
        self,
        *,
        filesystem,
        staging_path: str,
        path: str,
        expected_size: int,
        expected_sha256: str,
        overwrite: bool,
        uri: str,
        staging_verified: bool = False,
    ) -> None:
        if not staging_verified and not self._object_matches(
            filesystem,
            staging_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        ):
            raise OSError(f"staged S3 object failed integrity validation: {uri}")
        if not overwrite and self._path_exists(filesystem, path):
            raise FileExistsError(uri)
        try:
            filesystem.move(staging_path, path)
        except Exception:
            if self._object_matches(
                filesystem,
                path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            ):
                return
            raise
        if not self._object_matches(
            filesystem,
            path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        ):
            raise OSError(f"promoted S3 object failed integrity validation: {uri}")

    def _object_matches(
        self,
        filesystem,
        path: str,
        *,
        expected_size: int,
        expected_sha256: str,
    ) -> bool:
        info = filesystem.get_file_info(path)
        if (
            info.type == self._pyarrow_fs().FileType.NotFound
            or int(info.size) != expected_size
        ):
            return False
        with filesystem.open_input_file(path) as stream:
            return _stream_sha256(stream) == expected_sha256

    def _path_exists(self, filesystem, path: str) -> bool:  # noqa: ANN001
        return (
            filesystem.get_file_info(path).type != self._pyarrow_fs().FileType.NotFound
        )

    def _delete_path_if_present(self, filesystem, path: str) -> None:  # noqa: ANN001
        try:
            if self._path_exists(filesystem, path):
                filesystem.delete_file(path)
        except Exception:  # noqa: BLE001 - cleanup must not mask the primary write.
            return

    def _filesystem_and_path(self, uri: str):
        bucket, key = _split_s3_uri(uri)
        if bucket != self.bucket:
            raise ValueError(
                f"S3 URI bucket {bucket!r} does not match configured bucket {self.bucket!r}"
            )
        fs = self._filesystem_instance
        if fs is None:
            with self._filesystem_lock:
                fs = self._filesystem_instance
                if fs is None:
                    fs = self._pyarrow_fs().S3FileSystem(
                        access_key=self.access_key_id,
                        secret_key=self.secret_access_key,
                        region=self.region,
                        endpoint_override=self.endpoint_url,
                    )
                    self._filesystem_instance = fs
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
        except (
            ImportError
        ) as exc:  # pragma: no cover - pyarrow is a core dependency today.
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


def _file_info_s3_uri(bucket: str, path: str) -> str:
    normalized = str(path).lstrip("/")
    bucket_prefix = f"{bucket}/"
    if normalized.startswith(bucket_prefix):
        normalized = normalized.removeprefix(bucket_prefix)
    return f"s3://{bucket}/{normalized}"


def _write_frame(frame: pl.DataFrame, stream, *, compression: str | None) -> None:  # noqa: ANN001
    if compression is None:
        frame.write_parquet(stream)
        return
    frame.write_parquet(stream, compression=compression)


def _stream_sha256(stream) -> str:  # noqa: ANN001
    digest = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _parquet_object_metrics(filesystem, path: str) -> tuple[int, str]:  # noqa: ANN001
    import pyarrow.parquet as pq

    info = filesystem.get_file_info(path)
    size = int(info.size)
    if size <= 0:
        raise OSError("staged Parquet object is empty")
    with filesystem.open_input_file(path) as stream:
        parquet_file = pq.ParquetFile(stream)
        try:
            _ = parquet_file.metadata.num_rows
        finally:
            parquet_file.close()
    with filesystem.open_input_file(path) as stream:
        digest = _stream_sha256(stream)
    return size, digest


def _staging_path(path: str) -> str:
    return f"{path}.biominer-staging-{uuid4().hex}"
