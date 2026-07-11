from __future__ import annotations

import io
from pathlib import Path
import tempfile

import polars as pl
import pytest

import biominer.storage.s3 as s3_module
from biominer.storage.cloud import CloudStorage
from biominer.storage.local import LocalStorageBackend
from biominer.storage.parquet import write_parquet_batches, write_parquet_part
from biominer.storage.paths import (
    build_evidence_shard_uri,
    build_raw_flickr_response_uri,
    build_registry_current_pointer,
    build_registry_current_uri,
    build_registry_version_uri,
    build_report_uri,
    safe_path_component,
)
from biominer.storage.shard_paths import build_parquet_part_uri, build_parquet_shard_uri
from biominer.storage.s3 import S3StorageBackend
from biominer.storage.uri import is_cloud_uri, is_s3_uri, join_uri, normalize_local_uri


def test_local_storage_writes_and_reads_json(tmp_path) -> None:
    storage = LocalStorageBackend()
    target = tmp_path / "reports" / "manifest.json"

    written = storage.write_json(target, {"run_id": "r1", "rows": 3})

    assert written == str(target)
    assert storage.exists(target)
    assert storage.read_json(target) == {"run_id": "r1", "rows": 3}


def test_local_storage_accepts_file_uris(tmp_path) -> None:
    storage = LocalStorageBackend()
    target = (tmp_path / "manifest.json").as_uri()

    written = storage.write_json(target, {"ok": True})

    assert written == target
    assert storage.exists(target)
    assert storage.read_json(target) == {"ok": True}


def test_local_storage_writes_and_reads_text(tmp_path) -> None:
    storage = LocalStorageBackend()
    target = tmp_path / "reports" / "summary.md"

    written = storage.write_text(target, "# Summary\n")

    assert written == str(target)
    assert storage.exists(target)
    assert storage.read_text(target) == "# Summary\n"


def test_local_storage_writes_reads_and_scans_parquet_shards(tmp_path) -> None:
    storage = LocalStorageBackend()
    frame = pl.DataFrame({"photo_id": ["1", "2"], "score": [0.7, 0.4]})
    target = tmp_path / "evidence" / "stage=poll_once" / "batch=000001.parquet"

    written = storage.write_parquet_shard(target, frame)

    assert written == str(target)
    assert storage.read_parquet(target).to_dicts() == frame.to_dicts()
    lazy = storage.scan_parquet(target)
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().to_dicts() == frame.to_dicts()


def test_local_storage_iterates_parquet_in_bounded_ordered_batches(tmp_path) -> None:
    storage = LocalStorageBackend()
    target = tmp_path / "ordered.parquet"
    storage.write_parquet_shard(target, pl.DataFrame({"row_id": [0, 1, 2, 3, 4]}))

    batches = list(storage.iter_parquet_batches(target, batch_size=2))

    assert [batch.height for batch in batches] == [2, 2, 1]
    assert [row_id for batch in batches for row_id in batch["row_id"].to_list()] == [0, 1, 2, 3, 4]


def test_local_storage_rejects_invalid_parquet_batch_size(tmp_path) -> None:
    storage = LocalStorageBackend()
    target = tmp_path / "ordered.parquet"
    storage.write_parquet_shard(target, pl.DataFrame({"row_id": [0]}))

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(storage.iter_parquet_batches(target, batch_size=0))


def test_local_storage_writes_parquet_batches(tmp_path) -> None:
    storage = LocalStorageBackend()
    target = tmp_path / "evidence" / "stage=poll_once" / "batch=000001.parquet"

    written = storage.write_parquet_batches(
        target,
        (
            pl.DataFrame({"photo_id": ["1"], "score": [0.7]}),
            pl.DataFrame({"photo_id": ["2"], "score": [0.4]}),
        ),
    )

    assert isinstance(storage, CloudStorage)
    assert written == str(target)
    assert storage.read_parquet(target).to_dicts() == [
        {"photo_id": "1", "score": 0.7},
        {"photo_id": "2", "score": 0.4},
    ]


def test_write_parquet_batches_casts_each_part_to_the_explicit_schema(tmp_path) -> None:
    target = tmp_path / "parts.parquet"
    schema = {"label": pl.String, "bbox": pl.List(pl.Float64)}

    write_parquet_batches(
        (
            pl.DataFrame({"label": ["butterfly"], "bbox": [[0.0, 1.0]]}),
            pl.DataFrame({"label": [None], "bbox": [None]}),
        ),
        target,
        schema=schema,
    )

    frame = pl.read_parquet(target)
    assert frame.schema == schema
    assert frame.to_dicts() == [
        {"label": "butterfly", "bbox": [0.0, 1.0]},
        {"label": None, "bbox": None},
    ]


def test_local_parquet_part_defaults_to_zstd_and_refuses_overwrite(tmp_path) -> None:
    target = tmp_path / "evidence" / "stage=detect_objects" / "run_id=run-1" / "worker=w1" / "part=000001.parquet"
    frame = pl.DataFrame({"photo_id": ["1", "2"], "score": [0.7, 0.4]})

    written = write_parquet_part(frame, target)

    assert written.uri == str(target)
    assert written.row_count == 2
    assert written.byte_count is not None and written.byte_count > 0
    assert written.compression == "zstd"
    assert _parquet_column_compressions(target) == {"ZSTD"}
    with pytest.raises(FileExistsError):
        write_parquet_part(frame, target)


def test_local_storage_lists_shards_deterministically(tmp_path) -> None:
    storage = LocalStorageBackend()
    frame = pl.DataFrame({"x": [1]})
    prefix = tmp_path / "evidence"
    first = prefix / "worker=2" / "batch=000002.parquet"
    second = prefix / "worker=1" / "batch=000001.parquet"
    storage.write_parquet_shard(first, frame)
    storage.write_parquet_shard(second, frame)
    storage.write_json(prefix / "ignore.json", {"ignore": True})

    assert storage.list_shards(prefix) == sorted([str(first), str(second)])


def test_local_storage_exposes_no_append_api_and_targets_unique_shards(tmp_path) -> None:
    storage = LocalStorageBackend()
    assert not hasattr(storage, "append_parquet")

    shard_a = build_parquet_shard_uri(tmp_path, stage="poll_once", run_id="run-1", worker_id="w1", batch_id=1)
    shard_b = build_parquet_shard_uri(tmp_path, stage="poll_once", run_id="run-1", worker_id="w2", batch_id=1)

    assert shard_a != shard_b
    storage.write_parquet_shard(shard_a, pl.DataFrame({"worker": ["w1"]}))
    storage.write_parquet_shard(shard_b, pl.DataFrame({"worker": ["w2"]}))
    assert storage.read_parquet(shard_a)["worker"].to_list() == ["w1"]
    assert storage.read_parquet(shard_b)["worker"].to_list() == ["w2"]


def test_s3_storage_writes_parquet_without_materializing_payload(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)

    class NoGetValueBytesIO(io.BytesIO):
        def getvalue(self):  # noqa: ANN201
            raise AssertionError("S3 parquet upload must stream from a temp file")

    monkeypatch.setattr(s3_module, "BytesIO", NoGetValueBytesIO, raising=False)
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, "biominer/biominer/evidence/part.parquet"))

    written = backend.write_parquet_shard(
        "s3://biominer/biominer/evidence/part.parquet",
        pl.DataFrame({"photo_id": ["1", "2"], "score": [0.7, 0.4]}),
    )

    assert written == "s3://biominer/biominer/evidence/part.parquet"
    assert filesystem.paths == ["biominer/biominer/evidence/part.parquet"]
    assert stream.bytes_written > 0


def test_s3_storage_parquet_writes_do_not_require_local_temp_files(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)

    def fail_temp_file(*_args, **_kwargs):  # noqa: ANN202
        raise AssertionError("S3 parquet writes must not create local temporary parquet files")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_temp_file)
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, "biominer/biominer/evidence/part.parquet"))
    large_frame = pl.DataFrame({"photo_id": [str(index) for index in range(10_000)], "score": [0.42] * 10_000})

    backend.write_parquet_shard("s3://biominer/biominer/evidence/part.parquet", large_frame)
    backend.write_parquet_batches(
        "s3://biominer/biominer/evidence/part.parquet",
        (large_frame.slice(0, 5_000), large_frame.slice(5_000, 5_000)),
    )

    assert filesystem.paths == [
        "biominer/biominer/evidence/part.parquet",
        "biominer/biominer/evidence/part.parquet",
    ]
    assert stream.bytes_written > 0


def test_s3_storage_writes_zstd_parquet_parts(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/evidence/stage=detect_objects/run_id=run-1/worker=w1/part=000001.parquet"

    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, "biominer/biominer/evidence/part.parquet"))

    written = backend.write_parquet_part(uri, pl.DataFrame({"photo_id": ["1", "2"], "score": [0.7, 0.4]}))

    assert written.uri == uri
    assert written.row_count == 2
    assert written.byte_count is not None and written.byte_count > 0
    assert written.compression == "zstd"
    assert filesystem.paths == ["biominer/biominer/evidence/part.parquet"]
    assert _parquet_column_compressions(io.BytesIO(stream.payload)) == {"ZSTD"}
    with pytest.raises(FileExistsError):
        backend.write_parquet_part(uri, pl.DataFrame({"photo_id": ["3"]}))


def test_s3_storage_writes_and_reads_text(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/reports/evaluation_summary.md"
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, "biominer/biominer/reports/evaluation_summary.md"))

    written = backend.write_text(uri, "# Summary\n")

    assert written == uri
    assert backend.read_text(uri) == "# Summary\n"
    assert filesystem.paths == ["biominer/biominer/reports/evaluation_summary.md"]


def test_s3_storage_lists_shards_without_bucket_duplication(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    filesystem = _FakeListingS3Filesystem(
        [
            "biominer/biominer/runs/run_id=run-1/staging/canonical_source_records.parquet",
            "biominer/biominer/runs/run_id=run-1/staging/run_manifest.json",
            "biominer/biominer/runs/run_id=run-2/staging/canonical_source_records.parquet",
        ]
    )
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, "biominer/biominer/runs"))
    monkeypatch.setattr(backend, "_pyarrow_fs", lambda: _FakePyArrowFs)

    assert backend.list_shards("s3://biominer/biominer/runs") == [
        "s3://biominer/biominer/runs/run_id=run-1/staging/canonical_source_records.parquet",
        "s3://biominer/biominer/runs/run_id=run-2/staging/canonical_source_records.parquet",
    ]
    assert filesystem.selectors == [("biominer/biominer/runs", True)]


def test_s3_storage_lists_missing_prefix_as_empty(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    filesystem = _MissingListingS3Filesystem()
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, "biominer/biominer/missing"))
    monkeypatch.setattr(backend, "_pyarrow_fs", lambda: _FakePyArrowFs)

    assert backend.list_shards("s3://biominer/biominer/missing") == []


def test_uri_helpers_classify_and_join_paths(tmp_path) -> None:
    assert is_cloud_uri("s3://biominer/prefix/file.parquet")
    assert is_s3_uri("s3://biominer/prefix/file.parquet")
    assert not is_cloud_uri("relative/path.parquet")
    assert not is_s3_uri("file:///tmp/file.parquet")
    assert normalize_local_uri((tmp_path / "file.json").as_uri()) == tmp_path / "file.json"
    assert normalize_local_uri("relative/path.parquet") == Path("relative/path.parquet")
    assert join_uri("s3://bucket/base/", "evidence", "stage=poll_once") == "s3://bucket/base/evidence/stage=poll_once"
    assert join_uri("local/base/", "evidence", "batch=000001.parquet") == "local/base/evidence/batch=000001.parquet"


def test_build_parquet_shard_uri_is_stable_for_local_and_s3() -> None:
    assert build_parquet_shard_uri(
        "s3://biominer/biominer",
        stage="poll_once",
        run_id="run-2026",
        worker_id="worker-3",
        batch_id=7,
    ) == "s3://biominer/biominer/evidence/stage=poll_once/run_id=run-2026/worker=worker-3/batch=000007.parquet"
    assert build_parquet_shard_uri(
        "staging",
        stage="filter",
        run_id="r",
        worker_id="w",
        batch_id="abc",
    ) == "staging/evidence/stage=filter/run_id=r/worker=w/batch=abc.parquet"
    assert build_parquet_part_uri(
        "s3://biominer/biominer",
        stage="score_bioclip",
        run_id="run-2026",
        worker_id="worker-3",
        part_id=7,
    ) == "s3://biominer/biominer/evidence/stage=score_bioclip/run_id=run-2026/worker=worker-3/part=000007.parquet"


def test_build_evidence_shard_uri_local_and_s3() -> None:
    assert build_evidence_shard_uri(
        "staging",
        stage="poll_once",
        run_id="run-1",
        worker_id="worker-1",
        batch_id=1,
    ) == "staging/evidence/stage=poll_once/run_id=run-1/worker=worker-1/batch=000001.parquet"
    assert build_evidence_shard_uri(
        "s3://biominer/biominer",
        stage="poll_once",
        run_id="run-1",
        worker_id="worker-1",
        batch_id=1,
    ) == "s3://biominer/biominer/evidence/stage=poll_once/run_id=run-1/worker=worker-1/batch=000001.parquet"


def test_raw_flickr_response_uri_is_immutable_and_safe() -> None:
    uri = build_raw_flickr_response_uri(
        "s3://biominer/biominer",
        run_id="2026-07-02T000000Z",
        query_field="text",
        query_term="Papilio demoleus / lime butterfly",
        lane="normal_page",
        page=1,
        work_item_id="abc123",
    )

    assert safe_path_component("Papilio demoleus / lime butterfly") == "papilio_demoleus_lime_butterfly"
    assert uri == (
        "s3://biominer/biominer/raw/source=flickr/method=photos_search/"
        "run_id=2026-07-02T000000Z/field=text/term=papilio_demoleus_lime_butterfly/"
        "lane=normal_page/page=000001/work_item_id=abc123.json"
    )


def test_report_and_registry_uri_helpers_are_cloud_safe() -> None:
    assert build_report_uri("reports", run_id="run-1", report_name="step1_report") == "reports/run_id=run-1/step1_report.json"
    assert build_report_uri("s3://biominer/biominer", run_id="run-1", report_name="step1_report") == (
        "s3://biominer/biominer/reports/run_id=run-1/step1_report.json"
    )
    assert build_registry_version_uri("s3://biominer/biominer", registry_version="butterflies-v1", filename="taxa.parquet") == (
        "s3://biominer/biominer/registry/version=butterflies-v1/taxa.parquet"
    )
    assert build_registry_current_uri("s3://biominer/biominer", filename="manifest.json") == (
        "s3://biominer/biominer/registry/current/manifest.json"
    )
    assert build_registry_current_pointer(
        registry_version="butterflies-v1",
        registry_prefix="s3://biominer/biominer/registry/version=butterflies-v1",
        manifest_uri="s3://biominer/biominer/registry/version=butterflies-v1/manifest.json",
        promoted_at="2026-07-02T00:00:00Z",
    ) == {
        "registry_version": "butterflies-v1",
        "registry_prefix": "s3://biominer/biominer/registry/version=butterflies-v1",
        "manifest_uri": "s3://biominer/biominer/registry/version=butterflies-v1/manifest.json",
        "promoted_at": "2026-07-02T00:00:00Z",
    }


class _FakeS3Filesystem:
    def __init__(self, stream: "_FakeOutputStream") -> None:
        self.stream = stream
        self.paths: list[str] = []
        self.existing_paths: set[str] = set()

    def open_output_stream(self, path: str) -> "_FakeOutputStream":
        self.paths.append(path)
        self.existing_paths.add(path)
        self.stream.closed = False
        self.stream.payload.clear()
        self.stream.bytes_written = 0
        return self.stream

    def open_input_file(self, path: str):  # noqa: ANN201
        return io.BytesIO(bytes(self.stream.payload))

    def get_file_info(self, path: str):  # noqa: ANN201
        import pyarrow.fs as pafs

        if path in self.existing_paths:
            return _FakeFileInfo(type=pafs.FileType.File, size=self.stream.bytes_written, path=path)
        return _FakeFileInfo(type=pafs.FileType.NotFound, size=None, path=path)


class _FakeListingS3Filesystem:
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        self.selectors: list[tuple[str, bool]] = []

    def get_file_info(self, selector: "_FakeFileSelector") -> list["_FakeFileInfo"]:
        import pyarrow.fs as pafs

        self.selectors.append((selector.base_dir, selector.recursive))
        return [_FakeFileInfo(type=pafs.FileType.File, size=128, path=path) for path in self.paths]


class _MissingListingS3Filesystem:
    def get_file_info(self, selector: "_FakeFileSelector") -> list["_FakeFileInfo"]:
        raise FileNotFoundError(selector.base_dir)


class _FakeFileSelector:
    def __init__(self, base_dir: str, *, recursive: bool) -> None:
        self.base_dir = base_dir
        self.recursive = recursive


class _FakePyArrowFs:
    FileSelector = _FakeFileSelector


class _FakeFileInfo:
    def __init__(self, *, type, size: int | None, path: str = "") -> None:  # noqa: A002, ANN001
        self.type = type
        self.size = size
        self.path = path

    @property
    def is_file(self) -> bool:
        import pyarrow.fs as pafs

        return self.type == pafs.FileType.File


class _FakeOutputStream:
    def __init__(self) -> None:
        self.bytes_written = 0
        self.closed = False
        self.payload = bytearray()

    def __enter__(self) -> "_FakeOutputStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()
        return None

    def write(self, data: bytes) -> int:
        self.payload.extend(data)
        self.bytes_written += len(data)
        return len(data)

    def writable(self) -> bool:
        return not self.closed

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _parquet_column_compressions(source) -> set[str]:  # noqa: ANN001
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(source)
    compressions: set[str] = set()
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            compressions.add(str(row_group.column(column_index).compression).upper())
    return compressions
