from __future__ import annotations

from hashlib import sha256
import io
from pathlib import Path
import tempfile

import polars as pl
import pytest

import biominer.storage.local as local_module
import biominer.storage.s3 as s3_module
from biominer.storage.cloud import CloudStorage
from biominer.storage.local import LocalStorageBackend
from biominer.storage.parquet import (
    ParquetRowSource,
    write_parquet_batches,
    write_parquet_part,
)
from biominer.storage.paths import (
    build_evidence_shard_uri,
    build_raw_flickr_response_uri,
    build_registry_current_pointer,
    build_registry_current_uri,
    build_registry_version_uri,
    build_report_uri,
    safe_path_component,
)
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


def test_local_storage_streams_file_atomically_and_refuses_overwrite(tmp_path) -> None:
    storage = LocalStorageBackend()
    source = tmp_path / "source.jpg"
    source.write_bytes(b"image-source-bytes")
    target = tmp_path / "objects" / "reference.jpg"

    written = storage.write_file(
        target,
        source,
        content_type="image/jpeg",
        overwrite=False,
    )

    assert written == str(target)
    assert target.read_bytes() == b"image-source-bytes"
    assert storage.file_size(target) == len(b"image-source-bytes")
    assert (
        storage.file_sha256(target)
        == f"sha256:{sha256(b'image-source-bytes').hexdigest()}"
    )
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))

    source.write_bytes(b"replacement-bytes")
    with pytest.raises(FileExistsError):
        storage.write_file(target, source, overwrite=False)
    assert target.read_bytes() == b"image-source-bytes"
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))
    with pytest.raises(FileNotFoundError):
        storage.file_size(tmp_path / "missing.jpg")
    with pytest.raises(FileNotFoundError):
        storage.file_sha256(tmp_path / "missing.jpg")


def test_local_storage_hash_distinguishes_same_size_payloads(tmp_path) -> None:
    storage = LocalStorageBackend()
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"abcd")
    second.write_bytes(b"wxyz")

    assert storage.file_size(first) == storage.file_size(second)
    assert storage.file_sha256(first) != storage.file_sha256(second)


def test_local_storage_materializes_committed_file_without_overwrite(tmp_path) -> None:
    storage = LocalStorageBackend()
    source = tmp_path / "objects" / "source.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"committed-reference-bytes")
    destination = tmp_path / "work" / "source.jpg"

    assert storage.materialize_file(source, destination) == str(destination)
    assert destination.read_bytes() == b"committed-reference-bytes"
    with pytest.raises(FileExistsError):
        storage.materialize_file(source, destination)
    assert destination.read_bytes() == b"committed-reference-bytes"


def test_local_content_addressed_materialization_verifies_before_overwrite(
    tmp_path,
) -> None:
    storage = LocalStorageBackend()
    expected_payload = b"expected-archive"
    digest = sha256(expected_payload).hexdigest()
    source = tmp_path / f"handoff.sha256-{digest}.tar.gz"
    source.write_bytes(b"corrupt-archive")
    destination = tmp_path / "cache" / source.name
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"committed-cache")

    with pytest.raises(OSError, match="failed local SHA-256"):
        storage.materialize_content_addressed_file(
            source,
            destination,
            expected_sha256=f"sha256:{digest}",
            overwrite=True,
        )

    assert destination.read_bytes() == b"committed-cache"
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_local_storage_cleans_temporary_file_after_failed_copy(
    tmp_path,
    monkeypatch,
) -> None:
    storage = LocalStorageBackend()
    source = tmp_path / "source.jpg"
    source.write_bytes(b"image-source-bytes")
    target = tmp_path / "objects" / "reference.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"committed-bytes")

    def fail_after_partial_copy(source_stream, output_stream) -> None:
        output_stream.write(source_stream.read(5))
        raise OSError("simulated copy failure")

    monkeypatch.setattr(local_module, "copyfileobj", fail_after_partial_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        storage.write_file(target, source)
    assert target.read_bytes() == b"committed-bytes"
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


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
    assert [row_id for batch in batches for row_id in batch["row_id"].to_list()] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_local_storage_rejects_invalid_parquet_batch_size(tmp_path) -> None:
    storage = LocalStorageBackend()
    target = tmp_path / "ordered.parquet"
    storage.write_parquet_shard(target, pl.DataFrame({"row_id": [0]}))

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(storage.iter_parquet_batches(target, batch_size=0))


def test_parquet_row_source_is_bounded_and_reiterable(tmp_path) -> None:
    target = tmp_path / "rows.parquet"
    pl.DataFrame({"row_id": [0, 1, 2]}).write_parquet(target)
    rows = ParquetRowSource(target, batch_size=1)

    assert [row["row_id"] for row in rows] == [0, 1, 2]
    assert [row["row_id"] for row in rows] == [0, 1, 2]


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
    target = (
        tmp_path
        / "evidence"
        / "stage=detect_objects"
        / "run_id=run-1"
        / "worker=w1"
        / "part=000001.parquet"
    )
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


def test_local_storage_exposes_no_append_api_and_targets_unique_shards(
    tmp_path,
) -> None:
    storage = LocalStorageBackend()
    assert not hasattr(storage, "append_parquet")

    shard_a = build_evidence_shard_uri(
        tmp_path, stage="poll_once", run_id="run-1", worker_id="w1", batch_id=1
    )
    shard_b = build_evidence_shard_uri(
        tmp_path, stage="poll_once", run_id="run-1", worker_id="w2", batch_id=1
    )

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
    monkeypatch.setattr(
        backend,
        "_filesystem_and_path",
        lambda uri: (filesystem, "biominer/biominer/evidence/part.parquet"),
    )

    written = backend.write_parquet_shard(
        "s3://biominer/biominer/evidence/part.parquet",
        pl.DataFrame({"photo_id": ["1", "2"], "score": [0.7, 0.4]}),
    )

    assert written == "s3://biominer/biominer/evidence/part.parquet"
    assert len(filesystem.paths) == 1
    assert filesystem.paths[0].startswith(
        "biominer/biominer/evidence/part.parquet.biominer-staging-"
    )
    assert filesystem.moves == [
        (filesystem.paths[0], "biominer/biominer/evidence/part.parquet")
    ]
    assert stream.bytes_written > 0


def test_s3_storage_reuses_one_configured_filesystem_instance(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    filesystem = object()

    class FakePyArrowFs:
        calls = 0

        @classmethod
        def S3FileSystem(cls, **_kwargs):  # noqa: ANN206
            cls.calls += 1
            return filesystem

    monkeypatch.setattr(backend, "_pyarrow_fs", lambda: FakePyArrowFs)

    first, first_path = backend._filesystem_and_path("s3://biominer/one")
    second, second_path = backend._filesystem_and_path("s3://biominer/two")

    assert first is filesystem
    assert second is filesystem
    assert first_path == "biominer/one"
    assert second_path == "biominer/two"
    assert FakePyArrowFs.calls == 1


def test_s3_content_addressed_filesystem_disables_background_and_uses_delayed_open(
    monkeypatch,
) -> None:
    backend = S3StorageBackend(
        bucket="biominer",
        endpoint_url="https://s3.example.invalid",
        access_key_id="key-id",
        secret_access_key="secret",
        region="us-east-005",
    )
    filesystem = object()

    class FakePyArrowFs:
        calls: list[dict[str, object]] = []

        @classmethod
        def S3FileSystem(cls, **kwargs):  # noqa: ANN206
            cls.calls.append(kwargs)
            return filesystem

    monkeypatch.setattr(backend, "_pyarrow_fs", lambda: FakePyArrowFs)

    first, first_path = backend._content_addressed_filesystem_and_path(
        "s3://biominer/handoffs/one"
    )
    second, second_path = backend._content_addressed_filesystem_and_path(
        "s3://biominer/handoffs/two"
    )

    assert first is second is filesystem
    assert first_path == "biominer/handoffs/one"
    assert second_path == "biominer/handoffs/two"
    assert len(FakePyArrowFs.calls) == 1
    assert FakePyArrowFs.calls[0]["background_writes"] is False
    assert FakePyArrowFs.calls[0]["allow_delayed_open"] is True


def test_s3_content_addressed_filesystem_requires_explicit_region() -> None:
    backend = S3StorageBackend(bucket="biominer", region="auto")

    with pytest.raises(ValueError, match="explicit S3 region"):
        backend._content_addressed_filesystem_and_path(
            "s3://biominer/handoffs/archive"
        )


def test_s3_storage_parquet_writes_do_not_require_local_temp_files(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)

    def fail_temp_file(*_args, **_kwargs):  # noqa: ANN202
        raise AssertionError(
            "S3 parquet writes must not create local temporary parquet files"
        )

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_temp_file)
    monkeypatch.setattr(
        backend,
        "_filesystem_and_path",
        lambda uri: (filesystem, "biominer/biominer/evidence/part.parquet"),
    )
    large_frame = pl.DataFrame(
        {"photo_id": [str(index) for index in range(10_000)], "score": [0.42] * 10_000}
    )

    backend.write_parquet_shard(
        "s3://biominer/biominer/evidence/part.parquet", large_frame
    )
    backend.write_parquet_batches(
        "s3://biominer/biominer/evidence/part.parquet",
        (large_frame.slice(0, 5_000), large_frame.slice(5_000, 5_000)),
    )

    assert len(filesystem.paths) == 2
    assert all(
        path.startswith("biominer/biominer/evidence/part.parquet.biominer-staging-")
        for path in filesystem.paths
    )
    assert [destination for _, destination in filesystem.moves] == [
        "biominer/biominer/evidence/part.parquet",
        "biominer/biominer/evidence/part.parquet",
    ]
    assert stream.bytes_written > 0


def test_s3_storage_writes_zstd_parquet_parts(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/evidence/stage=detect_objects/run_id=run-1/worker=w1/part=000001.parquet"

    monkeypatch.setattr(
        backend,
        "_filesystem_and_path",
        lambda uri: (filesystem, "biominer/biominer/evidence/part.parquet"),
    )

    written = backend.write_parquet_part(
        uri, pl.DataFrame({"photo_id": ["1", "2"], "score": [0.7, 0.4]})
    )

    assert written.uri == uri
    assert written.row_count == 2
    assert written.byte_count is not None and written.byte_count > 0
    assert written.compression == "zstd"
    assert len(filesystem.paths) == 1
    assert filesystem.paths[0].startswith(
        "biominer/biominer/evidence/part.parquet.biominer-staging-"
    )
    assert _parquet_column_compressions(io.BytesIO(stream.payload)) == {"ZSTD"}
    with pytest.raises(FileExistsError):
        backend.write_parquet_part(uri, pl.DataFrame({"photo_id": ["3"]}))


def test_s3_storage_failed_parquet_overwrite_preserves_committed_inventory(
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/references/reference_media_objects.parquet"
    path = "biominer/biominer/references/reference_media_objects.parquet"
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, path))
    committed = pl.DataFrame({"reference_media_id": ["one"]})
    backend.write_parquet_shard(uri, committed)
    committed_bytes = bytes(filesystem.objects[path])

    def fail_partial_write(frame, output_stream, *, compression) -> None:  # noqa: ANN001
        output_stream.write(b"PAR1partial")
        raise OSError("simulated inventory upload failure")

    monkeypatch.setattr(s3_module, "_write_frame", fail_partial_write)
    with pytest.raises(OSError, match="simulated inventory upload failure"):
        backend.write_parquet_shard(
            uri,
            pl.DataFrame({"reference_media_id": ["replacement"]}),
        )

    assert bytes(filesystem.objects[path]) == committed_bytes
    assert backend.read_parquet(uri).equals(committed)
    assert len(filesystem.moves) == 1


def test_s3_storage_writes_and_reads_text(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/reports/evaluation_summary.md"
    monkeypatch.setattr(
        backend,
        "_filesystem_and_path",
        lambda uri: (filesystem, "biominer/biominer/reports/evaluation_summary.md"),
    )

    written = backend.write_text(uri, "# Summary\n")

    assert written == uri
    assert backend.read_text(uri) == "# Summary\n"
    assert len(filesystem.paths) == 1
    assert filesystem.paths[0].startswith(
        "biominer/biominer/reports/evaluation_summary.md.biominer-staging-"
    )
    assert filesystem.moves == [
        (filesystem.paths[0], "biominer/biominer/reports/evaluation_summary.md")
    ]


def test_s3_storage_failed_text_overwrite_preserves_committed_summary(
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/reports/evaluation_summary.md"
    path = "biominer/biominer/reports/evaluation_summary.md"
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, path))
    backend.write_text(uri, "committed\n")
    committed = bytes(filesystem.objects[path])
    real_write = stream.write

    def fail_after_partial_write(data: bytes) -> int:
        real_write(data[:4])
        raise OSError("simulated summary upload failure")

    monkeypatch.setattr(stream, "write", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated summary upload failure"):
        backend.write_text(uri, "replacement\n")
    assert bytes(filesystem.objects[path]) == committed
    assert backend.read_text(uri) == "committed\n"


def test_s3_storage_stages_json_and_preserves_checkpoint_on_failed_overwrite(
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/checkpoints/reference.json"
    path = "biominer/biominer/checkpoints/reference.json"
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, path))

    assert backend.write_json(uri, {"status": "committed", "rows": 1}) == uri
    assert backend.read_json(uri) == {"status": "committed", "rows": 1}
    committed = bytes(filesystem.objects[path])
    real_write = stream.write

    def fail_after_partial_write(data: bytes) -> int:
        real_write(data[:7])
        raise OSError("simulated checkpoint upload failure")

    monkeypatch.setattr(stream, "write", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated checkpoint upload failure"):
        backend.write_json(uri, {"status": "replacement", "rows": 2})

    assert bytes(filesystem.objects[path]) == committed
    assert backend.read_json(uri) == {"status": "committed", "rows": 1}
    assert len(filesystem.deleted_paths) == 1
    assert filesystem.deleted_paths[0].startswith(f"{path}.biominer-staging-")


def test_s3_storage_partial_first_checkpoint_does_not_poison_retry(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/checkpoints/reference.json"
    path = "biominer/biominer/checkpoints/reference.json"
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, path))
    real_write = stream.write

    def fail_after_partial_write(data: bytes) -> int:
        real_write(data[:7])
        raise OSError("simulated checkpoint upload failure")

    monkeypatch.setattr(stream, "write", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated checkpoint upload failure"):
        backend.write_json(uri, {"status": "committed"})
    assert not backend.exists(uri)

    monkeypatch.setattr(stream, "write", real_write)
    assert backend.write_json(uri, {"status": "committed"}) == uri
    assert backend.read_json(uri) == {"status": "committed"}


def test_s3_storage_streams_file_with_content_type_and_refuses_overwrite(
    tmp_path,
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/reference-media/source.jpg"
    source = tmp_path / "source.jpg"
    source.write_bytes(b"reference-image-bytes")
    monkeypatch.setattr(
        backend,
        "_filesystem_and_path",
        lambda uri: (filesystem, "biominer/biominer/reference-media/source.jpg"),
    )

    written = backend.write_file(
        uri,
        source,
        content_type="image/jpeg",
        overwrite=False,
    )

    assert written == uri
    assert bytes(stream.payload) == b"reference-image-bytes"
    assert filesystem.metadata == [{"Content-Type": "image/jpeg"}]
    assert filesystem.compressions == [None]
    assert backend.file_size(uri) == len(b"reference-image-bytes")
    assert (
        backend.file_sha256(uri)
        == f"sha256:{sha256(b'reference-image-bytes').hexdigest()}"
    )
    destination = tmp_path / "materialized" / "source.jpg"
    assert backend.materialize_file(uri, destination) == str(destination)
    assert destination.read_bytes() == b"reference-image-bytes"
    with pytest.raises(FileExistsError):
        backend.materialize_file(uri, destination)
    with pytest.raises(FileExistsError):
        backend.write_file(uri, source, overwrite=False)
    assert len(filesystem.paths) == 1
    assert filesystem.paths[0].startswith(
        "biominer/biominer/reference-media/source.jpg.biominer-staging-"
    )
    assert filesystem.moves == [
        (filesystem.paths[0], "biominer/biominer/reference-media/source.jpg")
    ]


def test_s3_content_addressed_upload_uses_one_final_write_without_remote_reads(
    tmp_path,
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    filesystem = _WriteOnlyS3Filesystem()
    source = tmp_path / "handoff.tar.gz"
    source.write_bytes(b"verified-handoff")
    digest = f"sha256:{sha256(source.read_bytes()).hexdigest()}"
    uri = (
        "s3://biominer/biominer/handoffs/"
        f"handoff.sha256-{digest.removeprefix('sha256:')}.tar.gz"
    )
    monkeypatch.setattr(
        backend,
        "_content_addressed_filesystem_and_path",
        lambda uri: (filesystem, uri.removeprefix("s3://")),
    )

    assert (
        backend.write_content_addressed_file(
            uri,
            source,
            expected_sha256=digest,
            content_type="application/gzip",
        )
        == uri
    )
    assert filesystem.output_paths == [uri.removeprefix("s3://")]
    assert bytes(filesystem.payload) == source.read_bytes()
    assert filesystem.metadata == [{"Content-Type": "application/gzip"}]


def test_s3_content_addressed_download_uses_one_stream_and_local_hash(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"verified-handoff"
    digest = f"sha256:{sha256(payload).hexdigest()}"
    uri = (
        "s3://biominer/biominer/handoffs/"
        f"handoff.sha256-{digest.removeprefix('sha256:')}.tar.gz"
    )
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    filesystem = _ReadOnlyS3Filesystem(payload)
    monkeypatch.setattr(
        backend,
        "_content_addressed_filesystem_and_path",
        lambda uri: (filesystem, uri.removeprefix("s3://")),
    )
    destination = tmp_path / "handoff.tar.gz"

    assert (
        backend.materialize_content_addressed_file(
            uri,
            destination,
            expected_sha256=digest,
        )
        == str(destination)
    )
    assert destination.read_bytes() == payload
    assert filesystem.input_paths == [uri.removeprefix("s3://")]
    assert filesystem.compressions == [None]


def test_s3_content_addressed_transfer_requires_digest_in_final_key(tmp_path) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    source = tmp_path / "handoff.tar.gz"
    source.write_bytes(b"verified-handoff")
    digest = f"sha256:{sha256(source.read_bytes()).hexdigest()}"

    with pytest.raises(ValueError, match="content-addressed"):
        backend.write_content_addressed_file(
            "s3://biominer/biominer/handoffs/handoff.tar.gz",
            source,
            expected_sha256=digest,
        )


def test_s3_storage_removes_partial_new_file_and_allows_retry(
    tmp_path,
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/reference-media/source.jpg"
    path = "biominer/biominer/reference-media/source.jpg"
    source = tmp_path / "source.jpg"
    source.write_bytes(b"reference-image-bytes")
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, path))
    real_copyfileobj = s3_module.copyfileobj

    def fail_after_partial_copy(source_stream, output_stream) -> None:
        output_stream.write(source_stream.read(5))
        raise OSError("simulated upload failure")

    monkeypatch.setattr(s3_module, "copyfileobj", fail_after_partial_copy)

    with pytest.raises(OSError, match="simulated upload failure"):
        backend.write_file(uri, source, overwrite=False)
    assert not backend.exists(uri)
    assert len(filesystem.deleted_paths) == 1
    assert filesystem.deleted_paths[0].startswith(f"{path}.biominer-staging-")

    monkeypatch.setattr(s3_module, "copyfileobj", real_copyfileobj)
    assert backend.write_file(uri, source, overwrite=False) == uri
    assert (
        backend.file_sha256(uri) == f"sha256:{sha256(source.read_bytes()).hexdigest()}"
    )


def test_s3_storage_does_not_promote_staged_file_when_copy_reports_failure(
    tmp_path,
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/reference-media/source.jpg"
    path = "biominer/biominer/reference-media/source.jpg"
    source = tmp_path / "source.jpg"
    source.write_bytes(b"reference-image-bytes")
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, path))

    def fail_after_complete_copy(source_stream, output_stream) -> None:
        output_stream.write(source_stream.read())
        raise OSError("simulated close failure")

    monkeypatch.setattr(s3_module, "copyfileobj", fail_after_complete_copy)

    with pytest.raises(OSError, match="simulated close failure"):
        backend.write_file(uri, source, overwrite=False)
    assert not backend.exists(uri)
    assert len(filesystem.deleted_paths) == 1
    assert filesystem.deleted_paths[0].startswith(f"{path}.biominer-staging-")


def test_s3_storage_hash_distinguishes_same_size_remote_payloads(
    tmp_path,
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/reference-media/source.bin"
    path = "biominer/biominer/reference-media/source.bin"
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcd")
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, path))

    backend.write_file(uri, source)
    first_sha256 = backend.file_sha256(uri)
    filesystem.objects[path][:] = b"wxyz"

    assert backend.file_size(uri) == source.stat().st_size
    assert backend.file_sha256(uri) != first_sha256


def test_s3_storage_failed_overwrite_does_not_delete_preexisting_object(
    tmp_path,
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/reference-media/source.jpg"
    path = "biominer/biominer/reference-media/source.jpg"
    source = tmp_path / "source.jpg"
    source.write_bytes(b"committed-image")
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, path))
    backend.write_file(uri, source)
    source.write_bytes(b"replacement-image")

    def fail_after_partial_copy(source_stream, output_stream) -> None:
        output_stream.write(source_stream.read(5))
        raise OSError("simulated overwrite failure")

    monkeypatch.setattr(s3_module, "copyfileobj", fail_after_partial_copy)

    with pytest.raises(OSError, match="simulated overwrite failure"):
        backend.write_file(uri, source)
    assert backend.exists(uri)
    assert bytes(filesystem.objects[path]) == b"committed-image"
    assert len(filesystem.deleted_paths) == 1
    assert filesystem.deleted_paths[0].startswith(f"{path}.biominer-staging-")


def test_s3_storage_rejects_corrupt_staging_before_replacing_existing_object(
    tmp_path,
    monkeypatch,
) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    stream = _FakeOutputStream()
    filesystem = _FakeS3Filesystem(stream)
    uri = "s3://biominer/biominer/reference-media/source.jpg"
    path = "biominer/biominer/reference-media/source.jpg"
    source = tmp_path / "source.jpg"
    source.write_bytes(b"committed-image")
    monkeypatch.setattr(backend, "_filesystem_and_path", lambda uri: (filesystem, path))
    backend.write_file(uri, source)
    committed = bytes(filesystem.objects[path])
    source.write_bytes(b"replacement-image")

    def silently_truncate(source_stream, output_stream) -> None:
        output_stream.write(source_stream.read(5))

    monkeypatch.setattr(s3_module, "copyfileobj", silently_truncate)

    with pytest.raises(OSError, match="staged S3 object failed integrity"):
        backend.write_file(uri, source)
    assert bytes(filesystem.objects[path]) == committed
    assert len(filesystem.moves) == 1


def test_s3_storage_lists_shards_without_bucket_duplication(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    filesystem = _FakeListingS3Filesystem(
        [
            "biominer/biominer/runs/run_id=run-1/staging/canonical_source_records.parquet",
            "biominer/biominer/runs/run_id=run-1/staging/run_manifest.json",
            "biominer/biominer/runs/run_id=run-2/staging/canonical_source_records.parquet",
        ]
    )
    monkeypatch.setattr(
        backend,
        "_filesystem_and_path",
        lambda uri: (filesystem, "biominer/biominer/runs"),
    )
    monkeypatch.setattr(backend, "_pyarrow_fs", lambda: _FakePyArrowFs)

    assert backend.list_shards("s3://biominer/biominer/runs") == [
        "s3://biominer/biominer/runs/run_id=run-1/staging/canonical_source_records.parquet",
        "s3://biominer/biominer/runs/run_id=run-2/staging/canonical_source_records.parquet",
    ]
    assert filesystem.selectors == [("biominer/biominer/runs", True)]


def test_s3_storage_lists_missing_prefix_as_empty(monkeypatch) -> None:
    backend = S3StorageBackend(bucket="biominer", prefix="biominer")
    filesystem = _MissingListingS3Filesystem()
    monkeypatch.setattr(
        backend,
        "_filesystem_and_path",
        lambda uri: (filesystem, "biominer/biominer/missing"),
    )
    monkeypatch.setattr(backend, "_pyarrow_fs", lambda: _FakePyArrowFs)

    assert backend.list_shards("s3://biominer/biominer/missing") == []


def test_uri_helpers_classify_and_join_paths(tmp_path) -> None:
    assert is_cloud_uri("s3://biominer/prefix/file.parquet")
    assert is_s3_uri("s3://biominer/prefix/file.parquet")
    assert not is_cloud_uri("relative/path.parquet")
    assert not is_s3_uri("file:///tmp/file.parquet")
    assert (
        normalize_local_uri((tmp_path / "file.json").as_uri()) == tmp_path / "file.json"
    )
    assert normalize_local_uri("relative/path.parquet") == Path("relative/path.parquet")
    assert (
        join_uri("s3://bucket/base/", "evidence", "stage=poll_once")
        == "s3://bucket/base/evidence/stage=poll_once"
    )
    assert (
        join_uri("local/base/", "evidence", "batch=000001.parquet")
        == "local/base/evidence/batch=000001.parquet"
    )


def test_build_evidence_shard_uri_local_and_s3() -> None:
    assert (
        build_evidence_shard_uri(
            "staging",
            stage="poll_once",
            run_id="run-1",
            worker_id="worker-1",
            batch_id=1,
        )
        == "staging/evidence/stage=poll_once/run_id=run-1/worker=worker-1/batch=000001.parquet"
    )
    assert (
        build_evidence_shard_uri(
            "s3://biominer/biominer",
            stage="poll_once",
            run_id="run-1",
            worker_id="worker-1",
            batch_id=1,
        )
        == "s3://biominer/biominer/evidence/stage=poll_once/run_id=run-1/worker=worker-1/batch=000001.parquet"
    )


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

    assert (
        safe_path_component("Papilio demoleus / lime butterfly")
        == "papilio_demoleus_lime_butterfly"
    )
    assert uri == (
        "s3://biominer/biominer/raw/source=flickr/method=photos_search/"
        "run_id=2026-07-02T000000Z/field=text/term=papilio_demoleus_lime_butterfly/"
        "lane=normal_page/page=000001/work_item_id=abc123.json"
    )


def test_report_and_registry_uri_helpers_are_cloud_safe() -> None:
    assert (
        build_report_uri("reports", run_id="run-1", report_name="step1_report")
        == "reports/run_id=run-1/step1_report.json"
    )
    assert build_report_uri(
        "s3://biominer/biominer", run_id="run-1", report_name="step1_report"
    ) == ("s3://biominer/biominer/reports/run_id=run-1/step1_report.json")
    assert build_registry_version_uri(
        "s3://biominer/biominer",
        registry_version="butterflies-v2",
        filename="taxa.parquet",
    ) == ("s3://biominer/biominer/registry/version=butterflies-v2/taxa.parquet")
    assert build_registry_current_uri(
        "s3://biominer/biominer", filename="manifest.json"
    ) == ("s3://biominer/biominer/registry/current/manifest.json")
    assert build_registry_current_pointer(
        registry_version="butterflies-v2",
        registry_prefix="s3://biominer/biominer/registry/version=butterflies-v2",
        manifest_uri="s3://biominer/biominer/registry/version=butterflies-v2/manifest.json",
        promoted_at="2026-07-02T00:00:00Z",
    ) == {
        "registry_version": "butterflies-v2",
        "registry_prefix": "s3://biominer/biominer/registry/version=butterflies-v2",
        "manifest_uri": "s3://biominer/biominer/registry/version=butterflies-v2/manifest.json",
        "promoted_at": "2026-07-02T00:00:00Z",
    }


class _FakeS3Filesystem:
    def __init__(self, stream: "_FakeOutputStream") -> None:
        self.stream = stream
        self.paths: list[str] = []
        self.existing_paths: set[str] = set()
        self.objects: dict[str, bytearray] = {}
        self.metadata: list[dict[str, str] | None] = []
        self.compressions: list[str | None] = []
        self.deleted_paths: list[str] = []
        self.moves: list[tuple[str, str]] = []

    def open_output_stream(
        self,
        path: str,
        compression: str | None = "detect",
        buffer_size: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> "_FakeOutputStream":
        self.paths.append(path)
        self.existing_paths.add(path)
        self.metadata.append(metadata)
        self.compressions.append(compression)
        self.stream.closed = False
        self.stream.payload.clear()
        self.stream.bytes_written = 0
        self.objects[path] = self.stream.payload
        return self.stream

    def open_input_file(self, path: str):  # noqa: ANN201
        return io.BytesIO(bytes(self.objects[path]))

    def get_file_info(self, path: str):  # noqa: ANN201
        import pyarrow.fs as pafs

        if path in self.existing_paths:
            return _FakeFileInfo(
                type=pafs.FileType.File, size=len(self.objects[path]), path=path
            )
        return _FakeFileInfo(type=pafs.FileType.NotFound, size=None, path=path)

    def delete_file(self, path: str) -> None:
        self.deleted_paths.append(path)
        self.existing_paths.discard(path)
        self.objects.pop(path, None)

    def move(self, source: str, destination: str) -> None:
        self.moves.append((source, destination))
        payload = self.objects.pop(source)
        self.existing_paths.discard(source)
        self.objects[destination] = bytearray(payload)
        self.existing_paths.add(destination)


class _WriteOnlyS3Filesystem:
    def __init__(self) -> None:
        self.payload = bytearray()
        self.output_paths: list[str] = []
        self.metadata: list[dict[str, str] | None] = []

    def open_output_stream(
        self,
        path: str,
        compression: str | None = "detect",
        buffer_size: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> "_FakeOutputStream":
        _ = compression, buffer_size
        self.output_paths.append(path)
        self.metadata.append(metadata)
        return _FakeOutputStream(self.payload)

    def __getattr__(self, name: str):  # noqa: ANN204
        raise AssertionError(f"content-addressed upload used remote operation {name}")


class _ReadOnlyS3Filesystem:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.input_paths: list[str] = []
        self.compressions: list[str | None] = []

    def open_input_stream(
        self,
        path: str,
        compression: str | None = "detect",
    ):  # noqa: ANN201
        self.input_paths.append(path)
        self.compressions.append(compression)
        return io.BytesIO(self.payload)

    def __getattr__(self, name: str):  # noqa: ANN204
        raise AssertionError(f"content-addressed download used remote operation {name}")


class _FakeListingS3Filesystem:
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        self.selectors: list[tuple[str, bool]] = []

    def get_file_info(self, selector: "_FakeFileSelector") -> list["_FakeFileInfo"]:
        import pyarrow.fs as pafs

        self.selectors.append((selector.base_dir, selector.recursive))
        return [
            _FakeFileInfo(type=pafs.FileType.File, size=128, path=path)
            for path in self.paths
        ]


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
    def __init__(self, payload: bytearray | None = None) -> None:
        self.bytes_written = 0
        self.closed = False
        self.payload = payload if payload is not None else bytearray()

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
