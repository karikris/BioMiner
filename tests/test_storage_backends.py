from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.storage.local import LocalStorageBackend
from biominer.storage.shard_paths import build_parquet_shard_uri
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
