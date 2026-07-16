from __future__ import annotations

import polars as pl

from biominer.storage.compaction import (
    MB,
    CompactionCandidate,
    compact_parquet_shards,
    plan_compaction_groups,
)
from biominer.storage.local import LocalStorageBackend
from biominer.storage.paths import build_compacted_evidence_uri, build_evidence_shard_uri
from biominer.workstore.sqlite import SQLiteWorkStore


def test_build_compacted_evidence_uri_local_and_s3() -> None:
    assert build_compacted_evidence_uri(
        "staging",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
        part_id=1,
    ) == "staging/evidence/stage=poll_once_compacted/registry_version=registry-v1/run_id=compact-1/part=000001.parquet"
    assert build_compacted_evidence_uri(
        "s3://biominer/biominer",
        source_stage="poll_once",
        registry_version=None,
        compaction_run_id="compact-1",
        part_id=1,
    ) == "s3://biominer/biominer/evidence/stage=poll_once_compacted/registry_version=unknown/run_id=compact-1/part=000001.parquet"


def test_plan_compaction_groups_is_deterministic() -> None:
    candidates = [
        CompactionCandidate(shard_id="b", uri="staging/b.parquet", row_count=2, byte_count=40 * MB, stage="poll_once", registry_version="v1"),
        CompactionCandidate(shard_id="a", uri="staging/a.parquet", row_count=1, byte_count=40 * MB, stage="poll_once", registry_version="v1"),
        CompactionCandidate(shard_id="c", uri="staging/c.parquet", row_count=3, byte_count=40 * MB, stage="poll_once", registry_version="v1"),
    ]

    first = plan_compaction_groups(
        candidates,
        base_prefix="staging",
        source_stage="poll_once",
        output_stage="poll_once_compacted",
        registry_version="v1",
        compaction_run_id="compact-1",
        target_file_mb=80,
        max_file_mb=120,
    )
    second = plan_compaction_groups(
        candidates,
        base_prefix="staging",
        source_stage="poll_once",
        output_stage="poll_once_compacted",
        registry_version="v1",
        compaction_run_id="compact-1",
        target_file_mb=80,
        max_file_mb=120,
    )

    assert first == second
    assert [[source.shard_id for source in group.source_shards] for group in first.groups] == [["a", "b"], ["c"]]
    assert [group.output_uri for group in first.groups] == [
        "staging/evidence/stage=poll_once_compacted/registry_version=v1/run_id=compact-1/part=000001.parquet",
        "staging/evidence/stage=poll_once_compacted/registry_version=v1/run_id=compact-1/part=000002.parquet",
    ]


def test_plan_compaction_groups_respects_target_and_max_size() -> None:
    candidates = [
        CompactionCandidate(shard_id=f"s{i}", uri=f"staging/s{i}.parquet", row_count=1, byte_count=120 * MB, stage="poll_once", registry_version="v1")
        for i in range(5)
    ]

    plan = plan_compaction_groups(
        candidates,
        base_prefix="staging",
        source_stage="poll_once",
        output_stage="poll_once_compacted",
        registry_version="v1",
        compaction_run_id="compact-1",
        target_file_mb=256,
        max_file_mb=512,
    )

    assert [len(group.source_shards) for group in plan.groups] == [3, 2]
    assert all(group.estimated_input_bytes <= 512 * MB for group in plan.groups if group.estimated_input_bytes is not None)


def test_local_compaction_writes_immutable_output_parts_and_keeps_sources(tmp_path) -> None:
    storage = LocalStorageBackend()
    source_prefix = tmp_path / "staging" / "evidence" / "stage=poll_once"
    output_prefix = tmp_path / "staging"
    source_uris = _write_source_shards(storage, source_prefix, count=3)

    result = compact_parquet_shards(
        storage=storage,
        workstore=None,
        input_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        job_name="flickr_poll_once",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
        target_file_mb=1,
        max_file_mb=1,
    )

    expected = build_compacted_evidence_uri(output_prefix, source_stage="poll_once", registry_version="registry-v1", compaction_run_id="compact-1", part_id=1)
    assert result.output_shards_written == 1
    assert result.rows_written == 3
    assert storage.exists(expected)
    assert storage.read_parquet(expected).height == 3
    assert all(storage.exists(uri) for uri in source_uris)


def test_local_compaction_does_not_append_on_restart(tmp_path) -> None:
    storage = LocalStorageBackend()
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    source_prefix = tmp_path / "staging" / "evidence" / "stage=poll_once"
    output_prefix = tmp_path / "staging"
    _write_source_shards(storage, source_prefix, count=2, workstore=store)

    first = compact_parquet_shards(
        storage=storage,
        workstore=store,
        input_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        job_name="flickr_poll_once",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
    )
    second = compact_parquet_shards(
        storage=storage,
        workstore=store,
        input_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        job_name="flickr_poll_once",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
    )
    output_uri = build_compacted_evidence_uri(output_prefix, source_stage="poll_once", registry_version="registry-v1", compaction_run_id="compact-1", part_id=1)

    assert first.rows_written == 2
    assert second.groups_written == 0
    assert second.skipped_groups == 0
    assert storage.read_parquet(output_uri).height == 2


def test_compact_with_dedupe_keys(tmp_path) -> None:
    storage = LocalStorageBackend()
    source_prefix = tmp_path / "staging" / "evidence" / "stage=poll_once"
    output_prefix = tmp_path / "staging"
    storage.write_parquet_shard(source_prefix / "a.parquet", pl.DataFrame({"source": ["flickr", "flickr"], "flickr_photo_id": ["1", "1"], "value": [1, 2]}))
    storage.write_parquet_shard(source_prefix / "b.parquet", pl.DataFrame({"source": ["flickr"], "flickr_photo_id": ["2"], "value": [3]}))

    compact_parquet_shards(
        storage=storage,
        workstore=None,
        input_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        job_name="flickr_poll_once",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
        dedupe_keys=["source", "flickr_photo_id"],
    )
    output_uri = build_compacted_evidence_uri(output_prefix, source_stage="poll_once", registry_version="registry-v1", compaction_run_id="compact-1", part_id=1)

    assert storage.read_parquet(output_uri).sort("flickr_photo_id")["flickr_photo_id"].to_list() == ["1", "2"]


def test_compact_without_dedupe_keys_preserves_rows(tmp_path) -> None:
    storage = LocalStorageBackend()
    source_prefix = tmp_path / "staging" / "evidence" / "stage=poll_once"
    output_prefix = tmp_path / "staging"
    storage.write_parquet_shard(source_prefix / "a.parquet", pl.DataFrame({"source": ["flickr", "flickr"], "flickr_photo_id": ["1", "1"]}))

    compact_parquet_shards(
        storage=storage,
        workstore=None,
        input_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        job_name="flickr_poll_once",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
    )
    output_uri = build_compacted_evidence_uri(output_prefix, source_stage="poll_once", registry_version="registry-v1", compaction_run_id="compact-1", part_id=1)

    assert storage.read_parquet(output_uri).height == 2


def test_workstore_registers_compaction_output_and_inputs(tmp_path) -> None:
    storage = LocalStorageBackend()
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    source_prefix = tmp_path / "staging" / "evidence" / "stage=poll_once"
    output_prefix = tmp_path / "staging"
    source_uris = _write_source_shards(storage, source_prefix, count=2, workstore=store)

    compact_parquet_shards(
        storage=storage,
        workstore=store,
        input_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        job_name="flickr_poll_once",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
    )
    compacted = store.list_committed_shards(job_name="flickr_poll_once", stage="poll_once_compacted", registry_version="registry-v1", run_id="compact-1")
    consumed = store.list_compacted_source_shard_ids(job_name="flickr_poll_once", stage="poll_once", registry_version="registry-v1")

    assert len(compacted) == 1
    assert compacted[0]["metadata"]["source_shard_count"] == 2
    assert consumed == {f"source-{index}" for index in range(len(source_uris))}


def test_dry_run_writes_nothing(tmp_path) -> None:
    storage = LocalStorageBackend()
    source_prefix = tmp_path / "staging" / "evidence" / "stage=poll_once"
    output_prefix = tmp_path / "staging"
    _write_source_shards(storage, source_prefix, count=2)

    result = compact_parquet_shards(
        storage=storage,
        workstore=None,
        input_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        job_name="flickr_poll_once",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
        dry_run=True,
    )

    assert result.groups_planned == 1
    assert result.output_shards_written == 0
    assert storage.list_shards(output_prefix / "evidence" / "stage=poll_once_compacted") == []


def test_schema_strict_fails_on_incompatible_columns(tmp_path) -> None:
    storage = LocalStorageBackend()
    source_prefix = tmp_path / "staging" / "evidence" / "stage=poll_once"
    output_prefix = tmp_path / "staging"
    storage.write_parquet_shard(source_prefix / "a.parquet", pl.DataFrame({"flickr_photo_id": ["1"], "value": [1]}))
    storage.write_parquet_shard(source_prefix / "b.parquet", pl.DataFrame({"flickr_photo_id": ["2"], "other": [2]}))

    result = compact_parquet_shards(
        storage=storage,
        workstore=None,
        input_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        job_name="flickr_poll_once",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
        schema_mode="strict",
    )

    assert result.failed_groups == 1
    assert result.output_shards_written == 0
    assert storage.list_shards(output_prefix / "evidence" / "stage=poll_once_compacted") == []


def test_report_written(tmp_path) -> None:
    storage = LocalStorageBackend()
    source_prefix = tmp_path / "staging" / "evidence" / "stage=poll_once"
    output_prefix = tmp_path / "staging"
    _write_source_shards(storage, source_prefix, count=2)

    result = compact_parquet_shards(
        storage=storage,
        workstore=None,
        input_prefix=str(source_prefix),
        output_prefix=str(output_prefix),
        job_name="flickr_poll_once",
        source_stage="poll_once",
        registry_version="registry-v1",
        compaction_run_id="compact-1",
    )
    report = storage.read_json(output_prefix / "reports" / "run_id=compact-1" / "compaction_poll_once.json")

    assert result.rows_written == 2
    assert report["rows_written"] == 2
    assert report["output_shards_written"] == 1
    assert report["dry_run"] is False


def _write_source_shards(
    storage: LocalStorageBackend,
    source_prefix,
    *,
    count: int,
    workstore: SQLiteWorkStore | None = None,
) -> list[str]:
    uris: list[str] = []
    for index in range(count):
        uri = build_evidence_shard_uri(
            source_prefix,
            stage="poll_once",
            run_id="source-run",
            worker_id=f"worker-{index}",
            batch_id=index + 1,
        )
        storage.write_parquet_shard(uri, pl.DataFrame({"source": ["flickr"], "flickr_photo_id": [str(index)], "value": [index]}))
        uris.append(uri)
        if workstore is not None:
            workstore.register_shard(
                shard_id=f"source-{index}",
                job_name="flickr_poll_once",
                stage="poll_once",
                run_id="source-run",
                registry_version="registry-v1",
                uri=uri,
                row_count=1,
                checksum=None,
                byte_count=100,
            )
    return uris
