from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sqlite3

import polars as pl

from biominer.storage.local import LocalStorageBackend
from biominer.storage.paths import build_evidence_shard_uri
from biominer.workstore.resume import prepare_resume_plan, repair_shard_manifest_from_storage
from biominer.workstore.sqlite import SQLiteWorkStore


def test_get_or_create_run_is_idempotent(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")

    first = store.get_or_create_run(
        job_name="flickr_poll_once",
        stage="poll_once",
        run_id="run-1",
        registry_version="registry-v1",
        config={"workers": 2},
    )
    second = store.get_or_create_run(
        job_name="flickr_poll_once",
        stage="poll_once",
        run_id="run-1",
        registry_version="registry-v1",
        config={"workers": 99},
    )

    assert first == second
    assert first["status"] == "planned"
    assert store.get_run(run_id="run-1") == first
    assert len(store.list_runs(job_name="flickr_poll_once", stage="poll_once", registry_version="registry-v1")) == 1


def test_prepare_resume_plan_skips_completed_and_claims_only_missing(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    store.enqueue_work(job_name="flickr_poll_once", stage="poll_once", registry_version="registry-v1", items=[{"work_key": "done", "page": 1}])
    store.claim_next_batch(worker_id="worker-old", job_name="flickr_poll_once", stage="poll_once", registry_version="registry-v1", limit=1)
    store.mark_completed(work_key="done", output_uri="shard.parquet", checksum="sha256:1", row_count=1)

    plan = prepare_resume_plan(
        workstore=store,
        storage=LocalStorageBackend(),
        job_name="flickr_poll_once",
        stage="poll_once",
        run_id="run-1",
        registry_version="registry-v1",
        planned_items=[{"work_key": "done", "page": 1}, {"work_key": "todo", "page": 2}],
        worker_id="worker-1",
        stale_after_seconds=3600,
    )

    assert plan.planned_count == 2
    assert plan.skipped_completed_count == 1
    assert plan.enqueued_count == 1
    assert plan.claimed_count == 1
    assert store.list_work_items(job_name="flickr_poll_once", stage="poll_once", registry_version="registry-v1", statuses=["claimed"])[0]["work_key"] == "todo"


def test_requeue_stale_claims_scopes_and_reclaims(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    store.enqueue_work(job_name="flickr_poll_once", stage="poll_once", registry_version=None, items=[{"work_key": "a"}])
    assert store.claim_next_batch(worker_id="worker-1", job_name="flickr_poll_once", stage="poll_once", registry_version=None, limit=1)[0]["work_key"] == "a"
    old_claim = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE biominer_work_items SET claimed_at = ?, error = 'kept context' WHERE work_key = 'a'", (old_claim,))

    assert store.requeue_stale_claims(job_name="flickr_poll_once", stage="poll_once", registry_version=None, stale_after_seconds=3600) == 1
    reclaimed = store.claim_next_batch(worker_id="worker-2", job_name="flickr_poll_once", stage="poll_once", registry_version=None, limit=1)

    assert reclaimed[0]["work_key"] == "a"
    with sqlite3.connect(store.path) as conn:
        row = conn.execute("SELECT status, claimed_by, attempt_count, error FROM biominer_work_items WHERE work_key = 'a'").fetchone()
    assert row == ("claimed", "worker-2", 2, None)


def test_claim_next_batch_does_not_double_claim(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    store.enqueue_work(
        job_name="flickr_poll_once",
        stage="poll_once",
        registry_version="registry-v1",
        items=[{"work_key": f"k{i}", "i": i} for i in range(5)],
    )

    first = store.claim_next_batch(worker_id="worker-a", job_name="flickr_poll_once", stage="poll_once", registry_version="registry-v1", limit=3)
    second = store.claim_next_batch(worker_id="worker-b", job_name="flickr_poll_once", stage="poll_once", registry_version="registry-v1", limit=3)

    first_keys = {item["work_key"] for item in first}
    second_keys = {item["work_key"] for item in second}
    assert len(first) == 3
    assert len(second) == 2
    assert first_keys.isdisjoint(second_keys)


def test_completed_keys_filter_by_job_stage_and_registry(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    for job, stage, registry, key in (
        ("job-a", "stage-a", "v1", "shared"),
        ("job-a", "stage-b", "v1", "shared-b"),
        ("job-a", "stage-a", "v2", "shared-v2"),
    ):
        store.enqueue_work(job_name=job, stage=stage, registry_version=registry, items=[{"work_key": key}])
        store.claim_next_batch(worker_id="worker", job_name=job, stage=stage, registry_version=registry, limit=1)
        store.mark_completed(work_key=key, output_uri=None, checksum=None, row_count=None)

    assert store.completed_keys(job_name="job-a", stage="stage-a", registry_version="v1") == {"shared"}


def test_repair_shard_manifest_from_local_storage(tmp_path) -> None:
    storage = LocalStorageBackend()
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    prefix = tmp_path / "staging"
    first_uri = build_evidence_shard_uri(prefix, stage="poll_once", run_id="run-1", worker_id="worker-1", batch_id=1)
    second_uri = build_evidence_shard_uri(prefix, stage="poll_once", run_id="run-1", worker_id="worker-1", batch_id=2)
    storage.write_parquet_shard(first_uri, pl.DataFrame({"x": [1]}))
    storage.write_parquet_shard(second_uri, pl.DataFrame({"x": [2, 3]}))
    store.register_shard(
        shard_id="known",
        job_name="flickr_poll_once",
        stage="poll_once",
        run_id="run-1",
        registry_version="registry-v1",
        uri=first_uri,
        row_count=1,
        checksum=None,
    )

    repaired = repair_shard_manifest_from_storage(
        storage=storage,
        workstore=store,
        job_name="flickr_poll_once",
        stage="poll_once",
        registry_version="registry-v1",
        run_id="run-1",
        shard_prefix=str(prefix),
    )
    shards = store.list_committed_shards(job_name="flickr_poll_once", stage="poll_once", registry_version="registry-v1", run_id="run-1")

    assert repaired == 1
    assert {shard["uri"] for shard in shards} == {first_uri, second_uri}
    assert {shard["row_count"] for shard in shards} == {1, 2}
