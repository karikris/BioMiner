from __future__ import annotations

import builtins
from datetime import UTC, datetime, timedelta
from typing import Any
import sqlite3

import pytest

from biominer.workstore.postgres import POSTGRES_CLAIM_SQL, POSTGRES_SCHEMA_SQL, PostgresWorkStore
from biominer.workstore.sqlite import SQLiteWorkStore


def test_sqlite_workstore_enqueue_claim_and_complete(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    inserted = store.enqueue_work(
        "poll_once",
        "registry-v1",
        [
            {"work_key": "b", "term": "beta"},
            {"work_key": "a", "term": "alpha"},
            {"work_key": "a", "term": "alpha duplicate"},
        ],
    )

    claimed = store.claim_next_batch("worker-1", 2)
    store.mark_completed("a", "staging/a.parquet", "sha256:a", 10)

    assert inserted == 2
    assert [item["work_key"] for item in claimed] == ["b", "a"]
    assert claimed[0]["payload"]["term"] == "beta"
    assert claimed[0]["status"] == "claimed"
    assert claimed[0]["claimed_by"] == "worker-1"
    assert claimed[0]["attempt_count"] == 1
    assert store.claim_next_batch("worker-2", 2) == []
    assert store.completed_keys("poll_once", "registry-v1") == {"a"}

    with sqlite3.connect(store.path) as conn:
        row = conn.execute("SELECT status, output_uri, checksum, row_count FROM biominer_work_items WHERE work_key = 'a'").fetchone()
    assert row == ("completed", "staging/a.parquet", "sha256:a", 10)


def test_sqlite_workstore_completed_keys_are_filtered(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    store.enqueue_work("poll_once", "registry-v1", [{"work_key": "a"}])
    store.enqueue_work("poll_once", "registry-v2", [{"work_key": "b"}])
    store.enqueue_work("filter", "registry-v1", [{"work_key": "c"}])
    for key in ("a", "b", "c"):
        store.claim_next_batch("worker", 1)
        store.mark_completed(key, None, None, None)

    assert store.completed_keys("poll_once", "registry-v1") == {"a"}
    assert store.completed_keys("poll_once", None) == set()


def test_sqlite_workstore_records_failures_and_requeues_stale_claims(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    store.enqueue_work("poll_once", None, [{"work_key": "a"}, {"work_key": "b"}])
    assert [item["work_key"] for item in store.claim_next_batch("worker-1", 2)] == ["a", "b"]
    store.mark_failed("b", "temporary failure")

    old_claim = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE biominer_work_items SET claimed_at = ? WHERE work_key = 'a'", (old_claim,))

    assert store.stale_claims_to_pending(3600) == 1
    assert [item["work_key"] for item in store.claim_next_batch("worker-2", 2)] == ["a"]

    with sqlite3.connect(store.path) as conn:
        failed = conn.execute("SELECT status, error FROM biominer_work_items WHERE work_key = 'b'").fetchone()
    assert failed == ("failed", "temporary failure")


def test_sqlite_workstore_derives_deterministic_work_key_when_missing(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")

    first = store.enqueue_work("poll_once", None, [{"term": "butterfly"}])
    second = store.enqueue_work("poll_once", None, [{"term": "butterfly"}])
    claimed = store.claim_next_batch("worker", 1)

    assert first == 1
    assert second == 0
    assert claimed[0]["work_key"].startswith("poll_once:")
    assert claimed[0]["payload"] == {"term": "butterfly"}


def test_sqlite_workstore_registers_shard_inventory(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")

    store.register_shard(
        job_name="poll_once",
        registry_version="registry-v1",
        stage="poll_once",
        run_id="run-1",
        worker_id="worker-1",
        uri="staging/evidence/stage=poll_once/run_id=run-1/worker=worker-1/batch=000001.parquet",
        checksum="sha256:abc",
        row_count=2,
        byte_count=123,
    )
    store.register_shard(
        job_name="poll_once",
        registry_version="registry-v1",
        stage="poll_once",
        run_id="run-1",
        worker_id="worker-1",
        uri="staging/evidence/stage=poll_once/run_id=run-1/worker=worker-1/batch=000001.parquet",
        checksum="sha256:abc",
        row_count=2,
        byte_count=123,
    )

    with sqlite3.connect(store.path) as conn:
        rows = conn.execute(
            """
            SELECT job_name, registry_version, stage, run_id, worker_id, uri, checksum, row_count, byte_count
            FROM biominer_parquet_shards
            """
        ).fetchall()

    assert rows == [
        (
            "poll_once",
            "registry-v1",
            "poll_once",
            "run-1",
            "worker-1",
            "staging/evidence/stage=poll_once/run_id=run-1/worker=worker-1/batch=000001.parquet",
            "sha256:abc",
            2,
            123,
        )
    ]


def test_postgres_schema_and_claim_sql_are_supabase_compatible() -> None:
    for table in (
        "biominer_runs",
        "biominer_work_items",
        "biominer_api_call_ledger",
        "biominer_parquet_shards",
        "biominer_compaction_inputs",
    ):
        assert table in POSTGRES_SCHEMA_SQL
    assert "FOR UPDATE SKIP LOCKED" in POSTGRES_CLAIM_SQL
    assert "jsonb" in POSTGRES_SCHEMA_SQL


def test_postgres_workstore_imports_without_psycopg_and_validates_lazily(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name == "psycopg" or name.startswith("psycopg."):
            raise ImportError("missing psycopg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    store = PostgresWorkStore("postgresql://user:pass@example.test/db")

    with pytest.raises(RuntimeError, match="psycopg"):
        store.claim_next_batch("worker", 1)


def test_postgres_workstore_contract_with_injected_connection() -> None:
    fake = _FakePostgres()
    store = PostgresWorkStore("postgresql://user:pass@example.test/db", connect=fake.connect)

    store.init_schema()
    run = store.get_or_create_run(
        job_name="poll_once",
        stage="metadata",
        run_id="run-1",
        registry_version="registry-v1",
        config={"max_api_calls": 10},
    )
    inserted = store.enqueue_work(
        "poll_once",
        "registry-v1",
        [
            {"work_key": "b", "term": "beta"},
            {"work_key": "a", "term": "alpha"},
            {"work_key": "a", "term": "duplicate"},
        ],
        stage="metadata",
    )
    claimed = store.claim_next_batch(
        "worker-1",
        2,
        job_name="poll_once",
        stage="metadata",
        registry_version="registry-v1",
    )
    store.mark_completed("a", "s3://biominer/evidence/a.parquet", "sha256:a", 4)
    store.register_shard(
        shard_id="shard-1",
        job_name="poll_once",
        registry_version="registry-v1",
        stage="metadata",
        run_id="run-1",
        worker_id="worker-1",
        uri="s3://biominer/evidence/shard-1.parquet",
        checksum="sha256:shard",
        row_count=1,
        byte_count=100,
        metadata={"kind": "canonical_delta"},
    )
    store.register_shard(
        shard_id="shard-duplicate-id",
        job_name="poll_once",
        registry_version="registry-v1",
        stage="metadata",
        run_id="run-1",
        worker_id="worker-1",
        uri="s3://biominer/evidence/shard-1.parquet",
        checksum="sha256:shard",
        row_count=1,
        byte_count=100,
        metadata={"kind": "canonical_delta"},
    )

    assert fake.schema_initialized
    assert run["config"] == {"max_api_calls": 10}
    assert store.get_run(run_id="run-1")["status"] == "planned"
    assert inserted == 2
    assert [item["work_key"] for item in claimed] == ["b", "a"]
    assert claimed[0]["payload"] == {"term": "beta"}
    assert claimed[0]["status"] == "claimed"
    assert claimed[0]["claimed_by"] == "worker-1"
    assert claimed[0]["attempt_count"] == 1
    assert store.completed_keys("poll_once", "registry-v1", stage="metadata") == {"a"}
    assert store.list_committed_shards(job_name="poll_once", stage="metadata", registry_version="registry-v1") == [
        {
            "shard_id": "shard-1",
            "job_name": "poll_once",
            "stage": "metadata",
            "run_id": "run-1",
            "registry_version": "registry-v1",
            "worker_id": "worker-1",
            "uri": "s3://biominer/evidence/shard-1.parquet",
            "row_count": 1,
            "byte_count": 100,
            "checksum": "sha256:shard",
            "metadata": {"kind": "canonical_delta"},
            "committed_at": "2026-07-04T00:00:00+00:00",
        }
    ]


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]] | None = None, *, rowcount: int = 0) -> None:
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _FakePostgres:
    def __init__(self) -> None:
        self.schema_initialized = False
        self.runs: dict[str, dict[str, Any]] = {}
        self.work_items: dict[str, dict[str, Any]] = {}
        self.shards: dict[str, dict[str, Any]] = {}

    def connect(self):
        return _FakePostgresConnection(self)


class _FakePostgresConnection:
    def __init__(self, db: _FakePostgres) -> None:
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> _FakeResult:
        params = params or ()
        normalized = " ".join(sql.split())
        if "CREATE TABLE IF NOT EXISTS biominer_runs" in normalized:
            self.db.schema_initialized = True
            return _FakeResult()
        if normalized.startswith("INSERT INTO biominer_runs"):
            run_id, job_name, stage, registry_version, status, started_at, config = params
            self.db.runs.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "job_name": job_name,
                    "stage": stage,
                    "registry_version": registry_version,
                    "status": status,
                    "started_at": str(started_at),
                    "ended_at": None,
                    "config_json": config,
                    "summary_json": None,
                },
            )
            return _FakeResult(rowcount=1)
        if normalized.startswith("SELECT * FROM biominer_runs WHERE run_id"):
            row = self.db.runs.get(params[0])
            return _FakeResult([row] if row else [])
        if normalized.startswith("INSERT INTO biominer_work_items"):
            work_key, job_name, stage, registry_version, status, payload, created_at = params
            if work_key in self.db.work_items:
                return _FakeResult(rowcount=0)
            self.db.work_items[work_key] = {
                "work_key": work_key,
                "job_name": job_name,
                "stage": stage,
                "registry_version": registry_version,
                "status": status,
                "payload_json": payload,
                "claimed_by": None,
                "claimed_at": None,
                "completed_at": None,
                "output_uri": None,
                "checksum": None,
                "row_count": None,
                "attempt_count": 0,
                "error": None,
                "created_at": str(created_at),
            }
            return _FakeResult(rowcount=1)
        if "FOR UPDATE SKIP LOCKED" in normalized:
            job_name, stage, registry_version, limit, worker_id = params
            rows = [
                row
                for row in self.db.work_items.values()
                if row["job_name"] == job_name
                and row["stage"] == stage
                and row["registry_version"] == registry_version
                and row["status"] == "pending"
            ][:limit]
            for row in rows:
                row["status"] = "claimed"
                row["claimed_by"] = worker_id
                row["claimed_at"] = "2026-07-04T00:00:00+00:00"
                row["attempt_count"] += 1
            return _FakeResult(rows)
        if normalized.startswith("UPDATE biominer_work_items SET status = %s, completed_at"):
            status, completed_at, output_uri, checksum, row_count, error, work_key = params
            row = self.db.work_items[work_key]
            row.update(
                {
                    "status": status,
                    "completed_at": str(completed_at),
                    "output_uri": output_uri,
                    "checksum": checksum,
                    "row_count": row_count,
                    "error": error,
                }
            )
            return _FakeResult(rowcount=1)
        if normalized.startswith("SELECT work_key FROM biominer_work_items"):
            job_name, status, stage, registry_version = params
            return _FakeResult(
                [
                    {"work_key": row["work_key"]}
                    for row in self.db.work_items.values()
                    if row["job_name"] == job_name
                    and row["status"] == status
                    and row["stage"] == stage
                    and row["registry_version"] == registry_version
                ]
            )
        if normalized.startswith("INSERT INTO biominer_parquet_shards"):
            (
                shard_id,
                job_name,
                registry_version,
                stage,
                run_id,
                worker_id,
                uri,
                row_count,
                byte_count,
                checksum,
                metadata,
                committed_at,
            ) = params
            if shard_id in self.db.shards or any(row["uri"] == uri for row in self.db.shards.values()):
                return _FakeResult(rowcount=0)
            self.db.shards.setdefault(
                shard_id,
                {
                    "shard_id": shard_id,
                    "job_name": job_name,
                    "registry_version": registry_version,
                    "stage": stage,
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "uri": uri,
                    "row_count": row_count,
                    "byte_count": byte_count,
                    "checksum": checksum,
                    "metadata_json": metadata,
                    "committed_at": "2026-07-04T00:00:00+00:00",
                },
            )
            return _FakeResult(rowcount=1)
        if normalized.startswith("SELECT * FROM biominer_parquet_shards"):
            job_name, stage, registry_version = params
            return _FakeResult(
                [
                    row
                    for row in self.db.shards.values()
                    if row["job_name"] == job_name
                    and row["stage"] == stage
                    and row["registry_version"] == registry_version
                ]
            )
        raise AssertionError(f"unexpected SQL: {normalized}")
