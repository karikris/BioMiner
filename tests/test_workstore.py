from __future__ import annotations

from datetime import UTC, datetime, timedelta
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


def test_postgres_schema_and_claim_sql_are_supabase_compatible() -> None:
    for table in (
        "biominer_runs",
        "biominer_work_items",
        "biominer_api_call_ledger",
        "biominer_parquet_shards",
    ):
        assert table in POSTGRES_SCHEMA_SQL
    assert "FOR UPDATE SKIP LOCKED" in POSTGRES_CLAIM_SQL
    assert "jsonb" in POSTGRES_SCHEMA_SQL


def test_postgres_workstore_imports_without_psycopg_and_validates_lazily() -> None:
    store = PostgresWorkStore("postgresql://user:pass@example.test/db")

    with pytest.raises(RuntimeError, match="psycopg"):
        store.claim_next_batch("worker", 1)
