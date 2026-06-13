from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sqlite3
import threading
from pathlib import Path

from biominer.flickr_fetch.query_planner import FlickrQuery
from biominer.flickr_fetch.metadata_poller import MetadataPollState, poll_once


def test_metadata_poller_creates_required_state_tables(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")

    with sqlite3.connect(state.path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert {"api_call_ledger", "flickr_work_items", "source_records", "source_record_query_hits", "image_triage_queue"}.issubset(tables)


def test_poll_once_fetches_metadata_only_dedupes_and_queues_image_urls(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    query = FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250)
    state.enqueue_work_item(query)
    calls: list[FlickrQuery] = []

    def fake_fetch(item: FlickrQuery) -> dict[str, object]:
        calls.append(item)
        return {
            "photos": {
                "total": "2",
                "photo": [
                    {"id": "1", "title": "butterfly", "url_l": "https://live.staticflickr.com/1_l.jpg"},
                    {"id": "1", "title": "duplicate", "url_l": "https://live.staticflickr.com/1_l.jpg"},
                    {"id": "2", "title": "butterfly", "url_m": "https://live.staticflickr.com/2_m.jpg"},
                ],
            }
        }

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence" / "poll.parquet",
        max_api_calls=3500,
        fetch_metadata=fake_fetch,
    )

    assert len(calls) == 1
    assert result.api_calls_made == 1
    assert result.raw_responses_written == 1
    assert result.source_records_inserted == 2
    assert result.duplicate_records_skipped == 1
    assert result.image_urls_queued == 2
    assert result.evidence_rows_written == 3
    assert list((tmp_path / "raw").rglob("*.json"))
    assert not list(tmp_path.rglob("*.jpg"))
    with sqlite3.connect(state.path) as conn:
        assert conn.execute("SELECT count(*) FROM source_records").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM image_triage_queue").fetchone()[0] == 2


def test_poll_once_preserves_duplicate_query_hits_for_source_record(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="Papilio", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    state.enqueue_work_item(FlickrQuery(term="Papilionidae", language="en", search_field="tags", lane="normal_page", page=1, per_page=250))

    def fake_fetch(item: FlickrQuery) -> dict[str, object]:
        return {
            "photos": {
                "total": "1",
                "photo": [
                    {
                        "id": "1",
                        "title": "swallowtail",
                        "url_l": "https://live.staticflickr.com/1_l.jpg",
                    }
                ],
            }
        }

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence" / "poll.parquet",
        max_api_calls=2,
        fetch_metadata=fake_fetch,
    )

    assert result.source_records_inserted == 1
    assert result.duplicate_records_skipped == 1
    assert result.query_hits_inserted == 2
    with sqlite3.connect(state.path) as conn:
        source_rows = conn.execute("SELECT query_field, query_term FROM source_records").fetchall()
        query_hits = conn.execute(
            "SELECT query_field, query_term FROM source_record_query_hits ORDER BY query_field, query_term"
        ).fetchall()
    assert source_rows == [("text", "Papilio")]
    assert query_hits == [("tags", "Papilionidae"), ("text", "Papilio")]


def test_export_source_records_with_query_provenance_lists_all_keywords(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="Papilio", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    state.enqueue_work_item(FlickrQuery(term="Papilionidae", language="en", search_field="tags", lane="normal_page", page=1, per_page=250))

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence" / "poll.parquet",
        max_api_calls=2,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "photo": [{"id": "1", "title": "swallowtail", "url_l": "https://live.staticflickr.com/1_l.jpg"}],
            }
        },
    )

    frame = state.source_records_with_query_provenance()
    row = frame.to_dicts()[0]

    assert row["first_query_label"] == "text:Papilio"
    assert row["first_query_field"] == "text"
    assert row["first_query_term"] == "Papilio"
    assert row["all_query_labels"] == ["tags:Papilionidae", "text:Papilio"]
    assert row["all_query_terms"] == ["Papilionidae", "Papilio"]
    assert row["all_query_fields"] == ["tags", "text"]
    assert row["query_hit_count"] == 2


def test_poll_once_records_count_probes_and_enqueues_pages(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="papillon", language="fr", search_field="tags", lane="count_probe", per_page=1))

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "501", "photo": []}},
    )

    assert result.api_calls_made == 1
    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT lane, per_page FROM flickr_work_items WHERE status = 'pending' ORDER BY page").fetchall()

    assert rows == [("normal_page", 250), ("normal_page", 250), ("normal_page", 250)]


def test_poll_once_enqueues_pages_for_count_probe_under_page_limit(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", per_page=1))

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "3501", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        pending = conn.execute("SELECT count(*) FROM flickr_work_items WHERE status = 'pending'").fetchone()[0]

    assert pending == 15


def test_poll_once_splits_count_probe_over_page_limit(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", per_page=1))

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": str(4000 * 250), "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT lane, per_page, json_extract(query_json, '$.split_reason') FROM flickr_work_items WHERE status = 'pending'").fetchall()

    assert rows
    assert {row[0] for row in rows} == {"count_probe"}
    assert {row[1] for row in rows} == {1}
    assert {row[2] for row in rows} == {"bbox"}


def test_poll_once_respects_3500_soft_budget_without_fetching(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", per_page=250))
    with sqlite3.connect(state.path) as conn:
        for index in range(3500):
            conn.execute(
                "INSERT INTO api_call_ledger(endpoint, work_item_id, status, created_at) VALUES (?, ?, ?, strftime('%s','now'))",
                ("flickr.photos.search", f"work-{index}", "ok"),
            )

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=3500,
        fetch_metadata=lambda query: (_ for _ in ()).throw(AssertionError("fetch should not be called")),
    )

    assert result.work_items_claimed == 0
    assert result.remaining_soft_budget == 0


def test_poll_once_requeues_stale_claimed_work(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    query = FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250)
    state.enqueue_work_item(query)
    claimed = state.claim_pending(limit=1)
    stale_time = datetime.now(UTC) - timedelta(hours=2)
    with sqlite3.connect(state.path) as conn:
        conn.execute("UPDATE flickr_work_items SET claimed_at = ? WHERE work_item_id = ?", (stale_time.isoformat(), claimed[0][0]))

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda item: {"photos": {"total": "0", "photo": []}},
        stale_claim_seconds=60,
    )

    assert result.stale_claims_requeued == 1
    assert result.work_items_claimed == 1


def test_poll_once_uses_parallel_workers_for_claimed_pages(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    for index in range(4):
        state.enqueue_work_item(FlickrQuery(term=f"butterfly-{index}", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    thread_names: set[str] = set()

    def fake_fetch(item: FlickrQuery) -> dict[str, object]:
        thread_names.add(threading.current_thread().name)
        return {"photos": {"total": "0", "photo": []}}

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=4,
        fetch_metadata=fake_fetch,
        workers=4,
    )

    assert result.work_items_claimed == 4
    assert result.api_calls_made == 4
    assert thread_names
