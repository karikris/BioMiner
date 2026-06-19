from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import sqlite3
import threading
from pathlib import Path

import httpx

from biominer.flickr_fetch.query_planner import BBOX_PAGE_SIZE, COUNT_PROBE_PAGE_SIZE, NORMAL_PAGE_SIZE, FlickrQuery, fixed_upload_date_slices
from biominer.flickr_fetch.metadata_poller import MetadataPollState, _payload_page, _payload_pages, _payload_perpage, poll_once


def test_metadata_poller_creates_required_state_tables(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")

    with sqlite3.connect(state.path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        work_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(flickr_work_items)").fetchall()
        }

    assert {"api_call_ledger", "flickr_work_items", "source_records", "source_record_query_hits", "image_triage_queue"}.issubset(tables)
    assert {
        "split_depth",
        "split_priority",
        "split_reason",
        "parent_query_hash",
        "parent_total",
        "date_kind",
        "min_date",
        "max_date",
        "bbox_index",
        "bbox_label",
        "term",
        "query_hash",
    }.issubset(work_columns)


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


def test_poll_once_plans_fixed_slice_pages_over_stable_result_threshold(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", per_page=1))

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "4001", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT lane, per_page, json_extract(query_json, '$.split_reason') FROM flickr_work_items WHERE status = 'pending' LIMIT 20").fetchall()
        pending_count = conn.execute("SELECT count(*) FROM flickr_work_items WHERE status = 'pending'").fetchone()[0]

    assert rows
    expected_slices = fixed_upload_date_slices(
        start_date="2004-02-10",
        end_date=datetime.now(UTC).date().isoformat(),
        slice_days=5,
        coarse_end_date=None,
        coarse_slice_days=None,
    )
    assert pending_count == len(expected_slices)
    assert {row[0] for row in rows} == {"normal_page"}
    assert {row[1] for row in rows} == {NORMAL_PAGE_SIZE}
    assert {row[2] for row in rows} == {"upload_date"}


def test_poll_once_enqueues_all_pages_for_4000_record_bbox_leaf(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", bbox="0,0,10,10", per_page=1))

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "4000", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT lane, page, per_page FROM flickr_work_items WHERE status = 'pending' ORDER BY page").fetchall()

    assert rows == [("bbox_page", page, BBOX_PAGE_SIZE) for page in range(1, 17)]


def test_poll_once_enqueues_all_pages_for_4000_record_standard_leaf(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="count_probe", has_geo=0, per_page=1))

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "4000", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT lane, page, per_page FROM flickr_work_items WHERE status = 'pending' ORDER BY page").fetchall()

    assert rows == [("normal_page", page, NORMAL_PAGE_SIZE) for page in range(1, 9)]


def test_payload_page_metadata_helpers_read_flickr_response_values() -> None:
    payload = {"photos": {"total": "740", "pages": "2", "page": "1", "perpage": "500", "photo": []}}

    assert _payload_pages(payload) == 2
    assert _payload_page(payload) == 1
    assert _payload_perpage(payload) == 500


def test_payload_page_metadata_helpers_default_missing_values() -> None:
    assert _payload_pages({"photos": {}}) == 0
    assert _payload_page({"photos": {}}) == 0
    assert _payload_perpage({"photos": {}}) == 0


def test_poll_once_reserves_api_call_before_fetch(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250))

    def fake_fetch(item: FlickrQuery) -> dict[str, object]:
        with sqlite3.connect(state.path) as conn:
            assert conn.execute("SELECT count(*) FROM api_call_ledger WHERE work_item_id != ''").fetchone()[0] == 1
        return {"photos": {"total": "0", "photo": []}}

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=fake_fetch,
    )

    assert result.api_calls_made == 1
    assert result.raw_responses_written == 1
    with sqlite3.connect(state.path) as conn:
        assert conn.execute("SELECT status FROM flickr_work_items").fetchone()[0] == "completed"


def test_poll_once_retries_transient_timeout_before_success(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    calls = 0

    def flaky_fetch(item: FlickrQuery) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.TimeoutException("timed out")
        return {"photos": {"total": "1", "pages": "1", "page": "1", "perpage": "250", "photo": [{"id": "1", "url_l": "https://live.staticflickr.com/1.jpg"}]}}

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=2,
        fetch_metadata=flaky_fetch,
        max_retries=1,
    )

    with sqlite3.connect(state.path) as conn:
        statuses = conn.execute("SELECT status, count(*) FROM api_call_ledger GROUP BY status ORDER BY status").fetchall()
    assert result.api_calls_made == 2
    assert result.raw_responses_written == 1
    assert statuses == [("failed", 1), ("ok", 1)]


def test_poll_once_does_not_retry_flickr_semantic_failure(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250))

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=3,
        fetch_metadata=lambda item: {"stat": "fail", "code": 100, "message": "Invalid API Key"},
        max_retries=2,
    )

    with sqlite3.connect(state.path) as conn:
        status, error = conn.execute("SELECT status, error FROM flickr_work_items").fetchone()
        api_calls = conn.execute("SELECT count(*) FROM api_call_ledger").fetchone()[0]
    assert result.api_calls_made == 1
    assert status == "failed"
    assert "Invalid API Key" in error
    assert api_calls == 1


def test_poll_once_does_not_retry_malformed_flickr_payload(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250))

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=3,
        fetch_metadata=lambda item: {"photos": {"total": "not-an-int", "photo": []}},
        max_retries=2,
    )

    with sqlite3.connect(state.path) as conn:
        status, error = conn.execute("SELECT status, error FROM flickr_work_items").fetchone()
        api_calls = conn.execute("SELECT count(*) FROM api_call_ledger").fetchone()[0]
    assert result.api_calls_made == 1
    assert status == "failed"
    assert "photos.total" in error
    assert api_calls == 1


def test_poll_once_retries_http_429_status(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    calls = 0

    def flaky_fetch(item: FlickrQuery) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            request = httpx.Request("GET", "https://www.flickr.com/services/rest/")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return {"photos": {"total": "0", "pages": "0", "page": "1", "perpage": "250", "photo": []}}

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=2,
        fetch_metadata=flaky_fetch,
        max_retries=1,
    )

    with sqlite3.connect(state.path) as conn:
        statuses = conn.execute("SELECT status, count(*) FROM api_call_ledger GROUP BY status ORDER BY status").fetchall()
    assert result.api_calls_made == 2
    assert result.raw_responses_written == 1
    assert statuses == [("failed", 1), ("ok", 1)]


def test_poll_once_does_not_claim_or_reserve_full_budget_before_fetch(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    for index in range(20):
        state.enqueue_work_item(
            FlickrQuery(
                term=f"butterfly-{index}",
                language="en",
                search_field="text",
                lane="normal_page",
                page=1,
                per_page=250,
            )
        )

    observed_before_first_fetch: dict[str, int] = {}

    def fake_fetch(item: FlickrQuery) -> dict[str, object]:
        if not observed_before_first_fetch:
            with sqlite3.connect(state.path) as conn:
                observed_before_first_fetch["claimed"] = conn.execute(
                    "SELECT count(*) FROM flickr_work_items WHERE status = 'claimed'"
                ).fetchone()[0]
                observed_before_first_fetch["reserved"] = conn.execute(
                    "SELECT count(*) FROM api_call_ledger WHERE status = 'reserved'"
                ).fetchone()[0]
        return {"photos": {"total": "0", "photo": []}}

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=20,
        workers=2,
        fetch_metadata=fake_fetch,
    )

    assert result.api_calls_made == 20
    assert observed_before_first_fetch == {"claimed": 2, "reserved": 2}
    with sqlite3.connect(state.path) as conn:
        assert conn.execute("SELECT count(*) FROM flickr_work_items WHERE status = 'claimed'").fetchone()[0] == 0


def test_poll_once_second_run_resumes_pending_pages_without_duplicates(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=2, per_page=250))

    def fake_fetch(item: FlickrQuery) -> dict[str, object]:
        return {
            "photos": {
                "total": "2",
                "photo": [{"id": str(item.page), "url_l": f"https://live.staticflickr.com/{item.page}.jpg"}],
            }
        }

    first = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=fake_fetch,
    )
    second = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=2,
        fetch_metadata=fake_fetch,
    )

    with sqlite3.connect(state.path) as conn:
        source_count = conn.execute("SELECT count(*) FROM source_records").fetchone()[0]
        statuses = dict(conn.execute("SELECT status, count(*) FROM flickr_work_items GROUP BY status").fetchall())
    assert first.work_items_claimed == 1
    assert second.work_items_claimed == 1
    assert source_count == 2
    assert statuses == {"completed": 2}


def test_claim_and_reserve_pending_uses_deterministic_date_slice_order(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    later = FlickrQuery(
        term="butterfly",
        language="en",
        search_field="text",
        lane="count_probe",
        per_page=1,
        min_taken_date="2025-01-01",
        max_taken_date="2025-12-31",
        split_reason="taken_date",
        split_depth=1,
    )
    older = FlickrQuery(
        term="butterfly",
        language="en",
        search_field="text",
        lane="count_probe",
        per_page=1,
        min_taken_date="2024-01-01",
        max_taken_date="2024-12-31",
        split_reason="taken_date",
        split_depth=1,
    )
    state.enqueue_work_item(later)
    state.enqueue_work_item(older)

    claimed = state.claim_and_reserve_pending(limit=1, max_api_calls=1, endpoint="flickr.photos.search")

    assert claimed[0][1].min_taken_date == "2024-01-01"
    with sqlite3.connect(state.path) as conn:
        assert conn.execute("SELECT count(*) FROM api_call_ledger WHERE status = 'reserved'").fetchone()[0] == 1


def test_claim_and_reserve_pending_orders_fixed_slice_pages_by_slice_then_page(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=2,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-06",
            max_upload_date="2007-01-10",
            slice_index=1,
        )
    )
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=8,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            slice_index=0,
        )
    )
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=1,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            slice_index=0,
        )
    )

    claimed = state.claim_and_reserve_pending(limit=3, max_api_calls=3, endpoint="flickr.photos.search")

    assert [(query.slice_index, query.page) for _work_id, query in claimed] == [(0, 1), (0, 8), (1, 2)]


def test_claim_and_reserve_pending_is_atomic_across_state_instances(tmp_path) -> None:
    state_db = tmp_path / "poller.sqlite"
    state = MetadataPollState(state_db)
    for index in range(10):
        state.enqueue_work_item(
            FlickrQuery(
                term=f"butterfly-{index}",
                language="en",
                search_field="text",
                lane="normal_page",
                page=1,
                per_page=500,
            )
        )

    def claim_once(index: int) -> int:
        claimed = MetadataPollState(state_db).claim_and_reserve_pending(
            limit=1,
            max_api_calls=3,
            endpoint="flickr.photos.search",
        )
        return len(claimed)

    with ThreadPoolExecutor(max_workers=8) as executor:
        claimed_counts = list(executor.map(claim_once, range(10)))

    with sqlite3.connect(state_db) as conn:
        reserved = conn.execute("SELECT count(*) FROM api_call_ledger WHERE status = 'reserved'").fetchone()[0]
        claimed_work = conn.execute("SELECT count(*) FROM flickr_work_items WHERE status = 'claimed'").fetchone()[0]

    assert sum(claimed_counts) == 3
    assert reserved == 3
    assert claimed_work == 3


def test_poll_once_records_page_payload_count_for_saturation_reporting(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=8,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            slice_index=0,
        )
    )
    payload = {
        "photos": {
            "total": "500",
            "photo": [
                {"id": str(index), "url_l": f"https://live.staticflickr.com/{index}.jpg"}
                for index in range(500)
            ],
        }
    }

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: payload,
    )

    with sqlite3.connect(state.path) as conn:
        row = conn.execute("SELECT records_returned FROM flickr_work_items WHERE page = 8").fetchone()
    assert result.source_records_inserted == 500
    assert row[0] == 500


def test_poll_once_records_page_response_metadata_for_reporting(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=1,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            split_reason="upload_date",
            split_depth=1,
            slice_index=0,
        )
    )

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {
            "photos": {"total": "9000", "pages": "20", "page": "1", "perpage": "500", "photo": []}
        },
    )

    with sqlite3.connect(state.path) as conn:
        row = conn.execute(
            "SELECT response_total, response_pages, response_page, response_perpage FROM flickr_work_items WHERE page = 1"
        ).fetchone()
    assert row == (9000, 20, 1, 500)


def test_poll_once_page_one_single_page_slice_enqueues_no_extra_pages(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=1,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            split_reason="upload_date",
            split_depth=1,
            slice_index=0,
        )
    )

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "500",
                "photo": [{"id": "1", "url_l": "https://live.staticflickr.com/1.jpg"}],
            }
        },
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT status, page FROM flickr_work_items ORDER BY page").fetchall()
    assert result.source_records_inserted == 1
    assert rows == [("completed", 1)]


def test_poll_once_page_one_enqueues_remaining_reported_pages(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=1,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            split_reason="upload_date",
            split_depth=1,
            slice_index=0,
        )
    )

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {
            "photos": {
                "total": "1200",
                "pages": "3",
                "page": "1",
                "perpage": "500",
                "photo": [{"id": "1", "url_l": "https://live.staticflickr.com/1.jpg"}],
            }
        },
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT status, page, per_page, min_date, max_date FROM flickr_work_items ORDER BY page").fetchall()
    assert rows == [
        ("completed", 1, 500, "2007-01-01", "2007-01-05"),
        ("pending", 2, 500, "2007-01-01", "2007-01-05"),
        ("pending", 3, 500, "2007-01-01", "2007-01-05"),
    ]


def test_poll_once_progress_callback_reports_claim_page_and_dynamic_enqueue(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=1,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            split_reason="upload_date",
            split_depth=1,
            slice_index=0,
        )
    )
    events: list[dict[str, object]] = []

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {
            "photos": {
                "total": "1200",
                "pages": "3",
                "page": "1",
                "perpage": "500",
                "photo": [{"id": "1", "url_l": "https://live.staticflickr.com/1.jpg"}],
            }
        },
        progress_callback=events.append,
    )

    event_names = [str(event["event"]) for event in events]
    assert "budget_checked" in event_names
    assert "work_claimed" in event_names
    assert "page_completed" in event_names
    assert "remaining_pages_enqueued" in event_names
    page_event = next(event for event in events if event["event"] == "page_completed")
    enqueue_event = next(event for event in events if event["event"] == "remaining_pages_enqueued")
    assert page_event["page"] == 1
    assert page_event["response_pages"] == 3
    assert page_event["records_inserted"] == 1
    assert enqueue_event["enqueued"] == 2
    assert enqueue_event["pages"] == [2, 3]


def test_poll_once_page_one_caps_dynamic_enqueue_at_page_sixteen_when_flickr_returns_250_perpage(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=1,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            split_reason="upload_date",
            split_depth=1,
            slice_index=0,
        )
    )

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "9000", "pages": "20", "page": "1", "perpage": "250", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, per_page FROM flickr_work_items WHERE status = 'pending' ORDER BY page").fetchall()
    assert rows == [(page, 250) for page in range(2, 17)]


def test_poll_once_page_one_caps_dynamic_enqueue_at_page_eight_when_flickr_returns_500_perpage(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=1,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            split_reason="upload_date",
            split_depth=1,
            slice_index=0,
        )
    )

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "9000", "pages": "20", "page": "1", "perpage": "500", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, per_page FROM flickr_work_items WHERE status = 'pending' ORDER BY page").fetchall()
    assert rows == [(page, 500) for page in range(2, 9)]


def test_poll_once_dynamic_enqueue_dedupes_existing_remaining_pages(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    page_one = FlickrQuery(
        term="butterfly",
        language="en",
        search_field="text",
        lane="normal_page",
        page=1,
        per_page=500,
        has_geo=0,
        min_upload_date="2007-01-01",
        max_upload_date="2007-01-05",
        split_reason="upload_date",
        split_depth=1,
        slice_index=0,
    )
    state.enqueue_work_item(page_one)
    state.enqueue_work_item(FlickrQuery(**{**page_one.__dict__, "page": 2}))

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "1200", "pages": "3", "page": "1", "perpage": "500", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, count(*) FROM flickr_work_items GROUP BY page ORDER BY page").fetchall()
    assert rows == [(1, 1), (2, 1), (3, 1)]


def test_poll_once_second_run_resumes_dynamically_enqueued_pages(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=1,
            per_page=500,
            has_geo=0,
            min_upload_date="2007-01-01",
            max_upload_date="2007-01-05",
            split_reason="upload_date",
            split_depth=1,
            slice_index=0,
        )
    )
    seen_pages: list[int] = []

    def fake_fetch(item: FlickrQuery) -> dict[str, object]:
        seen_pages.append(item.page)
        return {
            "photos": {
                "total": "1200",
                "pages": "3",
                "page": str(item.page),
                "perpage": "500",
                "photo": [{"id": str(item.page), "url_l": f"https://live.staticflickr.com/{item.page}.jpg"}],
            }
        }

    first = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=fake_fetch,
    )
    second = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=3,
        fetch_metadata=fake_fetch,
    )

    with sqlite3.connect(state.path) as conn:
        statuses = conn.execute("SELECT page, status FROM flickr_work_items ORDER BY page").fetchall()
    assert first.work_items_claimed == 1
    assert second.work_items_claimed == 2
    assert seen_pages == [1, 2, 3]
    assert statuses == [(1, "completed"), (2, "completed"), (3, "completed")]


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
    claimed = state.claim_and_reserve_pending(limit=1, max_api_calls=1, endpoint="flickr.photos.search")
    stale_time = datetime.now(UTC) - timedelta(hours=2)
    with sqlite3.connect(state.path) as conn:
        conn.execute("UPDATE flickr_work_items SET claimed_at = ? WHERE work_item_id = ?", (stale_time.isoformat(), claimed[0][0]))

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=2,
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
