from __future__ import annotations

import json
import sqlite3
import time

import httpx

from biominer.flickr_fetch.enrichment import GET_INFO_METHOD, enrich_once
from biominer.flickr_fetch.metadata_poller import MetadataPollState


def test_enrichment_stores_one_immutable_get_info_result_per_photo(tmp_path) -> None:
    state_db = tmp_path / "flickr.sqlite"
    _seed_source_record(state_db, photo_id="42")

    result = enrich_once(
        state_db=state_db,
        fetch_info=lambda photo_id: {
            "stat": "ok",
            "photo": {"id": photo_id, "title": {"_content": "Butterfly"}},
        },
        max_api_calls=10,
    )

    with sqlite3.connect(state_db) as conn:
        stored = conn.execute(
            """
            SELECT endpoint, payload_json, payload_sha256
            FROM flickr_enrichment_results
            WHERE flickr_photo_id = '42'
            """
        ).fetchone()
        ledger_endpoint = conn.execute(
            "SELECT endpoint FROM api_call_ledger ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    assert result.records_enriched == 1
    assert result.completed_records == 1
    assert stored is not None
    assert json.loads(stored[1])["photo"]["id"] == "42"
    assert len(stored[2]) == 64
    assert stored[0] == ledger_endpoint == GET_INFO_METHOD


def test_enrichment_resume_does_not_call_completed_photo_twice(tmp_path) -> None:
    state_db = tmp_path / "flickr.sqlite"
    _seed_source_record(state_db, photo_id="42")
    calls: list[str] = []

    first = enrich_once(
        state_db=state_db,
        fetch_info=lambda photo_id: calls.append(photo_id)
        or {"stat": "ok", "photo": {"id": photo_id}},
        max_api_calls=10,
    )
    second = enrich_once(
        state_db=state_db,
        fetch_info=lambda photo_id: calls.append(photo_id)
        or {"stat": "ok", "photo": {"id": photo_id}},
        max_api_calls=10,
    )

    assert first.records_enriched == 1
    assert second.work_items_enqueued == 0
    assert second.records_enriched == 0
    assert calls == ["42"]


def test_enrichment_obeys_discovery_calls_in_shared_hourly_ledger(tmp_path) -> None:
    state_db = tmp_path / "flickr.sqlite"
    _seed_source_record(state_db, photo_id="42")
    with sqlite3.connect(state_db) as conn:
        conn.execute(
            """
            INSERT INTO api_call_ledger (
                endpoint, work_item_id, status, created_at, started_at
            ) VALUES ('flickr.photos.search', 'search:1', 'ok', ?, ?)
            """,
            (time.time(), time.time()),
        )
    calls: list[str] = []

    result = enrich_once(
        state_db=state_db,
        fetch_info=lambda photo_id: calls.append(photo_id)
        or {"stat": "ok", "photo": {"id": photo_id}},
        max_api_calls=1,
    )

    assert result.remaining_soft_budget == 0
    assert result.pending_records == 1
    assert result.records_enriched == 0
    assert calls == []


def test_enrichment_retries_transient_failure_with_exponential_policy(tmp_path) -> None:
    state_db = tmp_path / "flickr.sqlite"
    _seed_source_record(state_db, photo_id="42")
    calls = 0

    def fetch(photo_id: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("temporary timeout")
        return {"stat": "ok", "photo": {"id": photo_id}}

    result = enrich_once(
        state_db=state_db,
        fetch_info=fetch,
        max_api_calls=10,
        max_retries=2,
        retry_backoff_seconds=0,
    )

    with sqlite3.connect(state_db) as conn:
        attempts = conn.execute(
            """
            SELECT attempt_count FROM flickr_enrichment_work_items
            WHERE flickr_photo_id = '42'
            """
        ).fetchone()[0]
        statuses = [
            row[0]
            for row in conn.execute(
                "SELECT status FROM api_call_ledger ORDER BY id"
            ).fetchall()
        ]
    assert result.records_enriched == 1
    assert result.api_calls_made == 2
    assert attempts == 2
    assert statuses == ["failed", "ok"]


def test_enrichment_keeps_exhausted_transient_failure_retryable(tmp_path) -> None:
    state_db = tmp_path / "flickr.sqlite"
    _seed_source_record(state_db, photo_id="42")

    result = enrich_once(
        state_db=state_db,
        fetch_info=lambda _photo_id: (_ for _ in ()).throw(
            httpx.ReadTimeout("temporary timeout")
        ),
        max_api_calls=10,
        max_retries=0,
        retry_backoff_seconds=2,
    )

    with sqlite3.connect(state_db) as conn:
        status, next_attempt_at, error = conn.execute(
            """
            SELECT status, next_attempt_at, error
            FROM flickr_enrichment_work_items
            WHERE flickr_photo_id = '42'
            """
        ).fetchone()
    assert result.records_deferred == 1
    assert status == "pending"
    assert next_attempt_at > time.time()
    assert "temporary timeout" in error


def _seed_source_record(state_db, *, photo_id: str) -> None:  # noqa: ANN001
    MetadataPollState(state_db)
    with sqlite3.connect(state_db) as conn:
        conn.execute(
            """
            INSERT INTO source_records (
                source, flickr_photo_id, image_url, image_url_kind,
                source_record_hash, query_term, query_language,
                query_field, raw_json, created_at
            ) VALUES ('flickr', ?, ?, 'url_l', ?, 'butterfly', 'en',
                      'text', '{}', '2026-07-21T00:00:00+00:00')
            """,
            (
                photo_id,
                f"https://live.staticflickr.com/example/{photo_id}.jpg",
                f"hash-{photo_id}",
            ),
        )
