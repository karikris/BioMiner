from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
import sqlite3
import threading
from pathlib import Path

import httpx
import polars as pl

from biominer.flickr_fetch.query_planner import (
    BBOX_PAGE_SIZE,
    COUNT_PROBE_PAGE_SIZE,
    DEFAULT_FIXED_SLICE_END_DATE,
    NORMAL_PAGE_SIZE,
    FlickrQuery,
    fixed_upload_date_slices,
)
from biominer.flickr_fetch.metadata_poller import MetadataPollState, _payload_page, _payload_pages, _payload_perpage, poll_once
from biominer.workstore.sqlite import SQLiteWorkStore


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
        api_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(api_call_ledger)").fetchall()
        }
        source_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(source_records)").fetchall()
        }

    assert {
        "api_call_ledger",
        "flickr_work_items",
        "source_records",
        "source_record_image_urls",
        "source_record_query_hits",
        "image_triage_queue",
    }.issubset(tables)
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
        "registry_version",
        "query_definition_id",
        "accepted_taxon_key",
        "accepted_scientific_name",
    }.issubset(work_columns)
    assert {"started_at", "finished_at", "duration_sec", "http_status"}.issubset(api_columns)
    assert {
        "text_search_terms_json",
        "tag_search_terms_json",
        "all_query_labels_json",
        "query_definition_ids_json",
        "accepted_taxon_keys_json",
        "family_keys_json",
        "genus_keys_json",
        "species_keys_json",
        "registry_versions_json",
        "query_hit_count",
        "duplicate_query_hit_count",
        "last_seen_at",
    }.issubset(source_columns)


def test_metadata_poller_migrates_existing_source_records_and_reads_legacy_query_hits(tmp_path) -> None:
    db_path = tmp_path / "poller.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE source_records (
                source TEXT NOT NULL,
                flickr_photo_id TEXT NOT NULL,
                image_url TEXT NOT NULL,
                image_url_kind TEXT NOT NULL,
                source_record_hash TEXT NOT NULL,
                query_term TEXT NOT NULL,
                query_language TEXT NOT NULL,
                query_field TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (source, flickr_photo_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO source_records (
                source, flickr_photo_id, image_url, image_url_kind, source_record_hash,
                query_term, query_language, query_field, raw_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "flickr",
                "1",
                "https://live.staticflickr.com/1_l.jpg",
                "url_l",
                "hash",
                "Papilio",
                "en",
                "text",
                json.dumps({"id": "1", "url_l": "https://live.staticflickr.com/1_l.jpg"}),
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            CREATE TABLE source_record_query_hits (
                source TEXT NOT NULL,
                flickr_photo_id TEXT NOT NULL,
                image_url TEXT NOT NULL,
                query_field TEXT NOT NULL,
                query_term TEXT NOT NULL,
                query_language TEXT NOT NULL,
                query_lane TEXT NOT NULL,
                query_page INTEGER NOT NULL,
                first_seen_at TEXT NOT NULL,
                PRIMARY KEY (
                    source, flickr_photo_id, image_url,
                    query_field, query_term, query_language
                )
            )
            """
        )
        conn.execute(
            """
            INSERT INTO source_record_query_hits (
                source, flickr_photo_id, image_url, query_field, query_term,
                query_language, query_lane, query_page, first_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "flickr",
                "1",
                "https://live.staticflickr.com/1_l.jpg",
                "tags",
                "Papilionidae",
                "en",
                "normal_page",
                1,
                "2026-01-01T00:00:00+00:00",
            ),
        )

    state = MetadataPollState(db_path)

    with sqlite3.connect(state.path) as conn:
        source_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(source_records)").fetchall()
        }
    row = state.source_records_with_query_provenance().to_dicts()[0]

    assert "all_query_labels_json" in source_columns
    assert row["all_query_labels"] == ["tags:Papilionidae"]
    assert row["tag_search_terms"] == ["Papilionidae"]
    assert row["query_hit_count"] == 1


def test_metadata_state_persists_registry_query_provenance_idempotently(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    query = FlickrQuery(
        term="Papilio demoleus",
        language="la",
        search_field="tags",
        lane="normal_page",
        page=1,
        per_page=NORMAL_PAGE_SIZE,
        has_geo=0,
        min_upload_date="2026-01-01",
        max_upload_date="2026-01-05",
        split_reason="upload_date",
        split_depth=1,
        query_definition_id="q-tags",
        registry_version="registry-v1",
        accepted_taxon_key="gbif:100",
        accepted_scientific_name="Papilio demoleus",
        family_key="gbif:10",
        genus_key="gbif:90",
        species_key="gbif:100",
    )

    assert state.enqueue_work_item(query) == 1
    assert state.enqueue_work_item(query) == 0

    with sqlite3.connect(state.path) as conn:
        row = conn.execute(
            """
            SELECT registry_version, query_definition_id, accepted_taxon_key,
                   accepted_scientific_name, family_key, genus_key, species_key
            FROM flickr_work_items
            """
        ).fetchone()

    assert row == ("registry-v1", "q-tags", "gbif:100", "Papilio demoleus", "gbif:10", "gbif:90", "gbif:100")


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
    assert result.evidence_rows_written == 2
    assert result.evidence_rows_total == 2
    assert pl.read_parquet(tmp_path / "evidence" / "poll.parquet").height == 2
    assert not list((tmp_path / "evidence" / "poll_pages").glob("*.parquet"))
    assert list((tmp_path / "raw").rglob("*.json"))
    assert not list(tmp_path.rglob("*.jpg"))
    with sqlite3.connect(state.path) as conn:
        assert conn.execute("SELECT count(*) FROM source_records").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM image_triage_queue").fetchone()[0] == 2


def test_poll_once_compacts_evidence_shards_without_duplicate_existing_rows(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly-1", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    state.enqueue_work_item(FlickrQuery(term="butterfly-2", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    evidence_output = tmp_path / "evidence" / "poll.parquet"

    first = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=evidence_output,
        max_api_calls=1,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "250",
                "photo": [{"id": "1", "title": "first", "url_l": "https://live.staticflickr.com/1.jpg"}],
            }
        },
    )
    second = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=evidence_output,
        max_api_calls=3,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "250",
                "photo": [{"id": "2", "title": "second", "url_l": "https://live.staticflickr.com/2.jpg"}],
            }
        },
    )

    frame = pl.read_parquet(evidence_output)
    assert first.evidence_rows_written == 1
    assert first.evidence_rows_total == 1
    assert second.evidence_rows_written == 2
    assert second.evidence_rows_total == 2
    assert frame.height == 2


def test_poll_once_writes_immutable_local_raw_json_and_evidence_shards(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="Papilio demoleus / lime butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    work_store = SQLiteWorkStore(tmp_path / "workstore.sqlite")

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence" / "poll.parquet",
        max_api_calls=1,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "250",
                "photo": [{"id": "1", "title": "lime butterfly", "url_l": "https://live.staticflickr.com/1.jpg"}],
            }
        },
        run_id="run-1",
        worker_id="worker-001",
        storage_prefix=tmp_path / "staging",
        compact_after_run=False,
        work_store=work_store,
    )

    raw_files = list((tmp_path / "raw").rglob("*.json"))
    evidence_shards = list((tmp_path / "staging" / "evidence" / "stage=poll_once" / "run_id=run-1" / "worker=worker-001").glob("*.parquet"))
    assert result.evidence_rows_written == 1
    assert result.evidence_rows_total == 1
    assert raw_files
    assert "source=flickr/method=photos_search/run_id=run-1/field=text/term=papilio_demoleus_lime_butterfly" in raw_files[0].as_posix()
    assert evidence_shards
    assert not (tmp_path / "evidence" / "poll.parquet").exists()
    assert pl.read_parquet(evidence_shards).height == 1
    with sqlite3.connect(work_store.path) as conn:
        shard = conn.execute("SELECT stage, run_id, worker_id, uri, row_count FROM biominer_parquet_shards").fetchone()
    assert shard[0:3] == ("poll_once", "run-1", "worker-001")
    assert shard[3].endswith(".parquet")
    assert shard[4] == 1


def test_poll_once_no_compact_writes_canonical_delta_shard_not_full_snapshot(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="Papilio", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    work_store = SQLiteWorkStore(tmp_path / "workstore.sqlite")

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence" / "poll.parquet",
        max_api_calls=1,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "250",
                "photo": [{"id": "1", "title": "first", "url_l": "https://live.staticflickr.com/1.jpg"}],
            }
        },
        run_id="run-1",
        worker_id="worker-001",
        storage_prefix=tmp_path / "staging",
        compact_after_run=False,
        work_store=work_store,
    )
    state.enqueue_work_item(FlickrQuery(term="Danaus", language="en", search_field="text", lane="normal_page", page=1, per_page=250))

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence" / "poll.parquet",
        max_api_calls=3,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "250",
                "photo": [{"id": "2", "title": "second", "url_l": "https://live.staticflickr.com/2.jpg"}],
            }
        },
        run_id="run-2",
        worker_id="worker-001",
        storage_prefix=tmp_path / "staging",
        compact_after_run=False,
        work_store=work_store,
    )

    second_shards = list((tmp_path / "staging" / "evidence" / "stage=poll_once" / "run_id=run-2" / "worker=worker-001").glob("*.parquet"))
    second_frame = pl.read_parquet(second_shards)

    assert result.evidence_rows_written == 1
    assert result.evidence_rows_total == 1
    assert second_frame["flickr_photo_id"].to_list() == ["2"]


def test_poll_once_no_compact_dedupes_duplicate_photo_with_folded_provenance_in_shard(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="Papilio", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    state.enqueue_work_item(FlickrQuery(term="Papilionidae", language="en", search_field="tags", lane="normal_page", page=1, per_page=250))

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence" / "poll.parquet",
        max_api_calls=2,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "250",
                "photo": [{"id": "123", "title": "swallowtail", "url_l": "https://live.staticflickr.com/123_l.jpg"}],
            }
        },
        run_id="run-1",
        worker_id="worker-001",
        storage_prefix=tmp_path / "staging",
        compact_after_run=False,
    )

    shards = list((tmp_path / "staging" / "evidence" / "stage=poll_once" / "run_id=run-1" / "worker=worker-001").glob("*.parquet"))
    frame = pl.read_parquet(shards)
    row = frame.to_dicts()[0]

    assert result.evidence_rows_written == 1
    assert frame.height == 1
    assert row["flickr_photo_id"] == "123"
    assert row["text_search_terms"] == ["Papilio"]
    assert row["tag_search_terms"] == ["Papilionidae"]
    assert row["all_query_labels"] == ["text:Papilio", "tags:Papilionidae"]
    assert row["query_hit_count"] == 2


def test_poll_once_old_style_local_arguments_still_write_compacted_output(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    evidence_output = tmp_path / "evidence" / "poll.parquet"

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=evidence_output,
        max_api_calls=1,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "250",
                "photo": [{"id": "1", "title": "butterfly", "url_l": "https://live.staticflickr.com/1.jpg"}],
            }
        },
    )

    assert evidence_output.exists()
    assert pl.read_parquet(evidence_output).height == 1


def test_poll_once_folds_duplicate_query_terms_onto_source_record(tmp_path) -> None:
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
        row = conn.execute(
            """
            SELECT query_field, query_term, text_search_terms_json, tag_search_terms_json,
                   all_query_labels_json, query_hit_count, duplicate_query_hit_count
            FROM source_records
            """
        ).fetchone()
        query_hits = conn.execute("SELECT count(*) FROM source_record_query_hits").fetchone()[0]
    assert row[:2] == ("text", "Papilio")
    assert json.loads(row[2]) == ["Papilio"]
    assert json.loads(row[3]) == ["Papilionidae"]
    assert json.loads(row[4]) == ["text:Papilio", "tags:Papilionidae"]
    assert row[5:] == (2, 0)
    assert query_hits == 0


def test_poll_once_keeps_one_source_record_and_tracks_image_url_history(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="Papilio", language="en", search_field="text", lane="normal_page", page=1, per_page=250))

    def fake_fetch(item: FlickrQuery) -> dict[str, object]:
        suffix = "large" if item.search_field == "text" else "medium"
        url_key = "url_l" if item.search_field == "text" else "url_m"
        return {
            "photos": {
                "total": "1",
                "photo": [
                    {
                        "id": "1",
                        "title": "swallowtail",
                        url_key: f"https://live.staticflickr.com/1_{suffix}.jpg",
                    }
                ],
            }
        }

    first = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=fake_fetch,
    )
    state.enqueue_work_item(FlickrQuery(term="Papilio", language="en", search_field="tags", lane="normal_page", page=1, per_page=250))
    second = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=3,
        fetch_metadata=fake_fetch,
    )

    with sqlite3.connect(state.path) as conn:
        source_rows = conn.execute("SELECT flickr_photo_id, image_url FROM source_records").fetchall()
        url_rows = conn.execute("SELECT image_url, image_url_kind FROM source_record_image_urls ORDER BY image_url").fetchall()
        provenance = conn.execute(
            "SELECT text_search_terms_json, tag_search_terms_json, all_query_labels_json, query_hit_count FROM source_records"
        ).fetchone()
    assert first.source_records_inserted == 1
    assert second.source_records_inserted == 0
    assert second.duplicate_records_skipped == 1
    assert source_rows == [("1", "https://live.staticflickr.com/1_large.jpg")]
    assert url_rows == [
        ("https://live.staticflickr.com/1_large.jpg", "url_l"),
        ("https://live.staticflickr.com/1_medium.jpg", "url_m"),
    ]
    assert json.loads(provenance[0]) == ["Papilio"]
    assert json.loads(provenance[1]) == ["Papilio"]
    assert json.loads(provenance[2]) == ["text:Papilio", "tags:Papilio"]
    assert provenance[3] == 2


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
    assert row["text_search_terms"] == ["Papilio"]
    assert row["tag_search_terms"] == ["Papilionidae"]
    assert row["all_query_labels"] == ["text:Papilio", "tags:Papilionidae"]
    assert row["all_query_terms"] == ["Papilio", "Papilionidae"]
    assert row["all_query_fields"] == ["text", "tags"]
    assert row["query_hit_count"] == 2
    assert row["duplicate_query_hit_count"] == 0


def test_poll_once_writes_one_evidence_row_with_folded_query_terms(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="Papilio", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    state.enqueue_work_item(FlickrQuery(term="Papilionidae", language="en", search_field="tags", lane="normal_page", page=1, per_page=250))
    evidence_output = tmp_path / "evidence" / "poll.parquet"

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=evidence_output,
        max_api_calls=2,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "250",
                "photo": [{"id": "123", "title": "swallowtail", "url_l": "https://live.staticflickr.com/123_l.jpg"}],
            }
        },
    )

    frame = pl.read_parquet(evidence_output)
    row = frame.to_dicts()[0]

    assert frame.height == 1
    assert row["text_search_terms"] == ["Papilio"]
    assert row["tag_search_terms"] == ["Papilionidae"]
    assert row["all_query_labels"] == ["text:Papilio", "tags:Papilionidae"]
    assert row["query_hit_count"] == 2


def test_poll_once_does_not_duplicate_same_field_term_on_rerun(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="Papilio", language="en", search_field="text", lane="normal_page", page=1, per_page=250))
    state.enqueue_work_item(FlickrQuery(term="Papilio", language="en", search_field="text", lane="normal_page", page=2, per_page=250))

    result = poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=2,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "2",
                "page": str(item.page),
                "perpage": "250",
                "photo": [{"id": "1", "title": "swallowtail", "url_l": "https://live.staticflickr.com/1_l.jpg"}],
            }
        },
    )

    with sqlite3.connect(state.path) as conn:
        row = conn.execute(
            "SELECT text_search_terms_json, all_query_labels_json, query_hit_count, duplicate_query_hit_count FROM source_records"
        ).fetchone()

    assert result.source_records_inserted == 1
    assert result.duplicate_records_skipped == 1
    assert result.query_hits_inserted == 1
    assert result.duplicate_query_hits_skipped == 1
    assert json.loads(row[0]) == ["Papilio"]
    assert json.loads(row[1]) == ["text:Papilio"]
    assert row[2:] == (1, 1)


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


def test_poll_once_enqueues_pages_for_count_probe_inside_result_window(tmp_path) -> None:
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
        end_date=DEFAULT_FIXED_SLICE_END_DATE,
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
        durations = [row[0] for row in conn.execute("SELECT duration_sec FROM api_call_ledger ORDER BY id").fetchall()]
    assert result.api_calls_made == 2
    assert result.raw_responses_written == 1
    assert statuses == [("failed", 1), ("ok", 1)]
    assert len(durations) == 2
    assert all(duration is not None and duration >= 0 for duration in durations)


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
                    "SELECT count(*) FROM api_call_ledger"
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
    assert 1 <= observed_before_first_fetch["claimed"] <= 2
    assert observed_before_first_fetch["reserved"] == 2
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
            page=4,
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

    assert [(query.slice_index, query.page) for _work_id, query in claimed] == [(0, 1), (0, 4), (1, 2)]


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


def test_poll_once_records_page_payload_count_for_reporting(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=4,
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
        row = conn.execute("SELECT records_returned FROM flickr_work_items WHERE page = 4").fetchone()
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
    assert page_event["known_work_items_for_query"] == 3
    assert page_event["missing_pages"] == []
    assert enqueue_event["enqueued"] == 2
    assert enqueue_event["target_pages"] == 3


def test_poll_once_enqueues_all_reported_pages_when_flickr_returns_250_perpage(tmp_path) -> None:
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
        fetch_metadata=lambda query: {"photos": {"total": "3500", "pages": "14", "page": "1", "perpage": "250", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, per_page, status FROM flickr_work_items ORDER BY page").fetchall()
    assert rows == [(1, 500, "completed"), *((page, 250, "pending") for page in range(2, 15))]


def test_poll_once_caps_executable_pages_at_flickr_accessible_window_and_warns(tmp_path) -> None:
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
        fetch_metadata=lambda query: {"photos": {"total": "9000", "pages": "20", "page": "1", "perpage": "250", "photo": []}},
        progress_callback=events.append,
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, per_page, status FROM flickr_work_items ORDER BY page").fetchall()
    page_event = next(event for event in events if event["event"] == "page_completed")
    assert rows == [(1, 500, "completed"), *((page, 250, "pending") for page in range(2, 17))]
    assert page_event["reported_pages"] == 20
    assert page_event["accessible_pages"] == 16
    assert page_event["known_work_items_for_query"] == 16
    assert any(event["event"] == "pagination_over_accessible_window" and event["response_pages"] == 20 for event in events)


def test_poll_once_uses_response_pages_not_requested_page_size_calculation(tmp_path) -> None:
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
        fetch_metadata=lambda query: {"photos": {"total": "2520", "pages": "11", "page": "1", "perpage": "250", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, per_page FROM flickr_work_items WHERE status = 'pending' ORDER BY page").fetchall()
    assert rows == [(page, 250) for page in range(2, 12)]


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


def test_poll_once_page_four_enqueues_all_missing_reported_pages(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(
        FlickrQuery(
            term="butterfly",
            language="en",
            search_field="text",
            lane="normal_page",
            page=4,
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
        fetch_metadata=lambda query: {"photos": {"total": "2520", "pages": "11", "page": "4", "perpage": "250", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, per_page, status FROM flickr_work_items ORDER BY page").fetchall()
    assert rows == [
        (1, 250, "pending"),
        (2, 250, "pending"),
        (3, 250, "pending"),
        (4, 500, "completed"),
        *((page, 250, "pending") for page in range(5, 12)),
    ]


def test_poll_once_dynamic_enqueue_preserves_existing_page_statuses(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    base = FlickrQuery(
        term="butterfly",
        language="en",
        search_field="text",
        lane="normal_page",
        page=4,
        per_page=500,
        has_geo=0,
        min_upload_date="2007-01-01",
        max_upload_date="2007-01-05",
        split_reason="upload_date",
        split_depth=1,
        slice_index=0,
    )
    for page in (1, 2, 4):
        state.enqueue_work_item(FlickrQuery(**{**base.__dict__, "page": page}))
    with sqlite3.connect(state.path) as conn:
        page_two_id = conn.execute("SELECT work_item_id FROM flickr_work_items WHERE page = 2").fetchone()[0]
    state.complete_work_item(page_two_id, records_returned=0, response_pages=11, response_page=2, response_perpage=500)

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "2520", "pages": "11", "page": "1", "perpage": "250", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, status, count(*) FROM flickr_work_items GROUP BY page, status ORDER BY page, status").fetchall()
    assert rows == [
        (1, "completed", 1),
        (2, "completed", 1),
        (3, "pending", 1),
        (4, "pending", 1),
        *((page, "pending", 1) for page in range(5, 12)),
    ]


def test_poll_once_concurrent_workers_do_not_duplicate_dynamically_enqueued_pages(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    base = FlickrQuery(
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
    state.enqueue_work_item(base)
    state.enqueue_work_item(FlickrQuery(**{**base.__dict__, "page": 4}))

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=2,
        workers=2,
        fetch_metadata=lambda query: {"photos": {"total": "3500", "pages": "14", "page": str(query.page), "perpage": "250", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, count(*) FROM flickr_work_items GROUP BY page ORDER BY page").fetchall()
    assert rows == [(page, 1) for page in range(1, 15)]


def test_poll_once_budget_limit_leaves_discovered_pages_pending(tmp_path) -> None:
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
        fetch_metadata=lambda query: {"photos": {"total": "3500", "pages": "14", "page": "1", "perpage": "250", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        pending_pages = conn.execute("SELECT page FROM flickr_work_items WHERE status = 'pending' ORDER BY page").fetchall()
    assert result.api_calls_made == 1
    assert [row[0] for row in pending_pages] == list(range(2, 15))


def test_poll_once_later_response_adds_more_pages(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    base = FlickrQuery(
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
    state.enqueue_work_item(base)
    pages_by_request = {1: 3, 2: 5}

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "1250", "pages": str(pages_by_request[query.page]), "page": str(query.page), "perpage": "250", "photo": []}},
    )
    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=2,
        fetch_metadata=lambda query: {"photos": {"total": "1250", "pages": str(pages_by_request.get(query.page, 5)), "page": str(query.page), "perpage": "250", "photo": []}},
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, count(*) FROM flickr_work_items GROUP BY page ORDER BY page").fetchall()
    assert rows == [(page, 1) for page in range(1, 6)]


def test_poll_once_later_response_with_fewer_pages_logs_discrepancy_without_deleting_work(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    base = FlickrQuery(
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
    state.enqueue_work_item(base)
    events: list[dict[str, object]] = []

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=1,
        fetch_metadata=lambda query: {"photos": {"total": "3500", "pages": "14", "page": "1", "perpage": "250", "photo": []}},
    )
    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=2,
        fetch_metadata=lambda query: {"photos": {"total": "2750", "pages": "11", "page": str(query.page), "perpage": "250", "photo": []}},
        progress_callback=events.append,
    )

    with sqlite3.connect(state.path) as conn:
        rows = conn.execute("SELECT page, count(*) FROM flickr_work_items GROUP BY page ORDER BY page").fetchall()
    assert rows == [(page, 1) for page in range(1, 15)]
    assert any(event["event"] == "pagination_metadata_changed" and event["level"] == "warning" for event in events)


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
