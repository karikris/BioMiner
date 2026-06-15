from __future__ import annotations

from datetime import UTC, datetime, timedelta
import importlib.util
from pathlib import Path
import sqlite3

import polars as pl

from biominer.flickr_fetch.metadata_poller import PollOnceResult
from biominer.reports.flickr_fetch import build_step1_fetch_report, write_step1_manifest
from biominer.flickr_fetch.metadata_poller import MetadataPollState
from biominer.flickr_fetch.query_planner import FlickrQuery


def _load_fetch_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_flickr_text_search.py"
    spec = importlib.util.spec_from_file_location("run_flickr_text_search", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_step1_manifest_records_background_run_contract(tmp_path) -> None:
    manifest_path = tmp_path / "reports" / "manifest.json"

    payload = write_step1_manifest(
        manifest_path,
        run_id="flickr_text_butterfly_20260612",
        command=["biominer", "poll-once"],
        expected_outputs={
            "state_db": "data/state/flickr_text_butterfly.sqlite",
            "raw_root": "data/raw/flickr_text_butterfly",
            "evidence_output": "staging/evidence/flickr_text_butterfly.parquet",
        },
        expected_pages=3362,
        status="running",
        pid=12345,
        started_at="2026-06-12T00:00:00+00:00",
        git_sha="abc123",
    )

    assert manifest_path.exists()
    assert payload["run_id"] == "flickr_text_butterfly_20260612"
    assert payload["status"] == "running"
    assert payload["pid"] == 12345
    assert payload["expected_pages"] == 3362
    assert payload["expected_outputs"]["evidence_output"].endswith("flickr_text_butterfly.parquet")
    assert "environment" in payload
    assert "secrets" not in payload["environment"]


def test_build_step1_fetch_report_includes_required_metrics(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "a.json").write_text("{}", encoding="utf-8")
    evidence = tmp_path / "evidence.parquet"
    evidence.write_bytes(b"parquet")
    state_db = tmp_path / "state.sqlite"
    state_db.write_bytes(b"sqlite")
    started_at = datetime(2026, 6, 12, tzinfo=UTC)
    ended_at = started_at + timedelta(seconds=20)
    result = PollOnceResult(
        state_db=state_db,
        raw_responses_written=2,
        evidence_rows_written=500,
        source_records_inserted=450,
        duplicate_records_skipped=50,
        query_hits_inserted=480,
        duplicate_query_hits_skipped=20,
        image_urls_queued=450,
        work_items_claimed=2,
        api_calls_made=2,
        remaining_soft_budget=3498,
        remaining_hard_budget=3598,
        stale_claims_requeued=0,
    )

    report = build_step1_fetch_report(
        run_id="run",
        command=["biominer", "poll-once"],
        result=result,
        raw_root=raw_root,
        evidence_output=evidence,
        query_provenance=pl.DataFrame(
            [
                {
                    "first_query_label": "text:Papilio",
                    "all_query_labels": ["text:Papilio", "tags:Papilio"],
                    "query_hit_count": 2,
                },
                {
                    "first_query_label": "text:Papilionidae",
                    "all_query_labels": ["text:Papilionidae"],
                    "query_hit_count": 1,
                },
            ]
        ),
        started_at=started_at,
        ended_at=ended_at,
        workers=4,
        expected_pages=3362,
        status="completed",
        pid=12345,
        git_sha="abc123",
    )

    assert report["status"] == "completed"
    assert report["api_budget"]["api_calls_used"] == 2
    assert report["api_budget"]["remaining_soft_budget"] == 3498
    assert report["throughput"]["records_per_call"] == 250
    assert report["timings"]["total_sec"] == 20
    assert report["storage_bytes"]["raw_json_bytes"] == 2
    assert report["storage_bytes"]["evidence_parquet_bytes"] == len(b"parquet")
    assert report["memory"]["max_rss_kb"] is not None
    assert report["gpu_memory"] == "not_instrumented"
    assert report["query_provenance"]["unique_query_labels_with_records"] == 3
    assert report["query_provenance"]["duplicate_records_with_additional_query_hits"] == 1
    assert report["query_provenance"]["top_query_labels_by_records"][0] == {"query_label": "tags:Papilio", "records": 1}


def test_build_step1_fetch_report_includes_split_progress_metrics(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "state.sqlite")
    completed = FlickrQuery(
        term="butterfly",
        language="en",
        search_field="text",
        lane="count_probe",
        min_upload_date="2020-01-01",
        max_upload_date="2020-12-31",
        split_reason="upload_date",
        split_depth=1,
    )
    pending = FlickrQuery(
        term="butterfly",
        language="en",
        search_field="text",
        lane="count_probe",
        min_upload_date="2021-01-01",
        max_upload_date="2021-12-31",
        split_reason="upload_date",
        split_depth=1,
    )
    page = FlickrQuery(term="butterfly", language="en", search_field="text", lane="normal_page", page=1, per_page=500, has_geo=0)
    completed_id = next_id = page_id = None
    state.enqueue_work_item(completed)
    state.enqueue_work_item(pending)
    state.enqueue_work_item(page)
    with sqlite3.connect(state.path) as conn:
        completed_id = conn.execute("SELECT work_item_id FROM flickr_work_items WHERE min_date = '2020-01-01'").fetchone()[0]
        page_id = conn.execute("SELECT work_item_id FROM flickr_work_items WHERE lane = 'normal_page'").fetchone()[0]
    state.complete_work_item(completed_id)
    state.complete_work_item(page_id)
    result = PollOnceResult(
        state_db=state.path,
        raw_responses_written=1,
        evidence_rows_written=500,
        source_records_inserted=450,
        duplicate_records_skipped=50,
        query_hits_inserted=450,
        duplicate_query_hits_skipped=0,
        image_urls_queued=450,
        work_items_claimed=1,
        api_calls_made=1,
        remaining_soft_budget=0,
        remaining_hard_budget=100,
        stale_claims_requeued=0,
    )

    report = build_step1_fetch_report(
        run_id="run",
        command=["biominer", "poll-once"],
        result=result,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        started_at=datetime(2026, 6, 12, tzinfo=UTC),
        ended_at=datetime(2026, 6, 12, 0, 0, 10, tzinfo=UTC),
        workers=1,
        expected_pages=1,
        status="completed",
    )

    assert report["api_budget"]["budget_limited_exit"] is True
    assert report["work"]["count_probes_completed"] == 1
    assert report["work"]["page_fetches_completed"] == 1
    assert report["work"]["split_probes_enqueued_by_reason"] == {"upload_date": 2}
    assert report["work"]["pending_count_probes"] == 1
    assert report["work"]["pending_page_fetches"] == 0
    assert report["work"]["completed_date_slices"] == 1
    assert report["work"]["pending_date_slices"] == 1
    assert report["work"]["last_completed_date_range"] == {"date_kind": "upload_date", "min_date": "2020-01-01", "max_date": "2020-12-31"}
    assert report["work"]["next_pending_date_range"] == {"date_kind": "upload_date", "min_date": "2021-01-01", "max_date": "2021-12-31"}
    assert report["rows"]["records_fetched"] == 500
    assert report["rows"]["records_inserted"] == 450
    assert report["throughput"]["records_per_page"] == 500


def test_enqueue_count_probe_supports_tag_search(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "state.sqlite")
    script = _load_fetch_script()

    inserted = script._enqueue_count_probe(state, term="butterfly", search_field="tags")

    assert inserted == 1
    with sqlite3.connect(state.path) as conn:
        rows = conn.execute(
            "SELECT json_extract(query_json, '$.search_field'), page, per_page, json_extract(query_json, '$.has_geo') "
            "FROM flickr_work_items ORDER BY page"
        ).fetchall()
    assert rows == [("tags", 1, 1, 0)]


def test_direct_page_enqueue_rejects_unsafe_ranges(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "state.sqlite")
    script = _load_fetch_script()

    try:
        script._enqueue_direct_pages(state, term="butterfly", pages=9, search_field="text", has_geo=0, unsafe=False)
    except ValueError as exc:
        assert "4000" in str(exc)
    else:  # pragma: no cover - test should fail before this branch.
        raise AssertionError("unsafe direct page range should be rejected")
