from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sqlite3

import polars as pl
import pytest

from biominer.flickr_fetch.metadata_poller import PollOnceResult
from biominer.reports.flickr_fetch import build_step1_fetch_report, write_step1_manifest
from biominer.flickr_fetch.metadata_poller import MetadataPollState
from biominer.flickr_fetch.query_planner import FlickrQuery


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
        evidence_rows_total=500,
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


@pytest.mark.filterwarnings("error:.*empty_as_null.*:DeprecationWarning")
@pytest.mark.parametrize("reverse", [False, True])
def test_fetch_report_preserves_label_counts_with_empty_null_and_duplicate_lists(
    tmp_path, reverse: bool,
) -> None:
    provenance = pl.DataFrame(
        {
            "all_query_labels": [
                [], None, [None], ["text:A", "tags:A", None, "text:A"],
                ["tags:A"], ["text:B", "text:A", "text:B"],
            ],
            "query_hit_count": [0, 0, 1, 3, 1, 3],
        },
        schema={"all_query_labels": pl.List(pl.String), "query_hit_count": pl.UInt32},
    )
    if reverse:
        provenance = provenance.reverse().with_columns(pl.col("all_query_labels").list.reverse())

    summary = _public_query_report(tmp_path, provenance)["query_provenance"]

    assert summary["unique_query_labels_with_records"] == 3
    assert summary["duplicate_records_with_additional_query_hits"] == 2
    assert summary["query_hit_count_distribution"] == [
        {"query_hit_count": 0, "records": 2},
        {"query_hit_count": 1, "records": 2},
        {"query_hit_count": 3, "records": 2},
    ]
    assert summary["top_query_labels_by_records"] == [
        {"query_label": "text:A", "records": 3},
        {"query_label": "tags:A", "records": 2},
        {"query_label": "text:B", "records": 2},
    ]


@pytest.mark.filterwarnings("error:.*empty_as_null.*:DeprecationWarning")
def test_fetch_report_counts_no_labels_without_erasing_observed_query_hit_counts(tmp_path) -> None:
    provenance = pl.DataFrame(
        {"all_query_labels": [[], None, [None]], "query_hit_count": [0, 0, 1]},
        schema={"all_query_labels": pl.List(pl.String), "query_hit_count": pl.UInt32},
    )

    summary = _public_query_report(tmp_path, provenance)["query_provenance"]

    assert summary == {
        "unique_query_labels_with_records": 0,
        "duplicate_records_with_additional_query_hits": 0,
        "query_hit_count_distribution": [
            {"query_hit_count": 0, "records": 2}, {"query_hit_count": 1, "records": 1},
        ],
        "top_query_labels_by_records": [],
    }


@pytest.mark.parametrize("provenance", [
    None, pl.DataFrame(schema={"all_query_labels": pl.List(pl.String), "query_hit_count": pl.UInt32}),
])
def test_fetch_report_marks_absent_query_provenance_uninstrumented(tmp_path, provenance) -> None:
    summary = _public_query_report(tmp_path, provenance)["query_provenance"]

    assert set(summary.values()) == {"not_instrumented"}


def _public_query_report(tmp_path, provenance):
    result = PollOnceResult(
        state_db=tmp_path / "state.sqlite", raw_responses_written=0,
        evidence_rows_written=0, evidence_rows_total=0, source_records_inserted=0,
        duplicate_records_skipped=0, query_hits_inserted=0, duplicate_query_hits_skipped=0,
        image_urls_queued=0, work_items_claimed=0, api_calls_made=0,
        remaining_soft_budget=1, remaining_hard_budget=1, stale_claims_requeued=0,
    )
    return build_step1_fetch_report(
        run_id="query-label-fixture", command=["biominer", "poll-once"], result=result,
        raw_root=tmp_path / "raw", evidence_output=tmp_path / "evidence.parquet",
        query_provenance=provenance, started_at=datetime(2026, 6, 12, tzinfo=UTC),
        ended_at=datetime(2026, 6, 12, tzinfo=UTC), workers=1, expected_pages=0,
        status="completed", git_sha="fixture",
    )


def test_build_step1_fetch_report_uses_recorded_api_call_timings(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "state.sqlite")
    with sqlite3.connect(state.path) as conn:
        for duration in (0.1, 0.2, 0.3):
            conn.execute(
                """
                INSERT INTO api_call_ledger(endpoint, work_item_id, status, created_at, started_at, finished_at, duration_sec)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("flickr.photos.search", f"work-{duration}", "ok", 1.0, 1.0, 1.0 + duration, duration),
            )
    result = PollOnceResult(
        state_db=state.path,
        raw_responses_written=3,
        evidence_rows_written=3,
        evidence_rows_total=3,
        source_records_inserted=3,
        duplicate_records_skipped=0,
        query_hits_inserted=3,
        duplicate_query_hits_skipped=0,
        image_urls_queued=3,
        work_items_claimed=3,
        api_calls_made=3,
        remaining_soft_budget=3497,
        remaining_hard_budget=3597,
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
        expected_pages=3,
        status="completed",
    )

    assert round(report["timings"]["avg_sec_per_call"], 6) == 0.2
    assert report["timings"]["p50_call_sec"] == 0.2
    assert report["timings"]["p95_call_sec"] == 0.3


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
    completed_id = page_id = None
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
        evidence_rows_total=500,
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


def test_build_step1_fetch_report_includes_dynamic_page_enqueue_metrics(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "state.sqlite")
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
    for page in range(2, 17):
        state.enqueue_work_item(FlickrQuery(**{**page_one.__dict__, "page": page, "per_page": 250}))
    with sqlite3.connect(state.path) as conn:
        conn.execute(
            """
            UPDATE flickr_work_items
            SET status = 'completed',
                records_returned = 500,
                response_total = 9000,
                response_pages = 20,
                response_page = 1,
                response_perpage = 250
            WHERE page = 1
            """
        )
    result = PollOnceResult(
        state_db=state.path,
        raw_responses_written=1,
        evidence_rows_written=500,
        evidence_rows_total=500,
        source_records_inserted=500,
        duplicate_records_skipped=0,
        query_hits_inserted=500,
        duplicate_query_hits_skipped=0,
        image_urls_queued=500,
        work_items_claimed=1,
        api_calls_made=1,
        remaining_soft_budget=3499,
        remaining_hard_budget=3599,
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
        expected_pages=20,
        status="completed",
    )

    assert report["work"]["slice_page1_completed"] == 1
    assert report["work"]["remaining_pages_enqueued_from_page1"] == 15
    assert report["work"]["empty_or_single_page_slices"] == 0
    assert report["work"]["reported_pagination_gaps"] == [
        {
            "date_kind": "upload_date",
            "min_date": "2007-01-01",
            "max_date": "2007-01-05",
            "response_pages": 20,
            "highest_known_page": 16,
            "known_pages": 16,
        }
    ]
    assert report["work"]["reported_over_16_page_slices"] == [
        {
            "lane": "normal_page",
            "term": "butterfly",
            "date_kind": "upload_date",
            "min_date": "2007-01-01",
            "max_date": "2007-01-05",
            "bbox_label": "",
            "response_pages": 20,
            "response_perpage": 250,
            "response_total": 9000,
        }
    ]
