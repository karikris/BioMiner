from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_papilio_demoleus_ranked_flickr_slices.py"
    spec = importlib.util.spec_from_file_location("run_papilio_demoleus_ranked_flickr_slices", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ranked_slice_plan_uses_requested_groups_and_dedupes_annual_terms() -> None:
    runner = _load_runner()

    queries = runner.build_ranked_slice_queries(end_date="2026-07-08")

    assert len(queries) == 247
    assert len({runner.query_hash(query) for query in queries}) == 247
    assert len(runner.six_year_query_specs()) == 10
    assert len(runner.one_year_query_specs()) == 9

    lime_text = [query for query in queries if query.term == "Lime Butterfly" and query.search_field == "text"]
    lime_tags = [query for query in queries if query.term == "Lime Butterfly" and query.search_field == "tags"]
    lemon_text = [query for query in queries if query.term == "Lemon Butterfly" and query.search_field == "text"]
    citrus_text = [query for query in queries if query.term == "Citrus Swallowtail" and query.search_field == "text"]
    citrus_tags = [query for query in queries if query.term == "Citrus Swallowtail" and query.search_field == "tags"]

    assert len(lime_text) == 23
    assert len(lemon_text) == 23
    assert len(lime_tags) == 4
    assert len(citrus_text) == 23
    assert len(citrus_tags) == 4
    assert all(query.page == 1 and query.per_page == 250 and query.has_geo == 1 for query in queries)
    assert all(query.lane == "normal_page" and query.split_reason == "upload_date" for query in queries)

    annual_ranges = runner.calendar_year_upload_ranges(start_date="2004-02-10", end_date="2026-07-08")
    assert annual_ranges[0] == ("2004-02-10", "2004-12-31")
    assert annual_ranges[1] == ("2005-01-01", "2005-12-31")
    assert annual_ranges[-1] == ("2026-01-01", "2026-07-08")
    assert len(annual_ranges) == 23
    assert runner.six_year_upload_ranges(end_date="2026-07-08") == (
        ("2004-02-10", "2009-12-31"),
        ("2010-01-01", "2015-12-31"),
        ("2016-01-01", "2021-12-31"),
        ("2022-01-01", "2026-07-08"),
    )


def test_ranked_slice_runner_seeds_page_one_work_items(tmp_path) -> None:
    runner = _load_runner()
    state_db = tmp_path / "flickr_poller.sqlite"

    inserted = runner.seed_ranked_slice_work(state_db=state_db, end_date="2026-07-08")

    assert inserted == 247
    with sqlite3.connect(state_db) as conn:
        total, page_one, page_two_plus = conn.execute(
            """
            SELECT count(*),
                   sum(CASE WHEN page = 1 THEN 1 ELSE 0 END),
                   sum(CASE WHEN page > 1 THEN 1 ELSE 0 END)
            FROM flickr_work_items
            """
        ).fetchone()
        lime_rows = conn.execute(
            """
            SELECT search_field, count(*), min(min_date), max(max_date)
            FROM flickr_work_items
            WHERE term = 'Lime Butterfly'
            GROUP BY search_field
            ORDER BY search_field
            """
        ).fetchall()

    assert (total, page_one, page_two_plus) == (247, 247, 0)
    assert lime_rows == [
        ("tags", 4, "2004-02-10", "2026-07-08"),
        ("text", 23, "2004-02-10", "2026-07-08"),
    ]


def test_previous_positive_keywords_are_queued_before_unmatched_keywords():
    import polars as pl
    runner = _load_runner()
    queries = runner.build_ranked_slice_queries(start_date="2026-01-01", end_date="2026-07-08")
    previous = pl.DataFrame({"all_query_labels": [["text:Small Citrus Butterfly"]]})
    prioritized = runner.prioritize_previous_matches(queries, previous)
    assert prioritized[0].term == "Small Citrus Butterfly"
    assert {(q.term, q.search_field) for q in prioritized} == {(q.term, q.search_field) for q in queries}
    assert all(q.has_geo == 1 for q in prioritized)


def test_non_geotagged_plan_preserves_keyword_fields_and_full_utc_days():
    from datetime import UTC, datetime

    from biominer.flickr_fetch.query_planner import flickr_search_params

    runner = _load_runner()
    queries = runner.build_ranked_slice_queries(
        start_date="2026-07-08", end_date="2026-11-17", has_geo=0,
        utc_upload_bounds=True,
    )
    assert len(queries) == len({runner.query_hash(q) for q in queries}) == 19
    assert len({q.term for q in queries}) == 15
    pairs = {(q.term, q.search_field) for q in queries}
    assert {("Papilio demoleus", "text"), ("Papilio demoleus", "tags"),
            ("無尾鳳蝶", "text"), ("無尾鳳蝶", "tags"),
            ("Lime Butterfly", "text"), ("Lime Butterfly", "tags"),
            ("Citrus Swallowtail", "text"), ("Citrus Swallowtail", "tags")} <= pairs
    assert all(q.has_geo == 0 and q.per_page == 500 for q in queries)
    assert all(flickr_search_params(q)["has_geo"] == 0 for q in queries)
    assert {q.min_upload_date for q in queries} == {
        str(int(datetime(2026, 7, 8, tzinfo=UTC).timestamp()))}
    assert {q.max_upload_date for q in queries} == {
        str(int(datetime(2026, 11, 17, 23, 59, 59, tzinfo=UTC).timestamp()))}


def test_non_geotagged_run_refuses_geotagged_api_records():
    runner = _load_runner()
    runner.validate_collection_records([{"latitude": 0, "longitude": 0, "accuracy": 0}, {}], has_geo=0)
    with pytest.raises(ValueError, match="has_geo=0"):
        runner.validate_collection_records([{"latitude": 0, "longitude": 30, "accuracy": 16}], has_geo=0)


def test_non_geotagged_cli_plan_never_loads_credentials(tmp_path, monkeypatch):
    import json
    import sys

    runner = _load_runner()
    def credentials_must_not_load(**kwargs):
        raise AssertionError("Plan-only loaded a credential")
    monkeypatch.setattr(runner, "_load_api_key", credentials_must_not_load)
    monkeypatch.setattr(sys, "argv", ["runner", "--run-id", "synthetic-nongeo",
        "--output-root", str(tmp_path), "--report-dir", str(tmp_path / "reports"),
        "--start-date", "2026-07-08", "--end-date", "2026-11-17",
        "--has-geo", "0", "--utc-upload-bounds", "--plan-only"])
    assert runner.main() == 0
    plan = json.loads((tmp_path / "reports/synthetic-nongeo_query_plan.json").read_text())
    assert plan["query_count"] == 19
    assert plan["geotag_policy"].startswith("has_geo=0")
    assert plan["upload_date_timezone"] == "UTC"
