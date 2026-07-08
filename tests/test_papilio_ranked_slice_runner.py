from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3


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
    assert all(query.page == 1 and query.per_page == 500 and query.has_geo == 0 for query in queries)
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
