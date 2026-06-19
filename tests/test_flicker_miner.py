from __future__ import annotations

import sqlite3
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("flicker_miner", Path(__file__).resolve().parents[1] / "flicker_miner.py")
assert _SPEC and _SPEC.loader
flicker_miner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(flicker_miner)

build_parser = flicker_miner.build_parser
seed_time_chunks = flicker_miner.seed_time_chunks


def test_flicker_miner_parser_accepts_text_keyword_days() -> None:
    args = build_parser().parse_args(["--text", "--keyword", "butterfly", "--days", "30"])

    assert args.search_field == "text"
    assert args.keyword == "butterfly"
    assert args.days == 30


def test_seed_time_chunks_creates_2024_eleven_30_day_batches_with_page_one_only(tmp_path) -> None:
    state_db = tmp_path / "poller.sqlite"

    inserted = seed_time_chunks(
        state_db=state_db,
        keyword="butterfly",
        search_field="text",
        days=30,
        start_date="2024-01-01",
        end_date="2024-11-25",
    )

    with sqlite3.connect(state_db) as conn:
        rows = conn.execute(
            """
            SELECT json_extract(query_json, '$.search_field'), term, page, per_page, min_date, max_date
            FROM flickr_work_items
            ORDER BY min_date
            """
        ).fetchall()

    assert inserted == 11
    assert len(rows) == 11
    assert rows[0] == ("text", "butterfly", 1, 500, "2024-01-01", "2024-01-30")
    assert rows[-1] == ("text", "butterfly", 1, 500, "2024-10-27", "2024-11-25")
    assert {row[2] for row in rows} == {1}


def test_2024_batches_can_expand_to_page_sixteen_from_flickr_page_metadata(tmp_path) -> None:
    from biominer.flickr_fetch.metadata_poller import poll_once

    state_db = tmp_path / "poller.sqlite"
    seed_time_chunks(
        state_db=state_db,
        keyword="butterfly",
        search_field="text",
        days=30,
        start_date="2024-01-01",
        end_date="2024-11-25",
    )

    result = poll_once(
        state_db=state_db,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence.parquet",
        max_api_calls=176,
        fetch_metadata=lambda query: {
            "photos": {
                "total": "9000",
                "pages": "20" if query.page == 1 else "16",
                "page": str(query.page),
                "perpage": "250",
                "photo": [{"id": f"{query.min_upload_date}-{query.page}", "url_l": f"https://live.staticflickr.com/{query.page}.jpg"}],
            }
        },
    )

    with sqlite3.connect(state_db) as conn:
        page_counts = conn.execute(
            """
            SELECT page, count(*)
            FROM flickr_work_items
            GROUP BY page
            ORDER BY page
            """
        ).fetchall()
        statuses = conn.execute("SELECT status, count(*) FROM flickr_work_items GROUP BY status").fetchall()

    assert page_counts == [(page, 11) for page in range(1, 17)]
    assert statuses == [("completed", 176)]
    assert result.work_items_claimed == 176
