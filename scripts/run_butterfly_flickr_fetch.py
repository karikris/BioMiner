from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

import httpx

from flickr_bio_occurrence.flickr.client import DEFAULT_EXTRAS
from flickr_bio_occurrence.flickr.endpoints import FLICKR_REST_BASE_URL, SEARCH_METHOD
from flickr_bio_occurrence.flickr.rate_limiter import FlickrRateLimiter, RateLimitExceeded
from flickr_bio_occurrence.flickr.butterfly_terms import (
    ButterflySearchTerm,
    estimate_minimum_fetch_hours,
    load_butterfly_dashboard_terms,
    safe_query_variant,
)


AUSTRALIA_BBOX = "112.92,-43.74,153.64,-10.05"
DEFAULT_OUTPUT_ROOT = Path("/home/toffe/BioMiner/data/flickr_butterfly_australia")


@dataclass(frozen=True)
class FetchConfig:
    output_root: Path
    dashboard_data_dir: Path
    api_key_env: str
    max_pages_per_term: int
    per_page: int
    soft_api_calls_per_hour: int
    hard_api_calls_per_hour: int
    start_taken_date: str
    end_taken_date: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dashboard-data-dir", default="/home/toffe/butterfly-dashboard/data")
    parser.add_argument("--api-key-env", default="FLICKR_API_KEY")
    parser.add_argument("--max-pages-per-term", type=int, default=16)
    parser.add_argument("--per-page", type=int, default=250)
    parser.add_argument("--soft-api-calls-per-hour", type=int, default=3200)
    parser.add_argument("--hard-api-calls-per-hour", type=int, default=3600)
    parser.add_argument("--start-taken-date", default="1950-01-01")
    parser.add_argument("--end-taken-date", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = FetchConfig(
        output_root=Path(args.output_root),
        dashboard_data_dir=Path(args.dashboard_data_dir),
        api_key_env=args.api_key_env,
        max_pages_per_term=args.max_pages_per_term,
        per_page=min(args.per_page, 250),
        soft_api_calls_per_hour=args.soft_api_calls_per_hour,
        hard_api_calls_per_hour=args.hard_api_calls_per_hour,
        start_taken_date=args.start_taken_date,
        end_taken_date=args.end_taken_date,
    )
    terms = load_butterfly_dashboard_terms(config.dashboard_data_dir)
    plan = build_plan(config, terms)
    write_json(config.output_root / "fetch_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"{config.api_key_env} is not set")
    run_fetch(config, terms, api_key)


def build_plan(config: FetchConfig, terms: list[ButterflySearchTerm]) -> dict[str, Any]:
    planned_calls = len(terms) * config.max_pages_per_term
    return {
        "official_api_method": SEARCH_METHOD,
        "dashboard_data_dir": str(config.dashboard_data_dir),
        "output_root": str(config.output_root),
        "terms": len(terms),
        "max_pages_per_term": config.max_pages_per_term,
        "planned_upper_bound_api_calls": planned_calls,
        "minimum_hours_at_3600_calls_per_hour": estimate_minimum_fetch_hours(
            planned_api_calls=planned_calls,
            api_calls_per_hour=config.hard_api_calls_per_hour,
        ),
        "soft_api_calls_per_hour": config.soft_api_calls_per_hour,
        "hard_api_calls_per_hour": config.hard_api_calls_per_hour,
        "per_page": config.per_page,
        "bbox": AUSTRALIA_BBOX,
        "start_taken_date": config.start_taken_date,
        "end_taken_date": config.end_taken_date,
    }


def run_fetch(config: FetchConfig, terms: list[ButterflySearchTerm], api_key: str) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    state = FetchState(config.output_root / "fetch_state.sqlite")
    limiter = FlickrRateLimiter(
        config.output_root / "rate_limits.sqlite",
        soft_api_calls_per_hour=config.soft_api_calls_per_hour,
        hard_api_calls_per_hour=config.hard_api_calls_per_hour,
        hard_photo_records_per_hour=3600,
    )
    with httpx.Client(timeout=30) as client:
        for term in terms:
            if state.term_is_exhausted(term.term):
                continue
            for page in range(1, config.max_pages_per_term + 1):
                work_item_id = f"{safe_query_variant(term.term)}:{page}"
                if state.is_done(work_item_id):
                    continue
                per_page = wait_for_photo_record_capacity(limiter, config.per_page)
                payload = fetch_one_page(config, client, limiter, api_key, term, page, work_item_id, per_page)
                photo_rows = payload.get("photos", {}).get("photo", [])
                photo_ids = [str(photo["id"]) for photo in photo_rows if isinstance(photo, dict) and "id" in photo]
                new_photo_ids = limiter.log_photo_records(photo_ids, work_item_id)
                raw_path = write_raw_payload(config.output_root, term, page, payload)
                pages = int(payload.get("photos", {}).get("pages") or page)
                state.mark_done(
                    work_item_id=work_item_id,
                    term=term.term,
                    term_source=term.source,
                    page=page,
                    raw_path=raw_path,
                    returned_records=len(photo_ids),
                    new_records=len(new_photo_ids),
                    flickr_pages=pages,
                )
                if not photo_ids or page >= pages:
                    break


def wait_for_photo_record_capacity(limiter: FlickrRateLimiter, requested: int) -> int:
    while True:
        allowed = limiter.reserve_photo_record_slots(requested)
        if allowed > 0:
            return allowed
        time.sleep(60)


def fetch_one_page(
    config: FetchConfig,
    client: httpx.Client,
    limiter: FlickrRateLimiter,
    api_key: str,
    term: ButterflySearchTerm,
    page: int,
    work_item_id: str,
    per_page: int,
) -> dict[str, Any]:
    while True:
        try:
            limiter.acquire_api_token(SEARCH_METHOD, work_item_id)
            break
        except RateLimitExceeded:
            time.sleep(60)
    response = client.get(
        FLICKR_REST_BASE_URL,
        params={
            "method": SEARCH_METHOD,
            "api_key": api_key,
            "text": term.term,
            "bbox": AUSTRALIA_BBOX,
            "min_taken_date": config.start_taken_date,
            "max_taken_date": config.end_taken_date,
            "has_geo": 1,
            "media": "photos",
            "content_types": "0",
            "safe_search": 1,
            "extras": DEFAULT_EXTRAS,
            "per_page": per_page,
            "page": page,
            "format": "json",
            "nojsoncallback": 1,
        },
    )
    status = "ok" if response.is_success else f"http_{response.status_code}"
    limiter.log_call(SEARCH_METHOD, work_item_id, status)
    response.raise_for_status()
    return response.json()


def write_raw_payload(output_root: Path, term: ButterflySearchTerm, page: int, payload: dict[str, Any]) -> Path:
    term_slug = safe_query_variant(term.term)
    target_dir = output_root / "raw" / "flickr" / "photos_search" / term.source / term_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"page={page:05d}.json"
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return target


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class FetchState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS completed_work_items (
                    work_item_id TEXT PRIMARY KEY,
                    term TEXT NOT NULL,
                    term_source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    raw_path TEXT NOT NULL,
                    returned_records INTEGER NOT NULL,
                    new_records INTEGER NOT NULL,
                    flickr_pages INTEGER NOT NULL,
                    completed_at REAL NOT NULL
                )
                """
            )

    def is_done(self, work_item_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM completed_work_items WHERE work_item_id = ?",
                (work_item_id,),
            ).fetchone() is not None

    def term_is_exhausted(self, term: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT max(page), max(flickr_pages), min(returned_records)
                FROM completed_work_items
                WHERE term = ?
                """,
                (term,),
            ).fetchone()
        if row is None or row[0] is None:
            return False
        max_page, flickr_pages, min_returned = int(row[0]), int(row[1]), int(row[2])
        return max_page >= flickr_pages or min_returned == 0

    def mark_done(
        self,
        *,
        work_item_id: str,
        term: str,
        term_source: str,
        page: int,
        raw_path: Path,
        returned_records: int,
        new_records: int,
        flickr_pages: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO completed_work_items
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    work_item_id,
                    term,
                    term_source,
                    page,
                    str(raw_path),
                    returned_records,
                    new_records,
                    flickr_pages,
                    time.time(),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, isolation_level=None)


if __name__ == "__main__":
    main()
