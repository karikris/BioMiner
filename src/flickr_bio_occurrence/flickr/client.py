from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import httpx

from flickr_bio_occurrence.flickr.endpoints import FLICKR_REST_BASE_URL, SEARCH_METHOD
from flickr_bio_occurrence.flickr.rate_limiter import FlickrRateLimiter
from flickr_bio_occurrence.flickr.work_items import WorkItem


DEFAULT_EXTRAS = (
    "description,license,date_upload,date_taken,geo,tags,machine_tags,owner_name,"
    "url_m,url_l,url_o,o_dims,last_update,media,views"
)


@dataclass(frozen=True)
class FlickrSearchResult:
    payload: dict[str, Any]
    raw_response_path: Path
    photo_ids: list[str]


class FlickrClient:
    def __init__(
        self,
        *,
        api_key: str,
        limiter: FlickrRateLimiter,
        http_client: httpx.Client | None = None,
        raw_output_root: str | Path = "data/raw/flickr",
        base_url: str = FLICKR_REST_BASE_URL,
        per_page: int = 250,
        extras: str = DEFAULT_EXTRAS,
    ) -> None:
        self.api_key = api_key
        self.limiter = limiter
        self.http_client = http_client or httpx.Client(timeout=30)
        self.raw_output_root = Path(raw_output_root)
        self.base_url = base_url
        self.per_page = min(per_page, 250)
        self.extras = extras

    def search_photos(self, work_item: WorkItem) -> FlickrSearchResult:
        self.limiter.acquire_api_token(SEARCH_METHOD, work_item.work_item_id)
        response = self.http_client.get(self.base_url, params=self._search_params(work_item))
        status = "ok" if response.is_success else f"http_{response.status_code}"
        self.limiter.log_call(SEARCH_METHOD, work_item.work_item_id, status)
        response.raise_for_status()
        payload = response.json()
        raw_path = self._write_raw_response(work_item, payload)
        photo_ids = [
            str(photo["id"])
            for photo in payload.get("photos", {}).get("photo", [])
            if "id" in photo
        ]
        allowed = self.limiter.reserve_photo_record_slots(len(photo_ids))
        photo_ids = photo_ids[:allowed]
        self.limiter.log_photo_records(photo_ids, work_item.work_item_id)
        return FlickrSearchResult(payload=payload, raw_response_path=raw_path, photo_ids=photo_ids)

    def _search_params(self, work_item: WorkItem) -> dict[str, str | int]:
        return {
            "method": SEARCH_METHOD,
            "api_key": self.api_key,
            "text": _query_text(work_item),
            "bbox": work_item.bbox,
            "min_taken_date": work_item.min_taken_date,
            "max_taken_date": work_item.max_taken_date,
            "has_geo": 1,
            "media": "photos",
            "content_types": "0",
            "safe_search": 1,
            "extras": self.extras,
            "per_page": self.per_page,
            "page": work_item.page,
            "format": "json",
            "nojsoncallback": 1,
        }

    def _write_raw_response(self, work_item: WorkItem, payload: dict[str, Any]) -> Path:
        target_dir = self.raw_output_root / "photos_search" / work_item.species_name / work_item.region_id / str(work_item.year) / f"{work_item.month:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{work_item.work_item_id}.json"
        target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return target


def _query_text(work_item: WorkItem) -> str:
    variant_to_term = {
        "scientific_name": work_item.species_name,
        "lime_butterfly": "lime butterfly",
        "chequered_swallowtail": "chequered swallowtail",
        "citrus_swallowtail": "citrus swallowtail",
    }
    return variant_to_term.get(work_item.query_variant, work_item.species_query_terms[0])
