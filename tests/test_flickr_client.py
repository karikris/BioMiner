from __future__ import annotations

import json

import httpx

from flickr_bio_occurrence.flickr.client import FlickrClient
from flickr_bio_occurrence.flickr.rate_limiter import FlickrRateLimiter
from flickr_bio_occurrence.flickr.work_items import WorkItem


def test_photos_search_uses_required_official_api_params(tmp_path) -> None:
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(200, json={"stat": "ok", "photos": {"photo": []}})

    client = FlickrClient(
        api_key="test-key",
        limiter=FlickrRateLimiter(tmp_path / "limits.sqlite"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        raw_output_root=tmp_path / "raw",
    )

    client.search_photos(_work_item())

    assert seen_request is not None
    params = dict(seen_request.url.params)
    assert params["method"] == "flickr.photos.search"
    assert params["text"] == "Papilio demoleus"
    assert params["bbox"] == "137.99,-29.18,153.55,-9.14"
    assert params["has_geo"] == "1"
    assert params["media"] == "photos"
    assert params["content_types"] == "0"
    assert params["safe_search"] == "1"
    assert params["format"] == "json"
    assert params["nojsoncallback"] == "1"
    assert int(params["per_page"]) <= 250


def test_search_photos_writes_raw_response_unchanged(tmp_path) -> None:
    payload = {"stat": "ok", "photos": {"photo": [{"id": "1", "title": "Papilio demoleus"}]}}
    client = FlickrClient(
        api_key="test-key",
        limiter=FlickrRateLimiter(tmp_path / "limits.sqlite"),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))),
        raw_output_root=tmp_path / "raw",
    )

    result = client.search_photos(_work_item())

    raw_path = result.raw_response_path
    assert raw_path.exists()
    assert json.loads(raw_path.read_text(encoding="utf-8")) == payload


def test_search_photos_counts_api_call_and_new_photo_records(tmp_path) -> None:
    payload = {"stat": "ok", "photos": {"photo": [{"id": "1"}, {"id": "2"}]}}
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", hard_api_calls_per_hour=5, hard_photo_records_per_hour=5)
    client = FlickrClient(
        api_key="test-key",
        limiter=limiter,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))),
        raw_output_root=tmp_path / "raw",
    )

    result = client.search_photos(_work_item())

    assert result.photo_ids == ["1", "2"]
    assert limiter.api_calls_in_window() == 1
    assert limiter.photo_records_in_window() == 2


def test_search_photos_returns_only_logged_photo_records(tmp_path) -> None:
    payload = {"stat": "ok", "photos": {"photo": [{"id": "1"}, {"id": "2"}, {"id": "3"}]}}
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", hard_api_calls_per_hour=5, hard_photo_records_per_hour=2)
    client = FlickrClient(
        api_key="test-key",
        limiter=limiter,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))),
        raw_output_root=tmp_path / "raw",
    )

    result = client.search_photos(_work_item())

    assert result.photo_ids == ["1", "2"]
    assert limiter.photo_records_in_window() == 2


def test_photos_search_maps_broader_query_variants_to_text(tmp_path) -> None:
    seen_text: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_text.append(dict(request.url.params)["text"])
        return httpx.Response(200, json={"stat": "ok", "photos": {"photo": []}})

    client = FlickrClient(
        api_key="test-key",
        limiter=FlickrRateLimiter(tmp_path / "limits.sqlite"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        raw_output_root=tmp_path / "raw",
    )

    for variant in ["butterfly", "papilio", "citrusbutterfly", "limebutterfly"]:
        client.search_photos(_work_item(query_variant=variant))

    assert seen_text == ["butterfly", "Papilio", "citrusbutterfly", "limebutterfly"]


def _work_item(query_variant: str = "scientific_name") -> WorkItem:
    return WorkItem(
        species_name="Papilio demoleus",
        species_query_terms=["Papilio demoleus", "lime butterfly"],
        region_id="AU_QLD",
        region_name="Queensland",
        bbox="137.99,-29.18,153.55,-9.14",
        year=2024,
        month=1,
        min_taken_date="2024-01-01",
        max_taken_date="2024-01-31",
        page=1,
        query_variant=query_variant,
    )
