from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from flickr_bio_occurrence.flickr.rate_limiter import FlickrRateLimiter, RateLimitExceeded


def test_rate_limiter_never_exceeds_3600_calls_per_hour(tmp_path) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", soft_api_calls_per_hour=3000, hard_api_calls_per_hour=3)

    for _ in range(3):
        limiter.acquire_api_token("flickr.photos.search", "work")

    try:
        limiter.acquire_api_token("flickr.photos.search", "work")
    except RateLimitExceeded:
        pass
    else:
        raise AssertionError("expected hard API call cap to fail closed")

    assert limiter.api_calls_in_window() == 3


def test_rate_limiter_never_exceeds_3600_photo_records_per_hour(tmp_path) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", hard_photo_records_per_hour=5)

    assert limiter.reserve_photo_record_slots(3) == 3
    limiter.log_photo_records(["1", "2", "3"], "work")
    assert limiter.reserve_photo_record_slots(3) == 2
    limiter.log_photo_records(["4", "5"], "work")
    assert limiter.reserve_photo_record_slots(1) == 0
    assert limiter.photo_records_in_window() == 5


def test_parallel_workers_share_global_limiter(tmp_path) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", soft_api_calls_per_hour=3000, hard_api_calls_per_hour=25)

    def acquire_once(index: int) -> bool:
        try:
            limiter.acquire_api_token("flickr.photos.search", f"work-{index}")
        except RateLimitExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(acquire_once, range(100)))

    assert sum(results) == 25
    assert limiter.api_calls_in_window() == 25
