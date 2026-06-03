from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from flickr_bio_occurrence.flickr.rate_limiter import DEFAULT_RATE_LIMIT_LEDGER_PATH, FlickrRateLimiter, RateLimitExceeded


def test_rate_limiter_never_exceeds_3600_calls_per_hour(tmp_path) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", soft_api_calls_per_hour=3200, hard_api_calls_per_hour=3)

    for _ in range(3):
        limiter.acquire_api_token("flickr.photos.search", "work")

    try:
        limiter.acquire_api_token("flickr.photos.search", "work")
    except RateLimitExceeded:
        pass
    else:
        raise AssertionError("expected hard API call cap to fail closed")

    assert limiter.api_calls_in_window() == 3


def test_rate_limiter_stops_at_configured_soft_api_calls_per_hour(tmp_path) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", soft_api_calls_per_hour=3, hard_api_calls_per_hour=3600)

    for _ in range(3):
        limiter.acquire_api_token("flickr.photos.search", "work")

    try:
        limiter.acquire_api_token("flickr.photos.search", "work")
    except RateLimitExceeded as exc:
        assert "soft API call cap reached" in str(exc)
    else:
        raise AssertionError("expected soft API call cap to stop scheduling")

    assert limiter.api_calls_in_window() == 3


def test_rate_limiter_never_exceeds_3600_photo_records_per_hour(tmp_path) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", hard_photo_records_per_hour=5)

    assert limiter.reserve_photo_record_slots(3) == 3
    assert limiter.log_photo_records(["1", "2", "3"], "work") == ["1", "2", "3"]
    assert limiter.reserve_photo_record_slots(3) == 2
    assert limiter.log_photo_records(["4", "5", "6"], "work") == ["4", "5"]
    assert limiter.reserve_photo_record_slots(1) == 0
    assert limiter.photo_records_in_window() == 5


def test_log_photo_records_returns_only_actually_logged_ids(tmp_path) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", hard_photo_records_per_hour=2)

    assert limiter.log_photo_records(["1", "2", "3"], "work") == ["1", "2"]
    assert limiter.log_photo_records(["4"], "work") == []
    assert limiter.photo_records_in_window() == 2


def test_log_photo_records_returns_only_new_photo_ids(tmp_path) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", hard_photo_records_per_hour=5)

    assert limiter.log_photo_records(["1", "2"], "work-a") == ["1", "2"]
    assert limiter.log_photo_records(["2", "3"], "work-b") == ["3"]
    assert limiter.photo_records_in_window() == 3


def test_parallel_workers_share_global_limiter(tmp_path) -> None:
    limiter = FlickrRateLimiter(tmp_path / "limits.sqlite", soft_api_calls_per_hour=3200, hard_api_calls_per_hour=25)

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


def test_default_limiter_uses_global_ledger_path() -> None:
    limiter = FlickrRateLimiter()

    assert limiter.ledger_path == DEFAULT_RATE_LIMIT_LEDGER_PATH


def test_multiple_limiter_instances_share_same_ledger(tmp_path) -> None:
    ledger = tmp_path / "global.sqlite"
    limiter_a = FlickrRateLimiter(ledger, hard_api_calls_per_hour=1)
    limiter_b = FlickrRateLimiter(ledger, hard_api_calls_per_hour=1)

    limiter_a.acquire_api_token("flickr.photos.search", "work-a")

    try:
        limiter_b.acquire_api_token("flickr.photos.search", "work-b")
    except RateLimitExceeded:
        pass
    else:
        raise AssertionError("expected second limiter instance to see shared hard cap")
