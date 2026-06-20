from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import httpx
import polars as pl

from biominer.common.artifacts import write_json_artifact, write_parquet_artifact
from biominer.common.concurrency import bounded_map_ordered
from biominer.common.http import RetryingHTTPClient


def test_bounded_map_ordered_limits_submitted_work_and_preserves_order() -> None:
    submitted_not_consumed = 0
    max_submitted_not_consumed = 0

    def record(value: int) -> int:
        return value * 10

    def items():
        nonlocal submitted_not_consumed, max_submitted_not_consumed
        for value in range(8):
            submitted_not_consumed += 1
            max_submitted_not_consumed = max(max_submitted_not_consumed, submitted_not_consumed)
            yield value

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = []
        for result in bounded_map_ordered(executor, record, items(), buffersize=2):
            submitted_not_consumed -= 1
            results.append(result)

    assert results == [0, 10, 20, 30, 40, 50, 60, 70]
    assert max_submitted_not_consumed <= 4


def test_retrying_http_client_honors_retry_after_before_success() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2.5"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    with RetryingHTTPClient(
        base_url="https://example.test",
        max_retries=2,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        jitter=lambda _attempt: 0.0,
    ) as client:
        response = client.get_json("/resource")

    assert response == {"ok": True}
    assert calls == 2
    assert sleeps == [2.5]
    assert client.request_attempt_count == 2
    assert client.retry_count == 1


def test_retrying_http_client_uses_exponential_backoff_with_jitter() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    with RetryingHTTPClient(
        base_url="https://example.test",
        max_retries=3,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        jitter=lambda attempt: 0.25 * (attempt + 1),
    ) as client:
        assert client.get_json("/resource") == {"ok": True}

    assert sleeps == [0.75, 1.5]
    assert client.retry_count == 2


def test_retrying_http_client_sends_configured_headers() -> None:
    seen_user_agents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_user_agents.append(request.headers["User-Agent"])
        return httpx.Response(200, json={"ok": True}, request=request)

    with RetryingHTTPClient(
        base_url="https://example.test",
        headers={"User-Agent": "BioMiner/test"},
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.get_json("/resource") == {"ok": True}

    assert seen_user_agents == ["BioMiner/test"]


def test_artifact_writers_return_size_and_sha256(tmp_path) -> None:
    json_meta = write_json_artifact(tmp_path / "manifest.json", {"b": 2, "a": 1})
    parquet_meta = write_parquet_artifact(tmp_path / "rows.parquet", pl.DataFrame({"id": [1, 2]}))

    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert parquet_meta.rows == 2
    assert json_meta.file_name == "manifest.json"
    assert json_meta.size_bytes > 0
    assert json_meta.sha256.startswith("sha256:")
    assert parquet_meta.file_name == "rows.parquet"
    assert parquet_meta.size_bytes > 0
    assert parquet_meta.sha256.startswith("sha256:")
