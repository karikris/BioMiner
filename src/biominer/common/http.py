from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import httpx


JSONPayload = dict[str, Any] | list[dict[str, Any]]
Sleep = Callable[[float], None]
Jitter = Callable[[int], float]


class RetryingHTTPClient:
    def __init__(
        self,
        *,
        base_url: str,
        max_retries: int = 5,
        timeout_seconds: float = 30.0,
        max_connections: int = 8,
        sleep: Sleep = time.sleep,
        jitter: Jitter | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self.request_attempt_count = 0
        self.retry_count = 0
        self._sleep = sleep
        self._jitter = jitter or (lambda _attempt: 0.0)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            transport=transport,
        )

    def get_json(self, path: str, *, params: dict[str, object] | None = None) -> JSONPayload:
        last_error: BaseException | None = None
        for retry_index in range(self.max_retries + 1):
            self.request_attempt_count += 1
            try:
                response = self._client.get(path, params=params)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
                    return payload
                raise ValueError(f"HTTP response for {path} must be a JSON object or an array of objects")
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc
                if not is_retryable_http_error(exc) or retry_index >= self.max_retries:
                    raise
                self.retry_count += 1
                self._sleep(retry_delay_seconds(exc, retry_index, jitter=self._jitter))
        assert last_error is not None
        raise last_error

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RetryingHTTPClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def is_retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status in {502, 503, 504}
    return False


def retry_delay_seconds(exc: BaseException, retry_index: int, *, jitter: Jitter | None = None) -> float:
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
    jitter_value = (jitter or (lambda _attempt: 0.0))(retry_index)
    return min(60.0, max(0.0, (0.5 * (2**retry_index)) + jitter_value))
