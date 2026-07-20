from __future__ import annotations

from collections.abc import Callable
import logging
import time

import httpx

from biominer.registry.gbif import GBIF_BASE_URL, GBIFClient, JSONPayload


logger = logging.getLogger(__name__)
GBIF_USER_AGENT = "BioMiner/0.1 (+https://github.com/karikris/BioMiner)"


class RetryingHTTPGet:
    def __init__(
        self,
        *,
        base_url: str = GBIF_BASE_URL,
        max_retries: int = 5,
        timeout_seconds: float = 30.0,
        max_connections: int = 8,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self.attempt_count = 0
        self.retry_count = 0
        self.rate_limit_count = 0
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            headers={"User-Agent": GBIF_USER_AGENT},
            transport=transport,
        )

    def __call__(self, path: str, params: dict[str, object]) -> JSONPayload:
        last_error: BaseException | None = None
        for retry_index in range(self.max_retries + 1):
            self.attempt_count += 1
            try:
                response = self._client.get(path, params=params)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
                    return payload
                raise ValueError(f"GBIF response for {path} must be a JSON object or an array of objects")
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code == 429
                ):
                    self.rate_limit_count += 1
                if not _is_retryable(exc) or retry_index >= self.max_retries:
                    raise
                self.retry_count += 1
                delay = _retry_delay(exc, retry_index)
                logger.warning(
                    "gbif.retry path=%s retry=%d/%d delay_seconds=%.2f error=%s",
                    path,
                    retry_index + 1,
                    self.max_retries,
                    delay,
                    type(exc).__name__,
                )
                self._sleep(delay)
        assert last_error is not None
        raise last_error

    def close(self) -> None:
        self._client.close()


class ProductionGBIFClient(GBIFClient):
    def __init__(
        self,
        *,
        base_url: str = GBIF_BASE_URL,
        max_retries: int = 5,
        max_connections: int = 8,
        timeout_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.transport = RetryingHTTPGet(
            base_url=base_url,
            max_retries=max_retries,
            max_connections=max_connections,
            timeout_seconds=timeout_seconds,
            sleep=sleep,
            transport=transport,
        )
        super().__init__(http_get=self.transport, base_url=base_url)

    @property
    def request_attempt_count(self) -> int:
        return self.transport.attempt_count

    @property
    def retry_count(self) -> int:
        return self.transport.retry_count

    @property
    def rate_limit_count(self) -> int:
        return self.transport.rate_limit_count

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> ProductionGBIFClient:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _retry_delay(exc: BaseException, retry_index: int) -> float:
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
    return min(60.0, 0.5 * (2**retry_index))
