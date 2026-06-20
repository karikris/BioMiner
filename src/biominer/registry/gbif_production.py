from __future__ import annotations

from collections.abc import Callable
import time

import httpx

from biominer.common.http import RetryingHTTPClient
from biominer.registry.gbif import GBIF_BASE_URL, GBIFClient, JSONPayload


class RetryingHTTPGet:
    def __init__(
        self,
        *,
        base_url: str = GBIF_BASE_URL,
        max_retries: int = 5,
        timeout_seconds: float = 30.0,
        max_connections: int = 8,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[int], float] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self._client = RetryingHTTPClient(
            base_url=base_url.rstrip("/"),
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            max_connections=max_connections,
            sleep=sleep,
            jitter=jitter,
            transport=transport,
        )

    def __call__(self, path: str, params: dict[str, object]) -> JSONPayload:
        return self._client.get_json(path, params=params)

    @property
    def attempt_count(self) -> int:
        return self._client.request_attempt_count

    @property
    def retry_count(self) -> int:
        return self._client.retry_count

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
        jitter: Callable[[int], float] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.transport = RetryingHTTPGet(
            base_url=base_url,
            max_retries=max_retries,
            max_connections=max_connections,
            timeout_seconds=timeout_seconds,
            sleep=sleep,
            jitter=jitter,
            transport=transport,
        )
        super().__init__(http_get=self.transport, base_url=base_url)

    @property
    def request_attempt_count(self) -> int:
        return self.transport.attempt_count

    @property
    def retry_count(self) -> int:
        return self.transport.retry_count

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> ProductionGBIFClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
