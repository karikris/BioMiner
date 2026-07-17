from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import ipaddress
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import re
import socket
import subprocess
from tempfile import NamedTemporaryFile
import threading
import time
from typing import Any, Iterator
from urllib.parse import urljoin, urlsplit, urlunsplit
from uuid import uuid4
import warnings

import httpcore
import httpx
from PIL import Image, UnidentifiedImageError
import polars as pl

from biominer.references.deduplication import (
    REFERENCE_PERCEPTUAL_HASH_VERSION,
    compute_reference_perceptual_hash,
)
from biominer.references.licensing import (
    ReferenceLicenceDecision,
    ReferenceLicencePolicy,
)
from biominer.references.schemas import (
    REFERENCE_MEDIA_OBJECTS_FILE,
    REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
    REFERENCE_MEDIA_RASTER_CONTENT_TYPES,
    reference_media_object_schema,
    reference_media_objects_frame,
    validate_reference_acquisition_selections,
    validate_reference_media_candidates,
    validate_reference_media_objects,
)
from biominer.storage.cloud import CloudStorage
from biominer.storage.paths import build_report_uri, safe_path_component
from biominer.storage.uri import join_uri


REFERENCE_MEDIA_DOWNLOADER_VERSION = "reference-media-downloader-v2"
REFERENCE_MEDIA_CHECKPOINT_VERSION = "reference-media-checkpoint-v2"
_LEGACY_REFERENCE_MEDIA_DOWNLOADER_VERSION = "reference-media-downloader-v1"
_LEGACY_REFERENCE_MEDIA_CHECKPOINT_VERSION = "reference-media-checkpoint-v1"
_LEGACY_REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION = "reference-media-objects-v1.0.0"
_REFERENCE_MEDIA_HASH_BACKFILL_VERSION = "reference-media-hash-backfill-v1"
REFERENCE_MEDIA_DOWNLOAD_REPORT_VERSION = "reference-media-download-report-v1"
REFERENCE_MEDIA_DOWNLOAD_REPORT_FILE = "reference_media_download_report.json"
REFERENCE_MEDIA_DOWNLOAD_SUMMARY_FILE = "reference_media_download_summary.md"

_LOGGER = logging.getLogger(__name__)
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_RETRY_STATUSES = (408, 425, 429, 500, 502, 503, 504)
_SUPPORTED_RETRY_STATUSES = frozenset(_DEFAULT_RETRY_STATUSES)
_INATURALIST_PHOTO_PATH = re.compile(
    r"\A/photos/(?P<photo_id>[0-9]+)/"
    r"(?P<style>original|large|medium|small|thumb|square)\."
    r"(?P<extension>jpe?g|png|gif)\Z",
    re.IGNORECASE,
)
_CONTENT_TYPE_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
    "image/x-tiff": "image/tiff",
}
_FORMAT_CONTENT_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
    "GIF": "image/gif",
}
_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/tiff": "tif",
    "image/gif": "gif",
}
_CHECKSUM_ALGORITHMS = frozenset({"md5", "sha1", "sha256"})
_DEFAULT_MAX_DECODE_MEMORY_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProviderMediaDownloadPolicy:
    source: str
    allowed_hosts: tuple[str, ...]
    policy_version: str = "provider-media-policy-v1"
    allowed_schemes: tuple[str, ...] = ("https",)
    max_concurrent_per_origin: int = 1
    min_request_interval_seconds: float = 0.0
    resolve_public_addresses: bool = True
    url_strategy: str = "direct"
    inaturalist_image_style: str = "large"

    def __post_init__(self) -> None:
        for field_name in ("source", "policy_version"):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} must be nonblank")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.allowed_hosts, tuple) or not all(
            isinstance(value, str) for value in self.allowed_hosts
        ):
            raise TypeError("allowed_hosts must be a tuple of strings")
        hosts = tuple(
            sorted(
                {
                    _normalise_host(value)
                    for value in self.allowed_hosts
                    if str(value or "").strip()
                }
            )
        )
        if not hosts:
            raise ValueError("provider policy allowed_hosts must be non-empty")
        object.__setattr__(self, "allowed_hosts", hosts)
        if not isinstance(self.allowed_schemes, tuple) or not all(
            isinstance(value, str) for value in self.allowed_schemes
        ):
            raise TypeError("allowed_schemes must be a tuple of strings")
        schemes = tuple(
            sorted(
                {
                    str(value or "").strip().casefold()
                    for value in self.allowed_schemes
                    if str(value or "").strip()
                }
            )
        )
        if not schemes or set(schemes) - {"http", "https"}:
            raise ValueError("allowed_schemes must contain only http and/or https")
        object.__setattr__(self, "allowed_schemes", schemes)
        _positive_int(
            self.max_concurrent_per_origin,
            field="max_concurrent_per_origin",
        )
        interval = _nonnegative_finite(
            self.min_request_interval_seconds,
            field="min_request_interval_seconds",
        )
        object.__setattr__(self, "min_request_interval_seconds", interval)
        if not isinstance(self.resolve_public_addresses, bool):
            raise TypeError("resolve_public_addresses must be boolean")
        if self.url_strategy not in {"direct", "inaturalist_photo"}:
            raise ValueError("unsupported provider URL strategy")
        if self.inaturalist_image_style not in {
            "original",
            "large",
            "medium",
            "small",
            "thumb",
        }:
            raise ValueError("unsupported iNaturalist image style")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "source": self.source,
                "allowed_hosts": self.allowed_hosts,
                "policy_version": self.policy_version,
                "allowed_schemes": self.allowed_schemes,
                "max_concurrent_per_origin": self.max_concurrent_per_origin,
                "min_request_interval_seconds": self.min_request_interval_seconds,
                "resolve_public_addresses": self.resolve_public_addresses,
                "url_strategy": self.url_strategy,
                "inaturalist_image_style": self.inaturalist_image_style,
            }
        )

    @property
    def validation_fingerprint(self) -> str:
        return _fingerprint(
            {
                "source": self.source,
                "allowed_hosts": self.allowed_hosts,
                "policy_version": self.policy_version,
                "allowed_schemes": self.allowed_schemes,
                "resolve_public_addresses": self.resolve_public_addresses,
                "url_strategy": self.url_strategy,
                "inaturalist_image_style": self.inaturalist_image_style,
            }
        )


def _default_provider_policies() -> tuple[ProviderMediaDownloadPolicy, ...]:
    return (
        ProviderMediaDownloadPolicy(
            source="iNaturalist",
            allowed_hosts=(
                "static.inaturalist.org",
                "inaturalist-open-data.s3.amazonaws.com",
            ),
            max_concurrent_per_origin=2,
            min_request_interval_seconds=0.25,
            url_strategy="inaturalist_photo",
            inaturalist_image_style="large",
        ),
    )


@dataclass(frozen=True, slots=True)
class ReferenceMediaDownloadConfig:
    workers: int = 8
    max_inflight: int = 32
    max_concurrent_decodes: int = 1
    max_attempts: int = 5
    max_redirects: int = 3
    max_source_bytes: int = 32 * 1024 * 1024
    max_decoded_pixels: int = 80_000_000
    max_decode_memory_bytes: int = _DEFAULT_MAX_DECODE_MEMORY_BYTES
    timeout_seconds: float = 30.0
    max_download_seconds: float = 300.0
    backoff_base_seconds: float = 0.5
    backoff_cap_seconds: float = 60.0
    max_retry_after_seconds: float = 3_600.0
    retry_statuses: tuple[int, ...] = _DEFAULT_RETRY_STATUSES
    allowed_content_types: tuple[str, ...] = tuple(
        sorted(REFERENCE_MEDIA_RASTER_CONTENT_TYPES)
    )
    provider_policies: tuple[ProviderMediaDownloadPolicy, ...] = field(
        default_factory=_default_provider_policies
    )
    user_agent: str = "BioMiner/0.1 reference-media-downloader"
    temporary_directory: Path | None = None
    git_sha: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "workers",
            "max_inflight",
            "max_concurrent_decodes",
            "max_attempts",
            "max_source_bytes",
            "max_decoded_pixels",
            "max_decode_memory_bytes",
        ):
            _positive_int(getattr(self, field_name), field=field_name)
        if self.max_inflight < self.workers:
            raise ValueError("max_inflight must be at least workers")
        if isinstance(self.max_redirects, bool) or not isinstance(
            self.max_redirects, int
        ):
            raise TypeError("max_redirects must be an integer")
        if self.max_redirects < 0:
            raise ValueError("max_redirects must be nonnegative")
        for field_name in (
            "timeout_seconds",
            "max_download_seconds",
            "backoff_base_seconds",
            "backoff_cap_seconds",
            "max_retry_after_seconds",
        ):
            value = _nonnegative_finite(getattr(self, field_name), field=field_name)
            if field_name in {"timeout_seconds", "max_download_seconds"} and value == 0:
                raise ValueError(f"{field_name} must be positive")
            if field_name == "max_retry_after_seconds" and value == 0:
                raise ValueError("max_retry_after_seconds must be positive")
            object.__setattr__(self, field_name, value)
        if self.backoff_cap_seconds < self.backoff_base_seconds:
            raise ValueError("backoff cap must be at least the base")
        if not isinstance(self.retry_statuses, tuple) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in self.retry_statuses
        ):
            raise TypeError("retry_statuses must be a tuple of integers")
        statuses = tuple(sorted(set(self.retry_statuses)))
        if not statuses or set(statuses) - _SUPPORTED_RETRY_STATUSES:
            raise ValueError(
                "retry_statuses contains a non-transient or unsupported status"
            )
        object.__setattr__(self, "retry_statuses", statuses)
        if not isinstance(self.allowed_content_types, tuple) or not all(
            isinstance(value, str) for value in self.allowed_content_types
        ):
            raise TypeError("allowed_content_types must be a tuple of strings")
        content_types = tuple(
            sorted(
                {_canonical_content_type(value) for value in self.allowed_content_types}
            )
        )
        if (
            not content_types
            or set(content_types) - REFERENCE_MEDIA_RASTER_CONTENT_TYPES
        ):
            raise ValueError(
                "allowed_content_types contains an unsupported raster MIME"
            )
        object.__setattr__(self, "allowed_content_types", content_types)
        if not isinstance(self.provider_policies, tuple) or not all(
            isinstance(value, ProviderMediaDownloadPolicy)
            for value in self.provider_policies
        ):
            raise TypeError(
                "provider_policies must be a tuple of ProviderMediaDownloadPolicy"
            )
        ownership: dict[tuple[str, str], str] = {}
        for policy in self.provider_policies:
            source = policy.source.casefold()
            for host in policy.allowed_hosts:
                key = (source, host)
                if key in ownership:
                    raise ValueError(
                        "provider policies overlap for source/host "
                        f"{policy.source!r}/{host!r}"
                    )
                ownership[key] = policy.fingerprint
        user_agent = str(self.user_agent or "").strip()
        if not user_agent:
            raise ValueError("user_agent must be nonblank")
        object.__setattr__(self, "user_agent", user_agent)
        if self.temporary_directory is not None:
            object.__setattr__(
                self,
                "temporary_directory",
                Path(self.temporary_directory),
            )

    @property
    def semantic_fingerprint(self) -> str:
        return _fingerprint(
            {
                "downloader_version": REFERENCE_MEDIA_DOWNLOADER_VERSION,
                "max_source_bytes": self.max_source_bytes,
                "max_decoded_pixels": self.max_decoded_pixels,
                "max_decode_memory_bytes": self.max_decode_memory_bytes,
                "allowed_content_types": self.allowed_content_types,
            }
        )


@dataclass(frozen=True, slots=True)
class ReferenceMediaDownloadResult:
    media_objects: pl.DataFrame
    report: dict[str, Any]
    media_objects_uri: str
    report_uri: str
    summary_uri: str


@dataclass(frozen=True, slots=True)
class _SelectedMedia:
    candidate: dict[str, object]
    candidate_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PendingDownload:
    selected: _SelectedMedia
    licence: ReferenceLicenceDecision
    provider_policy: ProviderMediaDownloadPolicy
    requested_url: str
    binding: dict[str, object]
    checkpoint_uri: str


@dataclass(frozen=True, slots=True)
class _PreparedDownload:
    pending: _PendingDownload
    path: Path | None
    content_type: str | None
    source_byte_count: int | None
    decoded_width: int | None
    decoded_height: int | None
    sha256: str | None
    perceptual_hash: str | None
    final_url: str | None
    attempt_count: int
    retry_count: int
    rate_limit_count: int
    retry_wait_seconds: float
    decode_status: str
    failure_reason: str | None

    @property
    def valid(self) -> bool:
        return self.decode_status == "valid" and self.path is not None


class _PayloadFailure(Exception):
    def __init__(
        self,
        reason: str,
        *,
        decode_status: str,
        content_type: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.decode_status = decode_status
        self.content_type = content_type


class _RetryableResponse(Exception):
    def __init__(
        self,
        status_code: int,
        *,
        retry_after_seconds: float | None,
    ) -> None:
        super().__init__(f"retryable HTTP status {status_code}")
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class _PermanentResponse(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _remaining_seconds(
    deadline: float,
    monotonic: Callable[[], float],
) -> float:
    return max(0.0, deadline - monotonic())


@dataclass(frozen=True, slots=True)
class _Redirect:
    url: str


class _OriginLimiter:
    def __init__(
        self,
        *,
        max_concurrent: int,
        minimum_interval_seconds: float,
        sleep: Callable[[float], None],
        monotonic: Callable[[], float],
    ) -> None:
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._minimum_interval = minimum_interval_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._schedule_lock = threading.Lock()
        self._next_request_at = 0.0
        self._blocked_reason: str | None = None

    def defer_for(self, seconds: float) -> None:
        with self._schedule_lock:
            self._next_request_at = max(
                self._next_request_at,
                self._monotonic() + seconds,
            )

    def block(self, reason: str) -> None:
        with self._schedule_lock:
            self._blocked_reason = reason

    @contextmanager
    def slot(self, *, deadline: float | None = None) -> Iterator[None]:
        if deadline is None:
            acquired = self._semaphore.acquire()
        else:
            remaining = max(0.0, deadline - self._monotonic())
            acquired = remaining > 0 and self._semaphore.acquire(timeout=remaining)
        if not acquired:
            raise _PermanentResponse("item_deadline_exceeded")
        try:
            current_hint: float | None = None
            while True:
                if deadline is None:
                    schedule_acquired = self._schedule_lock.acquire()
                else:
                    remaining = max(0.0, deadline - self._monotonic())
                    schedule_acquired = remaining > 0 and self._schedule_lock.acquire(
                        timeout=remaining
                    )
                if not schedule_acquired:
                    raise _PermanentResponse("item_deadline_exceeded")
                try:
                    if self._blocked_reason is not None:
                        raise _PermanentResponse(self._blocked_reason)
                    observed = self._monotonic()
                    current = (
                        observed
                        if current_hint is None
                        else max(observed, current_hint)
                    )
                    scheduled_at = max(self._next_request_at, current)
                    if deadline is not None and scheduled_at >= deadline:
                        raise _PermanentResponse("item_deadline_exceeded")
                    delay = scheduled_at - current
                    if not delay:
                        self._next_request_at = (
                            max(scheduled_at, self._monotonic())
                            + self._minimum_interval
                        )
                        break
                finally:
                    self._schedule_lock.release()
                if delay:
                    self._sleep(delay)
                    current_hint = scheduled_at
            yield
        finally:
            self._semaphore.release()


class _OriginLimiterRegistry:
    def __init__(
        self,
        *,
        policies: tuple[ProviderMediaDownloadPolicy, ...],
        sleep: Callable[[float], None],
        monotonic: Callable[[], float],
    ) -> None:
        self._policies = policies
        self._sleep = sleep
        self._monotonic = monotonic
        self._limiters: dict[tuple[str, str, int], _OriginLimiter] = {}
        self._lock = threading.Lock()

    def get(
        self,
        policy: ProviderMediaDownloadPolicy,
        url: str,
    ) -> _OriginLimiter:
        parsed = urlsplit(url)
        scheme = parsed.scheme.casefold()
        host = _normalise_host(parsed.hostname or "")
        port = parsed.port or (443 if scheme == "https" else 80)
        key = (scheme, host, port)
        with self._lock:
            limiter = self._limiters.get(key)
            if limiter is None:
                origin_policies = [
                    value
                    for value in self._policies
                    if host in value.allowed_hosts and scheme in value.allowed_schemes
                ]
                if policy not in origin_policies:
                    origin_policies.append(policy)
                limiter = _OriginLimiter(
                    max_concurrent=min(
                        value.max_concurrent_per_origin for value in origin_policies
                    ),
                    minimum_interval_seconds=max(
                        value.min_request_interval_seconds for value in origin_policies
                    ),
                    sleep=self._sleep,
                    monotonic=self._monotonic,
                )
                self._limiters[key] = limiter
            return limiter


@dataclass(slots=True)
class _InFlightResolution:
    completed: threading.Event = field(default_factory=threading.Event)
    values: Sequence[str] | None = None
    error: Exception | None = None
    timed_out: bool = False


class _HostValidator:
    def __init__(
        self,
        resolver: Callable[[str], Sequence[str]],
    ) -> None:
        self._resolver = resolver
        self._inflight: dict[str, _InFlightResolution] = {}
        self._lock = threading.Lock()

    def public_addresses(
        self,
        host: str,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[str, ...]:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            try:
                raw_addresses = self._resolve(host, timeout_seconds=timeout_seconds)
            except (OSError, socket.gaierror) as exc:
                raise ValueError("provider host resolution failed") from exc
            try:
                addresses = tuple(
                    sorted(
                        {str(ipaddress.ip_address(value)) for value in raw_addresses}
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "provider host resolution returned an invalid address"
                ) from exc
            if not addresses:
                raise ValueError("provider host did not resolve")
        else:
            addresses = (str(address),)
        if any(not ipaddress.ip_address(value).is_global for value in addresses):
            raise ValueError("provider host resolves to a non-public address")
        return addresses

    def require_public(
        self,
        host: str,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        self.public_addresses(host, timeout_seconds=timeout_seconds)

    def _resolve(
        self,
        host: str,
        *,
        timeout_seconds: float | None,
    ) -> Sequence[str]:
        if timeout_seconds is None:
            return self._resolver(host)
        if timeout_seconds <= 0:
            raise ValueError("provider host resolution timed out")
        with self._lock:
            state = self._inflight.get(host)
            if state is not None and state.timed_out:
                raise ValueError("provider host resolution timed out")
            start_resolution = state is None
            if state is None:
                state = _InFlightResolution()
                self._inflight[host] = state

        def resolve() -> None:
            try:
                state.values = self._resolver(host)
                state.error = None
                state.timed_out = False
            except Exception as exc:  # noqa: BLE001 - propagate on the caller thread.
                state.error = exc
            finally:
                state.completed.set()
                with self._lock:
                    if self._inflight.get(host) is state:
                        del self._inflight[host]

        if start_resolution:
            threading.Thread(
                target=resolve,
                name=f"reference-dns-{hashlib.sha256(host.encode()).hexdigest()[:8]}",
                daemon=True,
            ).start()
        if not state.completed.wait(timeout_seconds):
            with self._lock:
                if not state.completed.is_set():
                    state.timed_out = True
                    state.error = ValueError("provider host resolution timed out")
                    state.completed.set()
            raise ValueError("provider host resolution timed out")
        if state.error is not None:
            raise state.error
        if state.values is None:  # pragma: no cover - resolver contract guard.
            raise ValueError("provider host resolution returned no result")
        return state.values


class _DeadlineContext:
    def __init__(self, monotonic: Callable[[], float]) -> None:
        self._monotonic = monotonic
        self._local = threading.local()

    def set(self, deadline: float | None) -> None:
        self._local.deadline = deadline

    def clear(self) -> None:
        self._local.deadline = None

    def deadline(self) -> float | None:
        return getattr(self._local, "deadline", None)

    def bounded_timeout(
        self,
        timeout: float | None,
        *,
        error_type: type[httpcore.TimeoutException],
    ) -> float | None:
        deadline = self.deadline()
        if deadline is None:
            return timeout
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise error_type("item deadline exceeded")
        return remaining if timeout is None else min(timeout, remaining)


class _DeadlineNetworkStream(httpcore.NetworkStream):
    def __init__(
        self,
        stream: httpcore.NetworkStream,
        deadline_context: _DeadlineContext,
    ) -> None:
        self._stream = stream
        self._deadline_context = deadline_context

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._stream.read(
            max_bytes,
            timeout=self._deadline_context.bounded_timeout(
                timeout,
                error_type=httpcore.ReadTimeout,
            ),
        )

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._stream.write(
            buffer,
            timeout=self._deadline_context.bounded_timeout(
                timeout,
                error_type=httpcore.WriteTimeout,
            ),
        )

    def close(self) -> None:
        self._stream.close()

    def start_tls(
        self,
        ssl_context: Any,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        stream = self._stream.start_tls(
            ssl_context,
            server_hostname=server_hostname,
            timeout=self._deadline_context.bounded_timeout(
                timeout,
                error_type=httpcore.ConnectTimeout,
            ),
        )
        return _DeadlineNetworkStream(stream, self._deadline_context)

    def get_extra_info(self, info: str) -> Any:
        return self._stream.get_extra_info(info)


class _PinnedAddressNetworkBackend(httpcore.NetworkBackend):
    """Resolve once per connection and connect only to the validated IP literal."""

    def __init__(
        self,
        host_validator: _HostValidator,
        *,
        delegate: httpcore.NetworkBackend | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        pinned_hosts: frozenset[str] | None = None,
        deadline_context: _DeadlineContext | None = None,
    ) -> None:
        self._host_validator = host_validator
        self._delegate = delegate or httpcore.SyncBackend()
        self._monotonic = monotonic
        self._pinned_hosts = pinned_hosts
        self._deadline_context = deadline_context or _DeadlineContext(monotonic)

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        normalised_host = _normalise_host(host)
        if self._pinned_hosts is not None and normalised_host not in self._pinned_hosts:
            return _DeadlineNetworkStream(
                self._delegate.connect_tcp(
                    host,
                    port,
                    timeout=self._deadline_context.bounded_timeout(
                        timeout,
                        error_type=httpcore.ConnectTimeout,
                    ),
                    local_address=local_address,
                    socket_options=socket_options,
                ),
                self._deadline_context,
            )
        current = self._monotonic()
        operation_deadline = None if timeout is None else current + timeout
        request_deadline = self._deadline_context.deadline()
        deadlines = [
            value
            for value in (operation_deadline, request_deadline)
            if value is not None
        ]
        deadline = min(deadlines) if deadlines else None
        resolution_timeout = None if deadline is None else max(0.0, deadline - current)
        try:
            addresses = self._host_validator.public_addresses(
                normalised_host,
                timeout_seconds=resolution_timeout,
            )
        except ValueError as exc:
            raise httpcore.ConnectError(str(exc)) from exc
        last_error: httpcore.ConnectError | httpcore.ConnectTimeout | None = None
        for address in addresses:
            remaining = (
                None if deadline is None else max(0.0, deadline - self._monotonic())
            )
            if remaining == 0:
                raise httpcore.ConnectTimeout(
                    "provider address connect deadline exceeded"
                ) from last_error
            try:
                return _DeadlineNetworkStream(
                    self._delegate.connect_tcp(
                        address,
                        port,
                        timeout=remaining,
                        local_address=local_address,
                        socket_options=socket_options,
                    ),
                    self._deadline_context,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise httpcore.ConnectError("provider host did not resolve")  # pragma: no cover

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        del path, timeout, socket_options
        raise httpcore.ConnectError("UNIX sockets are not allowed")

    def sleep(self, seconds: float) -> None:
        self._delegate.sleep(seconds)


_HTTPCORE_HTTPX_EXCEPTIONS: tuple[
    tuple[type[Exception], type[httpx.TransportError]], ...
] = (
    (httpcore.ConnectTimeout, httpx.ConnectTimeout),
    (httpcore.ReadTimeout, httpx.ReadTimeout),
    (httpcore.WriteTimeout, httpx.WriteTimeout),
    (httpcore.PoolTimeout, httpx.PoolTimeout),
    (httpcore.ConnectError, httpx.ConnectError),
    (httpcore.ReadError, httpx.ReadError),
    (httpcore.WriteError, httpx.WriteError),
    (httpcore.ProxyError, httpx.ProxyError),
    (httpcore.UnsupportedProtocol, httpx.UnsupportedProtocol),
    (httpcore.LocalProtocolError, httpx.LocalProtocolError),
    (httpcore.RemoteProtocolError, httpx.RemoteProtocolError),
    (httpcore.ProtocolError, httpx.ProtocolError),
    (httpcore.NetworkError, httpx.NetworkError),
    (httpcore.TimeoutException, httpx.TimeoutException),
)


@contextmanager
def _map_httpcore_transport_exceptions() -> Iterator[None]:
    try:
        yield
    except Exception as exc:
        for source_type, target_type in _HTTPCORE_HTTPX_EXCEPTIONS:
            if isinstance(exc, source_type):
                raise target_type(str(exc)) from exc
        raise


class _HTTPXCoreResponseStream(httpx.SyncByteStream):
    def __init__(
        self,
        stream: Iterable[bytes],
        *,
        on_close: Callable[[], None],
    ) -> None:
        self._stream = stream
        self._on_close = on_close

    def __iter__(self) -> Iterator[bytes]:
        try:
            with _map_httpcore_transport_exceptions():
                yield from self._stream
        finally:
            self._on_close()

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if close is not None:
                close()
        finally:
            self._on_close()


class _PinnedAddressHTTPTransport(httpx.BaseTransport):
    """HTTPX transport whose httpcore pool cannot re-resolve provider hosts."""

    def __init__(
        self,
        host_validator: _HostValidator,
        *,
        max_connections: int,
        network_backend: httpcore.NetworkBackend | None = None,
        pinned_hosts: frozenset[str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._deadline_context = _DeadlineContext(monotonic)
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpx.create_ssl_context(verify=True, trust_env=False),
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            network_backend=_PinnedAddressNetworkBackend(
                host_validator,
                delegate=network_backend,
                monotonic=monotonic,
                pinned_hosts=pinned_hosts,
                deadline_context=self._deadline_context,
            ),
            retries=0,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if not isinstance(request.stream, httpx.SyncByteStream):
            raise TypeError("pinned HTTP transport requires a synchronous byte stream")
        deadline = request.extensions.get("biominer_deadline")
        self._deadline_context.set(
            float(deadline) if isinstance(deadline, (int, float)) else None
        )
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        try:
            with _map_httpcore_transport_exceptions():
                response = self._pool.handle_request(core_request)
        except Exception:
            self._deadline_context.clear()
            raise
        if not isinstance(response.stream, Iterable):  # pragma: no cover
            self._deadline_context.clear()
            raise TypeError("httpcore returned a non-iterable response stream")
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_HTTPXCoreResponseStream(
                response.stream,
                on_close=self._deadline_context.clear,
            ),
            extensions=response.extensions,
        )

    def close(self) -> None:
        self._pool.close()


def download_reference_media(
    selections: pl.DataFrame,
    media_candidates: pl.DataFrame,
    *,
    storage: CloudStorage,
    output_prefix: str,
    config: ReferenceMediaDownloadConfig | None = None,
    licence_policy: ReferenceLicencePolicy | None = None,
    http_client: httpx.Client | None = None,
    run_id: str | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    resolve_host: Callable[[str], Sequence[str]] | None = None,
) -> ReferenceMediaDownloadResult:
    effective_now = now or _utc_now
    started_at = _aware_utc(effective_now())
    effective_run_id = str(run_id or "").strip() or (
        "reference-download-"
        + started_at.strftime("%Y%m%dT%H%M%S%fZ-")
        + uuid4().hex[:12]
    )
    effective_config = config or ReferenceMediaDownloadConfig()
    effective_licence_policy = licence_policy or ReferenceLicencePolicy()
    first_now = True

    def run_now() -> datetime:
        nonlocal first_now
        if first_now:
            first_now = False
            return started_at
        return effective_now()

    try:
        return _download_reference_media_impl(
            selections,
            media_candidates,
            storage=storage,
            output_prefix=output_prefix,
            config=effective_config,
            licence_policy=effective_licence_policy,
            http_client=http_client,
            run_id=effective_run_id,
            now=run_now,
            sleep=sleep,
            monotonic=monotonic,
            resolve_host=resolve_host,
        )
    except Exception as exc:
        _persist_failed_download_report(
            storage=storage,
            output_prefix=output_prefix,
            run_id=effective_run_id,
            started_at=started_at,
            ended_at=_aware_utc(effective_now()),
            selections=selections,
            media_candidates=media_candidates,
            config=effective_config,
            licence_policy=effective_licence_policy,
            error=exc,
        )
        raise


def _download_reference_media_impl(
    selections: pl.DataFrame,
    media_candidates: pl.DataFrame,
    *,
    storage: CloudStorage,
    output_prefix: str,
    config: ReferenceMediaDownloadConfig | None = None,
    licence_policy: ReferenceLicencePolicy | None = None,
    http_client: httpx.Client | None = None,
    run_id: str | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    resolve_host: Callable[[str], Sequence[str]] | None = None,
) -> ReferenceMediaDownloadResult:
    config = config or ReferenceMediaDownloadConfig()
    licence_policy = licence_policy or ReferenceLicencePolicy()
    now = now or _utc_now
    output_prefix = str(output_prefix or "").rstrip("/")
    if not output_prefix:
        raise ValueError("output_prefix must be nonblank")
    mock_transport = (
        None if http_client is None else getattr(http_client, "_transport", None)
    )
    if http_client is not None and type(mock_transport) is not httpx.MockTransport:
        raise TypeError(
            "http_client is restricted to MockTransport-backed test doubles; "
            "production downloads use the pinned-address transport"
        )
    started_at = _aware_utc(now())
    run_id = str(run_id or "").strip() or (
        "reference-download-"
        + started_at.strftime("%Y%m%dT%H%M%S%fZ-")
        + uuid4().hex[:12]
    )
    media_objects_uri = join_uri(output_prefix, REFERENCE_MEDIA_OBJECTS_FILE)
    report_uri, summary_uri = _download_report_uris(output_prefix, run_id)
    _log_event(
        "reference_media_download_start",
        run_id=run_id,
        selection_rows=selections.height,
        media_candidate_rows=media_candidates.height,
        output_prefix=output_prefix,
        workers=config.workers,
        max_inflight=config.max_inflight,
        licence_policy_version=licence_policy.version,
    )
    selected = _selected_media(selections, media_candidates)
    input_provenance = _reference_input_provenance(
        selections,
        media_candidates,
        selected,
    )
    host_validator = _HostValidator(resolve_host or _resolve_host)
    limiters = _OriginLimiterRegistry(
        policies=config.provider_policies,
        sleep=sleep,
        monotonic=monotonic,
    )
    decode_limiter = threading.BoundedSemaphore(config.max_concurrent_decodes)
    _log_event(
        "reference_media_download_inputs_validated",
        run_id=run_id,
        selected_count=len(selected),
        selection_fingerprint=input_provenance["selection_fingerprint"],
        selected_candidates_fingerprint=input_provenance[
            "selected_candidates_fingerprint"
        ],
    )

    rows: list[dict[str, object]] = []
    pending_downloads: list[_PendingDownload] = []
    resumed_count = 0
    for item in selected:
        decision = licence_policy.evaluate(
            media_licence=item.candidate["licence"],
            licence_uri=item.candidate["licence_uri"],
            attribution=item.candidate["attribution"],
        )
        if decision.status not in {"allowed", "research_only"}:
            rows.append(
                _failure_row(
                    item,
                    licence_status=decision.status,
                    decode_status="not_attempted",
                    reason=decision.reason or "licence_not_allowed",
                    attempt_count=0,
                    content_type=None,
                    config_fingerprint=config.semantic_fingerprint,
                    licence_policy_fingerprint=licence_policy.fingerprint,
                )
            )
            continue
        provider_policy = _provider_policy(item.candidate, config.provider_policies)
        if provider_policy is None:
            rows.append(
                _failure_row(
                    item,
                    licence_status=decision.status,
                    decode_status="not_attempted",
                    reason="provider_policy_missing",
                    attempt_count=0,
                    content_type=None,
                    config_fingerprint=config.semantic_fingerprint,
                    licence_policy_fingerprint=licence_policy.fingerprint,
                )
            )
            continue
        try:
            requested_url = _provider_request_url(
                item.candidate,
                provider_policy,
            )
        except ValueError as exc:
            rows.append(
                _failure_row(
                    item,
                    licence_status=decision.status,
                    decode_status="not_attempted",
                    reason=f"provider_constraint:{exc}",
                    attempt_count=0,
                    content_type=None,
                    config_fingerprint=config.semantic_fingerprint,
                    licence_policy_fingerprint=licence_policy.fingerprint,
                )
            )
            continue
        checksum_reason = _checksum_preflight_reason(item.candidate)
        if checksum_reason is not None:
            rows.append(
                _failure_row(
                    item,
                    licence_status=decision.status,
                    decode_status="not_attempted",
                    reason=checksum_reason,
                    attempt_count=0,
                    content_type=None,
                    config_fingerprint=config.semantic_fingerprint,
                    licence_policy_fingerprint=licence_policy.fingerprint,
                )
            )
            continue
        checkpoint_uri = _checkpoint_uri(output_prefix, item.candidate)
        binding = _checkpoint_binding(
            item,
            decision=decision,
            provider_policy=provider_policy,
            requested_url=requested_url,
            config=config,
            licence_policy=licence_policy,
        )
        if storage.exists(checkpoint_uri):
            rows.append(
                _load_committed_checkpoint(
                    storage,
                    checkpoint_uri,
                    expected_binding=binding,
                    expected_evidence=_expected_checkpoint_evidence(
                        item.candidate,
                        requested_url=requested_url,
                    ),
                    provider_policy=provider_policy,
                    expected_provider_media_id=str(item.candidate["provider_media_id"]),
                    config=config,
                    now=now,
                )
            )
            resumed_count += 1
            _log_event(
                "reference_media_download_resume",
                run_id=run_id,
                reference_media_id=item.candidate["reference_media_id"],
                checkpoint_uri=checkpoint_uri,
            )
            continue
        if item.candidate["download_status"] == "complete":
            raise ValueError(
                "complete reference media candidate is missing its committed checkpoint: "
                f"{item.candidate['reference_media_id']}"
            )
        pending_downloads.append(
            _PendingDownload(
                selected=item,
                licence=decision,
                provider_policy=provider_policy,
                requested_url=requested_url,
                binding=binding,
                checkpoint_uri=checkpoint_uri,
            )
        )

    uses_mock_transport = mock_transport is not None
    client = httpx.Client(
        transport=(
            mock_transport
            or _PinnedAddressHTTPTransport(
                host_validator,
                max_connections=config.max_inflight,
                pinned_hosts=frozenset(
                    host
                    for policy in config.provider_policies
                    if policy.resolve_public_addresses
                    for host in policy.allowed_hosts
                ),
                monotonic=monotonic,
            )
        ),
        headers={"User-Agent": config.user_agent},
        timeout=httpx.Timeout(config.timeout_seconds),
        follow_redirects=False,
        trust_env=False,
    )
    prepared_results: list[_PreparedDownload] = []
    try:
        if pending_downloads:

            def worker(pending: _PendingDownload) -> _PreparedDownload:
                try:
                    return _fetch_and_validate(
                        pending,
                        config=config,
                        client=client,
                        limiters=limiters,
                        host_validator=host_validator,
                        decode_limiter=decode_limiter,
                        prevalidate_dns=uses_mock_transport,
                        isolate_decode=not uses_mock_transport,
                        now=now,
                        sleep=sleep,
                        monotonic=monotonic,
                    )
                except Exception as exc:  # noqa: BLE001 - retain per-item evidence.
                    _LOGGER.exception(
                        "reference media worker failed for %s",
                        pending.selected.candidate["reference_media_id"],
                    )
                    return _prepared_failure(
                        pending,
                        attempts=0,
                        retries=0,
                        rate_limits=0,
                        retry_wait=0.0,
                        decode_status="download_failed",
                        reason=f"worker_exception:{type(exc).__name__}",
                    )

            with ThreadPoolExecutor(
                max_workers=config.workers,
                thread_name_prefix="reference-media",
            ) as executor:
                for prepared in executor.map(
                    worker,
                    pending_downloads,
                    buffersize=config.max_inflight,
                ):
                    prepared_results.append(prepared)
                    try:
                        rows.append(
                            _commit_prepared(
                                prepared,
                                storage=storage,
                                output_prefix=output_prefix,
                                now=now,
                                run_id=run_id,
                                config=config,
                                licence_policy=licence_policy,
                            )
                        )
                    finally:
                        if prepared.path is not None:
                            prepared.path.unlink(missing_ok=True)
    finally:
        for prepared in prepared_results:
            if prepared.path is not None:
                prepared.path.unlink(missing_ok=True)
        client.close()

    run_frame = reference_media_objects_frame(rows)
    frame = _merge_media_object_inventory(
        storage,
        media_objects_uri,
        current=run_frame,
    )
    storage.write_parquet_shard(media_objects_uri, frame, overwrite=True)
    ended_at = _aware_utc(now())
    report = _download_report(
        run_frame,
        inventory_frame=frame,
        input_provenance=input_provenance,
        selected=selected,
        prepared=prepared_results,
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        resumed_count=resumed_count,
        config=config,
        licence_policy=licence_policy,
        media_objects_uri=media_objects_uri,
        report_uri=report_uri,
        summary_uri=summary_uri,
    )
    storage.write_text(summary_uri, _download_markdown(report))
    storage.write_json(report_uri, report)
    _log_event(
        "reference_media_download_complete",
        run_id=run_id,
        status=report["status"],
        selected_count=report["counts"]["selected"],
        committed_count=report["counts"]["committed"],
        quarantined_count=report["counts"]["quarantined"],
        elapsed_seconds=report["elapsed_seconds"],
        media_objects_uri=media_objects_uri,
    )
    return ReferenceMediaDownloadResult(
        media_objects=frame,
        report=report,
        media_objects_uri=media_objects_uri,
        report_uri=report_uri,
        summary_uri=summary_uri,
    )


def _merge_media_object_inventory(
    storage: CloudStorage,
    media_objects_uri: str,
    *,
    current: pl.DataFrame,
) -> pl.DataFrame:
    if not storage.exists(media_objects_uri):
        return current
    existing = storage.read_parquet(media_objects_uri)
    if "schema_version" not in existing.columns:
        raise ValueError("reference media inventory schema version is missing")
    schema_versions = set(existing["schema_version"].drop_nulls().to_list())
    if schema_versions != {REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION}:
        raise ValueError("reference media inventory schema is incompatible")
    validate_reference_media_objects(existing)
    current_ids = set(current["reference_media_id"].to_list())
    rows_by_id = {
        str(row["reference_media_id"]): row for row in existing.iter_rows(named=True)
    }
    for source_row in current.iter_rows(named=True):
        row = dict(source_row)
        media_id = str(row["reference_media_id"])
        previous = rows_by_id.get(media_id)
        if (
            previous is not None
            and row["decode_status"] == "valid"
            and previous["decode_status"] == "valid"
            and row["object_fingerprint"] == previous["object_fingerprint"]
            and row["duplicate_group_id"] is None
            and previous["duplicate_group_id"] is not None
        ):
            for field_name in (
                "duplicate_group_id",
                "duplicate_type",
                "canonical_reference_media_id",
                "provider_mirror_ids",
            ):
                row[field_name] = previous[field_name]
        rows_by_id[media_id] = row
    return reference_media_objects_frame(list(rows_by_id.values()))


def _selected_media(
    selections: pl.DataFrame,
    media_candidates: pl.DataFrame,
) -> list[_SelectedMedia]:
    from biominer.references.prototype_acquisition import (
        prototype_reference_selection_schema,
    )

    prototype_selections = selections.schema == prototype_reference_selection_schema()
    if prototype_selections:
        from biominer.references.prototype_download import (
            validate_prototype_download_inputs,
        )

        validate_prototype_download_inputs(selections, media_candidates)
    else:
        validate_reference_acquisition_selections(selections)
        validate_reference_media_candidates(media_candidates)
    candidates = {
        str(row["reference_media_id"]): row
        for row in media_candidates.iter_rows(named=True)
    }
    values: dict[str, _SelectedMedia] = {}
    for selection in selections.iter_rows(named=True):
        media_id = str(selection["reference_media_id"])
        candidate = candidates.get(media_id)
        if candidate is None:
            raise ValueError(
                f"selected reference media is absent from candidates: {media_id}"
            )
        identity_fields = [
            "reference_observation_id",
            "source",
            "licence",
        ]
        if prototype_selections:
            identity_fields.extend(
                ("provider_media_id", "media_identifier", "attribution")
            )
        else:
            identity_fields.append("source_snapshot_version")
        for field_name in identity_fields:
            if selection[field_name] != candidate[field_name]:
                raise ValueError(
                    f"selected reference media has stale {field_name}: {media_id}"
                )
        if candidate["download_status"] not in {"pending", "complete"}:
            raise ValueError(
                "selected reference media is not download eligible: "
                f"{media_id} ({candidate['download_status']})"
            )
        values[media_id] = _SelectedMedia(
            candidate=dict(candidate),
            candidate_fingerprint=_candidate_source_fingerprint(candidate),
        )
    return sorted(
        values.values(),
        key=lambda value: str(value.candidate["reference_media_id"]),
    )


def _reference_input_provenance(
    selections: pl.DataFrame,
    media_candidates: pl.DataFrame,
    selected: Sequence[_SelectedMedia],
) -> dict[str, object]:
    return {
        "selection_rows": selections.height,
        "selection_fingerprint": _frame_fingerprint(selections),
        "media_candidate_rows": media_candidates.height,
        "selected_candidate_count": len(selected),
        "selected_candidates_fingerprint": _mapping_rows_fingerprint(
            [value.candidate for value in selected]
        ),
        "selected_candidate_sources_fingerprint": _fingerprint(
            {"candidates": [value.candidate_fingerprint for value in selected]}
        ),
        "acquisition_plan_ids": sorted(
            set(selections["acquisition_plan_id"].to_list())
        ),
        "sources": sorted({str(value.candidate["source"]) for value in selected}),
        "source_snapshot_versions": sorted(
            {str(value.candidate["source_snapshot_version"]) for value in selected}
        ),
    }


def _provider_policy(
    candidate: Mapping[str, object],
    policies: tuple[ProviderMediaDownloadPolicy, ...],
) -> ProviderMediaDownloadPolicy | None:
    try:
        parsed = urlsplit(str(candidate["media_identifier"]))
        host = _normalise_host(parsed.hostname or "")
    except ValueError:
        return None
    source = str(candidate["source"]).casefold()
    exact_matches = [
        policy
        for policy in policies
        if policy.source.casefold() == source and host in policy.allowed_hosts
    ]
    if len(exact_matches) > 1:
        raise ValueError("multiple provider policies match a media candidate")
    if exact_matches:
        return exact_matches[0]
    return None


def _provider_request_url(
    candidate: Mapping[str, object],
    policy: ProviderMediaDownloadPolicy,
) -> str:
    value = str(candidate["media_identifier"])
    _validate_provider_url(
        value,
        policy,
        host_validator=None,
        expected_provider_media_id=str(candidate["provider_media_id"]),
        resolve_public_addresses=False,
    )
    if policy.url_strategy == "direct":
        return value
    parsed = urlsplit(value)
    match = _INATURALIST_PHOTO_PATH.fullmatch(parsed.path)
    if match is None:
        raise ValueError("iNaturalist URL is not a sanctioned photo path")
    if match.group("photo_id") != str(candidate["provider_media_id"]):
        raise ValueError("iNaturalist photo ID does not match provider_media_id")
    rewritten_path = (
        f"/photos/{match.group('photo_id')}/{policy.inaturalist_image_style}."
        f"{match.group('extension').casefold()}"
    )
    return urlunsplit((parsed.scheme, parsed.netloc, rewritten_path, parsed.query, ""))


def _validate_provider_url(
    url: str,
    policy: ProviderMediaDownloadPolicy,
    *,
    host_validator: _HostValidator | None,
    expected_provider_media_id: str | None = None,
    resolve_public_addresses: bool = True,
    resolution_timeout_seconds: float | None = None,
) -> None:
    parsed = urlsplit(url)
    scheme = parsed.scheme.casefold()
    if scheme not in policy.allowed_schemes:
        raise ValueError("URL scheme is not allowed by provider policy")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("provider URL must not contain user information")
    host = _normalise_host(parsed.hostname or "")
    if host not in policy.allowed_hosts:
        raise ValueError("URL host is not allowed by provider policy")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("provider URL port is invalid") from exc
    default_port = 443 if scheme == "https" else 80
    if port is not None and port != default_port:
        raise ValueError("provider URL uses a nondefault port")
    if not parsed.path.startswith("/"):
        raise ValueError("provider URL must contain an absolute path")
    if parsed.fragment:
        raise ValueError("provider URL must not contain a fragment")
    if resolve_public_addresses and policy.resolve_public_addresses:
        if host_validator is None:
            raise ValueError("provider host validator is required")
        host_validator.require_public(
            host,
            timeout_seconds=resolution_timeout_seconds,
        )
    if policy.url_strategy == "inaturalist_photo":
        match = _INATURALIST_PHOTO_PATH.fullmatch(parsed.path)
        if match is None:
            raise ValueError("iNaturalist redirect left the sanctioned photo path")
        if (
            expected_provider_media_id is not None
            and match.group("photo_id") != expected_provider_media_id
        ):
            raise ValueError("iNaturalist photo ID does not match provider_media_id")


def _fetch_and_validate(
    pending: _PendingDownload,
    *,
    config: ReferenceMediaDownloadConfig,
    client: httpx.Client,
    limiters: _OriginLimiterRegistry,
    host_validator: _HostValidator,
    decode_limiter: threading.BoundedSemaphore,
    prevalidate_dns: bool,
    isolate_decode: bool,
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> _PreparedDownload:
    current_url = pending.requested_url
    redirects = 0
    attempts = 0
    retries = 0
    rate_limits = 0
    retry_wait = 0.0
    deadline = monotonic() + config.max_download_seconds

    def failed(reason: str) -> _PreparedDownload:
        return _prepared_failure(
            pending,
            attempts=attempts,
            retries=retries,
            rate_limits=rate_limits,
            retry_wait=retry_wait,
            decode_status="download_failed",
            reason=reason,
        )

    while True:
        remaining = _remaining_seconds(deadline, monotonic)
        if remaining <= 0:
            return failed("item_deadline_exceeded")
        if attempts >= config.max_attempts:
            return failed("attempt_budget_exhausted")
        limiter = limiters.get(pending.provider_policy, current_url)
        try:
            with limiter.slot(deadline=deadline):
                try:
                    try:
                        _validate_provider_url(
                            current_url,
                            pending.provider_policy,
                            host_validator=(
                                host_validator if prevalidate_dns else None
                            ),
                            expected_provider_media_id=str(
                                pending.selected.candidate["provider_media_id"]
                            ),
                            resolve_public_addresses=prevalidate_dns,
                            resolution_timeout_seconds=remaining,
                        )
                    except ValueError as exc:
                        raise _PermanentResponse(f"provider_constraint:{exc}") from exc
                    attempts += 1
                    response = _request_once(
                        current_url,
                        pending=pending,
                        config=config,
                        client=client,
                        now=now,
                        deadline=deadline,
                        monotonic=monotonic,
                        isolate_decode=isolate_decode,
                        decode_limiter=decode_limiter,
                    )
                except _RetryableResponse as exc:
                    if exc.status_code == 429:
                        rate_limits += 1
                    retry_after = exc.retry_after_seconds
                    if retry_after is not None:
                        if retry_after > config.max_retry_after_seconds:
                            reason = (
                                "retry_after_exceeds_operational_limit_http_"
                                f"{exc.status_code}"
                            )
                            limiter.block(reason)
                            raise _PermanentResponse(reason) from exc
                        limiter.defer_for(retry_after)
                    raise
        except _RetryableResponse as exc:
            if attempts >= config.max_attempts:
                return failed(f"retry_exhausted_http_{exc.status_code}")
            retry_number = retries + 1
            retries += 1
            delay = (
                _backoff_seconds(pending, retry_number, config)
                if exc.retry_after_seconds is None
                else max(0.0, exc.retry_after_seconds)
            )
            if delay >= _remaining_seconds(deadline, monotonic):
                return failed("item_deadline_exceeded")
            retry_wait += delay
            _log_event(
                "reference_media_download_retry",
                reference_media_id=pending.selected.candidate["reference_media_id"],
                status_code=exc.status_code,
                wait_seconds=delay,
                attempt=attempts,
            )
            if exc.retry_after_seconds is None:
                sleep(delay)
            continue
        except httpx.TransportError as exc:
            if _remaining_seconds(deadline, monotonic) <= 0:
                return failed("item_deadline_exceeded")
            if attempts >= config.max_attempts:
                return failed(f"transport_retry_exhausted:{type(exc).__name__}")
            retry_number = retries + 1
            retries += 1
            delay = _backoff_seconds(pending, retry_number, config)
            if delay >= _remaining_seconds(deadline, monotonic):
                return failed("item_deadline_exceeded")
            retry_wait += delay
            sleep(delay)
            continue
        except _PermanentResponse as exc:
            return _prepared_failure(
                pending,
                attempts=attempts,
                retries=retries,
                rate_limits=rate_limits,
                retry_wait=retry_wait,
                decode_status="download_failed",
                reason=exc.reason,
            )
        except _PayloadFailure as exc:
            return _prepared_failure(
                pending,
                attempts=attempts,
                retries=retries,
                rate_limits=rate_limits,
                retry_wait=retry_wait,
                decode_status=exc.decode_status,
                reason=exc.reason,
                content_type=exc.content_type,
            )

        if isinstance(response, _Redirect):
            redirects += 1
            if redirects > config.max_redirects:
                return _prepared_failure(
                    pending,
                    attempts=attempts,
                    retries=retries,
                    rate_limits=rate_limits,
                    retry_wait=retry_wait,
                    decode_status="download_failed",
                    reason="redirect_limit_exceeded",
                )
            try:
                _validate_provider_url(
                    response.url,
                    pending.provider_policy,
                    host_validator=None,
                    expected_provider_media_id=str(
                        pending.selected.candidate["provider_media_id"]
                    ),
                    resolve_public_addresses=False,
                )
            except ValueError as exc:
                return _prepared_failure(
                    pending,
                    attempts=attempts,
                    retries=retries,
                    rate_limits=rate_limits,
                    retry_wait=retry_wait,
                    decode_status="download_failed",
                    reason=f"redirect_provider_constraint:{exc}",
                )
            current_url = response.url
            continue
        return _PreparedDownload(
            pending=pending,
            path=response.path,
            content_type=response.content_type,
            source_byte_count=response.source_byte_count,
            decoded_width=response.decoded_width,
            decoded_height=response.decoded_height,
            sha256=response.sha256,
            perceptual_hash=response.perceptual_hash,
            final_url=current_url,
            attempt_count=attempts,
            retry_count=retries,
            rate_limit_count=rate_limits,
            retry_wait_seconds=retry_wait,
            decode_status="valid",
            failure_reason=None,
        )


def _request_once(
    url: str,
    *,
    pending: _PendingDownload,
    config: ReferenceMediaDownloadConfig,
    client: httpx.Client,
    now: Callable[[], datetime],
    deadline: float,
    monotonic: Callable[[], float],
    isolate_decode: bool,
    decode_limiter: threading.BoundedSemaphore,
) -> _PreparedDownload | _Redirect:
    remaining = _remaining_seconds(deadline, monotonic)
    if remaining <= 0:
        raise _PermanentResponse("item_deadline_exceeded")
    with client.stream(
        "GET",
        url,
        headers={
            "Accept": ", ".join(config.allowed_content_types),
            "User-Agent": config.user_agent,
        },
        timeout=min(config.timeout_seconds, remaining),
        follow_redirects=False,
        extensions={"biominer_deadline": deadline},
    ) as response:
        status = response.status_code
        if status in _REDIRECT_STATUSES:
            location = response.headers.get("Location")
            if not location:
                raise _PermanentResponse("redirect_missing_location")
            return _Redirect(urljoin(url, location))
        if status in config.retry_statuses:
            raise _RetryableResponse(
                status,
                retry_after_seconds=_retry_after_seconds(
                    response.headers.get("Retry-After"),
                    now=_aware_utc(now()),
                ),
            )
        if status != 200:
            raise _PermanentResponse(f"http_status_{status}")
        return _consume_response(
            response,
            pending=pending,
            config=config,
            deadline=deadline,
            monotonic=monotonic,
            isolate_decode=isolate_decode,
            decode_limiter=decode_limiter,
        )


def _consume_response(
    response: httpx.Response,
    *,
    pending: _PendingDownload,
    config: ReferenceMediaDownloadConfig,
    deadline: float,
    monotonic: Callable[[], float],
    isolate_decode: bool,
    decode_limiter: threading.BoundedSemaphore,
) -> _PreparedDownload:
    if _remaining_seconds(deadline, monotonic) <= 0:
        raise _PermanentResponse("item_deadline_exceeded")
    content_encoding = str(response.headers.get("Content-Encoding") or "").strip()
    if content_encoding and content_encoding.casefold() != "identity":
        raise _PayloadFailure(
            "unsupported_content_encoding",
            decode_status="invalid_content_type",
        )
    raw_content_type = response.headers.get("Content-Type")
    if not raw_content_type:
        raise _PayloadFailure(
            "missing_content_type",
            decode_status="invalid_content_type",
        )
    try:
        content_type = _canonical_content_type(raw_content_type)
    except ValueError as exc:
        raise _PayloadFailure(
            "invalid_content_type_header",
            decode_status="invalid_content_type",
        ) from exc
    if content_type not in config.allowed_content_types:
        raise _PayloadFailure(
            f"content_type_not_allowed:{content_type}",
            decode_status="invalid_content_type",
            content_type=content_type,
        )
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise _PayloadFailure(
                "invalid_content_length",
                decode_status="download_failed",
                content_type=content_type,
            ) from exc
        if declared_length < 0 or declared_length > config.max_source_bytes:
            raise _PayloadFailure(
                "source_payload_too_large",
                decode_status="download_failed",
                content_type=content_type,
            )
    if config.temporary_directory is not None:
        config.temporary_directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    byte_count = 0
    sha256 = hashlib.sha256()
    checksum_hasher = _source_checksum_hasher(pending.selected.candidate)
    head = bytearray()
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix="biominer-reference-",
            suffix=".part",
            dir=config.temporary_directory,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            chunks = (
                (response.content,)
                if response.is_stream_consumed
                else response.iter_raw()
            )
            for chunk in chunks:
                if _remaining_seconds(deadline, monotonic) <= 0:
                    raise _PermanentResponse("item_deadline_exceeded")
                if not chunk:
                    continue
                byte_count += len(chunk)
                if byte_count > config.max_source_bytes:
                    raise _PayloadFailure(
                        "source_payload_too_large",
                        decode_status="download_failed",
                        content_type=content_type,
                    )
                if len(head) < 512:
                    head.extend(chunk[: 512 - len(head)])
                sha256.update(chunk)
                if checksum_hasher is not None:
                    checksum_hasher.update(chunk)
                temporary.write(chunk)
                if _remaining_seconds(deadline, monotonic) <= 0:
                    raise _PermanentResponse("item_deadline_exceeded")
        if byte_count == 0:
            raise _PayloadFailure(
                "empty_image_payload",
                decode_status="invalid_content_type",
                content_type=content_type,
            )
        sniffed = _sniff_content_type(bytes(head))
        if sniffed == "text/html":
            raise _PayloadFailure(
                "html_payload_masquerading_as_image",
                decode_status="invalid_content_type",
                content_type=content_type,
            )
        if sniffed is None:
            raise _PayloadFailure(
                "unrecognised_image_signature",
                decode_status="invalid_content_type",
                content_type=content_type,
            )
        if sniffed != content_type:
            raise _PayloadFailure(
                f"content_type_signature_mismatch:{content_type}:{sniffed}",
                decode_status="invalid_content_type",
                content_type=content_type,
            )
        if _remaining_seconds(deadline, monotonic) <= 0:
            raise _PermanentResponse("item_deadline_exceeded")
        decode_wait = _remaining_seconds(deadline, monotonic)
        if not decode_limiter.acquire(timeout=decode_wait):
            raise _PermanentResponse("item_deadline_exceeded")
        try:
            decode_timeout = _remaining_seconds(deadline, monotonic)
            if isolate_decode:
                width, height, decoded_content_type, perceptual_hash = (
                    _decode_image_isolated(
                        temporary_path,
                        max_pixels=config.max_decoded_pixels,
                        max_memory_bytes=config.max_decode_memory_bytes,
                        timeout_seconds=decode_timeout,
                    )
                )
            else:
                width, height, decoded_content_type, perceptual_hash = _decode_image(
                    temporary_path,
                    max_pixels=config.max_decoded_pixels,
                )
        finally:
            decode_limiter.release()
        if _remaining_seconds(deadline, monotonic) <= 0:
            raise _PermanentResponse("item_deadline_exceeded")
        if decoded_content_type != content_type:
            raise _PayloadFailure(
                f"content_type_decoder_mismatch:{content_type}:{decoded_content_type}",
                decode_status="invalid_content_type",
                content_type=content_type,
            )
        if checksum_hasher is not None and not _source_checksum_matches(
            pending.selected.candidate,
            checksum_hasher.hexdigest(),
        ):
            raise _PayloadFailure(
                "source_checksum_mismatch",
                decode_status="decode_failed",
                content_type=content_type,
            )
        return _PreparedDownload(
            pending=pending,
            path=temporary_path,
            content_type=content_type,
            source_byte_count=byte_count,
            decoded_width=width,
            decoded_height=height,
            sha256="sha256:" + sha256.hexdigest(),
            perceptual_hash=perceptual_hash,
            final_url=str(response.url),
            attempt_count=0,
            retry_count=0,
            rate_limit_count=0,
            retry_wait_seconds=0.0,
            decode_status="valid",
            failure_reason=None,
        )
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _decode_image_isolated(
    path: Path,
    *,
    max_pixels: int,
    max_memory_bytes: int = _DEFAULT_MAX_DECODE_MEMORY_BYTES,
    timeout_seconds: float,
) -> tuple[int, int, str, str]:
    if timeout_seconds <= 0:
        raise _PermanentResponse("item_deadline_exceeded")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_decode_image_worker,
        args=(str(path), max_pixels, max_memory_bytes, sender),
        name="reference-image-decode",
        daemon=True,
    )
    started = time.monotonic()
    try:
        process.start()
        sender.close()
        if _darwin_decode_memory_watchdog_required():
            _join_decode_worker_with_watchdog(
                process,
                deadline=started + timeout_seconds,
                max_memory_bytes=max_memory_bytes,
            )
        else:
            process.join(max(0.0, timeout_seconds - (time.monotonic() - started)))
            if process.is_alive():
                _terminate_decode_worker(process)
                raise _PermanentResponse("item_deadline_exceeded")
        if not receiver.poll():
            raise _PayloadFailure(
                f"image_decode_worker_failed:exit_{process.exitcode}",
                decode_status="decode_failed",
            )
        payload = receiver.recv()
    finally:
        receiver.close()
        sender.close()
        if process.is_alive():  # pragma: no cover - exceptional parent cleanup.
            process.kill()
            process.join(timeout=1.0)
    if (
        not isinstance(payload, tuple)
        or not payload
        or payload[0] not in {"ok", "error"}
    ):
        raise _PayloadFailure(
            "image_decode_worker_returned_invalid_result",
            decode_status="decode_failed",
        )
    if payload[0] == "error":
        reason = str(payload[1]) if len(payload) > 1 else "image_decode_worker_failed"
        content_type = (
            str(payload[2]) if len(payload) > 2 and payload[2] is not None else None
        )
        raise _PayloadFailure(
            reason,
            decode_status="decode_failed",
            content_type=content_type,
        )
    if (
        len(payload) != 5
        or not isinstance(payload[1], int)
        or not isinstance(payload[2], int)
        or not isinstance(payload[3], str)
        or not isinstance(payload[4], str)
    ):
        raise _PayloadFailure(
            "image_decode_worker_returned_invalid_result",
            decode_status="decode_failed",
        )
    return payload[1], payload[2], payload[3], payload[4]


def _decode_image_worker(
    path: str,
    max_pixels: int,
    max_memory_bytes: int,
    sender: Any,
) -> None:
    try:
        _apply_decode_memory_limit(max_memory_bytes)
        result = _decode_image(Path(path), max_pixels=max_pixels)
    except _PayloadFailure as exc:
        payload: tuple[object, ...] = ("error", exc.reason, exc.content_type)
    except BaseException as exc:  # pragma: no cover - child crash containment.
        payload = ("error", f"image_decode_worker_exception:{type(exc).__name__}", None)
    else:
        payload = ("ok", *result)
    try:
        sender.send(payload)
    finally:
        sender.close()


def _apply_decode_memory_limit(max_memory_bytes: int) -> None:
    if os.name != "posix":
        return
    if _darwin_decode_memory_watchdog_required():
        return
    try:
        import resource

        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = (
            max_memory_bytes
            if hard == resource.RLIM_INFINITY
            else min(
                max_memory_bytes,
                hard,
            )
        )
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError("unable to enforce image decode memory limit") from exc


def _darwin_decode_memory_watchdog_required() -> bool:
    return os.name == "posix" and os.uname().sysname == "Darwin"


def _join_decode_worker_with_watchdog(
    process: multiprocessing.Process,
    *,
    deadline: float,
    max_memory_bytes: int,
) -> None:
    if process.pid is None:  # pragma: no cover - guarded by process.start().
        raise RuntimeError("image decode worker has no process identifier")
    while process.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_decode_worker(process)
            raise _PermanentResponse("item_deadline_exceeded")
        rss_bytes = _process_rss_bytes(
            process.pid,
            timeout_seconds=min(1.0, remaining),
        )
        if rss_bytes is None:
            process.join(timeout=0)
            if not process.is_alive():
                break
            _terminate_decode_worker(process)
            raise _PayloadFailure(
                "image_decode_memory_monitor_unavailable",
                decode_status="decode_failed",
            )
        if rss_bytes > max_memory_bytes:
            _terminate_decode_worker(process)
            raise _PayloadFailure(
                "image_decode_memory_limit_exceeded",
                decode_status="decode_failed",
            )
        process.join(timeout=min(0.02, remaining))
    process.join(timeout=0)


def _process_rss_bytes(pid: int, *, timeout_seconds: float) -> int | None:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(output) * 1024
    except ValueError:
        return None


def _terminate_decode_worker(process: multiprocessing.Process) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():  # pragma: no cover - terminate is normally sufficient.
        process.kill()
        process.join(timeout=1.0)


def _decode_image(path: Path, *, max_pixels: int) -> tuple[int, int, str, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                if int(getattr(image, "n_frames", 1)) != 1:
                    raise ValueError("multi-frame images are not supported")
                if width <= 0 or height <= 0:
                    raise ValueError("decoded image dimensions must be positive")
                if width * height > max_pixels:
                    raise ValueError("decoded image exceeds the configured pixel limit")
                image.verify()
            with Image.open(path) as image:
                if str(image.format or "").upper() != image_format:
                    raise ValueError("image decoder format changed after reopen")
                if image.size != (width, height):
                    raise ValueError("image dimensions changed after reopen")
                image.load()
                perceptual_hash = compute_reference_perceptual_hash(image)
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise _PayloadFailure(
            f"image_decode_failed:{type(exc).__name__}",
            decode_status="decode_failed",
        ) from exc
    content_type = _FORMAT_CONTENT_TYPES.get(image_format)
    if content_type is None:
        raise _PayloadFailure(
            f"unsupported_decoded_format:{image_format or 'unknown'}",
            decode_status="invalid_content_type",
        )
    return width, height, content_type, perceptual_hash


def _commit_prepared(
    prepared: _PreparedDownload,
    *,
    storage: CloudStorage,
    output_prefix: str,
    now: Callable[[], datetime],
    run_id: str,
    config: ReferenceMediaDownloadConfig,
    licence_policy: ReferenceLicencePolicy,
) -> dict[str, object]:
    item = prepared.pending.selected
    if not prepared.valid:
        return _failure_row(
            item,
            licence_status=prepared.pending.licence.status,
            decode_status=prepared.decode_status,
            reason=prepared.failure_reason or "download_failed",
            attempt_count=prepared.attempt_count,
            content_type=prepared.content_type,
            config_fingerprint=config.semantic_fingerprint,
            licence_policy_fingerprint=licence_policy.fingerprint,
        )
    assert prepared.path is not None
    assert prepared.sha256 is not None
    assert prepared.perceptual_hash is not None
    assert prepared.content_type is not None
    assert prepared.source_byte_count is not None
    assert prepared.decoded_width is not None
    assert prepared.decoded_height is not None
    digest = prepared.sha256.removeprefix("sha256:")
    object_uri = join_uri(
        output_prefix,
        "source_objects",
        "sha256",
        digest[:2],
        digest[2:4],
        f"{digest}.{_CONTENT_TYPE_EXTENSIONS[prepared.content_type]}",
    )
    downloaded_at = _aware_utc(now())
    row: dict[str, object] = {
        "schema_version": REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
        "reference_media_id": item.candidate["reference_media_id"],
        "source_object_uri": object_uri,
        "content_type": prepared.content_type,
        "source_byte_count": prepared.source_byte_count,
        "decoded_width": prepared.decoded_width,
        "decoded_height": prepared.decoded_height,
        "sha256": prepared.sha256,
        "perceptual_hash": prepared.perceptual_hash,
        "duplicate_group_id": None,
        "duplicate_type": None,
        "canonical_reference_media_id": None,
        "provider_mirror_ids": [],
        "downloaded_at": downloaded_at,
        "download_attempt_count": prepared.attempt_count,
        "licence_policy_status": prepared.pending.licence.status,
        "decode_status": "valid",
        "quarantine_reason": None,
    }
    row["object_fingerprint"] = _committed_object_fingerprint(
        prepared.pending.binding,
        row,
        final_url=prepared.final_url,
    )
    try:
        try:
            storage.write_file(
                object_uri,
                prepared.path,
                content_type=prepared.content_type,
                overwrite=False,
            )
        except FileExistsError:
            if not storage.exists(object_uri):
                raise
        if not storage.exists(object_uri):
            raise OSError("source object was not durable after upload")
        if storage.file_size(object_uri) != prepared.source_byte_count:
            raise OSError("durable source object size does not match downloaded bytes")
        if storage.file_sha256(object_uri) != prepared.sha256:
            raise OSError(
                "durable source object checksum does not match downloaded bytes"
            )
        if storage.exists(prepared.pending.checkpoint_uri):
            return _load_committed_checkpoint(
                storage,
                prepared.pending.checkpoint_uri,
                expected_binding=prepared.pending.binding,
                expected_evidence=_expected_checkpoint_evidence(
                    item.candidate,
                    requested_url=prepared.pending.requested_url,
                ),
                provider_policy=prepared.pending.provider_policy,
                expected_provider_media_id=str(item.candidate["provider_media_id"]),
                config=config,
                now=now,
            )
        committed_at = _aware_utc(now())
        checkpoint = {
            "schema_version": REFERENCE_MEDIA_CHECKPOINT_VERSION,
            "binding": prepared.pending.binding,
            "object": _jsonable(row),
            "download_evidence": {
                **_expected_checkpoint_evidence(
                    item.candidate,
                    requested_url=prepared.pending.requested_url,
                ),
                "final_url": prepared.final_url,
            },
            "commit": {
                "run_id": run_id,
                "committed_at": committed_at.isoformat(),
                "object_write_precedes_checkpoint": True,
            },
        }
        storage.write_json(prepared.pending.checkpoint_uri, checkpoint)
    except Exception as exc:  # noqa: BLE001 - convert per-item commit errors to evidence.
        return _failure_row(
            item,
            licence_status=prepared.pending.licence.status,
            decode_status="download_failed",
            reason=f"object_commit_failed:{type(exc).__name__}",
            attempt_count=prepared.attempt_count,
            content_type=prepared.content_type,
            config_fingerprint=config.semantic_fingerprint,
            licence_policy_fingerprint=licence_policy.fingerprint,
        )
    _log_event(
        "reference_media_object_committed",
        run_id=run_id,
        reference_media_id=item.candidate["reference_media_id"],
        source_object_uri=object_uri,
        checkpoint_uri=prepared.pending.checkpoint_uri,
        source_byte_count=prepared.source_byte_count,
        decoded_width=prepared.decoded_width,
        decoded_height=prepared.decoded_height,
        perceptual_hash=prepared.perceptual_hash,
    )
    return row


def _failure_row(
    item: _SelectedMedia,
    *,
    licence_status: str,
    decode_status: str,
    reason: str,
    attempt_count: int,
    content_type: str | None,
    config_fingerprint: str,
    licence_policy_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
        "reference_media_id": item.candidate["reference_media_id"],
        "source_object_uri": None,
        "content_type": content_type,
        "source_byte_count": None,
        "decoded_width": None,
        "decoded_height": None,
        "sha256": None,
        "perceptual_hash": None,
        "duplicate_group_id": None,
        "duplicate_type": None,
        "canonical_reference_media_id": None,
        "provider_mirror_ids": [],
        "downloaded_at": None,
        "download_attempt_count": attempt_count,
        "licence_policy_status": licence_status,
        "decode_status": decode_status,
        "quarantine_reason": reason,
        "object_fingerprint": _fingerprint(
            {
                "reference_media_id": item.candidate["reference_media_id"],
                "candidate_fingerprint": item.candidate_fingerprint,
                "configuration_fingerprint": config_fingerprint,
                "licence_policy_fingerprint": licence_policy_fingerprint,
                "licence_policy_status": licence_status,
                "decode_status": decode_status,
                "reason": reason,
                "attempt_count": attempt_count,
                "content_type": content_type,
            }
        ),
    }


def _prepared_failure(
    pending: _PendingDownload,
    *,
    attempts: int,
    retries: int,
    rate_limits: int,
    retry_wait: float,
    decode_status: str,
    reason: str,
    content_type: str | None = None,
) -> _PreparedDownload:
    return _PreparedDownload(
        pending=pending,
        path=None,
        content_type=content_type,
        source_byte_count=None,
        decoded_width=None,
        decoded_height=None,
        sha256=None,
        perceptual_hash=None,
        final_url=None,
        attempt_count=attempts,
        retry_count=retries,
        rate_limit_count=rate_limits,
        retry_wait_seconds=retry_wait,
        decode_status=decode_status,
        failure_reason=reason,
    )


def _checkpoint_binding(
    item: _SelectedMedia,
    *,
    decision: ReferenceLicenceDecision,
    provider_policy: ProviderMediaDownloadPolicy,
    requested_url: str,
    config: ReferenceMediaDownloadConfig,
    licence_policy: ReferenceLicencePolicy,
) -> dict[str, object]:
    return {
        "downloader_version": REFERENCE_MEDIA_DOWNLOADER_VERSION,
        "reference_media_id": item.candidate["reference_media_id"],
        "candidate_fingerprint": item.candidate_fingerprint,
        "configuration_fingerprint": config.semantic_fingerprint,
        "licence_policy_version": licence_policy.version,
        "licence_policy_fingerprint": licence_policy.fingerprint,
        "licence_policy_status": decision.status,
        "canonical_licence": decision.canonical_licence,
        "provider_policy_fingerprint": provider_policy.validation_fingerprint,
        "requested_url": requested_url,
    }


def _checkpoint_uri(
    output_prefix: str,
    candidate: Mapping[str, object],
) -> str:
    media_id = str(candidate["reference_media_id"])
    digest = hashlib.sha256(media_id.encode("utf-8")).hexdigest()
    return join_uri(
        output_prefix,
        "checkpoints",
        "reference_media",
        digest[:2],
        f"{digest}.json",
    )


def _load_committed_checkpoint(
    storage: CloudStorage,
    checkpoint_uri: str,
    *,
    expected_binding: Mapping[str, object],
    expected_evidence: Mapping[str, object],
    provider_policy: ProviderMediaDownloadPolicy,
    expected_provider_media_id: str,
    config: ReferenceMediaDownloadConfig,
    now: Callable[[], datetime],
) -> dict[str, object]:
    payload = storage.read_json(checkpoint_uri)
    if set(payload) != {
        "schema_version",
        "binding",
        "object",
        "download_evidence",
        "commit",
    }:
        raise ValueError("reference media checkpoint shape is incompatible")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        REFERENCE_MEDIA_CHECKPOINT_VERSION,
        _LEGACY_REFERENCE_MEDIA_CHECKPOINT_VERSION,
    }:
        raise ValueError("reference media checkpoint schema is incompatible")
    legacy = schema_version == _LEGACY_REFERENCE_MEDIA_CHECKPOINT_VERSION
    checkpoint_binding = (
        _legacy_checkpoint_binding(expected_binding, config=config)
        if legacy
        else dict(expected_binding)
    )
    if payload.get("binding") != checkpoint_binding:
        raise ValueError("reference media checkpoint binding is incompatible")
    raw_object = payload.get("object")
    if not isinstance(raw_object, dict):
        raise ValueError("reference media checkpoint object is missing")
    row = dict(raw_object)
    downloaded_at = row.get("downloaded_at")
    if not isinstance(downloaded_at, str):
        raise ValueError("reference media checkpoint downloaded_at is invalid")
    row["downloaded_at"] = _aware_utc(datetime.fromisoformat(downloaded_at))
    if legacy:
        _validate_legacy_media_object_row(row)
        validation_row = dict(row)
        validation_row["schema_version"] = REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION
        validation_row["perceptual_hash"] = (
            f"{REFERENCE_PERCEPTUAL_HASH_VERSION}:" + "0" * 32
        )
        frame = reference_media_objects_frame([validation_row])
    else:
        frame = reference_media_objects_frame([row])
    committed = frame.to_dicts()[0]
    if committed["reference_media_id"] != expected_binding["reference_media_id"]:
        raise ValueError("reference media checkpoint object identity is incompatible")
    if committed["licence_policy_status"] != expected_binding["licence_policy_status"]:
        raise ValueError("reference media checkpoint licence status is incompatible")
    evidence = payload.get("download_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        *expected_evidence,
        "final_url",
    }:
        raise ValueError("reference media checkpoint download evidence is incomplete")
    if any(evidence.get(key) != value for key, value in expected_evidence.items()):
        raise ValueError("reference media checkpoint download evidence is incompatible")
    final_url = str(evidence.get("final_url") or "").strip()
    if not final_url:
        raise ValueError("reference media checkpoint final URL is invalid")
    try:
        _validate_provider_url(
            final_url,
            provider_policy,
            host_validator=None,
            expected_provider_media_id=expected_provider_media_id,
            resolve_public_addresses=False,
        )
    except ValueError as exc:
        raise ValueError("reference media checkpoint final URL is invalid") from exc
    expected_fingerprint = (
        _legacy_committed_object_fingerprint(
            checkpoint_binding,
            row,
            final_url=final_url,
        )
        if legacy
        else _committed_object_fingerprint(
            expected_binding,
            committed,
            final_url=final_url,
        )
    )
    if row["object_fingerprint"] != expected_fingerprint:
        raise ValueError("reference media checkpoint object fingerprint is invalid")
    commit = payload.get("commit")
    if not isinstance(commit, dict) or set(commit) != {
        "run_id",
        "committed_at",
        "object_write_precedes_checkpoint",
    }:
        raise ValueError("reference media checkpoint commit evidence is incomplete")
    if commit.get("object_write_precedes_checkpoint") is not True:
        raise ValueError("reference media checkpoint commit marker is invalid")
    if not str(commit.get("run_id") or "").strip():
        raise ValueError("reference media checkpoint commit run ID is invalid")
    committed_at = commit.get("committed_at")
    if not isinstance(committed_at, str):
        raise ValueError("reference media checkpoint commit timestamp is invalid")
    try:
        parsed_committed_at = _aware_utc(datetime.fromisoformat(committed_at))
    except ValueError as exc:
        raise ValueError(
            "reference media checkpoint commit timestamp is invalid"
        ) from exc
    if parsed_committed_at < committed["downloaded_at"]:
        raise ValueError(
            "reference media checkpoint commit timestamp predates download"
        )
    object_uri = str(committed["source_object_uri"])
    if not storage.exists(object_uri):
        raise ValueError("reference media checkpoint points to a missing object")
    if storage.file_size(object_uri) != int(committed["source_byte_count"]):
        raise ValueError("reference media checkpoint object size is incompatible")
    if storage.file_sha256(object_uri) != str(committed["sha256"]):
        raise ValueError("reference media checkpoint object checksum is incompatible")
    if legacy:
        return _upgrade_legacy_committed_checkpoint(
            storage,
            checkpoint_uri=checkpoint_uri,
            payload=payload,
            legacy_row=row,
            expected_binding=expected_binding,
            final_url=final_url,
            config=config,
            now=now,
        )
    return committed


def _legacy_checkpoint_binding(
    expected_binding: Mapping[str, object],
    *,
    config: ReferenceMediaDownloadConfig,
) -> dict[str, object]:
    binding = dict(expected_binding)
    binding["downloader_version"] = _LEGACY_REFERENCE_MEDIA_DOWNLOADER_VERSION
    binding["configuration_fingerprint"] = _fingerprint(
        {
            "downloader_version": _LEGACY_REFERENCE_MEDIA_DOWNLOADER_VERSION,
            "max_source_bytes": config.max_source_bytes,
            "max_decoded_pixels": config.max_decoded_pixels,
            "allowed_content_types": config.allowed_content_types,
        }
    )
    return binding


def _validate_legacy_media_object_row(row: Mapping[str, object]) -> None:
    if set(row) != set(reference_media_object_schema()):
        raise ValueError("legacy reference media object shape is incompatible")
    if row.get("schema_version") != _LEGACY_REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION:
        raise ValueError("legacy reference media object schema is incompatible")
    if (
        any(
            row.get(field_name) is not None
            for field_name in (
                "perceptual_hash",
                "duplicate_group_id",
                "duplicate_type",
                "canonical_reference_media_id",
            )
        )
        or row.get("provider_mirror_ids") != []
    ):
        raise ValueError("legacy reference media object has unexpected derived state")


def _backfill_legacy_media_object(
    storage: CloudStorage,
    row: Mapping[str, object],
    *,
    config: ReferenceMediaDownloadConfig,
) -> dict[str, object]:
    _validate_legacy_media_object_row(row)
    upgraded = dict(row)
    upgraded["schema_version"] = REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION
    if upgraded["decode_status"] != "valid":
        return reference_media_objects_frame([upgraded]).to_dicts()[0]

    object_uri = str(upgraded["source_object_uri"])
    expected_size = int(upgraded["source_byte_count"])
    expected_sha256 = str(upgraded["sha256"])
    if not 0 < expected_size <= config.max_source_bytes:
        raise ValueError(
            "legacy reference media object exceeds the current source byte limit"
        )
    if upgraded["content_type"] not in config.allowed_content_types:
        raise ValueError(
            "legacy reference media object content type is not currently allowed"
        )
    if not storage.exists(object_uri):
        raise ValueError("legacy reference media object is missing")
    if storage.file_size(object_uri) != expected_size:
        raise ValueError("legacy reference media object size is incompatible")
    if storage.file_sha256(object_uri) != expected_sha256:
        raise ValueError("legacy reference media object checksum is incompatible")

    if config.temporary_directory is not None:
        config.temporary_directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix="biominer-reference-backfill-",
            suffix=".part",
            dir=config.temporary_directory,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        storage.materialize_file(object_uri, temporary_path, overwrite=True)
        if temporary_path.stat().st_size != expected_size:
            raise ValueError("materialized legacy object size is incompatible")
        if _path_sha256(temporary_path) != expected_sha256:
            raise ValueError("materialized legacy object checksum is incompatible")
        width, height, content_type, perceptual_hash = _decode_image_isolated(
            temporary_path,
            max_pixels=config.max_decoded_pixels,
            max_memory_bytes=config.max_decode_memory_bytes,
            timeout_seconds=config.max_download_seconds,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if (width, height) != (
        int(upgraded["decoded_width"]),
        int(upgraded["decoded_height"]),
    ):
        raise ValueError("legacy reference media decoded dimensions changed")
    if content_type != upgraded["content_type"]:
        raise ValueError("legacy reference media decoded content type changed")
    upgraded["perceptual_hash"] = perceptual_hash
    return reference_media_objects_frame([upgraded]).to_dicts()[0]


def _upgrade_legacy_committed_checkpoint(
    storage: CloudStorage,
    *,
    checkpoint_uri: str,
    payload: Mapping[str, object],
    legacy_row: Mapping[str, object],
    expected_binding: Mapping[str, object],
    final_url: str,
    config: ReferenceMediaDownloadConfig,
    now: Callable[[], datetime],
) -> dict[str, object]:
    upgraded = _backfill_legacy_media_object(
        storage,
        legacy_row,
        config=config,
    )
    upgraded["object_fingerprint"] = _committed_object_fingerprint(
        expected_binding,
        upgraded,
        final_url=final_url,
    )
    validated = reference_media_objects_frame([upgraded]).to_dicts()[0]
    original_commit = payload.get("commit")
    if not isinstance(original_commit, Mapping):
        raise ValueError("legacy reference media checkpoint commit is invalid")
    migrated_at = max(_aware_utc(now()), validated["downloaded_at"])
    migrated_payload = {
        "schema_version": REFERENCE_MEDIA_CHECKPOINT_VERSION,
        "binding": dict(expected_binding),
        "object": _jsonable(validated),
        "download_evidence": dict(payload["download_evidence"]),
        "commit": {
            "run_id": str(original_commit["run_id"]),
            "committed_at": migrated_at.isoformat(),
            "object_write_precedes_checkpoint": True,
        },
    }
    storage.write_json(checkpoint_uri, migrated_payload)
    _log_event(
        "reference_media_checkpoint_migrated",
        checkpoint_uri=checkpoint_uri,
        reference_media_id=validated["reference_media_id"],
        from_schema=_LEGACY_REFERENCE_MEDIA_CHECKPOINT_VERSION,
        to_schema=REFERENCE_MEDIA_CHECKPOINT_VERSION,
    )
    return validated


def _expected_checkpoint_evidence(
    candidate: Mapping[str, object],
    *,
    requested_url: str,
) -> dict[str, object]:
    return {
        "requested_url": requested_url,
        "attribution": candidate["attribution"],
        "creator": candidate["creator"],
        "rights_holder": candidate["rights_holder"],
        "licence": candidate["licence"],
        "licence_uri": candidate["licence_uri"],
        "source_snapshot_version": candidate["source_snapshot_version"],
    }


def _committed_object_fingerprint(
    binding: Mapping[str, object],
    row: Mapping[str, object],
    *,
    final_url: object,
) -> str:
    return _fingerprint(
        {
            "reference_media_id": binding["reference_media_id"],
            "candidate_fingerprint": binding["candidate_fingerprint"],
            "configuration_fingerprint": binding["configuration_fingerprint"],
            "licence_policy_fingerprint": binding["licence_policy_fingerprint"],
            "provider_policy_fingerprint": binding["provider_policy_fingerprint"],
            "source_object_uri": row["source_object_uri"],
            "content_type": row["content_type"],
            "source_byte_count": row["source_byte_count"],
            "decoded_width": row["decoded_width"],
            "decoded_height": row["decoded_height"],
            "sha256": row["sha256"],
            "perceptual_hash": row["perceptual_hash"],
            "licence_policy_status": row["licence_policy_status"],
            "final_url": final_url,
        }
    )


def _legacy_committed_object_fingerprint(
    binding: Mapping[str, object],
    row: Mapping[str, object],
    *,
    final_url: object,
) -> str:
    return _fingerprint(
        {
            "reference_media_id": binding["reference_media_id"],
            "candidate_fingerprint": binding["candidate_fingerprint"],
            "configuration_fingerprint": binding["configuration_fingerprint"],
            "licence_policy_fingerprint": binding["licence_policy_fingerprint"],
            "provider_policy_fingerprint": binding["provider_policy_fingerprint"],
            "source_object_uri": row["source_object_uri"],
            "content_type": row["content_type"],
            "source_byte_count": row["source_byte_count"],
            "decoded_width": row["decoded_width"],
            "decoded_height": row["decoded_height"],
            "sha256": row["sha256"],
            "licence_policy_status": row["licence_policy_status"],
            "final_url": final_url,
        }
    )


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _checksum_preflight_reason(candidate: Mapping[str, object]) -> str | None:
    algorithm = candidate.get("source_checksum_algorithm")
    if algorithm is None:
        return None
    normalized = _normalise_checksum_algorithm(algorithm)
    if normalized not in _CHECKSUM_ALGORITHMS:
        return f"unsupported_source_checksum_algorithm:{normalized}"
    checksum = str(candidate.get("source_checksum") or "").strip().casefold()
    checksum = checksum.removeprefix(f"{normalized}:")
    expected_length = {"md5": 32, "sha1": 40, "sha256": 64}[normalized]
    if len(checksum) != expected_length or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        return "invalid_source_checksum"
    return None


def _source_checksum_hasher(
    candidate: Mapping[str, object],
) -> Any | None:
    algorithm = candidate.get("source_checksum_algorithm")
    if algorithm is None:
        return None
    normalized = _normalise_checksum_algorithm(algorithm)
    if normalized == "md5":
        return hashlib.md5(usedforsecurity=False)
    return hashlib.new(normalized)


def _source_checksum_matches(
    candidate: Mapping[str, object],
    actual: str,
) -> bool:
    algorithm = _normalise_checksum_algorithm(candidate["source_checksum_algorithm"])
    expected = str(candidate["source_checksum"]).strip().casefold()
    return expected.removeprefix(f"{algorithm}:") == actual.casefold()


def _normalise_checksum_algorithm(value: object) -> str:
    return str(value or "").strip().casefold().replace("-", "")


def _sniff_content_type(head: bytes) -> str | None:
    stripped = head.lstrip().lower()
    if stripped.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        return "text/html"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


def _canonical_content_type(value: object) -> str:
    content_type = str(value or "").split(";", 1)[0].strip().casefold()
    if not content_type or "/" not in content_type:
        raise ValueError("content type must be a MIME type")
    return _CONTENT_TYPE_ALIASES.get(content_type, content_type)


def _retry_after_seconds(value: str | None, *, now: datetime) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(text)
        except TypeError, ValueError, OverflowError:
            return None
        if retry_at.tzinfo is None or retry_at.utcoffset() is None:
            return None
        retry_at = _aware_utc(retry_at)
        seconds = (retry_at - now).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _backoff_seconds(
    pending: _PendingDownload,
    retry_number: int,
    config: ReferenceMediaDownloadConfig,
) -> float:
    base = min(
        config.backoff_cap_seconds,
        config.backoff_base_seconds * (2 ** max(0, retry_number - 1)),
    )
    seed = f"{pending.selected.candidate['reference_media_id']}:{retry_number}".encode()
    fraction = int.from_bytes(hashlib.sha256(seed).digest()[:2], "big") / 65535
    return min(config.backoff_cap_seconds, base * (0.9 + 0.2 * fraction))


def _download_report_uris(output_prefix: str, run_id: str) -> tuple[str, str]:
    readable_run_id = safe_path_component(run_id)[:80].rstrip("_-") or "run"
    report_run_id = (
        f"{readable_run_id}-{hashlib.sha256(run_id.encode('utf-8')).hexdigest()[:12]}"
    )
    return (
        build_report_uri(
            output_prefix,
            run_id=report_run_id,
            report_name=REFERENCE_MEDIA_DOWNLOAD_REPORT_FILE.removesuffix(".json"),
        ),
        build_report_uri(
            output_prefix,
            run_id=report_run_id,
            report_name=REFERENCE_MEDIA_DOWNLOAD_SUMMARY_FILE.removesuffix(".md"),
            suffix="md",
        ),
    )


def _download_settings(
    config: ReferenceMediaDownloadConfig,
    licence_policy: ReferenceLicencePolicy,
) -> dict[str, object]:
    return {
        "workers": config.workers,
        "max_inflight": config.max_inflight,
        "max_concurrent_decodes": config.max_concurrent_decodes,
        "max_attempts": config.max_attempts,
        "max_redirects": config.max_redirects,
        "max_source_bytes": config.max_source_bytes,
        "max_decoded_pixels": config.max_decoded_pixels,
        "max_decode_memory_bytes": config.max_decode_memory_bytes,
        "timeout_seconds": config.timeout_seconds,
        "max_download_seconds": config.max_download_seconds,
        "backoff_base_seconds": config.backoff_base_seconds,
        "backoff_cap_seconds": config.backoff_cap_seconds,
        "max_retry_after_seconds": config.max_retry_after_seconds,
        "retry_statuses": list(config.retry_statuses),
        "allowed_content_types": list(config.allowed_content_types),
        "user_agent": config.user_agent,
        "licence_policy": {
            "version": licence_policy.version,
            "broadly_reusable": list(licence_policy.broadly_reusable),
            "research_only": list(licence_policy.research_only),
            "attribution_required": list(licence_policy.attribution_required),
            "licence_aliases": [
                {"source": source, "canonical": canonical}
                for source, canonical in licence_policy.licence_aliases
            ],
            "fingerprint": licence_policy.fingerprint,
        },
        "provider_policies": [
            {
                "source": value.source,
                "policy_version": value.policy_version,
                "allowed_hosts": list(value.allowed_hosts),
                "allowed_schemes": list(value.allowed_schemes),
                "max_concurrent_per_origin": value.max_concurrent_per_origin,
                "min_request_interval_seconds": (value.min_request_interval_seconds),
                "resolve_public_addresses": value.resolve_public_addresses,
                "url_strategy": value.url_strategy,
                "inaturalist_image_style": value.inaturalist_image_style,
                "fingerprint": value.fingerprint,
                "validation_fingerprint": value.validation_fingerprint,
            }
            for value in sorted(
                config.provider_policies,
                key=lambda policy: (
                    policy.source.casefold(),
                    policy.allowed_hosts,
                    policy.validation_fingerprint,
                ),
            )
        ],
        "perceptual_hash_version": REFERENCE_PERCEPTUAL_HASH_VERSION,
    }


def _persist_failed_download_report(
    *,
    storage: CloudStorage,
    output_prefix: str,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
    selections: pl.DataFrame,
    media_candidates: pl.DataFrame,
    config: ReferenceMediaDownloadConfig,
    licence_policy: ReferenceLicencePolicy,
    error: Exception,
) -> None:
    prefix = str(output_prefix or "").rstrip("/")
    persisted = False
    failure_report_uri: str | None = None
    try:
        if not prefix:
            raise ValueError("output_prefix must be nonblank")
        failure_report_uri, failure_summary_uri = _download_report_uris(prefix, run_id)
        failure_provenance: dict[str, object] = {
            "selection_rows": selections.height,
            "selection_fingerprint": "not_instrumented",
            "media_candidate_rows": media_candidates.height,
            "selected_candidate_count": None,
            "selected_candidates_fingerprint": "not_instrumented",
            "selected_candidate_sources_fingerprint": "not_instrumented",
            "acquisition_plan_ids": "not_instrumented",
            "sources": "not_instrumented",
            "source_snapshot_versions": "not_instrumented",
        }
        failure_inputs_validated = False
        try:
            failure_selected = _selected_media(selections, media_candidates)
            failure_provenance = _reference_input_provenance(
                selections,
                media_candidates,
                failure_selected,
            )
            failure_inputs_validated = True
        except Exception:  # noqa: BLE001 - the primary failure may be validation.
            failure_selected = []
        report: dict[str, Any] = {
            "schema_version": REFERENCE_MEDIA_DOWNLOAD_REPORT_VERSION,
            "command": "references.download_media",
            "run_id": run_id,
            "pid": os.getpid(),
            "git_sha": config.git_sha or _git_sha(),
            "status": "failed",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "elapsed_seconds": max(
                0.0,
                (ended_at - started_at).total_seconds(),
            ),
            "inputs": {
                "selected_rows": (
                    len(failure_selected) if failure_inputs_validated else None
                ),
                **failure_provenance,
                "licence_policy_version": licence_policy.version,
                "licence_policy_fingerprint": licence_policy.fingerprint,
                "configuration_fingerprint": config.semantic_fingerprint,
            },
            "settings": _download_settings(config, licence_policy),
            "counts": {
                "selected": None,
                "rows_out": None,
                "inventory_rows": None,
                "committed": None,
                "resumed": None,
                "quarantined": None,
                "allowed": None,
                "research_only": None,
                "licence_quarantined": None,
                "licence_denied": None,
                "http_requests": "not_instrumented",
                "retries": "not_instrumented",
                "rate_limit_events": "not_instrumented",
                "errors": 1,
            },
            "decode_status_counts": {},
            "licence_status_counts": {},
            "source_counts": {},
            "quarantine_reason_counts": {},
            "source_error_counts": {},
            "bytes": {
                "source_objects": "not_instrumented",
                "artifact": "not_instrumented",
                "checkpoints": "not_instrumented",
            },
            "performance": {
                "objects_per_second": None,
                "retry_wait_seconds": "not_instrumented",
                "records_per_call": "not_instrumented",
                "request_seconds_avg": "not_instrumented",
                "request_seconds_p50": "not_instrumented",
                "request_seconds_p95": "not_instrumented",
                "rss_bytes": "not_instrumented",
                "peak_memory_bytes": "not_instrumented",
                "gpu_memory_bytes": "not_instrumented",
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:1_000],
            },
            "artifacts": {
                "media_objects": join_uri(prefix, REFERENCE_MEDIA_OBJECTS_FILE),
                "report": failure_report_uri,
                "summary": failure_summary_uri,
            },
        }
        summary = "\n".join(
            [
                "# Reference media download",
                "",
                f"- Run: `{run_id}`",
                "- Status: `failed`",
                f"- Error: `{type(error).__name__}`",
                "",
            ]
        )
        storage.write_text(failure_summary_uri, summary)
        storage.write_json(failure_report_uri, report)
        persisted = True
    except Exception as report_error:  # noqa: BLE001 - preserve the primary failure.
        _log_event(
            "reference_media_download_failure_report_error",
            run_id=run_id,
            error_type=type(report_error).__name__,
            error_message=str(report_error)[:1_000],
        )
    _log_event(
        "reference_media_download_failed",
        run_id=run_id,
        status="failed",
        error_type=type(error).__name__,
        error_message=str(error)[:1_000],
        failure_report_uri=failure_report_uri,
        failure_report_persisted=persisted,
    )


def _download_report(
    frame: pl.DataFrame,
    *,
    inventory_frame: pl.DataFrame,
    input_provenance: Mapping[str, object],
    selected: list[_SelectedMedia],
    prepared: list[_PreparedDownload],
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
    resumed_count: int,
    config: ReferenceMediaDownloadConfig,
    licence_policy: ReferenceLicencePolicy,
    media_objects_uri: str,
    report_uri: str,
    summary_uri: str,
) -> dict[str, Any]:
    rows = frame.to_dicts()
    decode_counts = Counter(str(row["decode_status"]) for row in rows)
    licence_counts = Counter(str(row["licence_policy_status"]) for row in rows)
    source_by_id = {
        str(item.candidate["reference_media_id"]): str(item.candidate["source"])
        for item in selected
    }
    source_counts = Counter(
        source_by_id[str(row["reference_media_id"])] for row in rows
    )
    failure_count = sum(row["decode_status"] != "valid" for row in rows)
    http_request_count = sum(value.attempt_count for value in prepared)
    quarantine_reason_counts = Counter(
        str(row["quarantine_reason"]) for row in rows if row["decode_status"] != "valid"
    )
    source_error_counts = Counter(
        source_by_id[str(row["reference_media_id"])]
        for row in rows
        if row["decode_status"] != "valid"
    )
    elapsed = max(0.0, (ended_at - started_at).total_seconds())
    committed_bytes = sum(
        int(row["source_byte_count"] or 0)
        for row in {
            str(value["source_object_uri"]): value
            for value in rows
            if value["decode_status"] == "valid"
        }.values()
    )
    return {
        "schema_version": REFERENCE_MEDIA_DOWNLOAD_REPORT_VERSION,
        "command": "references.download_media",
        "run_id": run_id,
        "pid": os.getpid(),
        "git_sha": config.git_sha or _git_sha(),
        "status": "complete" if failure_count == 0 else "complete_with_errors",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": elapsed,
        "inputs": {
            "selected_rows": len(selected),
            **input_provenance,
            "licence_policy_version": licence_policy.version,
            "licence_policy_fingerprint": licence_policy.fingerprint,
            "configuration_fingerprint": config.semantic_fingerprint,
        },
        "settings": _download_settings(config, licence_policy),
        "counts": {
            "selected": len(selected),
            "rows_out": frame.height,
            "inventory_rows": inventory_frame.height,
            "committed": decode_counts["valid"],
            "resumed": resumed_count,
            "quarantined": failure_count,
            "allowed": licence_counts["allowed"],
            "research_only": licence_counts["research_only"],
            "licence_quarantined": licence_counts["quarantined"],
            "licence_denied": licence_counts["denied"],
            "http_requests": http_request_count,
            "retries": sum(value.retry_count for value in prepared),
            "rate_limit_events": sum(value.rate_limit_count for value in prepared),
            "errors": failure_count,
        },
        "decode_status_counts": dict(sorted(decode_counts.items())),
        "licence_status_counts": dict(sorted(licence_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "quarantine_reason_counts": dict(sorted(quarantine_reason_counts.items())),
        "source_error_counts": dict(sorted(source_error_counts.items())),
        "bytes": {
            "source_objects": committed_bytes,
            "artifact": "not_instrumented",
            "checkpoints": "not_instrumented",
        },
        "performance": {
            "objects_per_second": (
                decode_counts["valid"] / elapsed if elapsed > 0 else None
            ),
            "retry_wait_seconds": sum(value.retry_wait_seconds for value in prepared),
            "records_per_call": (
                len(prepared) / http_request_count if http_request_count > 0 else None
            ),
            "request_seconds_avg": "not_instrumented",
            "request_seconds_p50": "not_instrumented",
            "request_seconds_p95": "not_instrumented",
            "rss_bytes": "not_instrumented",
            "peak_memory_bytes": "not_instrumented",
            "gpu_memory_bytes": "not_instrumented",
        },
        "artifacts": {
            "media_objects": media_objects_uri,
            "report": report_uri,
            "summary": summary_uri,
        },
        "error": None,
    }


def _download_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    return "\n".join(
        [
            "# Reference media download",
            "",
            f"- Run: `{report['run_id']}`",
            f"- Status: `{report['status']}`",
            f"- Selected: {counts['selected']}",
            f"- Committed: {counts['committed']}",
            f"- Resumed: {counts['resumed']}",
            f"- Quarantined or failed: {counts['quarantined']}",
            f"- HTTP requests: {counts['http_requests']}",
            f"- Retries: {counts['retries']}",
            f"- Media objects: `{report['artifacts']['media_objects']}`",
            "",
        ]
    )


def _candidate_source_fingerprint(row: Mapping[str, object]) -> str:
    lifecycle_fields = {
        "download_status",
        "verification_status",
        "exclusion_reason",
        "licence_policy_status",
    }
    return _fingerprint(
        {
            str(key): _jsonable(value)
            for key, value in row.items()
            if key not in lifecycle_fields
        }
    )


def _fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _jsonable(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    schema = json.dumps(
        {name: str(data_type) for name, data_type in frame.schema.items()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest.update(len(schema).to_bytes(8, "big"))
    digest.update(schema)
    for row in frame.iter_rows(named=True):
        encoded = json.dumps(
            _jsonable(row),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def _mapping_rows_fingerprint(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(
            _jsonable(row),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def _jsonable(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _normalise_host(value: object) -> str:
    host = str(value or "").strip().casefold().rstrip(".")
    if not host:
        raise ValueError("provider host must be nonblank")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("provider host is invalid") from exc


def _resolve_host(host: str) -> Sequence[str]:
    return tuple(
        sorted(
            {
                str(address[4][0])
                for address in socket.getaddrinfo(
                    host,
                    None,
                    type=socket.SOCK_STREAM,
                )
            }
        )
    )


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return parsed


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except OSError, subprocess.SubprocessError:
        return None


def _log_event(event: str, **payload: object) -> None:
    _LOGGER.info(
        "%s",
        json.dumps(
            {"event": event, **payload},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


__all__ = [
    "REFERENCE_MEDIA_CHECKPOINT_VERSION",
    "REFERENCE_MEDIA_DOWNLOADER_VERSION",
    "REFERENCE_MEDIA_DOWNLOAD_REPORT_FILE",
    "REFERENCE_MEDIA_DOWNLOAD_REPORT_VERSION",
    "REFERENCE_MEDIA_DOWNLOAD_SUMMARY_FILE",
    "ProviderMediaDownloadPolicy",
    "ReferenceMediaDownloadConfig",
    "ReferenceMediaDownloadResult",
    "download_reference_media",
]
