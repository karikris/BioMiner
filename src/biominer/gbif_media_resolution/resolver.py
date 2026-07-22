from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
import hashlib
import ipaddress
import json
import random
import socket
import time
from urllib.parse import parse_qsl, urljoin, urlsplit

import httpx
from PIL import ImageFile

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_media_resolution.adapters import (
    DEFAULT_PROVIDER_ADAPTERS,
    ProviderURLResolver,
)
from biominer.gbif_media_resolution.models import (
    ResolutionAttempt,
    ResolutionInput,
    ResolutionResult,
    ResolutionStatus,
)
from biominer.references.downloader import (
    create_pinned_address_http_transport,
    resolve_host_addresses,
)


RESOLVER_VERSION = "biominer-gbif-media-url-resolver/v1"
ALLOWED_IMAGE_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/gif"}
)
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
NON_IMAGE_HOSTS = frozenset({"youtube.com", "www.youtube.com", "youtu.be"})
SIGNED_QUERY_KEYS = frozenset(
    {
        "expires",
        "signature",
        "sig",
        "token",
        "x-amz-algorithm",
        "x-amz-credential",
        "x-amz-date",
        "x-amz-expires",
        "x-amz-signature",
        "x-goog-signature",
    }
)


@dataclass(frozen=True, slots=True)
class ResolverConfig:
    max_attempts: int = 5
    max_redirects: int = 5
    timeout_seconds: float = 30.0
    max_html_bytes: int = 2 * 1024 * 1024
    max_probe_bytes: int = 256 * 1024
    backoff_base_seconds: float = 0.5
    backoff_cap_seconds: float = 60.0
    user_agent: str = "BioMiner/0.1 GBIF-media-URL-resolver"
    minimum_origin_interval_seconds: float = 0.0
    circuit_failure_threshold: int = 10
    circuit_cooldown_seconds: float = 15 * 60.0
    circuit_max_cycles: int = 3

    def __post_init__(self) -> None:
        for field in (
            "max_attempts",
            "max_redirects",
            "max_html_bytes",
            "max_probe_bytes",
            "circuit_failure_threshold",
            "circuit_max_cycles",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in (
            "timeout_seconds",
            "backoff_base_seconds",
            "backoff_cap_seconds",
            "minimum_origin_interval_seconds",
            "circuit_cooldown_seconds",
        ):
            value = float(getattr(self, field))
            if value < 0 or (field == "timeout_seconds" and value == 0):
                raise ValueError(f"{field} is invalid")
        if self.backoff_cap_seconds < self.backoff_base_seconds:
            raise ValueError("backoff cap must be at least the base")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "version": RESOLVER_VERSION,
                **{
                    field: getattr(self, field)
                    for field in self.__dataclass_fields__
                },
            }
        )


@dataclass(frozen=True, slots=True)
class ValidatedURL:
    url: str
    scheme: str
    host: str
    port: int
    addresses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Fetched:
    requested_url: str
    final_url: str
    content: bytes
    declared_content_type: str | None
    status_code: int
    redirect_count: int
    etag: str | None
    last_modified: str | None


class _ResolutionFailure(Exception):
    def __init__(self, status: ResolutionStatus, reason: str) -> None:
        self.status = status
        self.reason = reason
        super().__init__(reason)


class _StructuredMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tiers: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
        self.json_ld: list[str] = []
        self._json_script = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {str(key).casefold(): str(value or "") for key, value in attrs}
        lowered = tag.casefold()
        if lowered == "meta":
            key = (values.get("property") or values.get("name") or "").casefold()
            content = values.get("content", "").strip()
            if content and key == "og:image":
                self.tiers[1].append(content)
            elif content and key == "twitter:image":
                self.tiers[2].append(content)
        elif lowered == "link":
            rel = {part.casefold() for part in values.get("rel", "").split()}
            href = values.get("href", "").strip()
            if href and "image_src" in rel:
                self.tiers[3].append(href)
        elif lowered == "script" and values.get("type", "").casefold() == "application/ld+json":
            self._json_script = True
            self._script_parts = []

    def handle_data(self, data: str) -> None:
        if self._json_script:
            self._script_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._json_script:
            self._json_script = False
            self.json_ld.append("".join(self._script_parts))
            self._script_parts = []


def extract_structured_image_candidates(content: bytes, *, base_url: str) -> tuple[str, ...]:
    parser = _StructuredMetadataParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    for raw in parser.json_ld:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        _collect_json_ld_images(value, parser.tiers[0])
    for tier in range(4):
        candidates = tuple(
            dict.fromkeys(
                urljoin(base_url, item.strip())
                for item in parser.tiers[tier]
                if item.strip()
            )
        )
        if candidates:
            return candidates
    return ()


def _collect_json_ld_images(value: object, output: list[str]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_json_ld_images(item, output)
        return
    if not isinstance(value, dict):
        return
    type_value = value.get("@type")
    types = {str(item).casefold() for item in type_value} if isinstance(type_value, list) else {str(type_value).casefold()}
    if "imageobject" in types:
        content_url = value.get("contentUrl")
        if isinstance(content_url, str) and content_url.strip():
            output.append(content_url)
    for nested in value.values():
        if isinstance(nested, (dict, list)):
            _collect_json_ld_images(nested, output)


def validate_public_http_url(
    url: str,
    *,
    resolve_host: Callable[[str], Sequence[str]] = resolve_host_addresses,
) -> ValidatedURL:
    parsed = urlsplit(str(url).strip())
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise ValueError("URL scheme must be HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not contain user information")
    host = (parsed.hostname or "").rstrip(".").casefold()
    if not host:
        raise ValueError("URL host is missing")
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("URL port is invalid") from exc
    if port not in {80, 443}:
        raise ValueError("URL uses a nonstandard port")
    if parsed.fragment:
        raise ValueError("URL fragments are not accepted")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            raw = resolve_host(host)
        except (OSError, socket.gaierror) as exc:
            raise ValueError("host resolution failed") from exc
        try:
            addresses = tuple(sorted({str(ipaddress.ip_address(item)) for item in raw}))
        except ValueError as exc:
            raise ValueError("host resolution returned an invalid address") from exc
    else:
        addresses = (str(literal),)
    if not addresses:
        raise ValueError("host did not resolve")
    if any(not ipaddress.ip_address(item).is_global for item in addresses):
        raise ValueError("host resolves to a non-public address")
    return ValidatedURL(str(url).strip(), scheme, host, port, addresses)


def sniff_image_content_type(content: bytes) -> str | None:
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if content.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def _canonical_content_type(value: str | None) -> str | None:
    if value is None:
        return None
    result = value.split(";", 1)[0].strip().casefold()
    return {
        "image/jpg": "image/jpeg",
        "image/pjpeg": "image/jpeg",
        "image/x-png": "image/png",
        "image/x-tiff": "image/tiff",
    }.get(result, result or None)


def _is_ephemeral_url(url: str) -> bool:
    keys = {key.casefold() for key, _ in parse_qsl(urlsplit(url).query, keep_blank_values=True)}
    return bool(keys & SIGNED_QUERY_KEYS)


class MediaURLResolver:
    def __init__(
        self,
        *,
        config: ResolverConfig | None = None,
        http_client: httpx.Client | None = None,
        resolve_host: Callable[[str], Sequence[str]] | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        provider_adapters: tuple[ProviderURLResolver, ...] = DEFAULT_PROVIDER_ADAPTERS,
        request_guard: Callable[[str], AbstractContextManager[None]] | None = None,
    ) -> None:
        self.config = config or ResolverConfig()
        self._resolve_host = resolve_host or resolve_host_addresses
        self._now = now or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._monotonic = monotonic
        self._provider_adapters = provider_adapters
        self._request_guard = request_guard or (lambda _host: nullcontext())
        self._attempts: list[ResolutionAttempt] = []
        self._source_row_id = ""
        self._origin_last_request: dict[str, float] = {}
        self._origin_failures: dict[str, int] = {}
        self._origin_cycles: dict[str, int] = {}
        self._origin_blocked_until: dict[str, float] = {}
        self._owns_client = http_client is None
        if http_client is not None:
            self._client = http_client
            self._prevalidate_dns = True
        else:
            self._client = httpx.Client(
                transport=create_pinned_address_http_transport(
                    max_connections=1,
                    resolve_host=self._resolve_host,
                    monotonic=monotonic,
                ),
                headers={"User-Agent": self.config.user_agent},
                timeout=httpx.Timeout(self.config.timeout_seconds),
                follow_redirects=False,
                trust_env=False,
            )
        self._prevalidate_dns = False

    @property
    def semantic_fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "resolver_version": RESOLVER_VERSION,
                "config_fingerprint": self.config.fingerprint,
                "provider_adapters": [
                    {"adapter_id": adapter.adapter_id, "version": adapter.version}
                    for adapter in self._provider_adapters
                ],
            }
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> MediaURLResolver:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def resolve(self, item: ResolutionInput) -> tuple[ResolutionResult, tuple[ResolutionAttempt, ...]]:
        self._attempts = []
        self._source_row_id = item.source_row_id
        if item.rights_blocked:
            return self._terminal(item, ResolutionStatus.RIGHTS_BLOCKED, "rights_policy", "explicit_item_rights_restriction"), ()
        if item.host in NON_IMAGE_HOSTS or (item.media_type or "").strip().casefold() in {"movingimage", "sound"}:
            return self._terminal(item, ResolutionStatus.NON_IMAGE_MEDIA, "preflight", "reference_is_not_still_image"), ()

        primary_failure: _ResolutionFailure | None = None
        try:
            reference = self._fetch(item.media_references, phase="reference", method="reference_probe", max_bytes=self.config.max_html_bytes)
            detected = sniff_image_content_type(reference.content)
            if detected is not None:
                if _is_ephemeral_url(item.media_references):
                    raise _ResolutionFailure(
                        ResolutionStatus.UNRESOLVED_INVALID_IMAGE,
                        "direct_reference_is_ephemeral",
                    )
                detected = self._require_valid_image(reference)
                result = self._resolved(item, reference, stable_candidate_url=item.media_references, method="reference_direct", detected=detected)
                return result, tuple(self._attempts)
            if reference.declared_content_type and reference.declared_content_type != "text/html":
                raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_INVALID_IMAGE, f"reference_content_type:{reference.declared_content_type}")
            candidates = extract_structured_image_candidates(reference.content, base_url=reference.final_url)
            if len(candidates) > 1:
                primary_failure = _ResolutionFailure(ResolutionStatus.UNRESOLVED_AMBIGUOUS_CANDIDATES, "multiple_structured_image_candidates")
            elif len(candidates) == 1:
                candidate = candidates[0]
                if _is_ephemeral_url(candidate):
                    primary_failure = _ResolutionFailure(ResolutionStatus.UNRESOLVED_INVALID_IMAGE, "ephemeral_candidate_url")
                else:
                    try:
                        probed = self._fetch(candidate, phase="candidate", method="structured_metadata", max_bytes=self.config.max_probe_bytes, image_probe=True)
                        detected = self._require_valid_image(probed)
                        return self._resolved(item, probed, stable_candidate_url=candidate, method="structured_metadata", detected=detected), tuple(self._attempts)
                    except _ResolutionFailure as exc:
                        primary_failure = exc
            else:
                primary_failure = _ResolutionFailure(ResolutionStatus.UNRESOLVED_NOT_FOUND, "structured_image_metadata_absent")
        except _ResolutionFailure as exc:
            primary_failure = exc

        adapter_failure: _ResolutionFailure | None = None
        try:
            adapter_candidate = self._provider_candidate(item)
            if adapter_candidate is not None:
                candidate, adapter = adapter_candidate
                if _is_ephemeral_url(candidate):
                    raise _ResolutionFailure(
                        ResolutionStatus.UNRESOLVED_INVALID_IMAGE,
                        "provider_adapter_candidate_is_ephemeral",
                    )
                probed = self._fetch(
                    candidate,
                    phase="candidate",
                    method=adapter.adapter_id,
                    max_bytes=self.config.max_probe_bytes,
                    image_probe=True,
                )
                detected = self._require_valid_image(probed)
                return self._resolved(
                    item,
                    probed,
                    stable_candidate_url=candidate,
                    method=adapter.adapter_id,
                    detected=detected,
                    adapter_version=adapter.version,
                ), tuple(self._attempts)
        except _ResolutionFailure as exc:
            adapter_failure = exc

        try:
            candidate = self._gbif_candidate(item)
            if candidate is not None:
                if _is_ephemeral_url(candidate):
                    raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_INVALID_IMAGE, "gbif_identifier_is_ephemeral")
                probed = self._fetch(candidate, phase="candidate", method="gbif_occurrence_api", max_bytes=self.config.max_probe_bytes, image_probe=True)
                detected = self._require_valid_image(probed)
                return self._resolved(item, probed, stable_candidate_url=candidate, method="gbif_occurrence_api", detected=detected), tuple(self._attempts)
        except _ResolutionFailure as exc:
            if primary_failure is None or exc.status is ResolutionStatus.RETRY_EXHAUSTED:
                primary_failure = exc

        if primary_failure is None and adapter_failure is not None:
            primary_failure = adapter_failure

        failure = primary_failure or _ResolutionFailure(ResolutionStatus.UNRESOLVED_NOT_FOUND, "no_direct_image_candidate")
        return self._terminal(item, failure.status, "resolution_exhausted", failure.reason), tuple(self._attempts)

    def _provider_candidate(
        self,
        item: ResolutionInput,
    ) -> tuple[str, ProviderURLResolver] | None:
        for adapter in self._provider_adapters:
            if not adapter.supports(item):
                continue
            discovery = adapter.discovery(item)
            candidates = discovery.direct_candidates
            if discovery.request_url is not None:
                response = self._fetch(
                    discovery.request_url,
                    phase="provider_api",
                    method=adapter.adapter_id,
                    max_bytes=self.config.max_html_bytes,
                    accept=discovery.accept,
                )
                candidates = adapter.parse(item, response.content)
            candidates = tuple(dict.fromkeys(value.strip() for value in candidates if value.strip()))
            if len(candidates) > 1:
                raise _ResolutionFailure(
                    ResolutionStatus.UNRESOLVED_AMBIGUOUS_CANDIDATES,
                    f"{adapter.adapter_id}:multiple_candidates",
                )
            if candidates:
                return candidates[0], adapter
        return None

    def _gbif_candidate(self, item: ResolutionInput) -> str | None:
        fetched = self._fetch(
            f"https://api.gbif.org/v1/occurrence/{item.gbif_id}",
            phase="gbif_api",
            method="gbif_occurrence_api",
            max_bytes=self.config.max_html_bytes,
            accept="application/json",
        )
        try:
            payload = json.loads(fetched.content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, "gbif_response_invalid_json") from exc
        if not isinstance(payload, dict):
            raise _ResolutionFailure(
                ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE,
                "gbif_response_invalid_shape",
            )
        if str(payload.get("key", "")).strip() != item.gbif_id:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, "gbif_occurrence_identity_mismatch")
        matches: list[str] = []
        for media in payload.get("media") or []:
            if not isinstance(media, dict):
                continue
            if str(media.get("references") or "").strip() != item.media_references.strip():
                continue
            identifier = str(media.get("identifier") or "").strip()
            if identifier:
                matches.append(identifier)
        distinct = tuple(dict.fromkeys(matches))
        if len(distinct) > 1:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_AMBIGUOUS_CANDIDATES, "gbif_multiple_identifiers_for_reference")
        return distinct[0] if distinct else None

    def _fetch(
        self,
        url: str,
        *,
        phase: str,
        method: str,
        max_bytes: int,
        image_probe: bool = False,
        accept: str | None = None,
    ) -> _Fetched:
        requested_url = url
        current_url = url
        redirects = 0
        retry_number = 0
        while retry_number < self.config.max_attempts:
            validated = self._validate(current_url)
            self._wait_for_origin(validated.host)
            started = self._timestamp()
            headers = {"Accept": accept or (", ".join(sorted(ALLOWED_IMAGE_TYPES)) if image_probe else "text/html, image/*;q=0.9")}
            if image_probe:
                headers["Range"] = f"bytes=0-{max_bytes - 1}"
            try:
                with self._request_guard(validated.host), self._client.stream(
                        "GET",
                        current_url,
                        headers=headers,
                        timeout=self.config.timeout_seconds,
                        follow_redirects=False,
                    ) as response:
                    status = response.status_code
                    declared = _canonical_content_type(response.headers.get("Content-Type"))
                    if status in REDIRECT_STATUSES:
                        location = response.headers.get("Location")
                        if not location:
                            self._record(started, phase, method, current_url, str(response.url), None, status, "rejected", "redirect_missing_location", declared, 0, response, retry_number)
                            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, "redirect_missing_location")
                        target = urljoin(current_url, location)
                        self._record(started, phase, method, current_url, target, current_url, status, "redirect", None, declared, 0, response, retry_number)
                        redirects += 1
                        if redirects > self.config.max_redirects:
                            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, "redirect_limit_exceeded")
                        self._validate(target)
                        current_url = target
                        continue
                    if status in RETRY_STATUSES:
                        self._record(started, phase, method, current_url, str(response.url), None, status, "retry", f"http_status_{status}", declared, 0, response, retry_number)
                        self._record_transient(validated.host)
                        retry_number += 1
                        if retry_number >= self.config.max_attempts:
                            raise _ResolutionFailure(ResolutionStatus.RETRY_EXHAUSTED, f"retry_exhausted_http_{status}")
                        self._sleep(self._retry_delay(response, retry_number, current_url))
                        continue
                    effective_max_bytes = (
                        min(max_bytes, self.config.max_probe_bytes)
                        if declared in ALLOWED_IMAGE_TYPES
                        else max_bytes
                    )
                    content = _read_bounded(
                        response,
                        max_bytes=effective_max_bytes,
                    )
                    if status not in {200, 206}:
                        reason = f"http_status_{status}"
                        self._record(started, phase, method, current_url, str(response.url), None, status, "rejected", reason, declared, len(content), response, retry_number, _prefix_sha256(content))
                        if status in {401, 403, 451}:
                            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_ACCESS_DENIED, reason)
                        if status in {404, 410}:
                            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_NOT_FOUND, reason)
                        raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, reason)
                    self._record_success(validated.host)
                    self._record(started, phase, method, current_url, str(response.url), None, status, "received", None, declared, len(content), response, retry_number, _prefix_sha256(content))
                    return _Fetched(requested_url, str(response.url), content, declared, status, redirects, response.headers.get("ETag"), response.headers.get("Last-Modified"))
            except _ResolutionFailure:
                raise
            except httpx.TransportError as exc:
                self._record_transport(started, phase, method, current_url, retry_number, exc)
                self._record_transient(validated.host)
                retry_number += 1
                if retry_number >= self.config.max_attempts:
                    raise _ResolutionFailure(ResolutionStatus.RETRY_EXHAUSTED, f"transport_retry_exhausted:{type(exc).__name__}") from exc
                self._sleep(self._backoff(retry_number, current_url))
        raise _ResolutionFailure(ResolutionStatus.RETRY_EXHAUSTED, "attempt_budget_exhausted")

    def _validate(self, url: str) -> ValidatedURL:
        if self._prevalidate_dns:
            try:
                return validate_public_http_url(url, resolve_host=self._resolve_host)
            except ValueError as exc:
                raise _ResolutionFailure(
                    ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE,
                    "unsafe_or_invalid_url",
                ) from exc
        parsed = urlsplit(url)
        scheme = parsed.scheme.casefold()
        if scheme not in {"http", "https"} or parsed.username is not None or parsed.password is not None or not parsed.hostname:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, "unsafe_or_invalid_url")
        try:
            port = parsed.port or (443 if scheme == "https" else 80)
        except ValueError as exc:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, "invalid_url_port") from exc
        if port not in {80, 443} or parsed.fragment:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, "unsafe_or_invalid_url")
        return ValidatedURL(url, scheme, parsed.hostname.casefold(), port, ())

    def _require_valid_image(self, fetched: _Fetched) -> str:
        detected = sniff_image_content_type(fetched.content)
        if fetched.declared_content_type not in ALLOWED_IMAGE_TYPES:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_INVALID_IMAGE, f"image_content_type_not_allowed:{fetched.declared_content_type}")
        if detected is None:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_INVALID_IMAGE, "unrecognized_image_signature")
        if fetched.declared_content_type != detected:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_INVALID_IMAGE, f"content_type_signature_mismatch:{fetched.declared_content_type}:{detected}")
        decoded = _decode_probe_content_type(fetched.content)
        if decoded is None:
            raise _ResolutionFailure(
                ResolutionStatus.UNRESOLVED_INVALID_IMAGE,
                "image_probe_decoder_rejected",
            )
        if decoded != detected:
            raise _ResolutionFailure(
                ResolutionStatus.UNRESOLVED_INVALID_IMAGE,
                f"image_signature_decoder_mismatch:{detected}:{decoded}",
            )
        return detected

    def _resolved(
        self,
        item: ResolutionInput,
        fetched: _Fetched,
        *,
        stable_candidate_url: str,
        method: str,
        detected: str,
        adapter_version: str = RESOLVER_VERSION,
    ) -> ResolutionResult:
        if fetched.declared_content_type not in ALLOWED_IMAGE_TYPES or fetched.declared_content_type != detected:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_INVALID_IMAGE, "direct_reference_is_not_valid_image")
        return self._result(
            item,
            status=ResolutionStatus.RESOLVED,
            method=method,
            stable_candidate_url=stable_candidate_url,
            validated_final_url=fetched.final_url,
            declared_content_type=fetched.declared_content_type,
            detected_content_type=detected,
            content=fetched.content,
            terminal_reason=None,
            adapter_version=adapter_version,
        )

    def _terminal(self, item: ResolutionInput, status: ResolutionStatus, method: str, reason: str) -> ResolutionResult:
        return self._result(item, status=status, method=method, stable_candidate_url=None, validated_final_url=None, declared_content_type=None, detected_content_type=None, content=b"", terminal_reason=reason, adapter_version=RESOLVER_VERSION)

    def _result(
        self,
        item: ResolutionInput,
        *,
        status: ResolutionStatus,
        method: str,
        stable_candidate_url: str | None,
        validated_final_url: str | None,
        declared_content_type: str | None,
        detected_content_type: str | None,
        content: bytes,
        terminal_reason: str | None,
        adapter_version: str = RESOLVER_VERSION,
    ) -> ResolutionResult:
        resolved_at = self._timestamp()
        redirect_count = sum(1 for attempt in self._attempts if attempt.outcome == "redirect")
        prefix_hash = "sha256:" + hashlib.sha256(content).hexdigest() if content else None
        identity = {
            "contract": RESOLVER_VERSION,
            "source_row_id": item.source_row_id,
            "status": status.value,
            "method": method,
            "stable_candidate_url": stable_candidate_url,
            "validated_final_url": validated_final_url,
            "attempt_ids": [attempt.attempt_id for attempt in self._attempts],
            "config_fingerprint": self.config.fingerprint,
        }
        return ResolutionResult(
            source_row_id=item.source_row_id,
            source_artifact_sha256=item.source_artifact_sha256,
            gbif_id=item.gbif_id,
            media_references=item.media_references,
            reference_host=item.host,
            media_type=item.media_type,
            media_format=item.media_format,
            media_license=item.media_license,
            occurrence_license=item.occurrence_license,
            license_basis=item.license_basis,
            status=status,
            method=method,
            stable_candidate_url=stable_candidate_url,
            validated_final_url=validated_final_url,
            redirect_count=redirect_count,
            declared_content_type=declared_content_type,
            detected_content_type=detected_content_type,
            bytes_sampled=len(content),
            probe_prefix_sha256=prefix_hash,
            content_sha256=None,
            content_hash_status="deferred",
            adapter_version=adapter_version,
            attempt_count=len(self._attempts),
            terminal_reason=terminal_reason,
            resolved_at=resolved_at,
            provenance_fingerprint=canonical_semantic_fingerprint(identity),
        )

    def _record(
        self,
        started: str,
        phase: str,
        method: str,
        requested_url: str,
        response_url: str | None,
        redirect_from: str | None,
        status_code: int | None,
        outcome: str,
        error: str | None,
        declared: str | None,
        byte_count: int,
        response: httpx.Response,
        retry_number: int,
        response_prefix_sha256: str | None = None,
    ) -> None:
        sequence = len(self._attempts) + 1
        ended = self._timestamp()
        attempt_id = canonical_semantic_fingerprint(
            {"source_row_id": self._source_row_id, "sequence": sequence, "phase": phase, "method": method, "requested_url": requested_url, "started_at": started}
        )
        self._attempts.append(
            ResolutionAttempt(
                attempt_id=attempt_id,
                source_row_id=self._source_row_id,
                sequence=sequence,
                phase=phase,
                method=method,
                requested_url=requested_url,
                response_url=response_url,
                redirect_from=redirect_from,
                status_code=status_code,
                outcome=outcome,
                error=error,
                declared_content_type=declared,
                response_prefix_sha256=response_prefix_sha256,
                response_byte_count=byte_count,
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
                retry_number=retry_number,
                started_at=started,
                ended_at=ended,
            )
        )

    def _record_transport(self, started: str, phase: str, method: str, url: str, retry_number: int, exc: Exception) -> None:
        sequence = len(self._attempts) + 1
        ended = self._timestamp()
        self._attempts.append(
            ResolutionAttempt(
                attempt_id=canonical_semantic_fingerprint({"source_row_id": self._source_row_id, "sequence": sequence, "phase": phase, "method": method, "requested_url": url, "started_at": started}),
                source_row_id=self._source_row_id,
                sequence=sequence,
                phase=phase,
                method=method,
                requested_url=url,
                response_url=None,
                redirect_from=None,
                status_code=None,
                outcome="retry",
                error=f"transport_error:{type(exc).__name__}",
                declared_content_type=None,
                response_prefix_sha256=None,
                response_byte_count=0,
                etag=None,
                last_modified=None,
                retry_number=retry_number,
                started_at=started,
                ended_at=ended,
            )
        )

    def _timestamp(self) -> str:
        value = self._now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("resolver clock must return an aware datetime")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def _wait_for_origin(self, host: str) -> None:
        now = self._monotonic()
        blocked_until = self._origin_blocked_until.get(host, 0.0)
        if self._origin_cycles.get(host, 0) >= self.config.circuit_max_cycles:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, "provider_quarantined")
        if blocked_until > now:
            raise _ResolutionFailure(ResolutionStatus.UNRESOLVED_PROVIDER_UNAVAILABLE, "provider_circuit_open")
        last = self._origin_last_request.get(host)
        if last is not None:
            delay = self.config.minimum_origin_interval_seconds - (now - last)
            if delay > 0:
                self._sleep(delay)
        self._origin_last_request[host] = self._monotonic()

    def _record_transient(self, host: str) -> None:
        count = self._origin_failures.get(host, 0) + 1
        self._origin_failures[host] = count
        if count >= self.config.circuit_failure_threshold:
            self._origin_failures[host] = 0
            self._origin_cycles[host] = self._origin_cycles.get(host, 0) + 1
            self._origin_blocked_until[host] = self._monotonic() + self.config.circuit_cooldown_seconds

    def _record_success(self, host: str) -> None:
        self._origin_failures[host] = 0

    def _retry_delay(self, response: httpx.Response, retry_number: int, url: str) -> float:
        value = response.headers.get("Retry-After")
        if value:
            try:
                return min(float(value), self.config.backoff_cap_seconds)
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(value)
                    return min(max(0.0, (parsed - self._now()).total_seconds()), self.config.backoff_cap_seconds)
                except (TypeError, ValueError, OverflowError):
                    pass
        return self._backoff(retry_number, url)

    def _backoff(self, retry_number: int, url: str) -> float:
        maximum = min(self.config.backoff_cap_seconds, self.config.backoff_base_seconds * (2 ** (retry_number - 1)))
        seed = hashlib.sha256(f"{self._source_row_id}:{url}:{retry_number}".encode()).digest()
        return random.Random(seed).uniform(0.0, maximum)


def _read_bounded(response: httpx.Response, *, max_bytes: int) -> bytes:
    output = bytearray()
    chunks: Iterable[bytes] = (
        (response.content,) if response.is_stream_consumed else response.iter_bytes()
    )
    for chunk in chunks:
        if not chunk:
            continue
        remaining = max_bytes - len(output)
        if remaining <= 0:
            break
        output.extend(chunk[:remaining])
        if len(output) >= max_bytes:
            break
    return bytes(output)


def _decode_probe_content_type(content: bytes) -> str | None:
    parser = ImageFile.Parser()
    try:
        parser.feed(content)
    except (OSError, SyntaxError, ValueError):
        return None
    image = parser.image
    if image is None:
        return None
    if image.width <= 0 or image.height <= 0 or image.width * image.height > 80_000_000:
        return None
    return {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "TIFF": "image/tiff",
        "GIF": "image/gif",
    }.get(str(image.format or "").upper())


def _prefix_sha256(content: bytes) -> str | None:
    return "sha256:" + hashlib.sha256(content).hexdigest() if content else None
