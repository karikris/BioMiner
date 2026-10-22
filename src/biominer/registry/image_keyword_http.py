"""Bounded HTTP and crash-safe request accounting for local name enrichment.

SQLite is an explicit local job ledger, never a production work-state backend.
Only request fingerprints and public endpoint metadata are stored in the ledger;
response JSON is compressed in a bounded cache. Reservations survive failures.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl
import zlib

import httpx


MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_CACHE_BYTES = 8 * 1024 * 1024
MYMEMORY_REQUEST_BYTES = 1024 * 1024
_SENSITIVE = re.compile(
    r"^(?:key|(?:x[-_])?api[-_]?key|access[-_]?key|token|(?:access|refresh|auth)[-_]?token"
    r"|secret|client[-_]?secret|password|passwd|authorization|auth|email(?:[-_]?address)?|de)$",
    re.I,
)
_RETRY_STATUSES = {429, 502, 503, 504}


class BudgetDeferred(RuntimeError):
    """A persistent quota or invocation limit prevents another attempt."""


class KeywordHTTPError(RuntimeError):
    """A deliberately value-free transport or response failure."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class KeywordHTTP:
    """One sequential coordinator's HTTP client and durable quota ledger.

    ``max_retries`` counts retries after the first attempt. Cache hits consume no
    request quota. All calls, including cache hits, count toward the per-job call
    limit; ``reset_job_receipts()`` starts the next species/source job. A caller
    must select a source with ``with http.source(key)``.
    Closing this object closes only a client it created itself.
    """

    def __init__(
        self,
        state_path: Path,
        *,
        max_retries: int = 3,
        min_interval: float = 1.0,
        max_requests: int = 250_000,
        max_job_calls: int = 10_000,
        logger: Callable[[dict[str, Any]], None] | None = None,
        client: Any = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if (not 0 <= max_retries <= 3 or not math.isfinite(min_interval)
                or min_interval < 0 or max_requests < 0 or max_job_calls < 1):
            raise ValueError("invalid HTTP bounds")
        self.max_retries, self.min_interval, self.max_requests = max_retries, min_interval, max_requests
        self.max_job_calls = max_job_calls
        self.job_call_count = 0
        self.logger, self.clock, self.sleep = logger, clock, sleep
        # HTTPX INFO logging contains complete URLs, including API credentials.
        # This coordinator emits its own value-free request events instead.
        for logger_name in ("httpx", "httpcore"):
            transport_logger = logging.getLogger(logger_name)
            if transport_logger.getEffectiveLevel() < logging.WARNING:
                transport_logger.setLevel(logging.WARNING)
        self.request_count = 0
        self.job_receipts: list[dict[str, Any]] = []
        self._source: str | None = None
        self._owns_client = client is None
        self.client = client if client is not None else httpx.Client(
            timeout=httpx.Timeout(45.0, connect=15.0), follow_redirects=False,
            headers={"User-Agent": "BioMiner-name-enrichment/0.1 (source-bound research)"},
            trust_env=False,
        )
        state_path = Path(state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(state_path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS request_attempts (
                id INTEGER PRIMARY KEY, source TEXT NOT NULL, host TEXT NOT NULL,
                path TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                reserved_at REAL NOT NULL, status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS quota_usage (
                bucket TEXT NOT NULL, period TEXT NOT NULL,
                requests INTEGER NOT NULL, words INTEGER NOT NULL, bandwidth INTEGER NOT NULL,
                PRIMARY KEY(bucket, period)
            );
            CREATE TABLE IF NOT EXISTS rate_slots (
                scope TEXT PRIMARY KEY, next_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS response_cache (
                request_sha256 TEXT PRIMARY KEY, body BLOB NOT NULL,
                response_sha256 TEXT NOT NULL, retrieved_at REAL NOT NULL,
                status INTEGER NOT NULL
            );
        """)

    @contextmanager
    def source(self, source_key: str) -> Iterator[KeywordHTTP]:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", source_key):
            raise ValueError("invalid source key")
        previous = self._source
        self._source = source_key
        try:
            yield self
        finally:
            self._source = previous

    def reset_job_receipts(self) -> None:
        self.job_receipts.clear()
        self.job_call_count = 0

    def get(self, base_url: str) -> Callable[[str, Mapping[str, Any]], Any]:
        return lambda path, params: self._request("GET", self._join(base_url, path), params=params)

    def post(self, base_url: str) -> Callable[[str, Mapping[str, Any]], Any]:
        return lambda path, payload: self._request("POST", self._join(base_url, path), payload=payload)

    def graphql(self, url: str) -> Callable[[str, Mapping[str, Any]], Any]:
        return lambda query, variables: self._request("POST", url, payload={"query": query, "variables": variables})

    @staticmethod
    def _join(base_url: str, path: str) -> str:
        result = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
        if urlsplit(result).netloc != urlsplit(base_url).netloc:
            raise ValueError("endpoint host changed")
        return result

    def _emit(self, event: dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger(event)

    def _reserve(self, host: str, path: str, fingerprint: str, words: int, mymemory: bool) -> tuple[int, float]:
        if self.request_count >= self.max_requests:
            raise BudgetDeferred("invocation_request_limit")
        now = self.clock()
        stamp = datetime.fromtimestamp(now, timezone.utc)
        day, month = stamp.strftime("%Y-%m-%d"), stamp.strftime("%Y-%m")
        source = str(self._source)
        quotas = [("source:" + source, day, None, None, None)]
        if "inaturalist" in host or source in {"inat", "inaturalist"}:
            quotas.append(("inaturalist", day, 10_000, None, None))
        wikimedia = any(host == suffix or host.endswith("." + suffix) for suffix in (
            "wikimedia.org", "wikipedia.org", "wikidata.org", "wikispecies.org",
        ))
        if wikimedia or mymemory or source in {"wikimedia", "wikispecies", "commons", "wikidata", "mymemory"}:
            quotas.append(("wikimedia_mymemory", day, 10_000, None, None))
        if mymemory:
            quotas.append(("mymemory", month, 10_000, 10_000, 10 * 1024**3))
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for bucket, period, request_limit, word_limit, byte_limit in quotas:
                row = self.db.execute(
                    "SELECT requests, words, bandwidth FROM quota_usage WHERE bucket=? AND period=?",
                    (bucket, period),
                ).fetchone() or (0, 0, 0)
                extra_words = words if bucket == "mymemory" else 0
                extra_bytes = MYMEMORY_REQUEST_BYTES if bucket == "mymemory" else 0
                if ((request_limit is not None and row[0] + 1 > request_limit)
                        or (word_limit is not None and row[1] + extra_words > word_limit)
                        or (byte_limit is not None and row[2] + extra_bytes > byte_limit)):
                    raise BudgetDeferred("persistent_" + bucket + "_limit")
                self.db.execute("""
                    INSERT INTO quota_usage VALUES (?, ?, 1, ?, ?)
                    ON CONFLICT(bucket, period) DO UPDATE SET requests=requests+1,
                    words=words+excluded.words, bandwidth=bandwidth+excluded.bandwidth
                """, (bucket, period, extra_words, extra_bytes))
            scopes = ("host:" + host, "source:" + source)
            slot = now
            for scope in scopes:
                row = self.db.execute("SELECT next_at FROM rate_slots WHERE scope=?", (scope,)).fetchone()
                if row:
                    slot = max(slot, row[0])
            for scope in scopes:
                self.db.execute("INSERT INTO rate_slots VALUES (?, ?) ON CONFLICT(scope) DO UPDATE SET next_at=excluded.next_at",
                                (scope, slot + self.min_interval))
            attempt = self.db.execute("INSERT INTO request_attempts(source,host,path,request_sha256,reserved_at,status) VALUES (?,?,?,?,?,?)",
                                      (source, host, path, fingerprint, now, "reserved")).lastrowid
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.request_count += 1
        return int(attempt), max(0.0, slot - now)

    def _finish(self, attempt: int, status: str, metadata: dict[str, Any]) -> None:
        self.db.execute("UPDATE request_attempts SET status=? WHERE id=?", (status, attempt))
        self._emit({**metadata, "event": "http_attempt", "status": status, "request_count": self.request_count})

    def _retry_delay(self, response: Any, retry: int) -> float:
        value = response.headers.get("Retry-After", "") if response is not None else ""
        try:
            delay = float(value)
        except (ValueError, TypeError):
            try:
                delay = parsedate_to_datetime(value).timestamp() - self.clock()
            except (ValueError, TypeError, OverflowError):
                delay = float(2**retry)
        # A remote server cannot make this bounded worker sleep indefinitely.
        if not math.isfinite(delay) or delay > 3600:
            raise BudgetDeferred("retry_after_exceeds_one_hour")
        return max(0.0, delay)

    def _wait_retry(self, response: Any, retry: int, host: str) -> None:
        delay = self._retry_delay(response, retry)
        until = self.clock() + delay
        # Preserve server backoff even if this process stops while sleeping.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for scope in ("host:" + host, "source:" + str(self._source)):
                self.db.execute("""INSERT INTO rate_slots VALUES (?, ?)
                    ON CONFLICT(scope) DO UPDATE SET next_at=MAX(next_at,excluded.next_at)
                """, (scope, until))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.sleep(delay)

    def _request(self, method: str, url: str, *, params: Mapping[str, Any] | None = None,
                 payload: Mapping[str, Any] | None = None) -> Any:
        if self._source is None:
            raise ValueError("source context required")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("HTTPS public endpoint required")
        host, path = parsed.hostname.lower(), parsed.path or "/"
        # URL query arguments are accepted for existing adapters but never stored.
        all_params = dict(parse_qsl(parsed.query))
        all_params.update(params or {})
        clean_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        mymemory = host == "mymemory.translated.net" or host.endswith(".mymemory.translated.net") or self._source == "mymemory"
        if mymemory and str(all_params.get("mt", "0")) != "0":
            raise ValueError("machine translation must remain disabled")
        if mymemory:
            all_params["mt"] = "0"
        safe_params = {k: v for k, v in all_params.items() if not _SENSITIVE.search(k)}
        # Sensitive fields are omitted recursively, including GraphQL variables.
        def sanitized(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {k: sanitized(v) for k, v in value.items() if not _SENSITIVE.search(str(k))}
            if isinstance(value, (list, tuple)):
                return [sanitized(v) for v in value]
            return value
        fingerprint = hashlib.sha256(_json_bytes({"source": self._source, "method": method,
            "host": host, "path": path, "params": safe_params, "payload": sanitized(payload)})).hexdigest()
        metadata = {"source": self._source, "host": host, "path": path, "request_sha256": fingerprint}
        if self.job_call_count >= self.max_job_calls:
            raise KeywordHTTPError("job_call_limit")
        self.job_call_count += 1
        cached = self.db.execute("SELECT body,response_sha256,retrieved_at,status FROM response_cache WHERE request_sha256=?", (fingerprint,)).fetchone()
        if cached:
            raw = self._decompress(cached[0])
            if hashlib.sha256(raw).hexdigest() != cached[1]:
                raise KeywordHTTPError("cache_digest_mismatch")
            value = self._decode(raw)
            self.job_receipts.append({**metadata, "response_sha256": cached[1], "retrieved_at": cached[2], "status": cached[3], "cache_hit": True})
            return value
        words = len(str(all_params.get("q", "")).split())
        for retry in range(self.max_retries + 1):
            attempt, delay = self._reserve(host, path, fingerprint, words, mymemory)
            if delay:
                self.sleep(delay)
            response = None
            try:
                kwargs = {"params": all_params}
                if payload is not None:
                    kwargs["json"] = payload
                with self.client.stream(method, clean_url, **kwargs) as response:
                    status = response.status_code
                    if status in _RETRY_STATUSES:
                        self._finish(attempt, str(status), metadata)
                        if retry == self.max_retries:
                            raise KeywordHTTPError("retry_limit_http_" + str(status))
                        self._wait_retry(response, retry, host)
                        continue
                    if not 200 <= status < 300:
                        self._finish(attempt, str(status), metadata)
                        raise KeywordHTTPError("http_" + str(status))
                    chunks, size = [], 0
                    for chunk in response.iter_bytes(chunk_size=64 * 1024):
                        size += len(chunk)
                        if size > MAX_BODY_BYTES or (mymemory and size > MYMEMORY_REQUEST_BYTES):
                            raise KeywordHTTPError("response_body_limit")
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    value = self._decode(raw)
                    retrieved = self.clock()
                    digest = hashlib.sha256(raw).hexdigest()
                    self._cache(fingerprint, raw, digest, retrieved, status)
                    self._finish(attempt, str(status), metadata)
                    self.job_receipts.append({**metadata, "response_sha256": digest, "retrieved_at": retrieved, "status": status, "cache_hit": False})
                    return value
            except httpx.TransportError:
                self._finish(attempt, "network_error", metadata)
                if retry == self.max_retries:
                    raise KeywordHTTPError("network_retry_limit") from None
                self._wait_retry(None, retry, host)
            except KeywordHTTPError:
                self.db.execute("UPDATE request_attempts SET status='response_error' WHERE id=? AND status='reserved'", (attempt,))
                raise
        raise AssertionError("unreachable retry state")

    @staticmethod
    def _decompress(body: bytes) -> bytes:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(body, MAX_BODY_BYTES + 1)
        if len(raw) > MAX_BODY_BYTES or not decoder.eof or decoder.unused_data:
            raise KeywordHTTPError("cache_body_limit_or_corruption")
        return raw

    @staticmethod
    def _decode(raw: bytes) -> Any:
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            raise KeywordHTTPError("non_json_response") from None
        if not isinstance(value, (dict, list)):
            raise KeywordHTTPError("unexpected_json_shape")
        if isinstance(value, dict) and (value.get("error") or value.get("errors")):
            raise KeywordHTTPError("provider_error_response")
        return value

    def _cache(self, fingerprint: str, raw: bytes, digest: str, retrieved: float, status: int) -> None:
        body = zlib.compress(raw)
        if len(body) > MAX_CACHE_BYTES:
            return
        self.db.execute("BEGIN IMMEDIATE")
        try:
            while self.db.execute("SELECT COALESCE(SUM(length(body)),0) FROM response_cache").fetchone()[0] + len(body) > MAX_CACHE_BYTES:
                self.db.execute("DELETE FROM response_cache WHERE request_sha256=(SELECT request_sha256 FROM response_cache ORDER BY retrieved_at,request_sha256 LIMIT 1)")
            self.db.execute("INSERT OR REPLACE INTO response_cache VALUES (?,?,?,?,?)", (fingerprint, body, digest, retrieved, status))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.db.close()
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> KeywordHTTP:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
