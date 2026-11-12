"""Bounded local media cache for the experimental EmbeddingGemma benchmark.

Bytes are validated before create-only content-addressed publication. The
coordinator atomically checkpoints an index containing no source URLs or error
messages. Successful cached files are hash-checked on every resume; corrupt
objects are retained for inspection and excluded from downstream work.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import tempfile
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import wraps
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import polars as pl
from PIL import Image, ImageOps, UnidentifiedImageError

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.embeddinggemma.common import run_lock
from biominer.storage.content_address import sha256_file

SCHEMA_VERSION = "biominer-embeddinggemma-media-v1"
_INDEX_SCHEMA = {
    "item_id": pl.String,
    "status": pl.String,
    "image_hash": pl.String,
    "local_path": pl.String,
    "width": pl.Int64,
    "height": pl.Int64,
    "error_type": pl.String,
    "source_fingerprint": pl.String,
    "attempts": pl.Int64,
    "bytes_downloaded": pl.Int64,
}
_MAX_RETRY_DELAY = 30.0
_USER_AGENT = "BioMiner-EmbeddingGemma-experiment/1.0"
_LOGGER = logging.getLogger(__name__)


def _source_fingerprint(item: dict) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "item_id": item["item_id"],
            "image_url": item.get("image_url") or "",
            "fallback_url": item.get("fallback_url"),
        }
    )


def read_cached_items(items: pl.DataFrame, cache_root: Path) -> pl.DataFrame:
    """Read-only scoring selection; verify source binding and bytes without requests."""
    cache_root = Path(cache_root).resolve()
    with run_lock(cache_root):
        previous = _read_index(cache_root / "index.parquet")
        rows = []
        for item in items.sort("item_id").iter_rows(named=True):
            fingerprint = _source_fingerprint(item)
            old = previous.get(item["item_id"])
            if old and old["source_fingerprint"] != fingerprint:
                raise ValueError(
                    "cache item source fingerprint changed; use a new cache root"
                )
            if old and old["status"] in {"ok", "cache_corrupt"}:
                rows.append(_verify_cached(old, cache_root))
            elif old:
                rows.append(old)
            else:
                rows.append(
                    {
                        **_result(
                            {
                                "item_id": item["item_id"],
                                "source_fingerprint": fingerprint,
                            }
                        ),
                        "status": "not_cached",
                    }
                )
        return pl.from_dicts(rows, schema=_INDEX_SCHEMA)


def _exclusive_cache(function):
    @wraps(function)
    def locked(items, cache_root, **kwargs):
        with run_lock(Path(cache_root)):
            return function(items, cache_root, **kwargs)

    return locked


@_exclusive_cache
def cache_items(
    items: pl.DataFrame,
    cache_root: Path,
    *,
    workers: int = 4,
    limit: int | None = None,
    http_client=None,
    retries: int = 2,
    max_image_bytes: int = 20_000_000,
    timeout: float = 30,
) -> pl.DataFrame:
    """Cache supplied items and return the selected status rows, sorted by ID.

    ``http_client`` may be an injected HTTPX-compatible synchronous client with
    ``stream()``; it is never closed by this function. ``limit`` caps the sorted
    input selection, including resume hits. The persistent ``index.parquet``
    also retains items from earlier calls. At most ``workers`` futures exist;
    checkpoints lose at most 31 completed status updates on interruption, while
    already published content objects remain intact.
    """
    _validate_bounds(workers, limit, retries, max_image_bytes, timeout)
    required = {"item_id", "image_url"}
    if not required.issubset(items.columns):
        raise ValueError("cache input must contain item_id and image_url")
    selected = items.sort("item_id")
    if (
        selected["item_id"].null_count()
        or selected["item_id"].n_unique() != selected.height
    ):
        raise ValueError("cache input item IDs must be nonnull and unique")
    if limit is not None:
        selected = selected.head(limit)
    work = []
    for item in selected.iter_rows(named=True):
        item_id = item["item_id"]
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("cache input item IDs must be nonempty strings")
        work.append(
            {
                "item_id": item_id,
                "image_url": item.get("image_url") or "",
                "fallback_url": item.get("fallback_url"),
                "source_fingerprint": _source_fingerprint(item),
            }
        )
    cache_root = Path(cache_root).resolve()
    index_path = cache_root / "index.parquet"
    previous = _read_index(index_path)
    for item in work:
        old = previous.get(item["item_id"])
        if old and old["source_fingerprint"] != item["source_fingerprint"]:
            raise ValueError(
                "cache item source fingerprint changed; use a new cache root"
            )
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "objects").mkdir(exist_ok=True)
    pending = []
    changed = False
    for item in work:
        old = previous.get(item["item_id"])
        if old and old["status"] in ("ok", "cache_corrupt"):
            checked = _verify_cached(old, cache_root)
            previous[item["item_id"]] = checked
            changed = changed or checked != old
        else:
            pending.append(item)
    client_context = (
        nullcontext(http_client)
        if http_client is not None
        else httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
            limits=httpx.Limits(
                max_connections=workers, max_keepalive_connections=workers
            ),
        )
    )
    completed_since_checkpoint = 0
    downloaded = 0
    checkpoint_every = min(32, max(1, workers * 4))
    if pending:
        with (
            client_context as client,
            ThreadPoolExecutor(max_workers=workers) as executor,
        ):
            iterator = iter(pending)
            futures = {}
            for _ in range(min(workers, len(pending))):
                item = next(iterator)
                futures[
                    executor.submit(
                        _cache_one,
                        item,
                        cache_root,
                        client,
                        retries,
                        max_image_bytes,
                        timeout,
                    )
                ] = item
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                # The coordinator owns all index mutations and publication.
                for future in done:
                    item = futures.pop(future)
                    result = future.result()
                    previous[item["item_id"]] = result
                    changed = True
                    completed_since_checkpoint += 1
                    downloaded += 1
                    if downloaded == 1 or downloaded % 100 == 0:
                        _LOGGER.warning(
                            "Image cache completed=%s/%s", downloaded, len(pending)
                        )
                    if completed_since_checkpoint >= checkpoint_every:
                        _write_index(index_path, previous)
                        completed_since_checkpoint = 0
                        changed = False
                    next_item = next(iterator, None)
                    if next_item is not None:
                        futures[
                            executor.submit(
                                _cache_one,
                                next_item,
                                cache_root,
                                client,
                                retries,
                                max_image_bytes,
                                timeout,
                            )
                        ] = next_item
    if changed or not index_path.exists():
        _write_index(index_path, previous)
    return pl.from_dicts(
        [previous[item["item_id"]] for item in work], schema=_INDEX_SCHEMA
    )


def _cache_one(
    item: dict, root: Path, client, retries: int, max_bytes: int, timeout: float
) -> dict:
    result = _result(item)
    urls = list(
        dict.fromkeys(url for url in (item["image_url"], item["fallback_url"]) if url)
    )
    if not urls:
        return {**result, "status": "unavailable", "error_type": "missing_image_url"}
    for url in urls:
        if not _valid_url(url):
            result.update(status="unavailable", error_type="invalid_image_url")
            continue
        for attempt in range(retries + 1):
            result["attempts"] += 1
            started = time.monotonic()
            try:
                with client.stream(
                    "GET",
                    url,
                    timeout=timeout,
                    follow_redirects=True,
                    headers={"User-Agent": _USER_AGENT},
                ) as response:
                    code = response.status_code
                    if code != 200:
                        retryable = code in (408, 425, 429) or 500 <= code < 600
                        result.update(
                            status="rate_limited" if code == 429 else "unavailable",
                            error_type=f"http_{code}",
                        )
                        if retryable and attempt < retries:
                            delay = _retry_delay(
                                response.headers.get("Retry-After"), attempt
                            )
                            if delay is None:
                                result.update(
                                    status="rate_limited"
                                    if code == 429
                                    else "unavailable",
                                    error_type="retry_after_exceeds_budget",
                                )
                                return result
                        else:
                            break
                    else:
                        result.update(status="unavailable", error_type=None)
                        declared = response.headers.get("Content-Length")
                        if (
                            declared
                            and declared.isdigit()
                            and int(declared) > max_bytes
                        ):
                            result.update(
                                status="too_large",
                                error_type="max_image_bytes_exceeded",
                            )
                            break
                        data = bytearray()
                        # Preserve the transport's bounded chunks so a small limit
                        # does not buffer an entire response before detecting it.
                        for chunk in response.iter_bytes():
                            if time.monotonic() - started > timeout:
                                raise TimeoutError
                            result["bytes_downloaded"] += len(chunk)
                            if len(data) + len(chunk) > max_bytes:
                                result.update(
                                    status="too_large",
                                    error_type="max_image_bytes_exceeded",
                                )
                                break
                            data.extend(chunk)
                        if result["status"] == "too_large":
                            break
                        try:
                            width, height = _dimensions(BytesIO(data))
                        except (
                            UnidentifiedImageError,
                            OSError,
                            ValueError,
                            SyntaxError,
                            Image.DecompressionBombError,
                            Image.DecompressionBombWarning,
                        ):
                            result.update(
                                status="decode_error",
                                error_type="invalid_image_content",
                            )
                            break
                        image_hash = "sha256:" + hashlib.sha256(data).hexdigest()
                        try:
                            path = _publish_bytes(root, image_hash, data)
                        except _CorruptObject:
                            result.update(
                                status="cache_corrupt",
                                error_type="content_hash_mismatch",
                                image_hash=image_hash,
                                local_path=str(_object_path(root, image_hash)),
                            )
                            return result
                        except OSError:
                            result.update(
                                status="cache_error", error_type="cache_write_error"
                            )
                            return result
                        result.update(
                            status="ok",
                            error_type=None,
                            image_hash=image_hash,
                            local_path=str(path),
                            width=width,
                            height=height,
                        )
                        return result
                # Retry delay occurs after closing the response/connection.
                time.sleep(delay)
            except httpx.RequestError, TimeoutError:
                result.update(status="download_error", error_type="network_error")
                if attempt < retries:
                    time.sleep(_retry_delay(None, attempt))
                    continue
                break
            except httpx.InvalidURL, httpx.UnsupportedProtocol, ValueError:
                result.update(status="unavailable", error_type="invalid_image_url")
                break
        # A fallback URL may recover unavailable, oversized, or undecodable media.
    return result


def _result(item: dict) -> dict:
    return {
        "item_id": item["item_id"],
        "status": "unavailable",
        "image_hash": None,
        "local_path": None,
        "width": None,
        "height": None,
        "error_type": None,
        "source_fingerprint": item["source_fingerprint"],
        "attempts": 0,
        "bytes_downloaded": 0,
    }


def _dimensions(source) -> tuple[int, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(source) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            if width <= 0 or height <= 0:
                raise ValueError("invalid image dimensions")
            return width, height


def _object_path(root: Path, image_hash: str) -> Path:
    digest = image_hash.removeprefix("sha256:")
    return root / "objects" / digest[:2] / (digest + ".image")


class _CorruptObject(Exception):
    pass


def _publish_bytes(root: Path, image_hash: str, data: bytes | bytearray) -> Path:
    path = _object_path(root, image_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != image_hash:
            raise _CorruptObject
        return path
    descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Hard-link publication is atomic and create-only, including when
            # two workers discover the same bytes concurrently.
            os.link(temporary, path)
        except FileExistsError:
            if sha256_file(path) != image_hash:
                raise _CorruptObject from None
        return path
    finally:
        Path(temporary).unlink(missing_ok=True)


def _verify_cached(row: dict, root: Path) -> dict:
    path = Path(row["local_path"] or "")
    expected_hash = row["image_hash"] or ""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash):
        return {
            **row,
            "status": "cache_corrupt",
            "error_type": "invalid_content_hash",
            "width": None,
            "height": None,
        }
    if path.resolve() != _object_path(root, expected_hash).resolve():
        return {
            **row,
            "status": "cache_corrupt",
            "error_type": "invalid_cache_path",
            "width": None,
            "height": None,
        }
    try:
        if not path.is_file() or sha256_file(path) != expected_hash:
            return {
                **row,
                "status": "cache_corrupt",
                "error_type": "content_hash_mismatch",
                "width": None,
                "height": None,
            }
        width, height = _dimensions(path)
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return {
            **row,
            "status": "cache_corrupt",
            "error_type": "invalid_cached_image",
            "width": None,
            "height": None,
        }
    return {**row, "status": "ok", "error_type": None, "width": width, "height": height}


def _retry_delay(value: str | None, attempt: int) -> float | None:
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                delay = max(0.0, (parsed - datetime.now(UTC)).total_seconds())
            except ValueError, TypeError, OverflowError:
                delay = 0.25 * (2 ** min(attempt, 4))
        if math.isfinite(delay) and delay >= 0:
            return delay if delay <= _MAX_RETRY_DELAY else None
    return min(4.0, 0.25 * (2 ** min(attempt, 4)))


def _valid_url(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme.casefold() in ("http", "https")
            and bool(parsed.netloc)
            and not parsed.username
            and not parsed.password
        )
    except ValueError:
        return False


def _validate_bounds(workers, limit, retries, max_bytes, timeout) -> None:
    for value, label, minimum in (
        (workers, "workers", 1),
        (retries, "retries", 0),
        (max_bytes, "max_image_bytes", 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{label} is outside its finite integer bounds")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise ValueError("limit must be a nonnegative integer or None")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (float, int))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be finite and positive")


def _read_index(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        frame = pl.read_parquet(path)
    except OSError, pl.exceptions.PolarsError:
        raise ValueError("cache status index is unreadable") from None
    if (
        frame.schema != pl.Schema(_INDEX_SCHEMA)
        or frame["item_id"].null_count()
        or frame["item_id"].n_unique() != frame.height
    ):
        raise ValueError("cache status index schema or identity is incompatible")
    return {row["item_id"]: row for row in frame.iter_rows(named=True)}


def _write_index(path: Path, rows: dict[str, dict]) -> None:
    frame = pl.from_dicts([rows[key] for key in sorted(rows)], schema=_INDEX_SCHEMA)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".index-", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.write_parquet(temporary, compression="zstd")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
