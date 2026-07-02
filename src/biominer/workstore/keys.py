from __future__ import annotations

from typing import Any
import hashlib
import json


def stable_work_key(payload: dict[str, Any], *, prefix: str | None = None) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}" if prefix else digest


def flickr_poll_once_work_key(query_payload: dict[str, Any]) -> str:
    return stable_work_key(query_payload, prefix="flickr_poll_once")


def uri_shard_id(uri: str) -> str:
    return hashlib.sha256(uri.encode("utf-8")).hexdigest()
