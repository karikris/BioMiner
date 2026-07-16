from __future__ import annotations

from typing import Any
import hashlib
import json


def publication_lock_digest(key: str) -> bytes:
    if not isinstance(key, str):
        raise TypeError("publication lock key must be a string")
    if not key:
        raise ValueError("publication lock key must not be empty")
    return hashlib.sha256(key.encode("utf-8")).digest()


def stable_work_key(payload: dict[str, Any], *, prefix: str | None = None) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}" if prefix else digest


def scoped_work_item_key(job_name: str, stage: str, registry_version: str | None, payload: dict[str, Any]) -> str:
    canonical = {
        "job_name": job_name,
        "stage": stage,
        "registry_version": registry_version,
        "payload": payload,
    }
    digest = stable_work_key(canonical)[:24]
    return f"{job_name}:{digest}"


def uri_shard_id(uri: str) -> str:
    return hashlib.sha256(uri.encode("utf-8")).hexdigest()
