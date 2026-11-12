"""Small local artifact primitives for the explicit experimental backend."""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from uuid import uuid4


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def producer_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
    ).strip()


def source_fingerprint() -> str:
    """Identify the actual workflow sources, including an uncommitted checkout."""
    from biominer.common.semantic_hash import canonical_semantic_fingerprint

    return canonical_semantic_fingerprint(
        {
            path.name: file_sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        }
    )


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_parquet(path: Path, frame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.write_parquet(temp, compression="zstd")
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def run_lock(directory: Path):
    """OS releases this lock after a crash; no stale PID deletion is needed."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".worker.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another worker owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def exclusive_output(parameter: str = "output_dir"):
    """Keep identity checks and multi-file publication under the same OS lock."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def locked(*args, **kwargs):
            directory = Path(signature.bind(*args, **kwargs).arguments[parameter])
            with run_lock(directory):
                return function(*args, **kwargs)

        return locked

    return decorate
