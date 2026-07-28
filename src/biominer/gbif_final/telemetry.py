from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
from typing import Any
from uuid import uuid4

from biominer.common.semantic_hash import canonical_semantic_fingerprint


TELEMETRY_VERSION = "gbif-final-bounded-telemetry/v1"
SUCCESS_RECEIPT = "run_receipt.json"
FAILURE_RECEIPT = "failure_receipt.json"
EVENT_LOG = "progress.jsonl"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class BoundedRunTelemetry:
    """Persist one invocation's chained progress and process evidence."""

    def __init__(
        self,
        *,
        root_directory: str | Path,
        producer_git_sha: str,
        config: Mapping[str, object],
        run_id: str | None = None,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not producer_git_sha.strip():
            raise ValueError("producer_git_sha must be non-empty")
        resolved_run_id = run_id or _default_run_id()
        if not _RUN_ID.fullmatch(resolved_run_id):
            raise ValueError(f"invalid telemetry run ID: {resolved_run_id!r}")
        root = Path(root_directory).resolve()
        invocation = root / f"run_id={resolved_run_id}"
        if invocation.exists():
            raise FileExistsError(
                f"refusing to overwrite telemetry invocation: {invocation}"
            )
        invocation.mkdir(parents=True)
        _fsync_directory(invocation.parent)
        event_log = invocation / EVENT_LOG
        with event_log.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())

        self.run_id = resolved_run_id
        self.invocation_directory = invocation
        self.event_log = event_log
        self.producer_git_sha = producer_git_sha
        self.config = _normalized(config)
        self._event_sink = event_sink
        self._started_ns = time.monotonic_ns()
        self._io_start = _process_io()
        self._event_count = 0
        self._previous_event_sha256: str | None = None
        self._closed = False
        self.emit(
            "run_started",
            stage="orchestration",
            partition=None,
            checkpoint_path=str(invocation),
        )

    def emit_payload(self, payload: Mapping[str, object]) -> None:
        event = str(payload.get("event") or "").strip()
        if not event:
            raise ValueError("telemetry payload requires a non-empty event")
        self.emit(
            event,
            **{
                key: value
                for key, value in payload.items()
                if key != "event"
            },
        )

    def emit(self, event: str, **fields: object) -> None:
        if self._closed:
            raise RuntimeError("telemetry invocation is already closed")
        if not event.strip():
            raise ValueError("telemetry event must be non-empty")
        record: dict[str, object] = {
            "schema_version": TELEMETRY_VERSION,
            "run_id": self.run_id,
            "event_index": self._event_count,
            "event": event,
            "timestamp": _timestamp(),
            "elapsed_seconds": _elapsed_seconds(self._started_ns),
            "process_id": os.getpid(),
            "peak_rss_bytes": _peak_rss_bytes(),
            "process_io": _process_io(),
            "previous_event_sha256": self._previous_event_sha256,
            **_normalized(fields),
        }
        record["event_sha256"] = _fingerprint_without(
            record,
            "event_sha256",
        )
        encoded = _canonical_json(record) + b"\n"
        with self.event_log.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._previous_event_sha256 = str(record["event_sha256"])
        self._event_count += 1
        if self._event_sink is not None:
            self._event_sink(record)

    def finish(
        self,
        *,
        output_manifest: str | Path,
        rows: int,
        resumed_output: bool,
    ) -> dict[str, Any]:
        if rows <= 0:
            raise ValueError("completed telemetry rows must be positive")
        manifest = Path(output_manifest).resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        manifest_sha256 = _sha256(manifest)
        self.emit(
            "run_completed",
            stage="orchestration",
            partition=None,
            rows_read=rows,
            rows_written=rows,
            rows_passed=rows,
            rows_failed=0,
            rows_unresolved=0,
            rows_skipped_from_cache=rows if resumed_output else 0,
            requests_completed=0,
            retries=0,
            rate_limit_events=0,
            bytes_downloaded=0,
            network_scope="NOT_APPLICABLE",
            output_manifest=str(manifest),
            output_manifest_sha256=manifest_sha256,
            checkpoint_path=str(self.invocation_directory),
        )
        receipt = self._receipt_base(status="completed")
        receipt.update(
            {
                "output": {
                    "manifest_path": str(manifest),
                    "manifest_sha256": manifest_sha256,
                    "rows": rows,
                    "resumed_existing_output": resumed_output,
                },
                "validation": {
                    "event_chain_complete": True,
                    "output_manifest_checksum_recorded": True,
                    "receipt_written_last": True,
                },
            }
        )
        return self._seal_receipt(SUCCESS_RECEIPT, receipt)

    def fail(
        self,
        error: BaseException,
        *,
        stage: str,
        partition: int | None = None,
    ) -> dict[str, Any]:
        self.emit(
            "run_failed",
            stage=stage,
            partition=partition,
            error_type=type(error).__name__,
            error_message=str(error),
            checkpoint_path=str(self.invocation_directory),
        )
        receipt = self._receipt_base(status="failed")
        receipt.update(
            {
                "failure": {
                    "stage": stage,
                    "partition": partition,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                "validation": {
                    "event_chain_complete": True,
                    "failure_evidence_retained": True,
                    "receipt_written_last": True,
                },
            }
        )
        return self._seal_receipt(FAILURE_RECEIPT, receipt)

    def _receipt_base(self, *, status: str) -> dict[str, Any]:
        io_end = _process_io()
        return {
            "schema_version": TELEMETRY_VERSION,
            "run_id": self.run_id,
            "status": status,
            "generated_at": _timestamp(),
            "producer_git_sha": self.producer_git_sha,
            "config": self.config,
            "metrics": {
                "elapsed_seconds": _elapsed_seconds(self._started_ns),
                "peak_rss_bytes": _peak_rss_bytes(),
                "process_io_start": self._io_start,
                "process_io_end": io_end,
                "process_io_delta": _io_delta(self._io_start, io_end),
            },
            "event_log": {
                "path": EVENT_LOG,
                "event_count": self._event_count,
                "final_event_sha256": self._previous_event_sha256,
                "physical_sha256": _sha256(self.event_log),
                "physical_bytes": self.event_log.stat().st_size,
            },
            "receipt_policy": {
                "create_only": True,
                "event_log_hash_chained": True,
                "receipt_written_last": True,
            },
        }

    def _seal_receipt(
        self,
        filename: str,
        receipt: dict[str, Any],
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("telemetry invocation is already closed")
        receipt["receipt_fingerprint"] = _fingerprint_without(
            receipt,
            "receipt_fingerprint",
        )
        path = self.invocation_directory / filename
        _write_json_create_only(path, receipt)
        self._closed = True
        return validate_run_receipt(path)


def validate_run_receipt(
    receipt_path: str | Path,
    *,
    expected_output_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Independently revalidate a completed or failed invocation receipt."""

    path = Path(receipt_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != TELEMETRY_VERSION:
        raise RuntimeError(f"telemetry receipt schema mismatch: {path}")
    if payload.get("receipt_fingerprint") != _fingerprint_without(
        payload,
        "receipt_fingerprint",
    ):
        raise RuntimeError(f"telemetry receipt fingerprint mismatch: {path}")

    event_evidence = payload.get("event_log") or {}
    event_log = path.parent / str(event_evidence.get("path"))
    if (
        not event_log.is_file()
        or _sha256(event_log) != event_evidence.get("physical_sha256")
        or event_log.stat().st_size
        != int(event_evidence.get("physical_bytes") or -1)
    ):
        raise RuntimeError(f"telemetry event log checksum mismatch: {event_log}")
    final_event, event_count = _validate_event_chain(event_log)
    if (
        event_count != int(event_evidence.get("event_count") or -1)
        or final_event != event_evidence.get("final_event_sha256")
    ):
        raise RuntimeError(f"telemetry event chain mismatch: {event_log}")

    status = payload.get("status")
    if status == "completed":
        output = payload.get("output") or {}
        manifest = Path(str(output.get("manifest_path"))).resolve()
        if expected_output_manifest is not None and manifest != Path(
            expected_output_manifest
        ).resolve():
            raise RuntimeError("telemetry output manifest path mismatch")
        if (
            not manifest.is_file()
            or _sha256(manifest) != output.get("manifest_sha256")
        ):
            raise RuntimeError(
                f"telemetry output manifest checksum mismatch: {manifest}"
            )
        expected_terminal_event = "run_completed"
    elif status == "failed":
        expected_terminal_event = "run_failed"
    else:
        raise RuntimeError(f"unknown telemetry receipt status: {status!r}")
    records = event_log.read_text(encoding="utf-8").splitlines()
    terminal = json.loads(records[-1])
    if terminal.get("event") != expected_terminal_event:
        raise RuntimeError("telemetry event log has the wrong terminal event")
    if path.stat().st_mtime_ns < event_log.stat().st_mtime_ns:
        raise RuntimeError("telemetry receipt was not written last")
    validation = payload.get("validation") or {}
    if not validation or not all(validation.values()):
        raise RuntimeError("telemetry receipt validation is not PASS")
    return payload


def _validate_event_chain(path: Path) -> tuple[str | None, int]:
    previous: str | None = None
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise RuntimeError("telemetry event log contains a blank line")
            record = json.loads(line)
            if (
                record.get("schema_version") != TELEMETRY_VERSION
                or record.get("event_index") != count
                or record.get("previous_event_sha256") != previous
                or record.get("event_sha256")
                != _fingerprint_without(record, "event_sha256")
            ):
                raise RuntimeError(
                    f"telemetry event chain is invalid at index {count}"
                )
            previous = str(record["event_sha256"])
            count += 1
    if not count:
        raise RuntimeError("telemetry event log is empty")
    return previous, count


def _process_io() -> dict[str, int] | None:
    try:
        values = {
            key: int(value.strip())
            for key, value in (
                line.split(":", 1)
                for line in Path("/proc/self/io").read_text(
                    encoding="utf-8"
                ).splitlines()
            )
        }
    except (FileNotFoundError, OSError, ValueError):
        return None
    return {
        key: values[key]
        for key in (
            "rchar",
            "wchar",
            "syscr",
            "syscw",
            "read_bytes",
            "write_bytes",
            "cancelled_write_bytes",
        )
        if key in values
    }


def _io_delta(
    start: Mapping[str, int] | None,
    end: Mapping[str, int] | None,
) -> dict[str, int] | None:
    if start is None or end is None or set(start) != set(end):
        return None
    return {key: int(end[key]) - int(start[key]) for key in start}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"gbif-final-bounded-{timestamp}-{uuid4().hex[:12]}"


def _elapsed_seconds(started_ns: int) -> float:
    return round((time.monotonic_ns() - started_ns) / 1_000_000_000, 6)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalized(value: object) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _normalized(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _fingerprint_without(
    value: Mapping[str, object],
    excluded_key: str,
) -> str:
    return canonical_semantic_fingerprint(
        {
            key: item
            for key, item in value.items()
            if key != excluded_key
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json_create_only(
    path: Path,
    value: Mapping[str, object],
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BoundedRunTelemetry",
    "FAILURE_RECEIPT",
    "SUCCESS_RECEIPT",
    "TELEMETRY_VERSION",
    "validate_run_receipt",
]
