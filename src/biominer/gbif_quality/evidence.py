from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import pyarrow.parquet as pq


EVIDENCE_AUDIT_VERSION = "biominer-gbif-evidence-audit/v1"


def audit_evidence_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
    expected_source_snapshot_id: str | None = None,
    current_git_commit: str | None = None,
) -> dict[str, Any]:
    """Independently verify one evidence manifest and every artifact it names.

    ``current_git_commit`` is accepted only to make the non-dependency explicit:
    repository HEAD is not an invalidation key. Artifact bytes, recorded source
    identity, schema, row counts, and the producer commit are.
    """

    del current_git_commit
    path = Path(manifest_path).resolve()
    repository = Path(repository_root).resolve()
    failures: list[str] = []
    base = {
        "audit_version": EVIDENCE_AUDIT_VERSION,
        "manifest_path": str(path),
        "manifest_sha256": None,
        "schema_version": None,
        "producer_commit": None,
        "producer_commit_status": "FAIL",
        "source_snapshot_id": None,
        "source_snapshot_status": (
            "FAIL" if expected_source_snapshot_id is not None else "NOT_APPLICABLE"
        ),
        "artifact_count": 0,
        "artifact_pass_count": 0,
        "artifact_fail_count": 0,
        "manifest_last_status": "FAIL",
        "dependency_fingerprint": None,
        "artifacts": [],
        "failure_reasons": failures,
        "status": "FAIL",
    }
    if not path.is_file():
        failures.append("manifest_missing")
        return base
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        failures.append("manifest_unreadable")
        return base
    if not isinstance(manifest, dict):
        failures.append("manifest_not_an_object")
        return base

    base["manifest_sha256"] = _prefixed_sha256(path)
    schema_version = manifest.get("schema_version")
    base["schema_version"] = schema_version
    if not isinstance(schema_version, str) or not schema_version.strip():
        failures.append("schema_version_missing")

    producer_commit = _producer_commit(manifest)
    base["producer_commit"] = producer_commit
    if producer_commit is None:
        failures.append("producer_commit_missing")
    elif _commit_exists(repository, producer_commit):
        base["producer_commit_status"] = "PASS"
    else:
        failures.append("producer_commit_unavailable")

    source_snapshot = manifest.get("source_snapshot_id")
    base["source_snapshot_id"] = source_snapshot
    if expected_source_snapshot_id is not None:
        if source_snapshot == expected_source_snapshot_id:
            base["source_snapshot_status"] = "PASS"
        else:
            failures.append("source_snapshot_mismatch")

    entries = list(_artifact_entries(manifest))
    base["artifact_count"] = len(entries)
    if not entries:
        failures.append("artifact_inventory_missing")
    artifacts = [
        _audit_artifact(entry, manifest_directory=path.parent)
        for entry in entries
    ]
    base["artifacts"] = artifacts
    base["artifact_pass_count"] = sum(
        artifact["status"] == "PASS" for artifact in artifacts
    )
    base["artifact_fail_count"] = sum(
        artifact["status"] == "FAIL" for artifact in artifacts
    )
    for artifact in artifacts:
        failures.extend(artifact["failure_reasons"])

    manifest_mtime = path.stat().st_mtime_ns
    artifact_mtimes = [
        int(artifact["mtime_ns"])
        for artifact in artifacts
        if artifact["mtime_ns"] is not None
    ]
    if artifact_mtimes and manifest_mtime >= max(artifact_mtimes):
        base["manifest_last_status"] = "PASS"
    else:
        failures.append("manifest_not_written_last")

    failures[:] = list(dict.fromkeys(failures))
    fingerprint_payload = {
        "schema_version": schema_version,
        "producer_commit": producer_commit,
        "source_snapshot_id": source_snapshot,
        "manifest_sha256": base["manifest_sha256"],
        "artifacts": [
            {
                "path": artifact["path"],
                "observed_sha256": artifact["observed_sha256"],
                "row_count": artifact["observed_row_count"],
                "column_count": artifact["observed_column_count"],
                "row_group_rows": artifact["row_group_rows"],
            }
            for artifact in artifacts
        ],
    }
    base["dependency_fingerprint"] = _semantic_fingerprint(
        fingerprint_payload
    )
    base["status"] = "PASS" if not failures else "FAIL"
    return base


def _artifact_entries(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for field in ("artifacts", "artifact_inventory", "reports"):
        raw = manifest.get(field)
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict):
                    yield entry
        elif isinstance(raw, dict):
            for key in sorted(raw):
                entry = raw[key]
                if isinstance(entry, dict):
                    yield entry


def _audit_artifact(
    entry: dict[str, Any],
    *,
    manifest_directory: Path,
) -> dict[str, Any]:
    raw_path = entry.get("path")
    artifact_path = (
        Path(str(raw_path))
        if raw_path is not None
        else manifest_directory / "__missing_artifact_path__"
    )
    if not artifact_path.is_absolute():
        artifact_path = manifest_directory / artifact_path
    artifact_path = artifact_path.resolve()
    failures: list[str] = []
    result: dict[str, Any] = {
        "path": str(artifact_path),
        "recorded_sha256": _normalize_sha256(
            entry.get("sha256") or entry.get("physical_sha256")
        ),
        "observed_sha256": None,
        "checksum_status": "FAIL",
        "recorded_bytes": entry.get("physical_bytes"),
        "observed_bytes": None,
        "byte_count_status": "NOT_APPLICABLE",
        "recorded_row_count": entry.get("row_count"),
        "observed_row_count": None,
        "recorded_column_count": entry.get("column_count"),
        "observed_column_count": None,
        "recorded_row_group_count": entry.get("row_group_count"),
        "observed_row_group_count": None,
        "row_group_rows": None,
        "row_group_status": "NOT_APPLICABLE",
        "mtime_ns": None,
        "failure_reasons": failures,
        "status": "FAIL",
    }
    if raw_path is None:
        failures.append("artifact_path_missing")
        return result
    if not artifact_path.is_file():
        failures.append("artifact_missing")
        return result

    result["mtime_ns"] = artifact_path.stat().st_mtime_ns
    result["observed_bytes"] = artifact_path.stat().st_size
    expected_bytes = entry.get("physical_bytes")
    if expected_bytes is not None:
        if int(expected_bytes) == result["observed_bytes"]:
            result["byte_count_status"] = "PASS"
        else:
            result["byte_count_status"] = "FAIL"
            failures.append("artifact_byte_count_mismatch")

    observed_sha = _prefixed_sha256(artifact_path)
    result["observed_sha256"] = observed_sha
    expected_sha = result["recorded_sha256"]
    if expected_sha is not None and expected_sha == observed_sha:
        result["checksum_status"] = "PASS"
    else:
        failures.append(
            "artifact_checksum_missing"
            if expected_sha is None
            else "artifact_checksum_mismatch"
        )

    if artifact_path.suffix.casefold() == ".parquet":
        try:
            parquet = pq.ParquetFile(artifact_path)
            metadata = parquet.metadata
            row_group_rows = [
                metadata.row_group(index).num_rows
                for index in range(metadata.num_row_groups)
            ]
            result["observed_row_count"] = metadata.num_rows
            result["observed_column_count"] = len(parquet.schema_arrow)
            result["observed_row_group_count"] = metadata.num_row_groups
            result["row_group_rows"] = row_group_rows
            row_groups_complete = (
                sum(row_group_rows) == metadata.num_rows
                and (
                    (
                        metadata.num_rows == 0
                        and all(rows == 0 for rows in row_group_rows)
                    )
                    or (
                        metadata.num_rows > 0
                        and metadata.num_row_groups > 0
                        and all(rows > 0 for rows in row_group_rows)
                    )
                )
            )
            metadata_matches = all(
                recorded is None or int(recorded) == observed
                for recorded, observed in (
                    (entry.get("row_count"), metadata.num_rows),
                    (entry.get("column_count"), len(parquet.schema_arrow)),
                    (
                        entry.get("row_group_count"),
                        metadata.num_row_groups,
                    ),
                )
            )
            if row_groups_complete and metadata_matches:
                result["row_group_status"] = "PASS"
            else:
                result["row_group_status"] = "FAIL"
                if not row_groups_complete:
                    failures.append("parquet_row_groups_incomplete")
                if not metadata_matches:
                    failures.append("parquet_metadata_mismatch")
        except (OSError, ValueError, TypeError):
            result["row_group_status"] = "FAIL"
            failures.append("parquet_metadata_unreadable")

    result["status"] = "PASS" if not failures else "FAIL"
    return result


def _producer_commit(manifest: dict[str, Any]) -> str | None:
    for field in ("code_commit", "git_commit", "producer_git_sha"):
        value = manifest.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _commit_exists(repository: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _normalize_sha256(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().casefold()
    return (
        normalized
        if normalized.startswith("sha256:")
        else f"sha256:{normalized}"
    )


def _prefixed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _semantic_fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "EVIDENCE_AUDIT_VERSION",
    "audit_evidence_manifest",
]
