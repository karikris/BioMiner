from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq


RECOVERY_VERSION = "biominer-gbif-restart-validation/v1"
STAGE_MANIFESTS = (
    "manifest.json",
    "source_lineage/manifest.json",
    "source_lineage/identity_v2/manifest.json",
    "occurrence_quality/manifest.json",
    "media_assertion_quality/manifest.json",
    "quality_results/phase2/manifest.json",
    "derived_assertions/temporal/manifest.json",
    "derived_assertions/geography/manifest.json",
    "derived_assertions/geography_v3/manifest.json",
    "derived_assertions/taxonomy/manifest.json",
    "derived_assertions/biology/manifest.json",
    "quality_results/phase3/manifest.json",
    "quality_results/phase3_v3/manifest.json",
    "quality_results/phase4_pilot_preflight/manifest.json",
    "rights_and_attribution/manifest.json",
    "duplicates/manifest.json",
    "ai_readiness/manifest.json",
    "representativeness/manifest.json",
    "representativeness_concentration/manifest.json",
    "media_resources/manifest.json",
    "completeness_gates/manifest.json",
    "quality_results/review_capsules/manifest.json",
    "incremental_state/manifest.json",
    "incremental_validation/manifest.json",
    "freshness/manifest.json",
    "provider_enrichment/manifest.json",
    "provider_enrichment_v2/manifest.json",
    "performance/manifest.json",
    "canonical_resource_analysis/v1/manifest.json",
)
SCHEMA = pa.schema(
    [
        ("recovery_version", pa.string()),
        ("stage", pa.string()),
        ("manifest_path", pa.string()),
        ("artifact_count", pa.int64()),
        ("artifact_checksums_verified", pa.bool_()),
        ("manifest_last_validated", pa.bool_()),
        ("restart_action", pa.string()),
        ("restart_status", pa.string()),
        ("restart_reason", pa.string()),
    ]
)


def publish_restart_validation(
    *, data_root: str | Path, output_directory: str | Path, code_commit: str
) -> dict[str, object]:
    """Verify committed-stage skipping and unchanged-row restart behaviour."""

    data = Path(data_root).resolve()
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    rows: list[dict[str, object]] = []
    for relative in STAGE_MANIFESTS:
        manifest_path = data / relative
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        root = manifest_path.parent
        artifacts = manifest.get("artifacts", manifest.get("artifact_inventory", []))
        verified = bool(artifacts) and all(
            _artifact_matches(root, artifact) for artifact in artifacts
        )
        manifest_last = bool(
            manifest.get("manifest_policy", {}).get("written_last")
            or manifest.get("validation", {}).get("manifest_written_last")
            or relative == "manifest.json"
        )
        status = "PASS" if verified and manifest_last else "FAIL"
        rows.append(
            {
                "recovery_version": RECOVERY_VERSION,
                "stage": relative.removesuffix("/manifest.json").removesuffix("manifest.json")
                or "baseline",
                "manifest_path": str(manifest_path),
                "artifact_count": len(artifacts),
                "artifact_checksums_verified": verified,
                "manifest_last_validated": manifest_last,
                "restart_action": "SKIP_COMMITTED" if status == "PASS" else "BLOCK_RESTART",
                "restart_status": status,
                "restart_reason": (
                    "manifest-bound committed output is complete"
                    if status == "PASS"
                    else "committed output failed manifest or checksum validation"
                ),
            }
        )
    incremental = json.loads((data / "incremental_validation/manifest.json").read_text())
    unchanged = bool(
        incremental["validation"].get("unchanged_rows_not_queued")
        and incremental["validation"].get("unchanged_rerun_semantically_identical")
        and incremental["counts"].get("queue_rows") == 0
    )
    rows.append(
        {
            "recovery_version": RECOVERY_VERSION,
            "stage": "incremental_unchanged_rows",
            "manifest_path": str(data / "incremental_validation/manifest.json"),
            "artifact_count": 1,
            "artifact_checksums_verified": unchanged,
            "manifest_last_validated": True,
            "restart_action": "SKIP_UNCHANGED" if unchanged else "BLOCK_RESTART",
            "restart_status": "PASS" if unchanged else "FAIL",
            "restart_reason": (
                "zero unchanged rows queued and semantic fingerprints match"
                if unchanged
                else "unchanged-row restart validation failed"
            ),
        }
    )
    orphaned = sorted(
        str(path) for path in data.rglob(".*.staging") if path.is_dir()
    )
    if orphaned:
        rows.append(
            {
                "recovery_version": RECOVERY_VERSION,
                "stage": "orphaned_staging_scan",
                "manifest_path": str(data),
                "artifact_count": len(orphaned),
                "artifact_checksums_verified": False,
                "manifest_last_validated": False,
                "restart_action": "REVIEW_ORPHANED_STAGING",
                "restart_status": "CONFLICT",
                "restart_reason": json.dumps(orphaned),
            }
        )
    else:
        rows.append(
            {
                "recovery_version": RECOVERY_VERSION,
                "stage": "orphaned_staging_scan",
                "manifest_path": str(data),
                "artifact_count": 0,
                "artifact_checksums_verified": True,
                "manifest_last_validated": True,
                "restart_action": "CONTINUE",
                "restart_status": "PASS",
                "restart_reason": "no orphaned staging directory detected",
            }
        )
    if any(row["restart_status"] != "PASS" for row in rows):
        raise ValueError("restart validation found incomplete or conflicting state")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    output = staging / "restart_validation.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), output)
    artifact = _artifact(output)
    baseline = json.loads((data / "manifest.json").read_text())
    manifest = {
        "schema_version": RECOVERY_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": baseline["source_snapshot_id"],
        "counts": {
            "validated_stages": len(rows),
            "skipped_committed_stages": sum(
                row["restart_action"] == "SKIP_COMMITTED" for row in rows
            ),
            "unchanged_rows_queued": incremental["counts"]["queue_rows"],
            "orphaned_staging_directories": len(orphaned),
        },
        "validation": {
            "all_committed_artifacts_checksummed": all(
                bool(row["artifact_checksums_verified"]) for row in rows
            ),
            "all_committed_stages_skippable": all(
                row["restart_status"] == "PASS" for row in rows
            ),
            "unchanged_rows_not_reprocessed": unchanged,
            "no_orphaned_staging": not orphaned,
            "manifest_written_last": True,
        },
        "artifacts": [artifact],
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    if _sha256(output) != artifact["sha256"]:
        raise ValueError("restart validation checksum mismatch")
    os.replace(staging, destination)
    return manifest


def _artifact_matches(root: Path, artifact: dict[str, object]) -> bool:
    relative = artifact.get("path")
    expected = artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        return False
    path = root / relative
    return path.is_file() and _sha256(path) == expected


def _artifact(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["RECOVERY_VERSION", "SCHEMA", "STAGE_MANIFESTS", "publish_restart_validation"]
