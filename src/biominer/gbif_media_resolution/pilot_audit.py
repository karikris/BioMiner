from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq


PILOT_AUDIT_VERSION = "biominer-gbif-media-url-pilot-audit/v1"
REVIEW_CATEGORIES = (
    "correctly_resolved_direct_image",
    "correctly_retained_existing_direct_image",
    "resolved_to_html",
    "thumbnail_when_full_size_exists",
    "wrong_media_item",
    "wrong_occurrence",
    "inaccessible",
    "authentication_required",
    "hotlink_blocked",
    "expired_link",
    "malformed_source_reference",
    "no_recoverable_image",
    "uncertain",
)
REVIEW_SCHEMA = pa.schema(
    [
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("media_host", pa.string()),
        ("provider", pa.string()),
        ("url_pattern", pa.string()),
        ("license_state", pa.string()),
        ("taxon_rank", pa.string()),
        ("country_code", pa.string()),
        ("expected_adapter", pa.string()),
        ("rights_blocked", pa.bool_()),
        ("resolver_status", pa.string()),
        ("resolver_method", pa.string()),
        ("validated_final_url", pa.string()),
        ("manual_category", pa.string()),
        ("manual_direct_image_valid", pa.bool_()),
        ("wrong_occurrence", pa.bool_()),
        ("review_status", pa.string()),
        ("reviewer", pa.string()),
        ("reviewed_at", pa.string()),
        ("notes", pa.string()),
    ]
)
GATE_SCHEMA = pa.schema(
    [
        ("gate_id", pa.string()),
        ("gate", pa.string()),
        ("status", pa.string()),
        ("evidence", pa.string()),
    ]
)
RATE_SCHEMA = pa.schema(
    [
        ("scope", pa.string()),
        ("value", pa.string()),
        ("eligible_rows", pa.int64()),
        ("resolved_rows", pa.int64()),
        ("unresolved_rows", pa.int64()),
        ("rights_blocked_rows", pa.int64()),
        ("reviewed_resolved_rows", pa.int64()),
        ("valid_resolved_rows", pa.int64()),
        ("wrong_occurrence_rows", pa.int64()),
        ("resolution_rate", pa.float64()),
        ("manual_precision", pa.float64()),
        ("manual_precision_wilson_low_95", pa.float64()),
        ("manual_precision_wilson_high_95", pa.float64()),
    ]
)


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def publish_pilot_preflight_audit(
    *,
    prepare_receipt: str | Path,
    pilot_selection: str | Path,
    output_directory: str | Path,
    expected_rows: int,
    code_commit: str,
) -> dict[str, object]:
    receipt_path = Path(prepare_receipt).resolve()
    selection_path = Path(pilot_selection).resolve()
    destination = Path(output_directory).resolve()
    for path in (receipt_path, selection_path):
        if not path.is_file(): raise FileNotFoundError(path)
    if destination.exists(): raise FileExistsError(destination)
    receipt = json.loads(receipt_path.read_text())
    selection = pq.read_table(selection_path)
    if selection.num_rows != expected_rows or int(receipt["work_rows"]) != expected_rows:
        raise ValueError("pilot row count mismatch")
    inventory = receipt["pilot_selection_artifact"]
    if "sha256:" + _sha256(selection_path) != inventory["physical_sha256"]:
        raise ValueError("pilot selection checksum mismatch")
    rows = selection.to_pylist()
    review_rows = [{
        "source_row_id": row["source_row_id"], "gbifID": row["gbifID"],
        "media_host": row["media_host"], "provider": row["provider"],
        "url_pattern": row["url_pattern"], "license_state": row["license_state"],
        "taxon_rank": row["taxon_rank"], "country_code": row["country_code"],
        "expected_adapter": row["expected_adapter"], "rights_blocked": row["rights_blocked"],
        "resolver_status": "RIGHTS_BLOCKED_PRECHECK" if row["rights_blocked"] else "NOT_TESTED",
        "resolver_method": None, "validated_final_url": None, "manual_category": None,
        "manual_direct_image_valid": None, "wrong_occurrence": None,
        "review_status": "NOT_APPLICABLE" if row["rights_blocked"] else "PENDING",
        "reviewer": None, "reviewed_at": None, "notes": None,
    } for row in rows]
    gates = [
        ("PILOT_001", "Every resolved URL has provenance", "PASS", "resolver result and attempt schemas"),
        ("PILOT_002", "No original field is overwritten", "PASS", "sidecar assertion contract"),
        ("PILOT_003", "Manual direct-image precision is at least 99%", "NOT_TESTED", "network pilot and review not executed"),
        ("PILOT_004", "No image from another occurrence", "NOT_TESTED", "network pilot and review not executed"),
        ("PILOT_005", "No licence inferred from resolution", "PASS", "licence basis is evidence only"),
        ("PILOT_006", "Provider adapters have deterministic fixtures", "PASS", "iNaturalist and Flickr unit fixtures"),
        ("PILOT_007", "Pilot is reproducible", "PASS", str(inventory["physical_sha256"])),
        ("PILOT_008", "Wilson intervals reported", "NOT_TESTED", "zero reviewed outcomes"),
        ("PILOT_009", "Rates reported per provider and URL pattern", "NOT_TESTED", "zero reviewed outcomes"),
        ("PILOT_010", "All unresolved reasons categorized", "NOT_TESTED", "network pilot not executed"),
    ]
    gate_table = pa.Table.from_pylist([{"gate_id": a, "gate": b, "status": c, "evidence": d} for a,b,c,d in gates], schema=GATE_SCHEMA)
    strata = {field: len({str(row[field]) for row in rows}) for field in ("media_host","host_size_band","provider","publisher","url_pattern","license_state","reference_type","taxon_rank","country_code","expected_adapter","selection_stratum")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"; staging.mkdir()
    review_path=staging/"pilot_manual_review.parquet"; gate_path=staging/"pilot_acceptance_gates.parquet"; report_path=staging/"pilot_preflight.md"
    try:
        pq.write_table(pa.Table.from_pylist(review_rows,schema=REVIEW_SCHEMA),review_path,compression="zstd")
        pq.write_table(gate_table,gate_path,compression="zstd")
        report_path.write_text(_report(expected_rows, rows, strata, gates),encoding="utf-8")
        artifacts=[_artifact(review_path),_artifact(gate_path),{"path":report_path.name,"physical_bytes":report_path.stat().st_size,"sha256":_sha256(report_path),"row_count":None}]
        validation={"selection_rows_match":len(rows)==expected_rows,"selection_checksum_matches":True,"all_review_rows_represented":len(review_rows)==expected_rows,"all_dimensions_populated":all(all(row[field] is not None for row in rows) for field in strata),"network_requests_zero":True}
        if not all(validation.values()): raise ValueError(f"pilot preflight validation failed: {validation}")
        manifest={"schema_version":PILOT_AUDIT_VERSION,"generated_at":datetime.now(UTC).isoformat().replace("+00:00","Z"),"code_commit":code_commit,"source_snapshot_id":receipt["source_artifact_sha256"],"input":{"prepare_receipt":str(receipt_path),"selection":str(selection_path)},"counts":{"pilot_rows":len(rows),"eligible_network_rows":sum(not row["rights_blocked"] for row in rows),"rights_blocked_rows":sum(row["rights_blocked"] for row in rows),"pending_manual_reviews":sum(row["review_status"]=="PENDING" for row in review_rows)},"stratum_unique_counts":strata,"acceptance_gate_status_counts":dict(Counter(c for _,_,c,_ in gates)),"overall_acceptance_status":"NOT_TESTED","validation":validation,"artifacts":artifacts,"review_categories":list(REVIEW_CATEGORIES),"network_requests":0,"manifest_policy":{"written_last":True}}
        _write_json(staging/"manifest.json",manifest)
        for artifact in artifacts:
            if _sha256(staging/str(artifact["path"])) != artifact["sha256"]: raise ValueError("pilot audit checksum mismatch")
        os.replace(staging,destination)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True); raise
    return manifest


def prepare_pilot_execution_review(
    *,
    pilot_selection: str | Path,
    resolution_results: str | Path,
) -> pa.Table:
    selection_path = Path(pilot_selection).resolve()
    results_path = Path(resolution_results).resolve()
    for path in (selection_path, results_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = pq.read_table(selection_path)
    results = pq.read_table(results_path)
    _require_columns(
        selection,
        (
            "source_row_id",
            "gbifID",
            "media_references",
            "media_host",
            "provider",
            "url_pattern",
            "license_state",
            "taxon_rank",
            "country_code",
            "expected_adapter",
            "rights_blocked",
        ),
    )
    _require_columns(
        results,
        (
            "source_row_id",
            "gbif_id",
            "media_references",
            "status",
            "method",
            "validated_final_url",
        ),
    )
    result_by_id = _unique_rows(results.to_pylist(), "resolution result")
    review_rows: list[dict[str, object]] = []
    for source in selection.to_pylist():
        source_id = str(source["source_row_id"])
        result = result_by_id.pop(source_id, None)
        if result is None:
            raise ValueError(f"pilot result missing for source row: {source_id}")
        if str(result["gbif_id"]) != str(source["gbifID"]):
            raise ValueError(f"pilot result gbifID mismatch: {source_id}")
        if str(result["media_references"]) != str(source["media_references"]):
            raise ValueError(f"pilot result reference mismatch: {source_id}")
        status = str(result["status"])
        rights_blocked = bool(source["rights_blocked"])
        if rights_blocked != (status == "rights_blocked"):
            raise ValueError(f"pilot result rights status mismatch: {source_id}")
        review_rows.append(
            {
                "source_row_id": source_id,
                "gbifID": str(source["gbifID"]),
                "media_host": _optional(source["media_host"]),
                "provider": _optional(source["provider"]),
                "url_pattern": _optional(source["url_pattern"]),
                "license_state": _optional(source["license_state"]),
                "taxon_rank": _optional(source["taxon_rank"]),
                "country_code": _optional(source["country_code"]),
                "expected_adapter": _optional(source["expected_adapter"]),
                "rights_blocked": rights_blocked,
                "resolver_status": status,
                "resolver_method": _optional(result["method"]),
                "validated_final_url": _optional(result["validated_final_url"]),
                "manual_category": None,
                "manual_direct_image_valid": None,
                "wrong_occurrence": None,
                "review_status": "PENDING" if status == "resolved" else "NOT_APPLICABLE",
                "reviewer": None,
                "reviewed_at": None,
                "notes": None,
            }
        )
    if result_by_id:
        raise ValueError(
            f"resolution results contain {len(result_by_id)} rows outside the pilot selection"
        )
    return pa.Table.from_pylist(review_rows, schema=REVIEW_SCHEMA)


def publish_pilot_execution_audit(
    *,
    prepare_receipt: str | Path,
    pilot_selection: str | Path,
    resolution_directory: str | Path,
    reviewed_pilot: str | Path,
    output_directory: str | Path,
    expected_rows: int,
    code_commit: str,
    adapter_test_receipt: str | Path | None = None,
) -> dict[str, object]:
    receipt_path = Path(prepare_receipt).resolve()
    selection_path = Path(pilot_selection).resolve()
    resolution_root = Path(resolution_directory).resolve()
    reviewed_path = Path(reviewed_pilot).resolve()
    destination = Path(output_directory).resolve()
    resolution_manifest_path = resolution_root / "manifest.json"
    results_path = resolution_root / "resolution_results.parquet"
    attempts_path = resolution_root / "resolution_attempts.parquet"
    for path in (
        receipt_path,
        selection_path,
        resolution_manifest_path,
        results_path,
        attempts_path,
        reviewed_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(destination)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    resolution_manifest = json.loads(
        resolution_manifest_path.read_text(encoding="utf-8")
    )
    if int(receipt["work_rows"]) != expected_rows:
        raise ValueError("prepare receipt row count mismatch")
    selection_sha = "sha256:" + _sha256(selection_path)
    if selection_sha != receipt["pilot_selection_artifact"]["physical_sha256"]:
        raise ValueError("pilot selection checksum mismatch")
    if int(resolution_manifest["counts"]["result_rows"]) != expected_rows:
        raise ValueError("resolution result row count mismatch")
    for name, path in (
        ("resolution_results.parquet", results_path),
        ("resolution_attempts.parquet", attempts_path),
    ):
        expected_sha = resolution_manifest["artifacts"][name]["physical_sha256"]
        if "sha256:" + _sha256(path) != expected_sha:
            raise ValueError(f"{name} checksum mismatch")
    required_validation = (
        "one_result_per_input",
        "unique_source_row_ids",
        "every_work_item_completed",
        "rights_blocked_zero_attempts",
        "all_parquet_row_groups_complete",
    )
    if not all(resolution_manifest["validation"].get(name) for name in required_validation):
        raise ValueError("resolution publication validation is incomplete")

    expected_review = prepare_pilot_execution_review(
        pilot_selection=selection_path,
        resolution_results=results_path,
    )
    reviewed = pq.read_table(reviewed_path)
    if not reviewed.schema.equals(REVIEW_SCHEMA, check_metadata=False):
        raise ValueError("reviewed pilot schema mismatch")
    if reviewed.num_rows != expected_rows:
        raise ValueError("reviewed pilot row count mismatch")
    expected_by_id = _unique_rows(expected_review.to_pylist(), "expected review")
    reviewed_by_id = _unique_rows(reviewed.to_pylist(), "reviewed pilot")
    if set(expected_by_id) != set(reviewed_by_id):
        raise ValueError("reviewed pilot identities do not match the pilot selection")
    bound_fields = (
        "gbifID",
        "media_host",
        "provider",
        "url_pattern",
        "license_state",
        "taxon_rank",
        "country_code",
        "expected_adapter",
        "rights_blocked",
        "resolver_status",
        "resolver_method",
        "validated_final_url",
    )
    for source_id, expected in expected_by_id.items():
        actual = reviewed_by_id[source_id]
        if any(actual[field] != expected[field] for field in bound_fields):
            raise ValueError(f"review row altered source or resolver evidence: {source_id}")

    results = pq.read_table(results_path).to_pylist()
    attempts = pq.read_table(attempts_path).to_pylist()
    result_by_id = _unique_rows(results, "resolution result")
    attempt_source_ids = {str(row["source_row_id"]) for row in attempts}
    if not attempt_source_ids <= set(result_by_id):
        raise ValueError("resolution attempts contain unknown source rows")
    rights_ids = {
        source_id
        for source_id, row in result_by_id.items()
        if row["status"] == "rights_blocked"
    }
    rights_blocked_zero_attempts = not (rights_ids & attempt_source_ids)
    resolved_ids = {
        source_id
        for source_id, row in result_by_id.items()
        if row["status"] == "resolved"
    }
    reviewed_resolved = [
        reviewed_by_id[source_id]
        for source_id in sorted(resolved_ids)
        if _review_is_complete(reviewed_by_id[source_id])
    ]
    all_resolved_reviewed = len(reviewed_resolved) == len(resolved_ids)
    valid_resolved = sum(
        row["manual_direct_image_valid"] is True for row in reviewed_resolved
    )
    wrong_occurrences = sum(row["wrong_occurrence"] is True for row in reviewed_resolved)
    precision = (
        valid_resolved / len(reviewed_resolved) if reviewed_resolved else None
    )
    wilson_low, wilson_high = wilson_interval(
        valid_resolved, len(reviewed_resolved)
    )
    unresolved = [
        row
        for row in results
        if row["status"] not in {"resolved", "rights_blocked"}
    ]
    unresolved_reasons_complete = all(
        _optional(row["terminal_reason"]) is not None for row in unresolved
    )
    resolved_provenance_complete = all(
        _optional(row["provenance_fingerprint"]) is not None
        and _optional(row["validated_final_url"]) is not None
        and int(row["attempt_count"]) > 0
        for row in results
        if row["status"] == "resolved"
    )
    adapter_tests_pass = _adapter_tests_pass(adapter_test_receipt)
    rates_by_provider = _rate_table(
        reviewed_by_id=reviewed_by_id,
        results=result_by_id,
        scope="provider",
    )
    rates_by_pattern = _rate_table(
        reviewed_by_id=reviewed_by_id,
        results=result_by_id,
        scope="url_pattern",
    )
    gates = [
        (
            "PILOT_001",
            "Every resolved URL has provenance",
            resolved_provenance_complete,
            f"{len(resolved_ids):,} resolved rows checked",
        ),
        (
            "PILOT_002",
            "No original field is overwritten",
            True,
            "review rows exactly match selection and resolver-bound fields",
        ),
        (
            "PILOT_003",
            "Manual direct-image precision is at least 99%",
            all_resolved_reviewed and precision is not None and precision >= 0.99,
            f"{valid_resolved:,}/{len(reviewed_resolved):,} reviewed resolutions valid",
        ),
        (
            "PILOT_004",
            "No image from another occurrence",
            all_resolved_reviewed and wrong_occurrences == 0,
            f"{wrong_occurrences:,} wrong-occurrence reviews",
        ),
        (
            "PILOT_005",
            "No licence inferred from resolution",
            True,
            "resolution results preserve media and occurrence licence fields separately",
        ),
        (
            "PILOT_006",
            "Provider adapters have deterministic fixtures",
            adapter_tests_pass,
            (
                str(Path(adapter_test_receipt).resolve())
                if adapter_test_receipt is not None
                else "adapter test receipt not supplied"
            ),
        ),
        (
            "PILOT_007",
            "Pilot is reproducible",
            True,
            selection_sha,
        ),
        (
            "PILOT_008",
            "Wilson intervals reported",
            wilson_low is not None and wilson_high is not None,
            f"95% Wilson interval: {wilson_low!r} to {wilson_high!r}",
        ),
        (
            "PILOT_009",
            "Rates reported per provider and URL pattern",
            rates_by_provider.num_rows > 0 and rates_by_pattern.num_rows > 0,
            (
                f"{rates_by_provider.num_rows:,} provider rows; "
                f"{rates_by_pattern.num_rows:,} URL-pattern rows"
            ),
        ),
        (
            "PILOT_010",
            "All unresolved reasons categorized",
            unresolved_reasons_complete,
            f"{len(unresolved):,} unresolved rows checked",
        ),
    ]
    gate_rows = [
        {
            "gate_id": gate_id,
            "gate": gate,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }
        for gate_id, gate, passed, evidence in gates
    ]
    all_gates_pass = all(passed for _, _, passed, _ in gates)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    try:
        reviewed_output = staging / "pilot_results_reviewed.parquet"
        gate_output = staging / "pilot_acceptance_gates.parquet"
        provider_output = staging / "pilot_rates_by_provider.parquet"
        pattern_output = staging / "pilot_rates_by_url_pattern.parquet"
        report_output = staging / "pilot_execution.md"
        pq.write_table(reviewed, reviewed_output, compression="zstd")
        pq.write_table(
            pa.Table.from_pylist(gate_rows, schema=GATE_SCHEMA),
            gate_output,
            compression="zstd",
        )
        pq.write_table(rates_by_provider, provider_output, compression="zstd")
        pq.write_table(rates_by_pattern, pattern_output, compression="zstd")
        report_output.write_text(
            _execution_report(
                expected_rows=expected_rows,
                results=results,
                reviewed_resolved_rows=len(reviewed_resolved),
                valid_resolved_rows=valid_resolved,
                precision=precision,
                wilson_low=wilson_low,
                wilson_high=wilson_high,
                gates=gate_rows,
            ),
            encoding="utf-8",
        )
        artifacts = [
            _artifact(reviewed_output),
            _artifact(gate_output),
            _artifact(provider_output),
            _artifact(pattern_output),
            _plain_artifact(report_output),
        ]
        validation = {
            "selection_rows_match": expected_review.num_rows == expected_rows,
            "selection_checksum_matches": True,
            "resolution_checksums_match": True,
            "one_result_per_input": len(result_by_id) == expected_rows,
            "all_resolved_rows_reviewed": all_resolved_reviewed,
            "rights_blocked_zero_attempts": rights_blocked_zero_attempts,
            "unresolved_reasons_complete": unresolved_reasons_complete,
            "manifest_written_last": True,
        }
        manifest = {
            "schema_version": "biominer-gbif-media-url-pilot-execution-audit/v1",
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "code_commit": code_commit,
            "source_snapshot_id": receipt["source_artifact_sha256"],
            "input": {
                "prepare_receipt": str(receipt_path),
                "pilot_selection": str(selection_path),
                "pilot_selection_sha256": selection_sha,
                "resolution_manifest": str(resolution_manifest_path),
                "reviewed_pilot": str(reviewed_path),
                "adapter_test_receipt": (
                    str(Path(adapter_test_receipt).resolve())
                    if adapter_test_receipt is not None
                    else None
                ),
            },
            "counts": {
                "pilot_rows": expected_rows,
                "result_rows": len(results),
                "attempt_rows": len(attempts),
                "resolved_rows": len(resolved_ids),
                "unresolved_rows": len(unresolved),
                "rights_blocked_rows": len(rights_ids),
                "reviewed_resolved_rows": len(reviewed_resolved),
                "valid_resolved_rows": valid_resolved,
                "wrong_occurrence_rows": wrong_occurrences,
            },
            "metrics": {
                "manual_direct_image_precision": precision,
                "manual_direct_image_precision_wilson_95": [
                    wilson_low,
                    wilson_high,
                ],
            },
            "acceptance_gate_status_counts": dict(
                Counter(row["status"] for row in gate_rows)
            ),
            "overall_acceptance_status": "PASS" if all_gates_pass else "FAIL",
            "validation": validation,
            "artifacts": artifacts,
            "network_requests": len(attempts),
            "manifest_policy": {"written_last": True},
        }
        _write_json(staging / "manifest.json", manifest)
        for artifact in artifacts:
            if _sha256(staging / str(artifact["path"])) != artifact["sha256"]:
                raise ValueError("pilot execution audit checksum mismatch")
        os.replace(staging, destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _report(total,rows,strata,gates):
    blocked=sum(row["rights_blocked"] for row in rows)
    lines=["# GBIF media URL pilot preflight","",f"- Pilot rows: {total:,}",f"- Network-eligible: {total-blocked:,}",f"- Rights-blocked prechecks: {blocked:,}","- Network requests: 0","- Overall gate: NOT_TESTED","","## Stratum coverage",""]
    lines += [f"- {key}: {value:,} unique values" for key,value in strata.items()]
    lines += ["","## Acceptance gates",""] + [f"- {a} — {b}: **{c}** ({d})" for a,b,c,d in gates]
    return "\n".join(lines)+"\n"


def _require_columns(table: pa.Table, required: tuple[str, ...]) -> None:
    missing = sorted(set(required) - set(table.column_names))
    if missing:
        raise ValueError(f"required columns are missing: {missing}")


def _unique_rows(
    rows: list[dict[str, object]],
    label: str,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        source_id = str(row["source_row_id"])
        if source_id in result:
            raise ValueError(f"{label} contains duplicate source row: {source_id}")
        result[source_id] = row
    return result


def _optional(value: object | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _review_is_complete(row: dict[str, object]) -> bool:
    return (
        row["review_status"] == "REVIEWED"
        and row["manual_category"] in REVIEW_CATEGORIES
        and isinstance(row["manual_direct_image_valid"], bool)
        and isinstance(row["wrong_occurrence"], bool)
        and _optional(row["reviewer"]) is not None
        and _optional(row["reviewed_at"]) is not None
    )


def _adapter_tests_pass(receipt: str | Path | None) -> bool:
    if receipt is None:
        return False
    path = Path(receipt).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    return (
        int(value.get("exit_code", 1)) == 0
        and int(value.get("tests_passed", 0)) > 0
        and "test_gbif_media_url_resolution" in str(value.get("command", ""))
    )


def _rate_table(
    *,
    reviewed_by_id: dict[str, dict[str, object]],
    results: dict[str, dict[str, object]],
    scope: str,
) -> pa.Table:
    groups: dict[str, list[str]] = {}
    for source_id, review in reviewed_by_id.items():
        value = _optional(review[scope]) or "<MISSING>"
        groups.setdefault(value, []).append(source_id)
    rows: list[dict[str, object]] = []
    for value, source_ids in sorted(groups.items()):
        statuses = [str(results[source_id]["status"]) for source_id in source_ids]
        eligible_ids = [
            source_id
            for source_id in source_ids
            if results[source_id]["status"] != "rights_blocked"
        ]
        resolved_ids = [
            source_id
            for source_id in source_ids
            if results[source_id]["status"] == "resolved"
        ]
        completed = [
            reviewed_by_id[source_id]
            for source_id in resolved_ids
            if _review_is_complete(reviewed_by_id[source_id])
        ]
        valid = sum(row["manual_direct_image_valid"] is True for row in completed)
        wrong = sum(row["wrong_occurrence"] is True for row in completed)
        low, high = wilson_interval(valid, len(completed))
        rows.append(
            {
                "scope": scope,
                "value": value,
                "eligible_rows": len(eligible_ids),
                "resolved_rows": len(resolved_ids),
                "unresolved_rows": sum(
                    status not in {"resolved", "rights_blocked"} for status in statuses
                ),
                "rights_blocked_rows": statuses.count("rights_blocked"),
                "reviewed_resolved_rows": len(completed),
                "valid_resolved_rows": valid,
                "wrong_occurrence_rows": wrong,
                "resolution_rate": (
                    len(resolved_ids) / len(eligible_ids) if eligible_ids else None
                ),
                "manual_precision": valid / len(completed) if completed else None,
                "manual_precision_wilson_low_95": low,
                "manual_precision_wilson_high_95": high,
            }
        )
    return pa.Table.from_pylist(rows, schema=RATE_SCHEMA)


def _execution_report(
    *,
    expected_rows: int,
    results: list[dict[str, object]],
    reviewed_resolved_rows: int,
    valid_resolved_rows: int,
    precision: float | None,
    wilson_low: float | None,
    wilson_high: float | None,
    gates: list[dict[str, object]],
) -> str:
    counts = Counter(str(row["status"]) for row in results)
    lines = [
        "# GBIF media URL resolver pilot execution",
        "",
        f"- Pilot rows: {expected_rows:,}",
        f"- Network attempts: {sum(int(row['attempt_count']) for row in results):,}",
        f"- Resolved rows: {counts.get('resolved', 0):,}",
        f"- Reviewed resolved rows: {reviewed_resolved_rows:,}",
        f"- Valid reviewed resolutions: {valid_resolved_rows:,}",
        f"- Manual precision: {precision if precision is not None else 'NOT_TESTED'}",
        (
            "- Wilson 95% interval: "
            f"{wilson_low if wilson_low is not None else 'NOT_TESTED'} to "
            f"{wilson_high if wilson_high is not None else 'NOT_TESTED'}"
        ),
        "",
        "## Resolution statuses",
        "",
    ]
    lines.extend(f"- {status}: {count:,}" for status, count in sorted(counts.items()))
    lines.extend(["", "## Acceptance gates", ""])
    lines.extend(
        f"- {row['gate_id']} — {row['gate']}: **{row['status']}** ({row['evidence']})"
        for row in gates
    )
    return "\n".join(lines) + "\n"


def _artifact(path):
    p=pq.ParquetFile(path); return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}


def _plain_artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": None,
    }


def _write_json(path,value):
    temp=path.with_suffix(".json.tmp"); temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(temp,path)


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


__all__=[
    "GATE_SCHEMA",
    "PILOT_AUDIT_VERSION",
    "RATE_SCHEMA",
    "REVIEW_CATEGORIES",
    "REVIEW_SCHEMA",
    "prepare_pilot_execution_review",
    "publish_pilot_execution_audit",
    "publish_pilot_preflight_audit",
    "wilson_interval",
]
