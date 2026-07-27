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


def _report(total,rows,strata,gates):
    blocked=sum(row["rights_blocked"] for row in rows)
    lines=["# GBIF media URL pilot preflight","",f"- Pilot rows: {total:,}",f"- Network-eligible: {total-blocked:,}",f"- Rights-blocked prechecks: {blocked:,}","- Network requests: 0","- Overall gate: NOT_TESTED","","## Stratum coverage",""]
    lines += [f"- {key}: {value:,} unique values" for key,value in strata.items()]
    lines += ["","## Acceptance gates",""] + [f"- {a} — {b}: **{c}** ({d})" for a,b,c,d in gates]
    return "\n".join(lines)+"\n"


def _artifact(path):
    p=pq.ParquetFile(path); return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}


def _write_json(path,value):
    temp=path.with_suffix(".json.tmp"); temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(temp,path)


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


__all__=["GATE_SCHEMA","PILOT_AUDIT_VERSION","REVIEW_CATEGORIES","REVIEW_SCHEMA","publish_pilot_preflight_audit","wilson_interval"]
