from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REVIEW_SAMPLE_VERSION = "biominer-gbif-manual-review-sample/v1"
REVIEW_SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_version", pa.string()),
        ("sample_id", pa.string()),
        ("sample_seed", pa.string()),
        ("review_domain", pa.string()),
        ("review_stratum", pa.string()),
        ("source_artifact", pa.string()),
        ("source_record_id", pa.string()),
        ("gbifID", pa.string()),
        ("target_field", pa.string()),
        ("candidate_value", pa.string()),
        ("review_reason", pa.string()),
        ("sample_hash", pa.string()),
        ("review_status", pa.string()),
    ]
)


def build_manual_review_sample(
    *,
    temporal_assertions: str | Path,
    geographic_outcomes: str | Path,
    taxonomic_repairs: str | Path,
    biological_candidates: str | Path,
    sample_seed: str,
    max_per_stratum: int = 25,
) -> pa.Table:
    if max_per_stratum < 1:
        raise ValueError("max_per_stratum must be positive")
    candidates: list[dict[str, str | None]] = []
    for row in pq.read_table(temporal_assertions).to_pylist():
        candidates.append(_candidate(
            domain="temporal",
            stratum=f"temporal|{row['target_field']}",
            artifact=str(Path(temporal_assertions).name),
            source_id=str(row["assertion_id"]), gbif_id=str(row["gbifID"]),
            target=str(row["target_field"]), value=_optional(row["derived_value"]),
            reason="deterministic_derivation_audit",
        ))
    for row in pq.read_table(geographic_outcomes).to_pylist():
        topic, target, value, reason = _geography_review(row)
        candidates.append(_candidate(
            domain="geography", stratum=f"geography|{topic}",
            artifact=str(Path(geographic_outcomes).name),
            source_id=str(row["source_row_id"]), gbif_id=str(row["gbifID"]),
            target=target, value=value, reason=reason,
        ))
    for row in pq.read_table(taxonomic_repairs).to_pylist():
        candidates.append(_candidate(
            domain="taxonomy",
            stratum=f"taxonomy|{row['derivation_status']}|{row['derived_species'] or 'unresolved'}",
            artifact=str(Path(taxonomic_repairs).name),
            source_id=str(row["source_row_id"]), gbif_id=str(row["gbifID"]),
            target="derived_species", value=_optional(row["derived_species"]),
            reason=str(row["derivation_reason"]),
        ))
    for row in pq.read_table(biological_candidates).to_pylist():
        candidates.append(_candidate(
            domain="biology",
            stratum="|".join(("biology", str(row["target_field"]), str(row["candidate_status"]), str(row["derived_value"] or "unresolved"))),
            artifact=str(Path(biological_candidates).name),
            source_id=str(row["candidate_id"]), gbif_id=str(row["gbifID"]),
            target=str(row["target_field"]), value=_optional(row["derived_value"]),
            reason=str(row["candidate_reason"]),
        ))
    groups: dict[str, list[dict[str, str | None]]] = defaultdict(list)
    for row in candidates:
        score = hashlib.sha256(
            "|".join((sample_seed, str(row["source_artifact"]), str(row["source_record_id"]), str(row["review_stratum"]))).encode()
        ).hexdigest()
        row["sample_hash"] = "sha256:" + score
        groups[str(row["review_stratum"])].append(row)
    selected = []
    for stratum in sorted(groups):
        selected.extend(sorted(groups[stratum], key=lambda row: (str(row["sample_hash"]), str(row["source_record_id"])))[:max_per_stratum])
    output = []
    for row in selected:
        identity = "|".join((REVIEW_SAMPLE_VERSION, sample_seed, str(row["source_record_id"]), str(row["review_stratum"])))
        output.append({
            "sample_version": REVIEW_SAMPLE_VERSION,
            "sample_id": "sha256:" + hashlib.sha256(identity.encode()).hexdigest(),
            "sample_seed": sample_seed,
            **row,
            "review_status": "PENDING",
        })
    output.sort(key=lambda row: (str(row["review_domain"]), str(row["review_stratum"]), str(row["sample_hash"])))
    return pa.Table.from_pylist(output, schema=REVIEW_SAMPLE_SCHEMA)


def _geography_review(row: dict[str, object]) -> tuple[str, str, str | None, str]:
    if row["geographic_conflict_status"] == "CONFLICT":
        return "conflict", "geographic_conflict", None, "source_consensus_conflict"
    if row["country_derivation_status"] == "NOT_TESTED":
        return "country_unresolved", "derived_countryCode", None, str(row["country_derivation_reason"])
    if row["continent_derivation_status"] == "UNKNOWN":
        return "continent_unresolved", "derived_continent", None, str(row["continent_derivation_reason"])
    if row["gbif_region_derivation_status"] == "UNKNOWN":
        return "region_unresolved", "derived_gbifRegion", None, str(row["gbif_region_derivation_reason"])
    if row["derived_continent"] is not None:
        return "continent_derived", "derived_continent", str(row["derived_continent"]), "deterministic_derivation_audit"
    return "region_derived", "derived_gbifRegion", _optional(row["derived_gbifRegion"]), "deterministic_derivation_audit"


def _candidate(*, domain: str, stratum: str, artifact: str, source_id: str, gbif_id: str, target: str, value: str | None, reason: str) -> dict[str, str | None]:
    return {"review_domain": domain, "review_stratum": stratum, "source_artifact": artifact, "source_record_id": source_id, "gbifID": gbif_id, "target_field": target, "candidate_value": value, "review_reason": reason}


def _optional(value: object | None) -> str | None:
    return None if value is None else str(value)


__all__ = ["REVIEW_SAMPLE_SCHEMA", "REVIEW_SAMPLE_VERSION", "build_manual_review_sample"]
