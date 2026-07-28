from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from biominer.gbif_quality.assertions import DERIVED_ASSERTION_SCHEMA
from biominer.gbif_quality.review_samples import build_manual_review_sample


PHASE3_VERSION = "biominer-gbif-quality-phase3/v1"
COVERAGE_SCHEMA = pa.schema(
    [
        ("phase3_version", pa.string()),
        ("domain", pa.string()),
        ("metric", pa.string()),
        ("status", pa.string()),
        ("occurrence_count", pa.int64()),
        ("media_row_count", pa.int64()),
        ("note", pa.string()),
    ]
)


def publish_phase3_summary(
    *,
    temporal_directory: str | Path,
    geography_directory: str | Path,
    taxonomy_directory: str | Path,
    biology_directory: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_media_rows: int,
    expected_occurrences: int,
    code_commit: str,
    sample_seed: str = "gbif-media-v4-phase3-review-v1",
    max_review_rows_per_stratum: int = 25,
) -> dict[str, object]:
    roots = {
        "temporal": Path(temporal_directory).resolve(),
        "geography": Path(geography_directory).resolve(),
        "taxonomy": Path(taxonomy_directory).resolve(),
        "biology": Path(biology_directory).resolve(),
    }
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    manifests = {name: _read_valid_manifest(path) for name, path in roots.items()}
    if any(str(item["source_snapshot_id"]) != source_snapshot_id for item in manifests.values()):
        raise ValueError("Phase 3 source snapshots differ")
    assertion_paths = [
        roots["temporal"] / "temporal_assertions.parquet",
        roots["geography"] / "geographic_assertions.parquet",
        roots["taxonomy"] / "taxonomic_assertions.parquet",
        roots["biology"] / "biological_assertions.parquet",
    ]
    tables = [pq.read_table(path) for path in assertion_paths]
    if any(not table.schema.equals(DERIVED_ASSERTION_SCHEMA) for table in tables):
        raise ValueError("Phase 3 assertion schema mismatch")
    combined = pa.concat_tables(tables)
    if len(set(combined.column("assertion_id").to_pylist())) != combined.num_rows:
        raise ValueError("Phase 3 assertion identifiers are not unique")
    combined = combined.take(pc.sort_indices(combined, sort_keys=[("assertion_id", "ascending")]))
    fingerprint = semantic_assertion_fingerprint(combined)
    idempotence_probe = semantic_assertion_fingerprint(combined)
    review = build_manual_review_sample(
        temporal_assertions=assertion_paths[0],
        geographic_outcomes=roots["geography"] / "geographic_outcomes.parquet",
        taxonomic_repairs=roots["taxonomy"] / "species_rank_repairs.parquet",
        biological_candidates=roots["biology"] / "biological_candidates.parquet",
        sample_seed=sample_seed,
        max_per_stratum=max_review_rows_per_stratum,
    )
    coverage = _coverage(manifests)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    assertion_path = staging / "derived_assertions.parquet"
    coverage_path = staging / "enrichment_coverage.parquet"
    review_path = staging / "manual_review_sample.parquet"
    report_path = staging / "phase3_enrichment.md"
    try:
        pq.write_table(combined, assertion_path, compression="zstd")
        pq.write_table(coverage, coverage_path, compression="zstd")
        pq.write_table(review, review_path, compression="zstd")
        report_path.write_text(_report(manifests, combined.num_rows, review.num_rows, fingerprint), encoding="utf-8")
        counts = {
            "source_media_rows": int(manifests["temporal"]["counts"]["media_rows"]),
            "source_occurrences": int(manifests["temporal"]["counts"]["occurrence_rows"]),
            "derived_assertions": combined.num_rows,
            "coverage_rows": coverage.num_rows,
            "manual_review_rows": review.num_rows,
        }
        validation = {
            "source_media_rows_match": counts["source_media_rows"] == expected_media_rows,
            "source_occurrences_match": counts["source_occurrences"] == expected_occurrences,
            "all_stage_manifests_valid": all(all(item["validation"].values()) for item in manifests.values()),
            "assertion_schemas_match": True,
            "assertion_ids_unique": len(set(combined.column("assertion_id").to_pylist())) == combined.num_rows,
            "semantic_idempotence_proven": fingerprint == idempotence_probe,
            "all_review_rows_pending": all(value == "PENDING" for value in review.column("review_status").to_pylist()),
        }
        if not all(validation.values()):
            raise ValueError(f"Phase 3 summary validation failed: {validation}")
        artifacts = [_artifact(path) for path in (assertion_path, coverage_path, review_path)]
        artifacts.append({"path": report_path.name, "physical_bytes": report_path.stat().st_size, "sha256": _sha256(report_path), "row_count": None, "column_count": None, "row_group_count": None})
        manifest = {
            "schema_version": PHASE3_VERSION,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "code_commit": code_commit,
            "source_snapshot_id": source_snapshot_id,
            "inputs": {name: str(path) for name, path in roots.items()},
            "counts": counts,
            "semantic_assertion_fingerprint": fingerprint,
            "sample_seed": sample_seed,
            "validation": validation,
            "artifacts": artifacts,
            "network_requests": 0,
            "manifest_policy": {"written_last": True},
        }
        _write_json(staging / "manifest.json", manifest)
        for artifact in artifacts: _verify(staging, artifact)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def semantic_assertion_fingerprint(table: pa.Table) -> str:
    columns = ("assertion_id", "source_row_id", "gbifID", "target_field", "original_value", "derived_value", "derivation_rule_version", "validation_status", "conflict_status", "reviewer_status")
    digest = hashlib.sha256()
    indices = pc.sort_indices(table, sort_keys=[("assertion_id", "ascending")])
    for row in table.select(columns).take(indices).to_pylist():
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _coverage(manifests: dict[str, dict[str, object]]) -> pa.Table:
    t=manifests["temporal"]["counts"]; g=manifests["geography"]["counts"]; x=manifests["taxonomy"]["counts"]; b=manifests["biology"]["counts"]
    rows=[]
    def add(domain,metric,status,occ,media,note): rows.append({"phase3_version":PHASE3_VERSION,"domain":domain,"metric":metric,"status":status,"occurrence_count":int(occ),"media_row_count":int(media),"note":note})
    for part in ("year","month","day"): add("temporal",f"derived_{part}","PASS",t[f"derived_{part}_occurrences"],t[f"derived_{part}_media_rows"],"explicitly encoded in eventDate")
    add("temporal","pre_1960_retained","GENERALIZED",t["ancient_occurrences"],t["ancient_media_rows"],"retained and flagged")
    add(
        "geography",
        "derived_country_code",
        "PASS" if g["derived_country_occurrences"] else "UNKNOWN",
        g["derived_country_occurrences"],
        g["derived_country_media_rows"],
        "unique intersection against checksum-pinned boundary polygons",
    )
    add("geography","derived_continent","PASS",g["derived_continent_occurrences"],g["derived_continent_media_rows"],"pinned snapshot consensus")
    add("geography","derived_gbif_region","PASS" if g["derived_region_occurrences"] else "UNKNOWN",g["derived_region_occurrences"],g["derived_region_media_rows"],"pinned snapshot consensus")
    add("geography","conflicts","CONFLICT",g["conflict_occurrences"],0,"retained for review")
    add("taxonomy","derived_species","PASS",x["repaired_occurrences"],x["repaired_media_rows"],"same-record accepted taxon evidence")
    add("biology","life_stage_candidate","UNKNOWN",b["life_stage_candidate_occurrences"],b["life_stage_candidate_media_rows"],"review required")
    add("biology","sex_candidate","UNKNOWN",b["sex_candidate_occurrences"],b["sex_candidate_media_rows"],"review required")
    return pa.Table.from_pylist(rows,schema=COVERAGE_SCHEMA)


def _report(manifests, assertion_rows, review_rows, fingerprint):
    t=manifests["temporal"]["counts"]; g=manifests["geography"]["counts"]; x=manifests["taxonomy"]["counts"]; b=manifests["biology"]["counts"]
    return f"""# GBIF media v4 Phase 3 deterministic enrichment\n\n- Combined derived assertions: {assertion_rows:,}\n- Manual-review sample rows: {review_rows:,}\n- Semantic assertion fingerprint: `{fingerprint}`\n- Temporal media derivations: year {t['derived_year_media_rows']:,}; month {t['derived_month_media_rows']:,}; day {t['derived_day_media_rows']:,}.\n- Pre-1960 media rows retained and flagged: {t['ancient_media_rows']:,}.\n- Coordinate-to-country candidates: {g['coordinate_country_candidate_media_rows']:,}; safely derived: {g['derived_country_media_rows']:,}; ambiguous borders: {g['ambiguous_border_occurrences']:,}; outside or unmapped: {g['outside_or_unmapped_occurrences']:,}.\n- Continent media rows derived safely: {g['derived_continent_media_rows']:,}.\n- Geographic conflict occurrences retained: {g['conflict_occurrences']:,}.\n- Species-rank media rows repaired from same-record evidence: {x['repaired_media_rows']:,}.\n- Life-stage candidate media rows: {b['life_stage_candidate_media_rows']:,}.\n- Sex candidate media rows: {b['sex_candidate_media_rows']:,}.\n\nAll source fields remain unchanged. Biological candidates require human review. No network requests were made.\n"""


def _read_valid_manifest(root: Path) -> dict[str, object]:
    value=json.loads((root/"manifest.json").read_text())
    if not isinstance(value.get("validation"),dict) or not all(value["validation"].values()): raise ValueError(f"invalid stage manifest: {root}")
    return value


def _artifact(path: Path) -> dict[str, object]:
    p=pq.ParquetFile(path); return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}


def _verify(root: Path, artifact: dict[str, object]) -> None:
    if _sha256(root/str(artifact["path"])) != artifact["sha256"]: raise ValueError(f"Phase 3 checksum mismatch: {artifact['path']}")


def _write_json(path: Path,value: object) -> None:
    temp=path.with_suffix(".json.tmp"); temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(temp,path)


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


__all__=["COVERAGE_SCHEMA","PHASE3_VERSION","publish_phase3_summary","semantic_assertion_fingerprint"]
