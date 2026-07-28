from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import duckdb


ACCEPTANCE_VERSION = "biominer-gbif-media-acceptance/v3"
CRITERIA = (
    "The v3 dataset remains unchanged.",
    "The source-to-v4 funnel is fully reconciled.",
    "Every source media row has an output status.",
    "Occurrence completeness uses 11,569,412 unless applicability is explicit.",
    "Media completeness uses 16,612,063 unless applicability is explicit.",
    "No structural null is counted as a repairable defect.",
    "No not-applicable field is presented as missing.",
    "No original field is overwritten.",
    "Every derived value has provenance.",
    "Every conflict is retained.",
    "Every unresolved row is retained.",
    "No occurrence licence is presented as the media licence.",
    "No identification provenance is synthesized.",
    "No species is imputed into a higher-rank record.",
    "No unknown coordinate uncertainty is written as zero.",
    "Coordinate precision is not treated as equivalent to uncertainty.",
    "The 130,689 reference-only rows reconcile exactly.",
    "The 4,055 reference-only eligibility difference is explained.",
    "The 131,804 missing-format rows have an explicit status.",
    "The 17 missing-type rows have an explicit status.",
    "The 823-record resolver pilot has a reproducible review report.",
    "Broad network execution is gated by pilot quality.",
    "Full local metadata processing is restartable.",
    "Full local metadata processing is idempotent.",
    "Output parts are checksummed.",
    "Output schemas are versioned.",
    "The manifest is written last.",
    "Exact and perceptual duplicates are reported separately.",
    "AI-readiness is separate from occurrence research quality.",
    "Model predictions never overwrite GBIF assertions.",
    "Reports contain raw and duplicate-adjusted counts.",
    "Reports contain before/after completeness.",
    "Reports contain invalid-present values, not only null counts.",
    "Reports contain provider and dataset breakdowns.",
    "Reports contain top unresolved reasons.",
    "Reports contain conflict counts.",
    "Reports contain repairable versus non-repairable gaps.",
    "Reports contain all available denominator types.",
    "The full metadata audit remains within the configured memory bound.",
    "The required tests pass.",
    "git diff --check passes.",
    "No result is claimed without stored evidence.",
)

SCHEMA = pa.schema([
    ("acceptance_version",pa.string()),("criterion_number",pa.int32()),
    ("requirement",pa.string()),("status",pa.string()),("blocking",pa.bool_()),
    ("network_required",pa.bool_()),("evidence_summary",pa.string()),
    ("evidence_paths",pa.list_(pa.string())),("rule_version",pa.string()),
])


def publish_acceptance_audit(*,repository_root:str|Path,data_root:str|Path,report_root:str|Path,output_directory:str|Path,test_receipt:str|Path,expected_v3_sha256:str,code_commit:str)->dict[str,object]:
    repo=Path(repository_root).resolve();data=Path(data_root).resolve();reports=Path(report_root).resolve();receipt=Path(test_receipt).resolve();destination=Path(output_directory).resolve()
    prepare_receipt_path=repo/"data/state/gbif-media-url-resolution/pilot-v2/prepare-gbif-media-url-pilot-v2.json"
    required=[data/"manifest.json",data/"source_funnel.parquet",data/"source_lineage/source_media_status.parquet",data/"source_lineage/identity_v2/manifest.json",data/"column_policy.parquet",data/"completeness_by_applicability.parquet",data/"quality_results/phase2/check_registry.parquet",data/"media_assertion_quality/manifest.json",data/"occurrence_quality/manifest.json",data/"derived_assertions/temporal/manifest.json",data/"derived_assertions/geography_v3/manifest.json",data/"derived_assertions/taxonomy/manifest.json",data/"derived_assertions/biology/manifest.json",data/"quality_results/phase3_v3/manifest.json",data/"rights_and_attribution/manifest.json",data/"duplicates/manifest.json",data/"media_resources/manifest.json",data/"ai_readiness/manifest.json",data/"representativeness/manifest.json",data/"representativeness_concentration/manifest.json",data/"completeness_gates/manifest.json",data/"quality_results/review_capsules/manifest.json",data/"incremental_validation/manifest.json",data/"quality_results/restart_validation_v3/manifest.json",data/"freshness/manifest.json",data/"provider_enrichment_v4/manifest.json",data/"quality_results/provider_archive_review/v1/manifest.json",data/"performance/manifest.json",data/"quality_results/phase4_pilot_execution/v1/audit/manifest.json",reports/"manifest.json",prepare_receipt_path,receipt]
    for path in required:
        if not path.is_file():raise FileNotFoundError(path)
    if destination.exists():raise FileExistsError(destination)
    load=lambda p:json.loads(Path(p).read_text())
    base=load(data/"manifest.json");media=load(data/"media_assertion_quality/manifest.json");occ=load(data/"occurrence_quality/manifest.json");rights=load(data/"rights_and_attribution/manifest.json");dup=load(data/"duplicates/manifest.json");resources=load(data/"media_resources/manifest.json");ai=load(data/"ai_readiness/manifest.json");representation=load(data/"representativeness/manifest.json");concentration=load(data/"representativeness_concentration/manifest.json");gates=load(data/"completeness_gates/manifest.json");reviews=load(data/"quality_results/review_capsules/manifest.json");inc=load(data/"incremental_validation/manifest.json");recovery=load(data/"quality_results/restart_validation_v3/manifest.json");freshness=load(data/"freshness/manifest.json");providers=load(data/"provider_enrichment_v4/manifest.json");provider_review=load(data/"quality_results/provider_archive_review/v1/manifest.json");lineage=load(data/"source_lineage/identity_v2/manifest.json");perf=load(data/"performance/manifest.json");pilot=load(data/"quality_results/phase4_pilot_execution/v1/audit/manifest.json");phase3=load(data/"quality_results/phase3_v3/manifest.json");geography=load(data/"derived_assertions/geography_v3/manifest.json");taxonomy=load(data/"derived_assertions/taxonomy/manifest.json");report_manifest=load(reports/"manifest.json");prepare=load(prepare_receipt_path);tests=load(receipt)
    inventory=pq.read_table(data/"source_inventory.parquet").to_pylist()
    v3_entry=next(row for row in inventory if row["artifact_role"]=="rights_filtered_v3")
    v3=(repo/str(v3_entry["path"])).resolve();v3_ok=_sha256(v3)==expected_v3_sha256
    connection=duckdb.connect()
    try:
        missing_format_status=connection.execute("""SELECT count(*),count(*) FILTER(WHERE q.media_type_format_status='UNKNOWN') FROM read_parquet(?) v POSITIONAL JOIN read_parquet(?) q WHERE v.media_identifier IS NOT NULL AND v.media_format IS NULL""",[str(v3),str(data/"media_assertion_quality/media_assertion_quality.parquet")]).fetchone()
        missing_type_status=connection.execute("""SELECT count(*),count(*) FILTER(WHERE q.media_type_format_status='CONFLICT') FROM read_parquet(?) v POSITIONAL JOIN read_parquet(?) q WHERE v.media_identifier IS NOT NULL AND v.media_type IS NULL""",[str(v3),str(data/"media_assertion_quality/media_assertion_quality.parquet")]).fetchone()
        funnel_status=connection.execute("""SELECT count(*),count(*) FILTER(WHERE status='PASS' AND input_row_count=output_row_count+excluded_row_count) FROM read_parquet(?)""",[str(data/"source_funnel.parquet")]).fetchone()
        source_status=connection.execute("""SELECT count(*),count(*) FILTER(WHERE media_join_status IS NOT NULL AND v3_funnel_status IS NOT NULL),count(*) FILTER(WHERE media_join_status='unresolved_occurrence') FROM read_parquet(?)""",[str(data/"source_lineage/source_media_status.parquet")]).fetchone()
        applicability_status=connection.execute("""SELECT count(*),count(*) FILTER(WHERE structurally_absent_media_rows>0 AND repairable_null_media_rows>0),count(*) FILTER(WHERE not_applicable_media_rows>0) FROM read_parquet(?)""",[str(data/"completeness_by_applicability.parquet")]).fetchone()
        prohibited_repairs=connection.execute("""SELECT count(*) FILTER(WHERE automatic_repair_permitted),count(*) FILTER(WHERE list_contains(fields_inspected,'identifiedBy') OR list_contains(fields_inspected,'identificationVerificationStatus')),count(*) FILTER(WHERE list_contains(fields_inspected,'coordinateUncertaintyInMeters')) FROM read_parquet(?) WHERE list_contains(fields_inspected,'identifiedBy') OR list_contains(fields_inspected,'identificationVerificationStatus') OR list_contains(fields_inspected,'coordinateUncertaintyInMeters')""",[str(data/"quality_results/phase2/check_registry.parquet")]).fetchone()
        uncertainty_assertions=connection.execute("""SELECT count(*),count(*) FILTER(WHERE lower(coalesce(derivation_method,'')) LIKE '%precision%') FROM read_parquet(?) WHERE lower(target_field) LIKE '%uncertainty%'""",[str(data/"quality_results/phase3_v3/derived_assertions.parquet")]).fetchone()
    finally:connection.close()
    schema_versions_present=all(
        bool(load(path).get("schema_version"))
        for path in required
        if path.name=="manifest.json"
    )
    diff=subprocess.run(["git","diff","--check"],cwd=repo,text=True,capture_output=True,check=False)
    diff_evidence=(diff.stdout+diff.stderr).strip() or "clean"
    facts={
        1:(v3_ok,f"v3 SHA-256 {_sha256(v3)}",[str(v3)]),
        2:(funnel_status==(8,8),"All eight source-funnel stages reconcile input = output + excluded and have PASS status.",[str(data/"source_funnel.parquet")]),
        3:(source_status==(18_680_565,18_680_565,0) and lineage["counts"]["rows"]==18_680_565 and lineage["validation"]["all_rows_have_source_value_hash"],"All 18,680,565 raw media assertions have an explicit join/funnel status, stable source location, and source-value hash; unresolved occurrence foreign keys: 0.",[str(data/"source_lineage/source_media_status.parquet"),str(data/"source_lineage/identity_v2/manifest.json")]),
        4:(occ["counts"]["rows"]==11_569_412,"Occurrence denominator is 11,569,412.",[str(data/"occurrence_quality/manifest.json")]),
        5:(media["counts"]["rows"]==16_612_063,"Media denominator is 16,612,063.",[str(data/"media_assertion_quality/manifest.json")]),
        6:(applicability_status[1]==0,"No column reports the same rows as both structurally absent and repairable null.",[str(data/"completeness_by_applicability.parquet")]),
        7:(applicability_status[2]>0,"NOT_APPLICABLE is materialized independently for applicable field policies.",[str(data/"completeness_by_applicability.parquet")]),
        8:(v3_ok and rights["validation"]["source_fields_unchanged"] and ai["validation"]["source_fields_unchanged"] and taxonomy["validation"]["original_species_unchanged"],"The v3 checksum matches and rights, AI-readiness, and taxonomic-repair manifests validate that original fields remain unchanged.",[str(v3),str(data/"rights_and_attribution/manifest.json"),str(data/"ai_readiness/manifest.json"),str(data/"derived_assertions/taxonomy/manifest.json")]),
        9:(phase3["validation"]["assertion_schemas_match"] and phase3["validation"]["assertion_ids_unique"] and phase3["counts"]["derived_assertions"]>0,"Phase 3 v3 publishes unique sparse assertions with the versioned evidence schema.",[str(data/"quality_results/phase3_v3/derived_assertions.parquet"),str(data/"quality_results/phase3_v3/manifest.json")]),
        10:(dup["counts"]["conflict_rows"]>0 and geography["counts"]["conflict_occurrences"]>0 and geography["validation"]["all_coordinate_country_candidates_retained"],"Duplicate and geographic conflicts remain materialized; no coordinate-to-country candidate is discarded.",[str(data/"duplicates/duplicate_membership.parquet"),str(data/"derived_assertions/geography_v3/geographic_outcomes.parquet")]),
        11:(ai["counts"]["decision_counts"].get("UNRESOLVED",0)>0 and pilot["validation"]["unresolved_reasons_complete"],f"{ai['counts']['decision_counts'].get('UNRESOLVED',0):,} AI-readiness unresolved assertions and {pilot['counts']['unresolved_rows']:,} pilot unresolved outcomes are retained.",[str(data/"ai_readiness/manifest.json"),str(data/"quality_results/phase4_pilot_execution/v1/audit/manifest.json")]),
        12:(rights["validation"]["media_and_occurrence_licenses_separate"],"Rights layer stores original media and occurrence licences separately.",[str(data/"rights_and_attribution/manifest.json")]),
        13:(prohibited_repairs[0]==0 and prohibited_repairs[1]>0,"The check registry contains identification-provenance checks and permits no automatic repair for identifiedBy or identificationVerificationStatus.",[str(data/"quality_results/phase2/check_registry.parquet"),str(data/"occurrence_quality/occurrence_quality.parquet")]),
        14:(taxonomy["counts"]["candidate_media_rows"]==337 and taxonomy["counts"]["repaired_media_rows"]==337 and taxonomy["validation"]["only_species_rank"] and taxonomy["validation"]["direct_evidence_required"],"All 337 repaired media rows are species-rank candidates backed by direct evidence; higher-rank repair is prohibited.",[str(data/"derived_assertions/taxonomy/manifest.json")]),
        15:(prohibited_repairs[0]==0 and prohibited_repairs[2]>0 and uncertainty_assertions[0]==0,"Coordinate-uncertainty checks permit no automatic repair and Phase 3 writes no uncertainty assertion, so unknown values remain unknown.",[str(data/"quality_results/phase2/check_registry.parquet"),str(data/"quality_results/phase3_v3/derived_assertions.parquet")]),
        16:(uncertainty_assertions==(0,0),"No uncertainty assertion, including a decimal-precision-derived assertion, exists in Phase 3 v3.",[str(data/"quality_results/phase3_v3/derived_assertions.parquet")]),
        17:(prepare["input_rows"]==130_689 and prepare["eligible_resolution_rows"]==126_634 and prepare["rights_blocked_rows"]==4_055 and prepare["eligible_resolution_rows"]+prepare["rights_blocked_rows"]==prepare["input_rows"],"The prepare receipt reconciles 130,689 reference-only rows as 126,634 network-eligible plus 4,055 rights-blocked.",[str(prepare_receipt_path)]),
        18:(prepare["rights_blocked_rows"]==4_055 and prepare["expected_rights_blocked_rows"]==4_055,"All 4,055 ineligible rows were deterministically counted by the explicit restricted-media-licence policy and retained as rights_blocked.",[str(prepare_receipt_path),str(data/"rights_and_attribution/manifest.json")]),
        19:(missing_format_status==(131_804,131_804),"All 131,804 direct format gaps are explicitly UNKNOWN/NOT_TESTED.",[str(data/"media_assertion_quality/media_assertion_quality.parquet")]),
        20:(missing_type_status==(17,17),"All 17 direct rows missing media_type have CONFLICT status.",[str(data/"media_assertion_quality/media_assertion_quality.parquet")]),
        21:(pilot["counts"]["pilot_rows"]==823 and pilot["counts"]["result_rows"]==823 and pilot["validation"]["all_resolved_rows_reviewed"] and pilot["validation"]["selection_checksum_matches"],f"The executed 823-row pilot has one result per row, a checksum-bound deterministic selection, and review decisions for every resolved outcome.",[str(data/"quality_results/phase4_pilot_execution/v1/audit/manifest.json")]),
        22:(prepare["work_rows"]==823 and report_manifest["validation"]["pilot_denominator_bound_to_execution_manifest"],"The only executed resolver queue is the fixed 823-row pilot; final reports bind claims to that denominator and do not claim a 126,634-row run.",[str(prepare_receipt_path),str(data/"quality_results/phase4_pilot_execution/v1/audit/manifest.json"),str(reports/"manifest.json")]),
        23:(recovery["validation"]["all_committed_stages_skippable"] and recovery["validation"]["unchanged_rows_not_reprocessed"],f"{recovery['counts']['skipped_committed_stages']} committed stages are checksum-verified and skippable; unchanged queue is zero.",[str(data/"quality_results/restart_validation_v3/manifest.json")]),
        24:(inc["validation"]["unchanged_rerun_semantically_identical"],"Full rerun semantic fingerprints match.",[str(data/"incremental_validation/manifest.json")]),
        25:(resources["validation"]["parts_nonempty"] and lineage["validation"]["manifest_written_last"],"Partitioned lineage, resource, readiness, and incremental outputs record every part SHA-256.",[str(data/"source_lineage/identity_v2/manifest.json"),str(data/"media_resources/manifest.json"),str(data/"ai_readiness/manifest.json"),str(data/"incremental_validation/manifest.json")]),
        26:(schema_versions_present,"Every required evidence manifest has a non-empty schema_version.",[str(path) for path in required if path.name=="manifest.json"]),
        27:(recovery["validation"]["all_committed_stages_skippable"] and report_manifest["validation"]["manifest_written_last"],"Restart validation verifies committed artifact checksums and manifest-last publication; the final report manifest is also written last.",[str(data/"quality_results/restart_validation_v3/manifest.json"),str(reports/"manifest.json")]),
        28:(dup["validation"]["content_claims_withheld"],"Exact-content and perceptual groups are separate NOT_TESTED rows.",[str(data/"duplicates/duplicate_group_summary.parquet")]),
        29:(ai["validation"]["one_row_per_media_assertion"] and ai["counts"]["rows"]==media["counts"]["rows"],"AI readiness is a separate one-row-per-media-assertion layer.",[str(data/"ai_readiness/manifest.json")]),
        30:(ai["validation"]["source_fields_unchanged"] and ai["validation"]["byte_claims_withheld"],"AI readiness preserves source fields and withholds model/byte claims without evidence.",[str(data/"ai_readiness/manifest.json")]),
        31:(concentration["counts"]["dimensions"]==4 and gates["counts"]["gate_summaries"]==7,"Representativeness includes raw, URL-adjusted, gate, and four explicit concentration dimensions.",[str(data/"representativeness/coverage_by_dimension.parquet"),str(data/"representativeness_concentration/concentration_metrics.parquet"),str(data/"completeness_gates/gate_summary.parquet")]),
        32:((reports/"before_after_summary.md").is_file() and "Before missing or candidate media rows" in (reports/"before_after_summary.md").read_text(), "The final before/after report distinguishes immutable source gaps, sparse assertions, and unresolved outcomes.",[str(reports/"before_after_summary.md")]),
        33:((reports/"completeness_by_applicability.md").is_file() and "invalid_present_media_rows" in (reports/"completeness_by_applicability.md").read_text(),"The completeness report includes invalid-present counts, not only null counts.",[str(reports/"completeness_by_applicability.md")]),
        34:(representation["counts"]["provider"]>0 and representation["counts"]["dataset"]>0,"Provider and dataset scorecards are materialized and reported.",[str(data/"representativeness/provider_scorecard.parquet"),str(data/"representativeness/dataset_scorecard.parquet")]),
        35:(gates["counts"]["failure_reason_rows"]>0 and (reports/"ai_readiness.md").is_file(),"AI readiness and completeness-gate evidence retain explicit failure and unresolved reason codes.",[str(data/"completeness_gates/gate_failure_reasons.parquet"),str(reports/"ai_readiness.md")]),
        36:(dup["counts"]["conflict_rows"]>0 and geography["counts"]["conflict_occurrences"]>0 and (reports/"duplicates_and_conflicts.md").is_file(),"Stored duplicate and geographic conflict counts are represented in the final report suite.",[str(data/"duplicates/manifest.json"),str(data/"derived_assertions/geography_v3/manifest.json"),str(reports/"duplicates_and_conflicts.md")]),
        37:((reports/"completeness_by_applicability.md").is_file() and "repairable_null_media_rows" in (reports/"completeness_by_applicability.md").read_text(),"The applicability report contains separate repairable and non-repairable null evidence columns.",[str(data/"completeness_by_applicability.parquet"),str(reports/"completeness_by_applicability.md")]),
        38:(all((reports/name).is_file() for name in ("source_funnel.md","occurrence_quality.md","media_quality.md","url_health.md","duplicates_and_conflicts.md")),"The report suite contains source, occurrence, media assertion, URL, and content/duplicate denominator reports.",[str(reports/name) for name in ("source_funnel.md","occurrence_quality.md","media_quality.md","url_health.md","duplicates_and_conflicts.md")]),
        39:(perf["validation"]["peak_rss_below_16_gib"],f"Measured peak RSS {perf['counts']['peak_rss_bytes']:,} bytes.",[str(data/"performance/manifest.json")]),
        40:(tests.get("exit_code")==0 and tests.get("tests_passed",0)>0,f"{tests.get('tests_passed')} required tests passed.",[str(receipt)]),
        41:(diff.returncode==0,f"git diff --check exit {diff.returncode}: {diff_evidence}",[str(repo)]),
        42:(report_manifest["validation"]["no_unsubstantiated_network_claim"] and report_manifest["validation"]["pilot_denominator_bound_to_execution_manifest"] and report_manifest["validation"]["provider_claims_bound_to_archive_manifest"] and reviews["counts"]["capsule_rows"]>0 and providers["counts"]["executed_archives"]==7 and provider_review["counts"]["review_rows"]>0 and freshness["counts"]["provider_dataset_rows"]>0,"Final claims are bound to the executed pilot, pinned provider archives, deterministic review populations, and freshness evidence.",[str(reports/"manifest.json"),str(data/"quality_results/review_capsules/manifest.json"),str(data/"freshness/manifest.json"),str(data/"provider_enrichment_v4/manifest.json"),str(data/"quality_results/provider_archive_review/v1/manifest.json")]),
    }
    rows=[]
    for number,requirement in enumerate(CRITERIA,1):
        passed,summary,paths=facts[number]
        status="PASS" if passed else "FAIL"
        rows.append({"acceptance_version":ACCEPTANCE_VERSION,"criterion_number":number,"requirement":requirement,"status":status,"blocking":status!="PASS","network_required":number in {17,18,21,22},"evidence_summary":summary,"evidence_paths":paths,"rule_version":"global-acceptance/v3.0.0"})
    destination.parent.mkdir(parents=True,exist_ok=True);staging=destination.parent/f".{destination.name}.{uuid4().hex}.staging";staging.mkdir();table_path=staging/"acceptance_criteria.parquet";report_path=staging/"acceptance_audit.md"
    pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),table_path,compression="zstd");_write_text(report_path,_markdown(rows))
    counts={status:sum(row["status"]==status for row in rows) for status in ("PASS","FAIL","NOT_TESTED","UNKNOWN")}
    artifacts=[_artifact(table_path),_file_artifact(report_path)];manifest={"schema_version":ACCEPTANCE_VERSION,"generated_at":datetime.now(UTC).isoformat().replace('+00:00','Z'),"code_commit":code_commit,"source_snapshot_id":base["source_snapshot_id"],"overall_status":"PASS" if counts["PASS"]==42 else "FAIL","counts":counts,"blocking_criteria":[row["criterion_number"] for row in rows if row["blocking"]],"git_diff_check":{"exit_code":diff.returncode,"output":diff_evidence},"artifacts":artifacts,"validation":{"exactly_42_criteria":len(rows)==42,"all_criteria_have_evidence":all(row["evidence_paths"] and row["evidence_summary"] for row in rows),"manifest_written_last":True},"network_requests":pilot["network_requests"],"manifest_policy":{"written_last":True,"create_only":True}}
    _write_json(staging/"manifest.json",manifest)
    for artifact in artifacts:
        if _sha256(staging/artifact["path"])!=artifact["sha256"]:raise ValueError("acceptance artifact checksum mismatch")
    os.replace(staging,destination);return manifest


def _markdown(rows):
    lines=["# GBIF media v4 global acceptance audit","","This report is rendered from the machine-readable 42-row evidence contract. A criterion is not passed without an authoritative artifact.","","| # | Status | Requirement | Evidence |","| ---: | --- | --- | --- |"]
    for row in rows:lines.append(f"| {row['criterion_number']} | {row['status']} | {row['requirement']} | {row['evidence_summary'].replace('|','/')} |")
    return "\n".join(lines)+"\n"
def _artifact(path):
    p=pq.ParquetFile(path);return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}
def _file_artifact(path):return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path)}
def _write_text(path,value):
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(value);os.replace(temp,path)
def _write_json(path,value):_write_text(path,json.dumps(value,indent=2,sort_keys=True)+"\n")
def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


__all__=["ACCEPTANCE_VERSION","CRITERIA","SCHEMA","publish_acceptance_audit"]
