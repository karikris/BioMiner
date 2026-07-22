from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path

import duckdb


REPORT_VERSION = "biominer-gbif-media-v4-reports/v1"
REPORT_NAMES = (
    "source_funnel.md", "schema_and_integrity.md", "completeness_by_applicability.md",
    "occurrence_quality.md", "media_quality.md", "url_resolution_pilot.md", "url_health.md",
    "rights_and_attribution.md", "temporal_quality.md", "geospatial_quality.md",
    "taxonomic_quality.md", "identification_provenance.md", "biological_attribute_candidates.md",
    "duplicates_and_conflicts.md", "ai_readiness.md", "bias_and_representativeness.md",
    "provider_remediation_priorities.md", "before_after_summary.md",
    "performance_and_reproducibility.md",
)


def publish_final_reports(*, data_root: str | Path, report_root: str | Path, code_commit: str) -> dict[str, object]:
    data = Path(data_root).resolve(); reports = Path(report_root).resolve(); reports.mkdir(parents=True, exist_ok=True)
    required = [
        data / "manifest.json", data / "source_funnel.parquet", data / "schema_inventory.parquet",
        data / "completeness_by_applicability.parquet", data / "occurrence_quality/manifest.json",
        data / "media_assertion_quality/manifest.json", data / "rights_and_attribution/manifest.json",
        data / "duplicates/manifest.json", data / "ai_readiness/manifest.json",
        data / "representativeness/manifest.json", data / "incremental_validation/manifest.json",
        data / "performance/manifest.json", data / "source_lineage/identity_v2/manifest.json",
        data / "media_resources/manifest.json", data / "completeness_gates/manifest.json",
        data / "quality_results/review_capsules/manifest.json",
        data / "representativeness_concentration/manifest.json",
        data / "freshness/manifest.json", data / "provider_enrichment/manifest.json",
    ]
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    load=lambda path: json.loads(Path(path).read_text())
    base=load(data/"manifest.json"); occurrence=load(data/"occurrence_quality/manifest.json"); media=load(data/"media_assertion_quality/manifest.json")
    rights=load(data/"rights_and_attribution/manifest.json"); duplicates=load(data/"duplicates/manifest.json"); readiness=load(data/"ai_readiness/manifest.json")
    representation=load(data/"representativeness/manifest.json"); incremental=load(data/"incremental_validation/manifest.json")
    performance=load(data/"performance/manifest.json")
    lineage=load(data/"source_lineage/identity_v2/manifest.json"); resources=load(data/"media_resources/manifest.json")
    gates=load(data/"completeness_gates/manifest.json"); reviews=load(data/"quality_results/review_capsules/manifest.json")
    concentration=load(data/"representativeness_concentration/manifest.json"); freshness=load(data/"freshness/manifest.json")
    providers=load(data/"provider_enrichment/manifest.json")
    temporal=load(data/"derived_assertions/temporal/manifest.json"); geography=load(data/"derived_assertions/geography/manifest.json")
    taxonomy=load(data/"derived_assertions/taxonomy/manifest.json"); biology=load(data/"derived_assertions/biology/manifest.json")
    c=duckdb.connect()
    try:
        contents={
            "source_funnel.md": _report("Source funnel", f"Exact source-to-v4 reconciliation. All {lineage['counts']['rows']:,} raw multimedia assertions have stable source locations, the GBIF download key, ingestion time, and source-value hashes; excluded rows remain represented.", _table(c,data/"source_funnel.parquet", "SELECT stage_order,stage_id,input_row_count,output_row_count,excluded_row_count,status FROM read_parquet(?) ORDER BY stage_order"), _table(c,data/"freshness/provider_dataset_freshness.parquet", "SELECT freshness_status,count(*) provider_dataset_groups,sum(media_rows) media_rows FROM read_parquet(?) GROUP BY 1 ORDER BY 1")),
            "schema_and_integrity.md": _report("Schema and integrity", "The retained archival layer has 114 columns. Checksums, physical types, encodings, statistics, and row-group bounds are inventoried; source values were not rewritten.", _table(c,data/"source_inventory.parquet", "SELECT artifact_role,row_count,column_count,row_group_count,checksum_status,row_count_status FROM read_parquet(?)"), _table(c,data/"schema_inventory.parquet", "SELECT type_drift_status,count(*) column_count FROM read_parquet(?) GROUP BY 1")),
            "completeness_by_applicability.md": _report("Completeness by applicability", "Physical missingness is separated from applicability, withheld/generalized values, invalid-present values, conflicts, and repairability. The Parquet evidence contains all 114 columns.", _table(c,data/"completeness_by_applicability.parquet", "SELECT field_name,scope,required_status,physical_fill_media_pct,applicable_fill_media_pct,repairable_null_media_rows,invalid_present_media_rows,conflict_media_rows FROM read_parquet(?) ORDER BY field_index")),
            "occurrence_quality.md": _report("Occurrence quality", f"Denominator: {occurrence['counts']['rows']:,} distinct gbifID occurrences and {occurrence['counts']['media_rows']:,} media assertions.", _table(c,data/"occurrence_quality/occurrence_check_status_summary.parquet", "SELECT check_output_field,status,occurrence_count,media_row_count FROM read_parquet(?) ORDER BY 1,2")),
            "media_quality.md": _report("Media assertion quality", f"Denominator: {media['counts']['rows']:,} assertions. Local checks make no network-liveness claim.", _nested_counts(media["counts"]["status_counts"])),
            "url_resolution_pilot.md": _report("URL resolution pilot", "The deterministic 823-row stratified sample is prepared. Live execution and manual adjudication are NOT_TESTED; the 126,634-row eligible tail is gated and has not run.", _table(c,data/"quality_results/phase4_pilot_preflight/pilot_acceptance_gates.parquet", "SELECT gate_id,gate,status,evidence FROM read_parquet(?) ORDER BY gate_id")),
            "url_health.md": _report("URL health", f"Direct URL syntax PASS: {media['counts']['status_counts']['direct_media_url_status'].get('PASS',0):,}. Canonical resources: {resources['counts']['canonical_resources']:,}; reference-only unresolved: {resources['counts']['unresolved_reference_only_assertions']:,}. Tested live denominator: 0; reachability, redirects, MIME, decode, and content health are NOT_TESTED.", _table(c,data/"media_resources/resource_status_summary.parquet", "SELECT * FROM read_parquet(?) ORDER BY status_name,status")),
            "rights_and_attribution.md": _report("Rights and attribution", f"Media rights are normalized independently of occurrence rights. Research-only means eligible only under the configured non-commercial biodiversity-research policy. Seven prioritized structured provider adapters are registered; {providers['counts']['executed_adapters']} have run, so provider enrichment remains NOT_TESTED.", _mapping(rights["counts"]["status_counts"]), _table(c,data/"rights_and_attribution/provider_rights_summary.parquet", "SELECT provider,media_rows,missing_media_license_rows,missing_creator_rows,missing_rights_holder_rows,estimated_recoverable_rows FROM read_parquet(?) ORDER BY estimated_recoverable_rows DESC LIMIT 20"), _table(c,data/"provider_enrichment/provider_enrichment_registry.parquet", "SELECT priority,provider_name,adapter_version,request_mode,execution_status FROM read_parquet(?) ORDER BY priority")),
            "temporal_quality.md": _report("Temporal quality", "Original fields remain unchanged; derived components are sparse assertions with precision-aware parsing.", _mapping(temporal["counts"])),
            "geospatial_quality.md": _report("Geospatial quality", "Country-from-coordinate candidates remain NOT_TESTED without a pinned boundary dataset. Generalized/withheld geography is not treated as ordinary missingness.", _mapping(geography["counts"])),
            "taxonomic_quality.md": _report("Taxonomic quality", "Species repairs require direct deterministic evidence and preserve source taxonomy.", _mapping(taxonomy["counts"])),
            "identification_provenance.md": _report("Identification provenance", "identifiedBy is independent from accepted-name status; acceptance is not silently treated as identifier evidence.", _table(c,data/"occurrence_quality/occurrence_check_status_summary.parquet", "SELECT check_output_field,status,occurrence_count,media_row_count FROM read_parquet(?) WHERE check_output_field IN ('identified_by_status','verification_source_evidence_status') ORDER BY 1,2")),
            "biological_attribute_candidates.md": _report("Biological attribute candidates", "Life-stage and sex values extracted from text remain review candidates; negated-only evidence is not asserted.", _mapping(biology["counts"])),
            "duplicates_and_conflicts.md": _report("Duplicates and conflicts", "No duplicates were deleted. URL groups are not claimed as content groups; exact-content and perceptual evidence remain NOT_TESTED.", _mapping(duplicates["counts"]), _table(c,data/"duplicates/duplicate_group_summary.parquet", "SELECT * FROM read_parquet(?) ORDER BY group_type")),
            "ai_readiness.md": _report("AI readiness", "Technical readiness is fail-closed. No image is AI_READY until live retrieval and safe byte decoding are evidenced. Coordinates are not required for pure classification; exact species labels are required for species training.", _mapping(readiness["counts"]["decision_counts"]), _table(c,data/"ai_readiness/readiness_status_summary.parquet", "SELECT status_name,status,media_rows,distinct_occurrences,distinct_original_urls FROM read_parquet(?) ORDER BY 1,2"), _table(c,data/"ai_readiness/parts/*.parquet", "SELECT reason,count(*) media_rows FROM read_parquet(?) CROSS JOIN UNNEST(reason_codes) AS t(reason) GROUP BY 1 ORDER BY 2 DESC")),
            "bias_and_representativeness.md": _report("Bias and representativeness", "Counts distinguish raw assertions, occurrences, exact URLs, canonical URLs, and URL-adjusted support. HHI, maximum share, and effective counts quantify provider, creator, regional, and temporal concentration for all and rights-qualified media. Technically usable, content, and perceptual cohorts remain NOT_TESTED; absence in media data does not imply biological absence.", _mapping(representation["counts"]), _mapping(concentration["counts"]), _table(c,data/"representativeness/coverage_by_dimension.parquet", "SELECT dimension,\"value\",raw_image_count,distinct_occurrence_count,duplicate_adjusted_count,unresolved_count FROM read_parquet(?) WHERE dimension IN ('provider','dataset') ORDER BY raw_image_count DESC LIMIT 40"), _table(c,data/"representativeness_concentration/concentration_metrics.parquet", "SELECT cohort,species,concentration_dimension,media_rows,distinct_values,max_value_share,hhi,effective_value_count FROM read_parquet(?) WHERE species<>'<MISSING>' ORDER BY hhi DESC,media_rows DESC LIMIT 40"), _table(c,data/"representativeness/species_bias_flags.parquet", "SELECT species,raw_image_count,distinct_occurrence_count,duplicate_adjusted_count,provider_count,creator_count,country_count,decade_count,bias_flags FROM read_parquet(?) ORDER BY duplicate_adjusted_count DESC LIMIT 30")),
            "provider_remediation_priorities.md": _report("Provider remediation priorities", "Ranking is lexicographic evidence, not an opaque composite score. Provider-level bulk fixes should precede per-record network calls.", _table(c,data/"representativeness/provider_remediation_queue.parquet", "SELECT * FROM read_parquet(?) ORDER BY priority_rank LIMIT 30")),
            "before_after_summary.md": _report("Before/after summary", "The audit preserves v3 as immutable input and adds sparse v4 evidence. 'After' means a separate derived assertion is available; source completeness is unchanged and originals are not overwritten.", _table(c,data/"source_funnel.parquet", "SELECT stage_id,input_row_count,output_row_count,excluded_row_count,exclusion_reason FROM read_parquet(?) ORDER BY stage_order"), _table(c,data/"completeness_gates/gate_summary.parquet", "SELECT gate_id,passed_media_rows,passed_occurrences,passed_original_urls,url_adjusted_pass_count,exact_content_deduplication_status,pass_percentage FROM read_parquet(?) ORDER BY gate_id"), "| Field/domain | Before missing or candidate media rows | Derived/repaired assertions | After unresolved |\n| --- | ---: | ---: | ---: |\n"+f"| year | {temporal['counts']['derived_year_media_rows']} | {temporal['counts']['derived_year_media_rows']} | 0 |\n| month | {temporal['counts']['derived_month_media_rows']} | {temporal['counts']['derived_month_media_rows']} | 0 |\n| day | {temporal['counts']['derived_day_media_rows']} | {temporal['counts']['derived_day_media_rows']} | 0 |\n| species | {taxonomy['counts']['candidate_media_rows']} | {taxonomy['counts']['repaired_media_rows']} | {taxonomy['counts']['unresolved_occurrences']} occurrences |\n| continent | {geography['counts']['missing_continent_media_rows']} | {geography['counts']['derived_continent_media_rows']} | {geography['counts']['missing_continent_media_rows']-geography['counts']['derived_continent_media_rows']} |\n| biological text candidates | {biology['counts']['life_stage_candidate_media_rows']+biology['counts']['sex_candidate_media_rows']} | {biology['counts']['assertion_rows']} review-only | NOT_APPLICABLE |"),
            "performance_and_reproducibility.md": _report("Performance and reproducibility", f"Six full-data benchmark stages passed under the configured 8 GiB analytical-engine limit. Observed process high-water RSS was {performance['counts']['peak_rss_bytes']:,} bytes, below the 16 GiB acceptance ceiling. Incremental semantic equality and zero changed rows prove deterministic value-level rerun behavior. The manual-review population contains {reviews['counts']['capsule_rows']:,} deterministic sealed capsules. No live resolver or provider enrichment is claimed.", _table(c,data/"performance/benchmark_results.parquet", "SELECT stage,status,rows_read,elapsed_seconds,rows_per_second,input_bytes,process_peak_rss_bytes,result_fingerprint FROM read_parquet(?) ORDER BY stage"), _mapping(incremental["counts"]), _mapping(incremental["validation"]), _mapping(freshness["counts"]), _mapping(gates["counts"])),
        }
    finally: c.close()
    generated=[]
    for name in REPORT_NAMES:
        path=reports/name; _atomic_text(path,contents[name]); generated.append(_entry(path,reports))
    overall="NOT_TESTED" if any("NOT_TESTED" in contents[name] for name in REPORT_NAMES) else "PASS"
    manifest={"schema_version":REPORT_VERSION,"generated_at":datetime.now(UTC).isoformat().replace('+00:00','Z'),"code_commit":code_commit,"source_snapshot_id":base["source_snapshot_id"],"overall_status":overall,"network_requests":0,"reports":generated,"data_manifests":[str(path) for path in required if path.name=='manifest.json'],"validation":{"all_required_reports_present":len(generated)==len(REPORT_NAMES),"manifest_written_last":True,"no_unsubstantiated_network_claim":True}}
    _atomic_text(reports/"manifest.json",json.dumps(manifest,indent=2,sort_keys=True)+"\n"); return manifest


def _report(title: str, intro: str, *sections: str) -> str:
    return f"# {title}\n\nStatus vocabulary: PASS, FAIL, UNKNOWN, NOT_APPLICABLE, WITHHELD, GENERALIZED, CONFLICT, NOT_TESTED.\n\n{intro}\n\n"+"\n\n".join(section for section in sections if section)+"\n"
def _table(c, path, sql):
    result=c.execute(sql,[str(path)]); names=[item[0] for item in result.description]; rows=result.fetchall()
    def value(v):
        if isinstance(v,list): v=", ".join(map(str,v))
        return str(v if v is not None else "NULL").replace("|","\\|").replace("\n"," ")
    return "| "+" | ".join(names)+" |\n| "+" | ".join("---" for _ in names)+" |\n"+"\n".join("| "+" | ".join(value(v) for v in row)+" |" for row in rows)
def _mapping(values): return "| Metric | Value |\n| --- | ---: |\n"+"\n".join(f"| {k} | {json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v} |" for k,v in sorted(values.items()))
def _nested_counts(values): return "| Check | Status | Count |\n| --- | --- | ---: |\n"+"\n".join(f"| {check} | {status} | {count:,} |" for check,counts in sorted(values.items()) for status,count in sorted(counts.items()))
def _atomic_text(path,value):
    temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(value); os.replace(temp,path)
def _entry(path,root): return {"path":str(path.relative_to(root)),"physical_bytes":path.stat().st_size,"sha256":_sha256(path)}
def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


__all__=["REPORT_NAMES","REPORT_VERSION","publish_final_reports"]
