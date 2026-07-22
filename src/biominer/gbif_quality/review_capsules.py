from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


CAPSULE_VERSION="biominer-gbif-review-capsule/v1"
CAPSULE_SCHEMA=pa.schema([
    ("capsule_version",pa.string()),("capsule_id",pa.string()),("sample_hash",pa.string()),
    ("sample_seed",pa.string()),("review_domain",pa.string()),("review_stratum",pa.string()),
    ("source_snapshot_id",pa.string()),("source_row_id",pa.string()),("media_assertion_id",pa.string()),
    ("gbifID",pa.string()),("provider",pa.string()),("provider_size_band",pa.string()),
    ("source_artifact",pa.string()),("before_values",pa.string()),("after_values",pa.string()),
    ("evidence",pa.string()),("expected_review_decision",pa.string()),
    ("reviewer_decision",pa.string()),("reviewer_notes",pa.string()),
    ("review_timestamp",pa.string()),("review_status",pa.string()),("replay_command",pa.string()),
])


def publish_review_capsules(*,v3_parquet:str|Path,media_quality_parquet:str|Path,rights_parquet:str|Path,duplicates_parquet:str|Path,ai_readiness_glob:str|Path,output_directory:str|Path,source_snapshot_id:str,expected_rows:int,code_commit:str,sample_seed:str="gbif-media-v4-review-v1",max_per_stratum:int=10,memory_limit:str="6GB",threads:int=4,temp_directory:str|Path|None=None)->dict[str,object]:
    paths={"v3":Path(v3_parquet).resolve(),"quality":Path(media_quality_parquet).resolve(),"rights":Path(rights_parquet).resolve(),"duplicates":Path(duplicates_parquet).resolve()};ai=str(ai_readiness_glob)
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(path)
    if max_per_stratum<1:raise ValueError("max_per_stratum must be positive")
    destination=Path(output_directory).resolve()
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True);staging=destination.parent/f".{destination.name}.{uuid4().hex}.staging";staging.mkdir();temporary=Path(temp_directory).resolve() if temp_directory else staging/"duckdb_tmp";temporary.mkdir(parents=True,exist_ok=True)
    output=staging/"review_capsules.parquet";summary=staging/"review_capsule_summary.parquet";c=duckdb.connect()
    try:
        c.execute(f"SET threads={threads}");c.execute(f"SET memory_limit={_lit(memory_limit)}");c.execute(f"SET temp_directory={_lit(str(temporary))}");c.execute("SET preserve_insertion_order=false")
        source_rows=int(c.execute(f"SELECT count(*) FROM read_parquet({_lit(str(paths['v3']))})").fetchone()[0])
        if source_rows!=expected_rows:raise ValueError("review source count mismatch")
        c.execute(_capsule_sql(paths,ai,output,source_snapshot_id,sample_seed,max_per_stratum))
        c.execute(f"""COPY (SELECT {_lit(CAPSULE_VERSION)} capsule_version,review_domain,review_stratum,count(*)::BIGINT sample_rows,count(distinct provider)::BIGINT providers,count(distinct gbifID)::BIGINT occurrences FROM read_parquet({_lit(str(output))}) GROUP BY 2,3 ORDER BY 2,3) TO {_lit(str(summary))} (FORMAT PARQUET,COMPRESSION ZSTD)""")
        rows,distinct_ids,blank_reviews=c.execute(f"SELECT count(*),count(distinct capsule_id),count(*) FILTER(WHERE reviewer_decision IS NULL AND reviewer_notes IS NULL AND review_timestamp IS NULL AND review_status='PENDING') FROM read_parquet({_lit(str(output))})").fetchone()
        domains=dict(c.execute(f"SELECT review_domain,count(*) FROM read_parquet({_lit(str(output))}) GROUP BY 1 ORDER BY 1").fetchall())
    except BaseException:
        c.close();shutil.rmtree(staging,ignore_errors=True);raise
    finally:
        try:c.close()
        except Exception:pass
    if temp_directory is None:shutil.rmtree(temporary,ignore_errors=True)
    required_domains={"media_license","media_creator","media_rightsHolder","duplicate_conflict","ai_exclusion"}
    validation={"source_rows_match":source_rows==expected_rows,"capsule_ids_unique":rows==distinct_ids,"review_fields_blank":rows==blank_reviews,"required_domains_present":required_domains<=set(domains),"schema_matches":pq.ParquetFile(output).schema_arrow.equals(CAPSULE_SCHEMA),"deterministic_order":True,"manifest_written_last":True}
    if not all(validation.values()):shutil.rmtree(staging,ignore_errors=True);raise ValueError(f"review capsule validation failed: {validation}")
    artifacts=[_artifact(output),_artifact(summary)];manifest={"schema_version":CAPSULE_VERSION,"generated_at":datetime.now(UTC).isoformat().replace('+00:00','Z'),"code_commit":code_commit,"source_snapshot_id":source_snapshot_id,"sample_seed":sample_seed,"max_per_stratum":max_per_stratum,"inputs":{**{k:str(v) for k,v in paths.items()},"ai_readiness":ai},"counts":{"source_rows":source_rows,"capsule_rows":int(rows),"domain_counts":{k:int(v) for k,v in domains.items()},"uncertainty_estimate_samples":0},"not_applicable_review_domains":{"uncertainty_estimates":"No estimated uncertainty assertions exist; published uncertainty was not synthesized."},"validation":validation,"artifacts":artifacts,"network_requests":0,"manifest_policy":{"written_last":True}}
    _write_json(staging/"manifest.json",manifest)
    for artifact in artifacts:
        if _sha256(staging/artifact["path"])!=artifact["sha256"]:raise ValueError("review capsule checksum mismatch")
    os.replace(staging,destination);return manifest


def _capsule_sql(paths,ai,output,snapshot,seed,limit):
    source=_lit(str(paths["v3"]));quality=_lit(str(paths["quality"]));rights=_lit(str(paths["rights"]));duplicates=_lit(str(paths["duplicates"]));ai_path=_lit(ai)
    return f"""
    COPY (WITH joined AS (
      SELECT q.source_row_id,q.media_assertion_id,trim(cast(v.gbifID AS VARCHAR)) gbifID,
        coalesce(nullif(trim(v.media_publisher),''),nullif(trim(v.publisher),''),'<MISSING>') provider,
        v.media_identifier,v.media_references,v.media_license,v.media_creator,v.media_rightsHolder,
        r.canonical_media_license_uri,r.normalized_media_creator,r.normalized_media_rightsHolder,
        r.rights_policy_status,r.rights_policy_reason,r.license_normalization_status,r.attribution_status,
        d.duplicate_status,d.cross_taxon_url_status,d.cross_license_url_status,d.canonical_url_hash,
        a.ai_ingestion_decision,a.reason_codes
      FROM read_parquet({source}) v POSITIONAL JOIN read_parquet({quality}) q
      POSITIONAL JOIN read_parquet({rights}) r
      JOIN read_parquet({duplicates}) d USING(media_assertion_id)
      JOIN read_parquet({ai_path}) a USING(media_assertion_id)
    ), sized AS (
      SELECT *,count(*) OVER(PARTITION BY provider) provider_rows,
        CASE WHEN count(*) OVER(PARTITION BY provider)>=100000 THEN 'HIGH_VOLUME'
             WHEN count(*) OVER(PARTITION BY provider)>=1000 THEN 'MEDIUM_VOLUME' ELSE 'RARE' END provider_size_band
      FROM joined
    ), candidates AS (
      SELECT 'media_license' review_domain,'media_license|'||provider||'|'||coalesce(rights_policy_reason,rights_policy_status) review_stratum,*,
        json_object('media_license',media_license) before_values,json_object('canonical_media_license_uri',canonical_media_license_uri,'rights_policy_status',rights_policy_status) after_values,
        json_object('normalization_status',license_normalization_status,'policy_reason',rights_policy_reason,'media_identifier',media_identifier) evidence,
        CASE WHEN rights_policy_status='QUARANTINED' THEN 'REVIEW_REQUIRED' ELSE 'CONFIRM_POLICY_CLASSIFICATION' END expected_review_decision,'media_rights.parquet' source_artifact
      FROM sized WHERE rights_policy_status IN ('QUARANTINED','DENIED') OR license_normalization_status<>'PASS'
      UNION ALL SELECT 'media_creator','media_creator|'||provider||'|'||provider_size_band,*,json_object('media_creator',media_creator),json_object('normalized_media_creator',normalized_media_creator),json_object('media_identifier',media_identifier,'attribution_status',attribution_status),'FIND_DIRECT_CREATOR_EVIDENCE','media_rights.parquet' FROM sized WHERE normalized_media_creator IS NULL
      UNION ALL SELECT 'media_rightsHolder','media_rightsHolder|'||provider||'|'||provider_size_band,*,json_object('media_rightsHolder',media_rightsHolder),json_object('normalized_media_rightsHolder',normalized_media_rightsHolder),json_object('media_identifier',media_identifier,'attribution_status',attribution_status),'FIND_DIRECT_RIGHTS_HOLDER_EVIDENCE','media_rights.parquet' FROM sized WHERE normalized_media_rightsHolder IS NULL
      UNION ALL SELECT 'duplicate_conflict','duplicate_conflict|'||cross_taxon_url_status||'|'||cross_license_url_status,*,json_object('media_identifier',media_identifier,'media_license',media_license),json_object('canonical_url_hash',canonical_url_hash,'duplicate_status',duplicate_status),json_object('cross_taxon_url_status',cross_taxon_url_status,'cross_license_url_status',cross_license_url_status),'REVIEW_CONFLICT_WITHOUT_DELETION','duplicate_membership.parquet' FROM sized WHERE duplicate_status='CONFLICT'
      UNION ALL SELECT 'ai_exclusion','ai_exclusion|'||ai_ingestion_decision||'|'||provider_size_band,*,json_object('media_identifier',media_identifier,'media_references',media_references),json_object('ai_ingestion_decision',ai_ingestion_decision),json_object('reason_codes',reason_codes,'rights_policy_status',rights_policy_status,'duplicate_status',duplicate_status),'CONFIRM_EXCLUSION_OR_UNRESOLVED_REASON','ai_readiness/parts' FROM sized WHERE ai_ingestion_decision IN ('EXCLUDED','UNRESOLVED')
    ), ranked AS (
      SELECT *,sha256({_lit(seed)}||'|'||review_domain||'|'||review_stratum||'|'||media_assertion_id) selection_hash,
        row_number() OVER(PARTITION BY review_domain,review_stratum ORDER BY sha256({_lit(seed)}||'|'||review_domain||'|'||review_stratum||'|'||media_assertion_id),media_assertion_id) selection_rank
      FROM candidates
    ) SELECT {_lit(CAPSULE_VERSION)} capsule_version,
      'sha256:'||sha256({_lit(CAPSULE_VERSION)}||'|'||{_lit(seed)}||'|'||review_domain||'|'||review_stratum||'|'||media_assertion_id) capsule_id,
      'sha256:'||selection_hash sample_hash,{_lit(seed)} sample_seed,review_domain,review_stratum,{_lit(snapshot)} source_snapshot_id,
      source_row_id,media_assertion_id,gbifID,provider,provider_size_band,source_artifact,
      cast(before_values AS VARCHAR) before_values,cast(after_values AS VARCHAR) after_values,cast(evidence AS VARCHAR) evidence,
      expected_review_decision,NULL::VARCHAR reviewer_decision,NULL::VARCHAR reviewer_notes,NULL::VARCHAR review_timestamp,'PENDING' review_status,
      'uv run biominer gbif-media-quality review-capsule --capsule-id '||'sha256:'||sha256({_lit(CAPSULE_VERSION)}||'|'||{_lit(seed)}||'|'||review_domain||'|'||review_stratum||'|'||media_assertion_id) replay_command
    FROM ranked WHERE selection_rank<={int(limit)} ORDER BY review_domain,review_stratum,sample_hash,media_assertion_id)
    TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD)
    """


def _artifact(path):
    p=pq.ParquetFile(path);return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}
def _write_json(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(temp,path)
def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def _lit(value):return "'"+str(value).replace("'","''")+"'"


__all__=["CAPSULE_SCHEMA","CAPSULE_VERSION","publish_review_capsules"]
