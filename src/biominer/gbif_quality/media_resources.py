from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb
import pyarrow.parquet as pq


RESOURCE_VERSION="biominer-gbif-media-resource/v1"


def publish_media_resources(*,duplicates_parquet:str|Path,output_directory:str|Path,source_snapshot_id:str,expected_assertion_rows:int,code_commit:str,memory_limit:str="6GB",threads:int=4,temp_directory:str|Path|None=None,partitions:int=16)->dict[str,object]:
    duplicates=Path(duplicates_parquet).resolve();destination=Path(output_directory).resolve()
    if not duplicates.is_file():raise FileNotFoundError(duplicates)
    if destination.exists():raise FileExistsError(destination)
    if partitions<1:raise ValueError("partitions must be positive")
    if pq.ParquetFile(duplicates).metadata.num_rows!=expected_assertion_rows:raise ValueError("resource source row mismatch")
    destination.parent.mkdir(parents=True,exist_ok=True);staging=destination.parent/f".{destination.name}.{uuid4().hex}.staging";staging.mkdir();temporary=Path(temp_directory).resolve() if temp_directory else staging/"duckdb_tmp";temporary.mkdir(parents=True,exist_ok=True);parts=staging/"parts";summary=staging/"resource_status_summary.parquet"
    c=duckdb.connect()
    try:
        c.execute(f"SET threads={threads}");c.execute(f"SET memory_limit={_lit(memory_limit)}");c.execute(f"SET temp_directory={_lit(str(temporary))}");c.execute("SET preserve_insertion_order=false")
        c.execute(_resource_sql(duplicates,parts,source_snapshot_id,partitions))
        glob=str(parts/"**/*.parquet")
        resources,assertions,occurrences,cross_taxon,cross_license=c.execute(f"SELECT count(*),sum(assertion_count),sum(distinct_occurrence_count),count(*) FILTER(WHERE cross_taxon_status='CONFLICT'),count(*) FILTER(WHERE cross_license_status='CONFLICT') FROM read_parquet({_lit(glob)})").fetchone()
        unresolved=expected_assertion_rows-int(assertions)
        c.execute(f"""COPY (SELECT {_lit(RESOURCE_VERSION)} resource_version,status_name,status,count(*)::BIGINT resource_count,sum(assertion_count)::BIGINT assertion_count FROM (SELECT assertion_count,'MEDIA_REACHABLE' status_name,media_reachable_status status FROM read_parquet({_lit(glob)}) UNION ALL SELECT assertion_count,'MEDIA_TECHNICALLY_VALID',media_technically_valid_status FROM read_parquet({_lit(glob)}) UNION ALL SELECT assertion_count,'CONTENT_IDENTITY',content_identity_status FROM read_parquet({_lit(glob)}) UNION ALL SELECT assertion_count,'PERCEPTUAL_IDENTITY',perceptual_identity_status FROM read_parquet({_lit(glob)})) GROUP BY 2,3 ORDER BY 2,3) TO {_lit(str(summary))} (FORMAT PARQUET,COMPRESSION ZSTD)""")
    except BaseException:
        c.close();shutil.rmtree(staging,ignore_errors=True);raise
    finally:
        try:c.close()
        except Exception:pass
    if temp_directory is None:shutil.rmtree(temporary,ignore_errors=True)
    part_files=sorted(parts.glob("**/*.parquet"));validation={"assertion_rows_reconcile":int(assertions)+unresolved==expected_assertion_rows,"canonical_resources_unique":int(resources)==sum(pq.ParquetFile(path).metadata.num_rows for path in part_files),"network_claims_withheld":True,"content_claims_withheld":True,"parts_nonempty":bool(part_files),"manifest_written_last":True}
    if not all(validation.values()):shutil.rmtree(staging,ignore_errors=True);raise ValueError(f"resource validation failed: {validation}")
    artifacts=[*(_artifact(path,staging) for path in part_files),_artifact(summary,staging)];manifest={"schema_version":RESOURCE_VERSION,"generated_at":datetime.now(UTC).isoformat().replace('+00:00','Z'),"code_commit":code_commit,"source_snapshot_id":source_snapshot_id,"inputs":{"duplicates":str(duplicates)},"counts":{"source_assertion_rows":expected_assertion_rows,"canonical_resources":int(resources),"addressable_assertion_rows":int(assertions),"unresolved_reference_only_assertions":unresolved,"resource_distinct_occurrence_sum":int(occurrences),"cross_taxon_resources":int(cross_taxon),"cross_license_resources":int(cross_license)},"configuration":{"partitions":partitions,"network_execution":False,"image_byte_inspection":False},"validation":validation,"artifacts":artifacts,"network_requests":0,"manifest_policy":{"written_last":True}}
    _write_json(staging/"manifest.json",manifest)
    for artifact in artifacts:
        if _sha256(staging/artifact["path"])!=artifact["sha256"]:raise ValueError("resource checksum mismatch")
    os.replace(staging,destination);return manifest


def _resource_sql(source,output,snapshot,partitions):
    return f"""
    COPY (WITH grouped AS (
      SELECT canonical_url_hash,arg_min(canonical_url,media_assertion_id) canonical_url,
        arg_min(source_platform_group_id,media_assertion_id) source_platform_group_id,
        count(*)::BIGINT assertion_count,count(distinct gbifID)::BIGINT distinct_occurrence_count,
        max(coalesce(url_distinct_taxa,1))::BIGINT distinct_taxon_count,
        max(coalesce(url_distinct_licenses,1))::BIGINT distinct_license_count,
        CASE WHEN count(*) FILTER(WHERE cross_taxon_url_status='CONFLICT')>0 THEN 'CONFLICT' ELSE 'PASS' END cross_taxon_status,
        CASE WHEN count(*) FILTER(WHERE cross_license_url_status='CONFLICT')>0 THEN 'CONFLICT' ELSE 'PASS' END cross_license_status,
        min(media_assertion_id) representative_media_assertion_id,min(gbifID) representative_gbifID
      FROM read_parquet({_lit(str(source))}) WHERE canonical_url_hash IS NOT NULL GROUP BY 1
    ) SELECT (hash(canonical_url_hash)%{partitions})::INTEGER resource_partition,
      {_lit(RESOURCE_VERSION)} resource_version,{_lit(snapshot)} source_snapshot_id,
      'sha256:'||sha256('canonical-resource|'||canonical_url_hash) media_resource_id,
      canonical_url_hash,canonical_url,NULL::VARCHAR final_url_hash,NULL::VARCHAR final_url,
      NULL::BLOB content_sha256,NULL::VARCHAR perceptual_hash,source_platform_group_id,
      representative_media_assertion_id,representative_gbifID,assertion_count,distinct_occurrence_count,
      distinct_taxon_count,distinct_license_count,'PASS' canonical_url_status,
      'NOT_TESTED' media_reachable_status,'NOT_TESTED' direct_image_response_status,
      'NOT_TESTED' media_decodable_status,'NOT_TESTED' media_technically_valid_status,
      'NOT_TESTED' content_identity_status,'NOT_TESTED' perceptual_identity_status,
      cross_taxon_status,cross_license_status,NULL::VARCHAR declared_content_type,
      NULL::BIGINT content_length,NULL::VARCHAR etag,NULL::VARCHAR last_modified,
      NULL::VARCHAR retrieval_timestamp,'canonical_url_local_normalization' observation_method
    FROM grouped) TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(resource_partition),ROW_GROUP_SIZE 250000,FILENAME_PATTERN 'part-{{i}}')
    """


def _artifact(path,root):
    p=pq.ParquetFile(path);return {"path":str(path.relative_to(root)),"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}
def _write_json(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(temp,path)
def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def _lit(value):return "'"+str(value).replace("'","''")+"'"


__all__=["RESOURCE_VERSION","publish_media_resources"]
