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


GATE_VERSION="biominer-gbif-completeness-gates/v1"
GATES={
    "GATE_1_ADDRESSABLE":"MEDIA_ADDRESSABLE='PASS'",
    "GATE_2_TECHNICALLY_USABLE":"MEDIA_TECHNICALLY_VALID='PASS'",
    "GATE_3_RIGHTS_QUALIFIED":"RIGHTS_KNOWN='PASS' AND RIGHTS_ALLOWED='PASS'",
    "GATE_4_OCCURRENCE_CORE":"OCCURRENCE_CORE_COMPLETE='PASS'",
    "GATE_5_SPATIAL_ANALYSIS":"SPATIALLY_USABLE='PASS'",
    "GATE_6_SPECIES_TRAINING":"AI_CLASSIFICATION_READY='PASS'",
    "GATE_7_PROVENANCE_RICH":"OCCURRENCE_CORE_COMPLETE='PASS' AND MEDIA_TECHNICALLY_VALID='PASS' AND RIGHTS_KNOWN='PASS' AND RIGHTS_ALLOWED='PASS' AND creator_present AND uncertainty_present AND IDENTIFICATION_PROVENANCE_PRESENT='PASS'",
}
DIMENSIONS={"provider":"provider","country":"countryCode","family":"\"family\"","species":"species","taxon_rank":"taxonRank"}


def publish_gate_breakdowns(*,v3_parquet:str|Path,media_quality_parquet:str|Path,ai_readiness_glob:str|Path,output_directory:str|Path,source_snapshot_id:str,expected_rows:int,code_commit:str,memory_limit:str="6GB",threads:int=4,temp_directory:str|Path|None=None)->dict[str,object]:
    source=Path(v3_parquet).resolve();quality=Path(media_quality_parquet).resolve();ai=str(ai_readiness_glob);destination=Path(output_directory).resolve()
    for path in (source,quality):
        if not path.is_file():raise FileNotFoundError(path)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True);staging=destination.parent/f".{destination.name}.{uuid4().hex}.staging";staging.mkdir();temporary=Path(temp_directory).resolve() if temp_directory else staging/"duckdb_tmp";temporary.mkdir(parents=True,exist_ok=True);base=staging/"gate_base.parquet";summary=staging/"gate_summary.parquet";breakdowns=staging/"gate_breakdowns.parquet";failures=staging/"gate_failure_reasons.parquet"
    c=duckdb.connect()
    try:
        c.execute(f"SET threads={threads}");c.execute(f"SET memory_limit={_lit(memory_limit)}");c.execute(f"SET temp_directory={_lit(str(temporary))}");c.execute("SET preserve_insertion_order=false")
        c.execute(_base_sql(source,quality,ai,base));rows=int(c.execute(f"SELECT count(*) FROM read_parquet({_lit(str(base))})").fetchone()[0])
        if rows!=expected_rows:raise ValueError("gate base row mismatch")
        c.execute(_summary_sql(base,summary));c.execute(_breakdown_sql(base,breakdowns));c.execute(_failure_sql(base,failures))
        summary_rows=int(c.execute(f"SELECT count(*) FROM read_parquet({_lit(str(summary))})").fetchone()[0]);breakdown_rows=int(c.execute(f"SELECT count(*) FROM read_parquet({_lit(str(breakdowns))})").fetchone()[0]);failure_rows=int(c.execute(f"SELECT count(*) FROM read_parquet({_lit(str(failures))})").fetchone()[0])
    except BaseException:
        c.close();shutil.rmtree(staging,ignore_errors=True);raise
    finally:
        try:c.close()
        except Exception:pass
    base.unlink(missing_ok=True)
    if temp_directory is None:shutil.rmtree(temporary,ignore_errors=True)
    validation={"source_rows_match":rows==expected_rows,"seven_gate_summaries":summary_rows==7,"all_gate_dimensions_present":breakdown_rows>=7*len(DIMENSIONS),"failure_reasons_present":failure_rows>0,"content_dedup_claims_withheld":True,"manifest_written_last":True}
    if not all(validation.values()):shutil.rmtree(staging,ignore_errors=True);raise ValueError(f"gate validation failed: {validation}")
    artifacts=[_artifact(summary),_artifact(breakdowns),_artifact(failures)];manifest={"schema_version":GATE_VERSION,"generated_at":datetime.now(UTC).isoformat().replace('+00:00','Z'),"code_commit":code_commit,"source_snapshot_id":source_snapshot_id,"inputs":{"v3":str(source),"media_quality":str(quality),"ai_readiness":ai},"counts":{"source_rows":rows,"gate_summaries":summary_rows,"breakdown_rows":breakdown_rows,"failure_reason_rows":failure_rows},"configuration":{"gates":GATES,"dimensions":list(DIMENSIONS),"content_deduplication_status":"NOT_TESTED"},"validation":validation,"artifacts":artifacts,"network_requests":0,"manifest_policy":{"written_last":True}}
    _write_json(staging/"manifest.json",manifest)
    for artifact in artifacts:
        if _sha256(staging/artifact["path"])!=artifact["sha256"]:raise ValueError("gate checksum mismatch")
    os.replace(staging,destination);return manifest


def _base_sql(source,quality,ai,output):
    return f"""COPY (SELECT q.media_assertion_id,trim(cast(v.gbifID AS VARCHAR)) gbifID,
      coalesce(nullif(trim(v.media_publisher),''),nullif(trim(v.publisher),''),'<MISSING>') provider,
      coalesce(nullif(trim(v.countryCode),''),'<MISSING>') countryCode,
      coalesce(nullif(trim(v.family),''),'<MISSING>') "family",
      coalesce(nullif(trim(v.species),''),'<MISSING>') species,
      coalesce(nullif(trim(v.taxonRank),''),'<MISSING>') taxonRank,
      v.media_creator IS NOT NULL AND trim(v.media_creator)<>'' creator_present,
      try_cast(v.coordinateUncertaintyInMeters AS DOUBLE) IS NOT NULL uncertainty_present,
      a.original_url_hash,a.canonical_url_hash,a.reason_codes,
      a.MEDIA_ADDRESSABLE,a.MEDIA_TECHNICALLY_VALID,a.RIGHTS_KNOWN,a.RIGHTS_ALLOWED,
      a.OCCURRENCE_CORE_COMPLETE,a.SPATIALLY_USABLE,a.IDENTIFICATION_PROVENANCE_PRESENT,
      a.AI_CLASSIFICATION_READY
    FROM read_parquet({_lit(str(source))}) v POSITIONAL JOIN read_parquet({_lit(str(quality))}) q
    JOIN read_parquet({_lit(ai)}) a ON q.media_assertion_id=a.media_assertion_id)
    TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"""


def _summary_sql(base,output):
    queries=[]
    for gate,condition in GATES.items():
        queries.append(f"""SELECT {_lit(GATE_VERSION)} gate_version,{_lit(gate)} gate_id,count(*)::BIGINT media_rows,
          count(distinct gbifID)::BIGINT distinct_occurrences,count(distinct original_url_hash)::BIGINT distinct_original_urls,
          count(*) FILTER(WHERE {condition})::BIGINT passed_media_rows,
          count(distinct gbifID) FILTER(WHERE {condition})::BIGINT passed_occurrences,
          count(distinct original_url_hash) FILTER(WHERE {condition})::BIGINT passed_original_urls,
          (count(distinct canonical_url_hash) FILTER(WHERE {condition})+count(*) FILTER(WHERE {condition} AND canonical_url_hash IS NULL))::BIGINT url_adjusted_pass_count,
          NULL::BIGINT exact_content_deduplicated_pass_count,'NOT_TESTED' exact_content_deduplication_status,
          100.0*count(*) FILTER(WHERE {condition})/count(*) pass_percentage
        FROM read_parquet({_lit(str(base))})""")
    return f"COPY ({' UNION ALL '.join(queries)}) TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD)"


def _breakdown_sql(base,output):
    queries=[]
    for gate,condition in GATES.items():
        for dimension,field in DIMENSIONS.items():
            queries.append(f"""SELECT {_lit(GATE_VERSION)} gate_version,{_lit(gate)} gate_id,{_lit(dimension)} dimension,
              cast({field} AS VARCHAR) "value",count(*)::BIGINT media_rows,count(distinct gbifID)::BIGINT distinct_occurrences,
              count(distinct original_url_hash)::BIGINT distinct_original_urls,count(*) FILTER(WHERE {condition})::BIGINT passed_media_rows,
              count(distinct gbifID) FILTER(WHERE {condition})::BIGINT passed_occurrences,
              count(distinct original_url_hash) FILTER(WHERE {condition})::BIGINT passed_original_urls,
              (count(distinct canonical_url_hash) FILTER(WHERE {condition})+count(*) FILTER(WHERE {condition} AND canonical_url_hash IS NULL))::BIGINT url_adjusted_pass_count,
              100.0*count(*) FILTER(WHERE {condition})/count(*) pass_percentage
            FROM read_parquet({_lit(str(base))}) GROUP BY 4""")
    return f"COPY ({' UNION ALL '.join(queries)}) TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD)"


def _failure_sql(base,output):
    status_field={"GATE_1_ADDRESSABLE":"MEDIA_ADDRESSABLE","GATE_2_TECHNICALLY_USABLE":"MEDIA_TECHNICALLY_VALID","GATE_3_RIGHTS_QUALIFIED":"RIGHTS_ALLOWED","GATE_4_OCCURRENCE_CORE":"OCCURRENCE_CORE_COMPLETE","GATE_5_SPATIAL_ANALYSIS":"SPATIALLY_USABLE","GATE_6_SPECIES_TRAINING":"AI_CLASSIFICATION_READY","GATE_7_PROVENANCE_RICH":"AI_CLASSIFICATION_READY"}
    queries=[]
    for gate,condition in GATES.items():
        field=status_field[gate]
        queries.append(f"SELECT {_lit(GATE_VERSION)} gate_version,{_lit(gate)} gate_id,{_lit(field)} reason_code,{field} status,count(*)::BIGINT media_rows,count(distinct gbifID)::BIGINT occurrences FROM read_parquet({_lit(str(base))}) WHERE NOT ({condition}) GROUP BY {field}")
    return f"COPY ({' UNION ALL '.join(queries)} ORDER BY gate_id,media_rows DESC) TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD)"


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


__all__=["DIMENSIONS","GATES","GATE_VERSION","publish_gate_breakdowns"]
