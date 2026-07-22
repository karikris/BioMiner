from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


PERFORMANCE_VERSION = "biominer-gbif-media-performance/v1"
BENCHMARK_SCHEMA = pa.schema([
    ("performance_version",pa.string()),("stage",pa.string()),("status",pa.string()),
    ("rows_read",pa.int64()),("rows_written",pa.int64()),("elapsed_seconds",pa.float64()),
    ("rows_per_second",pa.float64()),("input_bytes",pa.int64()),("output_bytes",pa.int64()),
    ("process_peak_rss_bytes",pa.int64()),("memory_limit_bytes",pa.int64()),
    ("result_fingerprint",pa.string()),("note",pa.string()),
])


def publish_performance_benchmark(*,v3_parquet:str|Path,media_quality_parquet:str|Path,occurrence_quality_parquet:str|Path,ai_readiness_glob:str|Path,provider_scorecard_parquet:str|Path,incremental_current_glob:str|Path,incremental_previous_glob:str|Path,output_directory:str|Path,source_snapshot_id:str,expected_media_rows:int,expected_occurrences:int,code_commit:str,memory_limit:str="8GB",threads:int=4,temp_directory:str|Path|None=None)->dict[str,object]:
    paths={"v3":Path(v3_parquet).resolve(),"media_quality":Path(media_quality_parquet).resolve(),"occurrence_quality":Path(occurrence_quality_parquet).resolve(),"provider_scorecard":Path(provider_scorecard_parquet).resolve()}
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(path)
    globs={"ai":str(ai_readiness_glob),"current":str(incremental_current_glob),"previous":str(incremental_previous_glob)}
    destination=Path(output_directory).resolve()
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True);staging=destination.parent/f".{destination.name}.{uuid4().hex}.staging";staging.mkdir()
    temporary=Path(temp_directory).resolve() if temp_directory else staging/"duckdb_tmp";temporary.mkdir(parents=True,exist_ok=True)
    c=duckdb.connect();c.execute(f"SET threads={threads}");c.execute(f"SET memory_limit={_lit(memory_limit)}");c.execute(f"SET temp_directory={_lit(str(temporary))}")
    definitions=[
        ("full_metadata_audit",f"SELECT count(*),count(*) FILTER(WHERE media_identifier IS NOT NULL),count(distinct gbifID) FROM read_parquet({_lit(str(paths['v3']))})",expected_media_rows,_size(paths["v3"]),"Full v3 scan and occurrence cardinality."),
        ("occurrence_level_aggregation",f"SELECT count(*),sum(media_assertion_count) FROM read_parquet({_lit(str(paths['occurrence_quality']))})",expected_occurrences,_size(paths["occurrence_quality"]),"Occurrence-level quality aggregation."),
        ("media_level_aggregation",f"SELECT count(*),count(*) FILTER(WHERE overall_media_quality_status='PASS') FROM read_parquet({_lit(str(paths['media_quality']))})",expected_media_rows,_size(paths["media_quality"]),"Media quality aggregation."),
        ("provider_summary",f"SELECT count(*),sum(media_count),sum(estimated_recoverable_rows) FROM read_parquet({_lit(str(paths['provider_scorecard']))})",110,_size(paths["provider_scorecard"]),"Provider scorecard aggregation."),
        ("join_propagation",f"SELECT count(*),count(*) FILTER(WHERE a.ai_ingestion_decision='UNRESOLVED') FROM read_parquet({_lit(str(paths['media_quality']))}) q JOIN read_parquet({_lit(globs['ai'])}) a USING(media_assertion_id)",expected_media_rows,_size(paths["media_quality"])+_glob_size(globs["ai"]),"Stable media-identity join propagation."),
        ("incremental_rerun",f"SELECT count(*) FROM read_parquet({_lit(globs['current'])}) c JOIN read_parquet({_lit(globs['previous'])}) p USING(media_assertion_id) WHERE c.source_value_hash IS DISTINCT FROM p.source_value_hash OR c.media_url_value_hash IS DISTINCT FROM p.media_url_value_hash OR c.media_rights_value_hash IS DISTINCT FROM p.media_rights_value_hash OR c.spatial_value_hash IS DISTINCT FROM p.spatial_value_hash OR c.temporal_value_hash IS DISTINCT FROM p.temporal_value_hash OR c.identification_value_hash IS DISTINCT FROM p.identification_value_hash OR c.taxonomy_value_hash IS DISTINCT FROM p.taxonomy_value_hash OR c.provider_value_hash IS DISTINCT FROM p.provider_value_hash",expected_media_rows,_glob_size(globs["current"])+_glob_size(globs["previous"]),"Unchanged snapshot diff; result must be zero."),
    ]
    results=[]
    try:
        for stage,sql,rows_read,input_bytes,note in definitions:
            started=time.perf_counter();value=c.execute(sql).fetchall();elapsed=time.perf_counter()-started
            fingerprint="sha256:"+hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
            status="PASS"
            if stage=="incremental_rerun" and value!=[(0,)]:status="FAIL"
            peak=_peak_rss_bytes();limit=_parse_bytes(memory_limit)
            if peak>16*1024**3:status="FAIL"
            results.append({"performance_version":PERFORMANCE_VERSION,"stage":stage,"status":status,"rows_read":int(rows_read),"rows_written":0,"elapsed_seconds":elapsed,"rows_per_second":rows_read/elapsed if elapsed else None,"input_bytes":int(input_bytes),"output_bytes":0,"process_peak_rss_bytes":peak,"memory_limit_bytes":limit,"result_fingerprint":fingerprint,"note":note})
    finally:c.close()
    if temp_directory is None:shutil.rmtree(temporary,ignore_errors=True)
    output=staging/"benchmark_results.parquet";pq.write_table(pa.Table.from_pylist(results,schema=BENCHMARK_SCHEMA),output,compression="zstd")
    validation={"all_stages_pass":all(row["status"]=="PASS" for row in results),"peak_rss_below_16_gib":max(row["process_peak_rss_bytes"] for row in results)<=16*1024**3,"six_required_benchmarks":len(results)==6,"manifest_written_last":True}
    if not all(validation.values()):shutil.rmtree(staging,ignore_errors=True);raise ValueError(f"benchmark validation failed: {validation}")
    artifact=_artifact(output);manifest={"schema_version":PERFORMANCE_VERSION,"generated_at":datetime.now(UTC).isoformat().replace('+00:00','Z'),"code_commit":code_commit,"source_snapshot_id":source_snapshot_id,"configuration":{"memory_limit":memory_limit,"threads":threads},"counts":{"benchmarks":len(results),"peak_rss_bytes":max(row["process_peak_rss_bytes"] for row in results)},"validation":validation,"artifacts":[artifact],"network_requests":0,"manifest_policy":{"written_last":True}}
    _write_json(staging/"manifest.json",manifest)
    if _sha256(output)!=artifact["sha256"]:raise ValueError("benchmark checksum mismatch")
    os.replace(staging,destination);return manifest


def _parse_bytes(value):
    number=float(value[:-2]);unit=value[-2:].upper();return int(number*({"KB":1024,"MB":1024**2,"GB":1024**3}[unit]))
def _peak_rss_bytes():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
def _size(path):return Path(path).stat().st_size
def _glob_size(value):
    import glob
    return sum(Path(path).stat().st_size for path in glob.glob(value,recursive=True))
def _artifact(path):
    p=pq.ParquetFile(path);return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}
def _write_json(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(temp,path)
def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def _lit(value):return "'"+str(value).replace("'","''")+"'"


__all__=["BENCHMARK_SCHEMA","PERFORMANCE_VERSION","publish_performance_benchmark"]
