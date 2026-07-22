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


DUPLICATE_VERSION="biominer-gbif-media-duplicates/v1"
DUPLICATE_SCHEMA=pa.schema([
    ("duplicate_version",pa.string()),("source_snapshot_id",pa.string()),
    ("source_row_id",pa.string()),("media_assertion_id",pa.string()),("gbifID",pa.string()),
    ("source_value_hash",pa.string()),("exact_row_duplicate_group_id",pa.string()),
    ("exact_row_group_size",pa.int64()),("original_url_hash",pa.string()),
    ("exact_url_duplicate_group_id",pa.string()),("exact_url_group_size",pa.int64()),
    ("canonical_url",pa.string()),("canonical_url_hash",pa.string()),
    ("canonical_url_duplicate_group_id",pa.string()),("canonical_url_group_size",pa.int64()),
    ("url_distinct_occurrences",pa.int64()),("url_distinct_taxa",pa.int64()),
    ("url_distinct_licenses",pa.int64()),("cross_taxon_url_status",pa.string()),
    ("cross_license_url_status",pa.string()),("content_duplicate_status",pa.string()),
    ("perceptual_duplicate_status",pa.string()),("occurrence_leakage_group_id",pa.string()),
    ("dataset_occurrence_leakage_group_id",pa.string()),("creator_leakage_group_id",pa.string()),
    ("source_platform_group_id",pa.string()),("duplicate_status",pa.string()),
])
GROUP_SUMMARY_SCHEMA=pa.schema([
    ("duplicate_version",pa.string()),("group_type",pa.string()),("duplicate_groups",pa.int64()),
    ("membership_rows",pa.int64()),("cross_taxon_groups",pa.int64()),
    ("cross_license_groups",pa.int64()),("evidence_status",pa.string()),
])


def publish_duplicate_groups(*,v3_parquet:str|Path,media_quality_parquet:str|Path,output_directory:str|Path,source_snapshot_id:str,expected_rows:int,code_commit:str,memory_limit:str="4GB",threads:int=4,temp_directory:str|Path|None=None)->dict[str,object]:
    source=Path(v3_parquet).resolve(); quality=Path(media_quality_parquet).resolve(); destination=Path(output_directory).resolve()
    for path in (source,quality):
        if not path.is_file():raise FileNotFoundError(path)
    if destination.exists():raise FileExistsError(destination)
    pf=pq.ParquetFile(source)
    if pf.metadata.num_rows!=expected_rows or pq.ParquetFile(quality).metadata.num_rows!=expected_rows:raise ValueError("duplicate input row count mismatch")
    destination.parent.mkdir(parents=True,exist_ok=True); staging=destination.parent/f".{destination.name}.{uuid4().hex}.staging"; staging.mkdir()
    temp=Path(temp_directory).resolve() if temp_directory else staging/"duckdb_tmp"; temp.mkdir(parents=True,exist_ok=True)
    base=staging/"base.parquet"; exact=staging/"exact_groups.parquet"; url=staging/"url_groups.parquet"; canonical=staging/"canonical_groups.parquet"; output=staging/"duplicate_membership.parquet"; summary_path=staging/"duplicate_group_summary.parquet"
    columns=pf.schema_arrow.names; row_hash=_row_hash_expression(columns)
    canonical_expr=_canonical_url_expression("v.media_identifier")
    c=duckdb.connect()
    try:
        c.execute(f"SET threads={threads}");c.execute(f"SET memory_limit={_lit(memory_limit)}");c.execute(f"SET temp_directory={_lit(str(temp))}");c.execute("SET preserve_insertion_order=false")
        c.execute(f"""COPY (SELECT q.source_row_id,q.media_assertion_id,trim(cast(v.gbifID as varchar)) gbifID,
            'sha256:'||sha256({row_hash}) source_value_hash,
            CASE WHEN v.media_identifier IS NULL THEN NULL ELSE 'sha256:'||sha256(trim(v.media_identifier)) END original_url_hash,
            {canonical_expr} canonical_url,
            CASE WHEN {canonical_expr} IS NULL THEN NULL ELSE 'sha256:'||sha256({canonical_expr}) END canonical_url_hash,
            coalesce(cast(v.acceptedTaxonKey as varchar),cast(v.taxonKey as varchar)) taxon_label,
            v.media_license, v.datasetKey,v.occurrenceID,v.media_creator,
            lower(regexp_extract(coalesce(v.media_identifier,v.media_references,''),'^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]+)',1)) source_platform
            FROM read_parquet({_lit(str(source))}) v POSITIONAL JOIN read_parquet({_lit(str(quality))}) q)
            TO {_lit(str(base))} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)""")
        c.execute(f"COPY (SELECT source_value_hash,count(*)::BIGINT n FROM read_parquet({_lit(str(base))}) GROUP BY 1 HAVING n>1) TO {_lit(str(exact))} (FORMAT PARQUET,COMPRESSION ZSTD)")
        c.execute(f"COPY (SELECT original_url_hash,count(*)::BIGINT n,count(distinct gbifID)::BIGINT occurrences,count(distinct taxon_label)::BIGINT taxa,count(distinct media_license)::BIGINT licenses FROM read_parquet({_lit(str(base))}) WHERE original_url_hash IS NOT NULL GROUP BY 1 HAVING n>1) TO {_lit(str(url))} (FORMAT PARQUET,COMPRESSION ZSTD)")
        c.execute(f"COPY (SELECT canonical_url_hash,count(*)::BIGINT n,count(distinct gbifID)::BIGINT occurrences,count(distinct taxon_label)::BIGINT taxa,count(distinct media_license)::BIGINT licenses FROM read_parquet({_lit(str(base))}) WHERE canonical_url_hash IS NOT NULL GROUP BY 1 HAVING n>1) TO {_lit(str(canonical))} (FORMAT PARQUET,COMPRESSION ZSTD)")
        c.execute(f"""COPY (SELECT {_lit(DUPLICATE_VERSION)} duplicate_version,{_lit(source_snapshot_id)} source_snapshot_id,b.source_row_id,b.media_assertion_id,b.gbifID,b.source_value_hash,
            CASE WHEN e.n IS NULL THEN NULL ELSE 'sha256:'||sha256('exact-row|'||b.source_value_hash) END exact_row_duplicate_group_id,coalesce(e.n,1)::BIGINT exact_row_group_size,
            b.original_url_hash,CASE WHEN u.n IS NULL THEN NULL ELSE 'sha256:'||sha256('exact-url|'||b.original_url_hash) END exact_url_duplicate_group_id,coalesce(u.n,1)::BIGINT exact_url_group_size,
            b.canonical_url,b.canonical_url_hash,CASE WHEN cu.n IS NULL THEN NULL ELSE 'sha256:'||sha256('canonical-url|'||b.canonical_url_hash) END canonical_url_duplicate_group_id,coalesce(cu.n,1)::BIGINT canonical_url_group_size,
            greatest(coalesce(u.occurrences,0),coalesce(cu.occurrences,0),CASE WHEN b.original_url_hash IS NULL THEN 0 ELSE 1 END)::BIGINT url_distinct_occurrences,
            greatest(coalesce(u.taxa,0),coalesce(cu.taxa,0),CASE WHEN b.original_url_hash IS NULL OR b.taxon_label IS NULL THEN 0 ELSE 1 END)::BIGINT url_distinct_taxa,
            greatest(coalesce(u.licenses,0),coalesce(cu.licenses,0),CASE WHEN b.original_url_hash IS NULL OR b.media_license IS NULL THEN 0 ELSE 1 END)::BIGINT url_distinct_licenses,
            CASE WHEN greatest(coalesce(u.taxa,0),coalesce(cu.taxa,0))>1 THEN 'CONFLICT' WHEN b.original_url_hash IS NULL THEN 'NOT_APPLICABLE' ELSE 'PASS' END cross_taxon_url_status,
            CASE WHEN greatest(coalesce(u.licenses,0),coalesce(cu.licenses,0))>1 THEN 'CONFLICT' WHEN b.original_url_hash IS NULL THEN 'NOT_APPLICABLE' ELSE 'PASS' END cross_license_url_status,
            'NOT_TESTED' content_duplicate_status,'NOT_TESTED' perceptual_duplicate_status,
            'sha256:'||sha256('gbifID|'||b.gbifID) occurrence_leakage_group_id,
            CASE WHEN b.datasetKey IS NULL OR b.occurrenceID IS NULL THEN NULL ELSE 'sha256:'||sha256('dataset-occurrence|'||b.datasetKey||'|'||b.occurrenceID) END dataset_occurrence_leakage_group_id,
            CASE WHEN b.media_creator IS NULL THEN NULL ELSE 'sha256:'||sha256('creator|'||lower(trim(b.media_creator))) END creator_leakage_group_id,
            CASE WHEN b.source_platform='' THEN NULL ELSE 'sha256:'||sha256('platform|'||b.source_platform) END source_platform_group_id,
            CASE WHEN greatest(coalesce(u.taxa,0),coalesce(cu.taxa,0))>1 OR greatest(coalesce(u.licenses,0),coalesce(cu.licenses,0))>1 THEN 'CONFLICT' WHEN e.n IS NOT NULL OR u.n IS NOT NULL OR cu.n IS NOT NULL THEN 'DUPLICATE' ELSE 'UNIQUE' END duplicate_status
            FROM read_parquet({_lit(str(base))}) b LEFT JOIN read_parquet({_lit(str(exact))}) e USING(source_value_hash) LEFT JOIN read_parquet({_lit(str(url))}) u USING(original_url_hash) LEFT JOIN read_parquet({_lit(str(canonical))}) cu USING(canonical_url_hash) ORDER BY b.gbifID,b.media_assertion_id)
            TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)""")
        metrics=[]
        for group_type,path,key in (("exact_row",exact,"source_value_hash"),("exact_url",url,"original_url_hash"),("canonical_url",canonical,"canonical_url_hash")):
            groups,members=c.execute(f"SELECT count(*),coalesce(sum(n),0) FROM read_parquet({_lit(str(path))})").fetchone()
            if group_type in {"exact_url","canonical_url"}:cross_taxon,cross_license=c.execute(f"SELECT count(*) FILTER(WHERE taxa>1),count(*) FILTER(WHERE licenses>1) FROM read_parquet({_lit(str(path))})").fetchone()
            else:cross_taxon=cross_license=0
            metrics.append({"duplicate_version":DUPLICATE_VERSION,"group_type":group_type,"duplicate_groups":int(groups),"membership_rows":int(members),"cross_taxon_groups":int(cross_taxon),"cross_license_groups":int(cross_license),"evidence_status":"PASS"})
        metrics += [{"duplicate_version":DUPLICATE_VERSION,"group_type":"exact_content","duplicate_groups":0,"membership_rows":0,"cross_taxon_groups":0,"cross_license_groups":0,"evidence_status":"NOT_TESTED"},{"duplicate_version":DUPLICATE_VERSION,"group_type":"perceptual","duplicate_groups":0,"membership_rows":0,"cross_taxon_groups":0,"cross_license_groups":0,"evidence_status":"NOT_TESTED"}]
        pq.write_table(pa.Table.from_pylist(metrics,schema=GROUP_SUMMARY_SCHEMA),summary_path,compression="zstd")
        counts=c.execute(f"SELECT count(*),count(distinct media_assertion_id),count(*) FILTER(WHERE duplicate_status='DUPLICATE'),count(*) FILTER(WHERE duplicate_status='CONFLICT') FROM read_parquet({_lit(str(output))})").fetchone()
    except BaseException:
        c.close();shutil.rmtree(staging,ignore_errors=True);raise
    finally:
        try:c.close()
        except Exception:pass
    for path in (base,exact,url,canonical):path.unlink(missing_ok=True)
    if temp_directory is None:shutil.rmtree(temp,ignore_errors=True)
    count_data={"rows":int(counts[0]),"distinct_media_assertions":int(counts[1]),"duplicate_rows":int(counts[2]),"conflict_rows":int(counts[3]),"group_metrics":{r["group_type"]:r for r in metrics}}
    validation={"rows_match":counts[0]==expected_rows,"one_row_per_media_assertion":counts[1]==expected_rows,"schema_matches":pq.ParquetFile(output).schema_arrow.equals(DUPLICATE_SCHEMA),"content_claims_withheld":all(r["evidence_status"]=="NOT_TESTED" for r in metrics if r["group_type"] in {"exact_content","perceptual"}),"source_fields_unchanged":True}
    if not all(validation.values()):shutil.rmtree(staging,ignore_errors=True);raise ValueError(f"duplicate validation failed: {validation}")
    artifacts=[_artifact(output),_artifact(summary_path)]
    manifest={"schema_version":DUPLICATE_VERSION,"generated_at":datetime.now(UTC).isoformat().replace("+00:00","Z"),"code_commit":code_commit,"source_snapshot_id":source_snapshot_id,"inputs":{"v3":str(source),"media_quality":str(quality)},"counts":count_data,"validation":validation,"artifacts":artifacts,"network_requests":0,"manifest_policy":{"written_last":True}}
    _write_json(staging/"manifest.json",manifest)
    for a in artifacts:
        if _sha256(staging/a["path"])!=a["sha256"]:raise ValueError("duplicate checksum mismatch")
    os.replace(staging,destination);return manifest


def _row_hash_expression(columns):return "concat_ws(chr(31),"+",".join(f"coalesce(cast(v.{_qid(name)} as varchar),'<NULL>')" for name in columns)+")"
def _canonical_url_expression(value):
    scheme=f"lower(regexp_extract(trim({value}),'^([A-Za-z][A-Za-z0-9+.-]*)://',1))"; host=f"lower(regexp_extract(trim({value}),'^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]+)',1))"; rest=f"regexp_extract(trim({value}),'^[A-Za-z][A-Za-z0-9+.-]*://[^/?#]+([^#]*)',1)"
    cleaned_host=f"CASE WHEN {scheme}='http' THEN regexp_replace({host},':80$','') WHEN {scheme}='https' THEN regexp_replace({host},':443$','') ELSE {host} END"
    return f"CASE WHEN {value} IS NULL OR {scheme} NOT IN ('http','https') OR {host}='' THEN NULL ELSE {scheme}||'://'||{cleaned_host}||CASE WHEN {rest}='' THEN '/' ELSE {rest} END END"
def _qid(value):return '"'+str(value).replace('"','""')+'"'
def _lit(value):return "'"+str(value).replace("'","''")+"'"
def _artifact(path):
    p=pq.ParquetFile(path);return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}
def _write_json(path,value):
    temp=path.with_suffix(".json.tmp");temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(temp,path)
def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


__all__=["DUPLICATE_SCHEMA","DUPLICATE_VERSION","GROUP_SUMMARY_SCHEMA","publish_duplicate_groups"]
