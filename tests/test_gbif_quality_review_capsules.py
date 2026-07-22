from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.review_capsules import publish_review_capsules


def test_review_capsules_are_deterministic_and_keep_decisions_blank(tmp_path: Path) -> None:
    source=tmp_path/"source.parquet";quality=tmp_path/"quality.parquet";rights=tmp_path/"rights.parquet";duplicates=tmp_path/"duplicates.parquet";ai=tmp_path/"ai";ai.mkdir()
    pq.write_table(pa.Table.from_pylist([{"gbifID":"1","media_publisher":"Provider","publisher":"Publisher","media_identifier":"https://x/1.jpg","media_references":None,"media_license":None,"media_creator":None,"media_rightsHolder":None}]),source)
    pq.write_table(pa.table({"source_row_id":["r1"],"media_assertion_id":["m1"]}),quality)
    pq.write_table(pa.Table.from_pylist([{"canonical_media_license_uri":None,"normalized_media_creator":None,"normalized_media_rightsHolder":None,"rights_policy_status":"QUARANTINED","rights_policy_reason":"missing_media_license","license_normalization_status":"UNKNOWN","attribution_status":"UNKNOWN"}]),rights)
    pq.write_table(pa.Table.from_pylist([{"media_assertion_id":"m1","duplicate_status":"CONFLICT","cross_taxon_url_status":"CONFLICT","cross_license_url_status":"PASS","canonical_url_hash":"h"}]),duplicates)
    pq.write_table(pa.Table.from_pylist([{"media_assertion_id":"m1","ai_ingestion_decision":"UNRESOLVED","reason_codes":["MEDIA_RIGHTS_UNRESOLVED"]}]),ai/"part.parquet")
    kwargs=dict(v3_parquet=source,media_quality_parquet=quality,rights_parquet=rights,duplicates_parquet=duplicates,ai_readiness_glob=ai/"*.parquet",source_snapshot_id="s",expected_rows=1,code_commit="c",sample_seed="seed",max_per_stratum=2,threads=1)
    first=publish_review_capsules(output_directory=tmp_path/"first",**kwargs);second=publish_review_capsules(output_directory=tmp_path/"second",**kwargs)
    a=pq.read_table(tmp_path/"first/review_capsules.parquet");b=pq.read_table(tmp_path/"second/review_capsules.parquet")
    assert a.equals(b)
    assert set(a.column("review_domain").to_pylist())=={"media_license","media_creator","media_rightsHolder","duplicate_conflict","ai_exclusion"}
    assert set(a.column("review_status").to_pylist())=={"PENDING"}
    assert a.column("reviewer_decision").null_count==a.num_rows
    assert first["validation"]["capsule_ids_unique"] and second["validation"]["review_fields_blank"]
