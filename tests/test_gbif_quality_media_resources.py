from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from biominer.gbif_quality.media_resources import publish_media_resources


def test_media_resources_do_not_equate_urls_with_content(tmp_path: Path) -> None:
    source=tmp_path/"duplicates.parquet"
    pq.write_table(pa.Table.from_pylist([
        _row("m1","1","h","https://x/a.jpg","PASS"),
        _row("m2","2","h","https://x/a.jpg","CONFLICT"),
        _row("m3","3",None,None,"PASS"),
    ]),source)
    manifest=publish_media_resources(duplicates_parquet=source,output_directory=tmp_path/"out",source_snapshot_id="s",expected_assertion_rows=3,code_commit="c",threads=1,partitions=2)
    rows=ds.dataset(tmp_path/"out/parts",format="parquet",partitioning="hive").to_table().to_pylist()
    assert len(rows)==1 and rows[0]["assertion_count"]==2
    assert rows[0]["content_identity_status"]=="NOT_TESTED"
    assert rows[0]["perceptual_identity_status"]=="NOT_TESTED"
    assert rows[0]["cross_taxon_status"]=="CONFLICT"
    assert manifest["counts"]["unresolved_reference_only_assertions"]==1


def _row(media_id,gbif_id,url_hash,url,cross_taxon):
    return {"media_assertion_id":media_id,"gbifID":gbif_id,"canonical_url_hash":url_hash,"canonical_url":url,"source_platform_group_id":"platform","url_distinct_taxa":2 if cross_taxon=="CONFLICT" else 1,"url_distinct_licenses":1,"cross_taxon_url_status":cross_taxon,"cross_license_url_status":"PASS"}
