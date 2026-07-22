from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.duplicates import publish_duplicate_groups


def test_duplicate_groups_separate_url_taxon_and_content_evidence(tmp_path:Path)->None:
    source=tmp_path/"source.parquet";quality=tmp_path/"quality.parquet"
    pq.write_table(pa.Table.from_pylist([
        _row("1","https://EXAMPLE.org/a.jpg#one","A","CC BY 4.0"),
        _row("2","https://example.org/a.jpg#two","B","CC BY-NC 4.0"),
        _row("3","https://example.org/b.jpg","A","CC BY 4.0"),
    ]),source)
    pq.write_table(pa.table({"source_row_id":["r1","r2","r3"],"media_assertion_id":["m1","m2","m3"]}),quality)
    manifest=publish_duplicate_groups(v3_parquet=source,media_quality_parquet=quality,output_directory=tmp_path/"out",source_snapshot_id="s",expected_rows=3,code_commit="deadbeef")
    rows=pq.read_table(tmp_path/"out/duplicate_membership.parquet").to_pylist()
    assert rows[0]["canonical_url_hash"]==rows[1]["canonical_url_hash"]
    assert rows[0]["original_url_hash"]!=rows[1]["original_url_hash"]
    assert rows[0]["cross_taxon_url_status"]=="CONFLICT"
    assert rows[0]["cross_license_url_status"]=="CONFLICT"
    assert all(row["content_duplicate_status"]=="NOT_TESTED" for row in rows)
    assert manifest["counts"]["rows"]==3


def _row(gbif_id,url,taxon,license_value):
    return {"gbifID":gbif_id,"media_identifier":url,"media_references":None,"acceptedTaxonKey":taxon,"taxonKey":taxon,"media_license":license_value,"datasetKey":"d","occurrenceID":"o"+gbif_id,"media_creator":"creator"}
