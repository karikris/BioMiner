from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.rights import normalize_media_license, publish_media_rights


def test_media_license_normalization_is_explicit_and_fail_closed() -> None:
    allowed=normalize_media_license("https://creativecommons.org/licenses/by/4.0/")
    assert allowed.family=="cc-by" and allowed.policy_status=="ALLOWED"
    research=normalize_media_license("CC BY-NC 4.0")
    assert research.commercial=="NO" and research.policy_status=="RESEARCH_ONLY"
    denied=normalize_media_license("CC BY-ND 4.0")
    assert denied.derivatives=="NO" and denied.policy_status=="DENIED"
    assert normalize_media_license("All rights reserved").policy_status=="DENIED"
    assert normalize_media_license(None).policy_status=="QUARANTINED"
    assert normalize_media_license("provider terms").policy_status=="QUARANTINED"


def test_rights_publication_keeps_occurrence_license_separate(tmp_path: Path) -> None:
    source=tmp_path/"source.parquet"; quality=tmp_path/"quality.parquet"
    pq.write_table(pa.Table.from_pylist([
        _row("1","CC BY 4.0","CC0","A. Creator",None,"https://x/1.jpg"),
        _row("2",None,"CC BY 4.0",None,None,"https://x/2.jpg"),
        _row("3","All rights reserved","CC0",None,"Holder",None),
    ]),source)
    pq.write_table(pa.table({"source_row_id":["r1","r2","r3"],"media_assertion_id":["m1","m2","m3"]}),quality)
    manifest=publish_media_rights(v3_parquet=source,media_quality_parquet=quality,output_directory=tmp_path/"out",source_snapshot_id="snapshot",expected_rows=3,code_commit="deadbeef",batch_rows=2)
    rows=pq.read_table(tmp_path/"out/media_rights.parquet").to_pylist()
    assert rows[0]["rights_policy_status"]=="ALLOWED"
    assert rows[1]["original_media_license"] is None
    assert rows[1]["original_occurrence_license"]=="CC BY 4.0"
    assert rows[1]["rights_policy_status"]=="QUARANTINED"
    assert rows[2]["rights_policy_status"]=="DENIED"
    assert manifest["counts"]["rows"]==3


def _row(gbif_id,media_license,occurrence_license,creator,holder,url):
    return {"gbifID":gbif_id,"media_identifier":url,"media_license":media_license,"license":occurrence_license,"media_creator":creator,"media_rightsHolder":holder,"media_publisher":"Provider","publisher":"Publisher"}
