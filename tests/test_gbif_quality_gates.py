from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.gates import GATES, publish_gate_breakdowns


def test_completeness_gates_keep_content_deduplication_not_tested(tmp_path: Path) -> None:
    source=tmp_path/"source.parquet";quality=tmp_path/"quality.parquet";ai=tmp_path/"ai";ai.mkdir()
    pq.write_table(pa.Table.from_pylist([_source("1"),_source("2")]),source)
    pq.write_table(pa.table({"media_assertion_id":["m1","m2"]}),quality)
    pq.write_table(pa.Table.from_pylist([_ai("m1","PASS"),_ai("m2","NOT_TESTED")]),ai/"part.parquet")
    manifest=publish_gate_breakdowns(v3_parquet=source,media_quality_parquet=quality,ai_readiness_glob=ai/"*.parquet",output_directory=tmp_path/"out",source_snapshot_id="s",expected_rows=2,code_commit="c",threads=1)
    summary=pq.read_table(tmp_path/"out/gate_summary.parquet").to_pylist()
    assert len(summary)==len(GATES)==7
    assert {row["exact_content_deduplication_status"] for row in summary}=={"NOT_TESTED"}
    assert manifest["validation"]["all_gate_dimensions_present"]


def _source(gbif_id):
    return {"gbifID":gbif_id,"media_publisher":"Provider","publisher":"Publisher","countryCode":"AU","family":"Family","species":"Species one","taxonRank":"SPECIES","media_creator":"Creator","coordinateUncertaintyInMeters":"10"}


def _ai(media_id,technical):
    return {"media_assertion_id":media_id,"original_url_hash":media_id,"canonical_url_hash":media_id,"reason_codes":["IMAGE_BYTES_NOT_INSPECTED"] if technical!="PASS" else [],"MEDIA_ADDRESSABLE":"PASS","MEDIA_TECHNICALLY_VALID":technical,"RIGHTS_KNOWN":"PASS","RIGHTS_ALLOWED":"PASS","OCCURRENCE_CORE_COMPLETE":"PASS","SPATIALLY_USABLE":"PASS","IDENTIFICATION_PROVENANCE_PRESENT":"PASS","AI_CLASSIFICATION_READY":"PASS" if technical=="PASS" else "NOT_TESTED"}
