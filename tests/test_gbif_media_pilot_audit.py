from pathlib import Path
import json

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_media_resolution.pipeline import PILOT_SELECTION_SCHEMA
from biominer.gbif_media_resolution.pilot_audit import publish_pilot_preflight_audit, wilson_interval


def test_wilson_interval_is_bounded_and_empty_aware() -> None:
    assert wilson_interval(0,0)==(None,None)
    low,high=wilson_interval(99,100)
    assert 0 < low < 0.99 < high <= 1


def test_pilot_preflight_retains_every_row_without_network_claims(tmp_path: Path) -> None:
    selection=tmp_path/"pilot.parquet"
    rows=[_row("1",False),_row("2",True)]
    pq.write_table(pa.Table.from_pylist(rows,schema=PILOT_SELECTION_SCHEMA),selection)
    import hashlib
    sha=hashlib.sha256(selection.read_bytes()).hexdigest()
    receipt=tmp_path/"receipt.json"
    receipt.write_text(json.dumps({"work_rows":2,"source_artifact_sha256":"sha256:source","pilot_selection_artifact":{"physical_sha256":"sha256:"+sha}}))
    manifest=publish_pilot_preflight_audit(prepare_receipt=receipt,pilot_selection=selection,output_directory=tmp_path/"out",expected_rows=2,code_commit="deadbeef")
    assert manifest["overall_acceptance_status"]=="NOT_TESTED"
    assert manifest["counts"]["pending_manual_reviews"]==1
    assert manifest["network_requests"]==0


def _row(gbif_id,blocked):
    return {"source_row_id":"r"+gbif_id,"gbifID":gbif_id,"media_references":"https://example.org/"+gbif_id,"media_host":"example.org","host_population_rows":2,"host_size_band":"small","provider":"p","publisher":"p","dataset_name":"d","url_pattern":"extensionless_reference","license_state":"explicitly_restricted" if blocked else "item_media_license","reference_type":"html_or_unknown_reference","taxon_rank":"SPECIES","country_code":"AU","expected_adapter":"generic_structured_or_gbif","rights_blocked":blocked,"selection_stratum":"s","selection_hash":"sha256:h"+gbif_id}
