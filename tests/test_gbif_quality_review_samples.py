from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.assertions import assertion_table, build_assertion
from biominer.gbif_quality.biology import CANDIDATE_SCHEMA
from biominer.gbif_quality.geography import GEOGRAPHIC_OUTCOME_SCHEMA
from biominer.gbif_quality.review_samples import build_manual_review_sample
from biominer.gbif_quality.taxonomy import TAXONOMIC_REPAIR_SCHEMA


def test_manual_review_sample_is_deterministic_and_stratified(tmp_path: Path) -> None:
    timestamp="2026-01-01T00:00:00Z"
    assertions=[build_assertion(source_snapshot_version="s",source_row_id=f"r{i}",gbif_id=str(i),target_field="derived_year",original_value=None,derived_value=2000+i,evidence_source="eventDate",derivation_method="m",derivation_rule_version="v",confidence_class="DETERMINISTIC_DERIVATION",validation_status="PASS",conflict_status="PASS",retrieval_timestamp=timestamp) for i in range(5)]
    temporal=tmp_path/"temporal.parquet"; pq.write_table(assertion_table(assertions),temporal)
    geo=tmp_path/"geo.parquet"; pq.write_table(pa.Table.from_pylist([_geo()],schema=GEOGRAPHIC_OUTCOME_SCHEMA),geo)
    tax=tmp_path/"tax.parquet"; pq.write_table(pa.Table.from_pylist([_tax()],schema=TAXONOMIC_REPAIR_SCHEMA),tax)
    bio=tmp_path/"bio.parquet"; pq.write_table(pa.Table.from_pylist([_bio()],schema=CANDIDATE_SCHEMA),bio)
    kwargs=dict(temporal_assertions=temporal,geographic_outcomes=geo,taxonomic_repairs=tax,biological_candidates=bio,sample_seed="stable",max_per_stratum=2)
    first=build_manual_review_sample(**kwargs); second=build_manual_review_sample(**kwargs)
    assert first.equals(second)
    assert first.num_rows == 5
    assert set(first.column("review_domain").to_pylist()) == {"temporal","geography","taxonomy","biology"}


def _geo():
    return {name: None for name in GEOGRAPHIC_OUTCOME_SCHEMA.names} | {"geography_version":"v","source_snapshot_id":"s","source_row_id":"g","gbifID":"g","affected_media_rows":1,"country_derivation_status":"NOT_TESTED","country_derivation_reason":"no_boundary","continent_derivation_status":"NOT_APPLICABLE","continent_derivation_reason":"present","gbif_region_derivation_status":"NOT_APPLICABLE","gbif_region_derivation_reason":"present","geographic_conflict_status":"PASS","geographic_conflict_fields":[],"border_ambiguity_status":"NOT_TESTED"}


def _tax():
    return {name: None for name in TAXONOMIC_REPAIR_SCHEMA.names} | {"repair_version":"v","source_snapshot_id":"s","source_row_id":"t","gbifID":"t","affected_media_rows":1,"source_taxon_rank":"SPECIES","derived_species":"A b","derivation_status":"PASS","derivation_reason":"direct","backbone_snapshot":"s","reviewer_status":"NOT_REQUIRED"}


def _bio():
    return {name: None for name in CANDIDATE_SCHEMA.names} | {"candidate_version":"v","candidate_id":"b","source_snapshot_id":"s","source_row_id":"b","gbifID":"b","affected_media_rows":1,"target_field":"derived_sex","derived_value":"female","source_text_field":"occurrenceRemarks","source_text":"female","matched_text_spans":["female"],"extraction_rules":["r"],"rule_version":"v","language":"und","confidence":"MEDIUM","candidate_status":"CANDIDATE","candidate_reason":"match","review_status":"PENDING"}
