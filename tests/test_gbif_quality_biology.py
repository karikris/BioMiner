from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.biology import extract_controlled_value, publish_biological_candidates


def test_controlled_extraction_handles_negation_and_multiples() -> None:
    assert extract_controlled_value("fresh adult", target="lifeStage").value == "adult"
    assert extract_controlled_value("not an adult", target="lifeStage").value is None
    assert extract_controlled_value("male and female", target="sex").value == "mixed"
    risky = extract_controlled_value("adult on host plant", target="lifeStage")
    assert risky.confidence == "LOW"


def test_biological_candidates_never_replace_source_values(tmp_path: Path) -> None:
    source=tmp_path/"source.parquet"
    pq.write_table(pa.Table.from_pylist([
        _row("1", None, None, "adult male", None),
        _row("1", None, None, "adult male", None),
        _row("2", None, None, "not an adult", None),
        _row("3", "LARVA", "FEMALE", "adult male", None),
        _row("4", None, None, None, "female and male caterpillar"),
    ]), source)
    result=publish_biological_candidates(
        v3_parquet=source, output_directory=tmp_path/"out",
        source_snapshot_id="sha256:test", expected_media_rows=5,
        expected_occurrences=4, code_commit="deadbeef",
    )
    candidates=pq.read_table(result.candidate_path).to_pylist()
    assert all(row["original_value"] is None for row in candidates)
    assert any(row["candidate_status"] == "NEGATED_ONLY" for row in candidates)
    assert all(row["review_status"] == "PENDING" for row in candidates)
    assert pq.read_table(result.assertion_path).num_rows == 4


def _row(gbif_id, life_stage, sex, remarks, dynamic):
    return {"gbifID":gbif_id,"lifeStage":life_stage,"sex":sex,"occurrenceRemarks":remarks,"dynamicProperties":dynamic}
