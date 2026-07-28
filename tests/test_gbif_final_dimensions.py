from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_final.dimensions import (
    build_derived_assertion_dimension,
)


def _assertions(path: Path) -> Path:
    pq.write_table(
        pa.table(
            {
                "gbifID": ["g2", "g1", "g1"],
                "assertion_id": ["a3", "a2", "a1"],
                "target_field": ["species", "year", "day"],
                "original_value": [None, None, None],
                "derived_value": ["Beta", "2020", "4"],
                "evidence_source": ["taxonomy", "eventDate", "eventDate"],
                "derivation_method": ["pinned", "parse", "parse"],
                "derivation_rule_version": ["v1", "v1", "v1"],
                "confidence_class": [
                    "DIRECT_SOURCE",
                    "DETERMINISTIC_DERIVATION",
                    "DETERMINISTIC_DERIVATION",
                ],
                "validation_status": ["PASS", "PASS", "PASS"],
                "conflict_status": [
                    "NOT_APPLICABLE",
                    "NOT_APPLICABLE",
                    "NOT_APPLICABLE",
                ],
                "reviewer_status": ["NOT_REVIEWED", "NOT_REVIEWED", "NOT_REVIEWED"],
            }
        ),
        path,
        compression="zstd",
        row_group_size=2,
    )
    return path


def test_derived_assertion_dimension_is_unique_ordered_and_resumable(
    tmp_path: Path,
) -> None:
    source = _assertions(tmp_path / "assertions.parquet")
    output = tmp_path / "derived-by-occurrence.parquet"

    first = build_derived_assertion_dimension(
        source_assertions=source,
        output_path=output,
        producer_git_sha="deadbeef",
        batch_rows=1,
    )
    mtime = output.stat().st_mtime_ns
    second = build_derived_assertion_dimension(
        source_assertions=source,
        output_path=output,
        producer_git_sha="deadbeef",
        batch_rows=2,
    )

    assert first == second
    assert output.stat().st_mtime_ns == mtime
    assert first["artifact"]["row_count"] == 2
    table = pq.read_table(output)
    assert table["dimension_ordinal"].to_pylist() == [0, 1]
    assert table["gbifID"].to_pylist() == ["g1", "g2"]
    g1 = table["derived_quality_assertions"][0].as_py()
    assert [row["assertion_id"] for row in g1] == ["a1", "a2"]
    assert [row["target_field"] for row in g1] == ["day", "year"]
