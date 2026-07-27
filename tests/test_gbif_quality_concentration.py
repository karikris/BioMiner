from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_quality.concentration import publish_concentration_metrics


def test_concentration_metrics_are_explicit_and_bounded(tmp_path) -> None:
    source = tmp_path / "v3.parquet"
    quality = tmp_path / "quality.parquet"
    ai_dir = tmp_path / "ai"
    ai_dir.mkdir()
    output = tmp_path / "concentration"
    pq.write_table(
        pa.table(
            {
                "gbifID": ["1", "2", "3", "4"],
                "species": ["A", "A", "A", "B"],
                "media_publisher": ["P1", "P1", "P2", "P3"],
                "publisher": [None, None, None, None],
                "media_creator": ["C1", "C1", "C1", None],
                "gbifRegion": ["R1", "R1", "R2", "R2"],
                "continent": [None, None, None, None],
                "countryCode": ["X", "X", "Y", "Y"],
                "year": [2001, 2002, 2011, None],
            }
        ),
        source,
    )
    pq.write_table(pa.table({"media_assertion_id": ["m1", "m2", "m3", "m4"]}), quality)
    pq.write_table(
        pa.table(
            {
                "media_assertion_id": ["m1", "m2", "m3", "m4"],
                "RIGHTS_ALLOWED": ["PASS", "PASS", "FAIL", "PASS"],
            }
        ),
        ai_dir / "part.parquet",
    )

    manifest = publish_concentration_metrics(
        v3_parquet=source,
        media_quality_parquet=quality,
        ai_readiness_glob=ai_dir / "*.parquet",
        output_directory=output,
        source_snapshot_id="sha256:snapshot",
        expected_rows=4,
        code_commit="commit",
        memory_limit="256MB",
        threads=1,
    )

    rows = pq.read_table(output / "concentration_metrics.parquet").to_pylist()
    provider = next(
        row
        for row in rows
        if row["species"] == "A"
        and row["cohort"] == "ALL_MEDIA"
        and row["concentration_dimension"] == "provider"
    )
    assert provider["media_rows"] == 3
    assert provider["distinct_values"] == 2
    assert provider["max_value_share"] == pytest.approx(2 / 3)
    assert provider["hhi"] == pytest.approx(5 / 9)
    assert manifest["counts"]["dimensions"] == 4
    assert manifest["configuration"]["technically_usable_cohort_status"] == "NOT_TESTED"
