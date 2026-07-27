from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_quality.source_ledger import publish_source_media_ledger


def test_source_ledger_assigns_every_funnel_reason(tmp_path: Path) -> None:
    joined = tmp_path / "joined.parquet"
    normalized = tmp_path / "normalized.parquet"
    pq.write_table(
        pa.table(
            {
                "gbifID": ["1", "2", "3", "4", "5"],
                "year": ["1950", "2020", "2021", "2022", None],
            }
        ),
        joined,
    )
    pq.write_table(
        pa.table(
            {
                "identifiedBy": [None, "expert", "expert", None],
                "identificationVerificationStatus": ["pending", None, None, "accepted"],
                "media_identifier": ["https://x/2", "https://x/3", "https://x/4", None],
                "media_license": ["CC BY", "All rights reserved", "CC0", None],
            }
        ),
        normalized,
    )
    output = tmp_path / "ledger"

    result = publish_source_media_ledger(
        joined_parquet=joined,
        normalized_parquet=normalized,
        output_directory=output,
        source_snapshot_id="sha256:test",
        expected_counts={
            "raw_multimedia_rows": 5,
            "pre_1960_media_rows_excluded": 1,
            "legacy_cohort_rows_excluded": 1,
            "explicit_rights_rows_excluded": 1,
            "v3_media_rows": 2,
        },
        code_commit="abc",
        memory_limit="1GB",
    )

    assert all(result.manifest["validation"].values())
    table = pq.read_table(result.ledger_path)
    assert table.num_rows == 5
    assert len(set(table["source_row_id"].to_pylist())) == 5
    assert table["v3_funnel_status"].to_pylist() == [
        "EXCLUDED_PRE_1960",
        "EXCLUDED_OUTSIDE_IDENTIFIED_OR_ACCEPTED",
        "EXCLUDED_EXPLICIT_MEDIA_RIGHTS",
        "RETAINED_V3",
        "RETAINED_V3",
    ]
    assert table["local_quality_status"].to_pylist() == [
        "NOT_APPLICABLE",
        "NOT_APPLICABLE",
        "NOT_APPLICABLE",
        "NOT_TESTED",
        "NOT_TESTED",
    ]
    assert (output / "manifest.json").stat().st_mtime_ns >= result.ledger_path.stat().st_mtime_ns

    with pytest.raises(FileExistsError):
        publish_source_media_ledger(
            joined_parquet=joined,
            normalized_parquet=normalized,
            output_directory=output,
            source_snapshot_id="sha256:test",
            expected_counts={},
            code_commit="abc",
        )
