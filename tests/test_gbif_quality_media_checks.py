from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.media_checks import publish_media_assertion_quality
from biominer.gbif_quality.source_ledger import SOURCE_LEDGER_SCHEMA, SOURCE_LEDGER_VERSION


def test_media_checks_align_retained_source_rows_and_keep_unknowns(tmp_path: Path) -> None:
    v3 = tmp_path / "v3.parquet"
    ledger = tmp_path / "ledger.parquet"
    pq.write_table(
        pa.table(
            {
                "gbifID": ["1", "2", "3"],
                "media_identifier": ["https://x/1.jpg", None, "ftp://x/3.jpg"],
                "media_references": ["https://x/1", "https://x/2", None],
                "media_type": ["StillImage", "StillImage", "StillImage"],
                "media_format": ["image/jpeg", None, "text/html"],
                "media_license": ["CC BY 4.0", None, "Copyright statement unclear"],
            }
        ),
        v3,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _ledger_row(0, "excluded-0", "EXCLUDED_PRE_1960"),
                _ledger_row(1, "source-1", "RETAINED_V3"),
                _ledger_row(2, "source-2", "RETAINED_V3"),
                _ledger_row(3, "excluded-3", "EXCLUDED_OUTSIDE_IDENTIFIED_OR_ACCEPTED"),
                _ledger_row(4, "source-3", "RETAINED_V3"),
            ],
            schema=SOURCE_LEDGER_SCHEMA,
        ),
        ledger,
    )

    result = publish_media_assertion_quality(
        v3_parquet=v3,
        source_ledger_parquet=ledger,
        output_directory=tmp_path / "quality",
        source_snapshot_id="sha256:test",
        expected_rows=3,
        code_commit="abc",
        batch_rows=2,
    )

    assert all(result.manifest["validation"].values())
    rows = pq.read_table(result.quality_path).to_pylist()
    assert [row["source_row_id"] for row in rows] == ["source-1", "source-2", "source-3"]
    assert [row["direct_media_url_status"] for row in rows] == ["PASS", "UNKNOWN", "FAIL"]
    assert [row["media_type_format_status"] for row in rows] == ["PASS", "NOT_APPLICABLE", "CONFLICT"]
    assert [row["media_rights_status"] for row in rows] == ["PASS", "UNKNOWN", "UNKNOWN"]
    assert [row["overall_media_quality_status"] for row in rows] == ["PASS", "UNKNOWN", "CONFLICT"]
    assert len({row["media_assertion_id"] for row in rows}) == 3


def _ledger_row(position: int, source_row_id: str, status: str) -> dict[str, object]:
    retained = status == "RETAINED_V3"
    return {
        "ledger_version": SOURCE_LEDGER_VERSION,
        "source_snapshot_id": "sha256:test",
        "source_file": "multimedia.txt",
        "source_sort_position": position,
        "source_row_id": source_row_id,
        "gbifID": str(position),
        "media_join_status": "resolved_occurrence",
        "v3_funnel_status": status,
        "exclusion_reason": "NONE" if retained else "TEST",
        "local_quality_status": "NOT_TESTED" if retained else "NOT_APPLICABLE",
    }
