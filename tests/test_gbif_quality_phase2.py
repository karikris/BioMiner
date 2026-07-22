from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.phase2 import publish_phase2_summary
from biominer.gbif_quality.source_ledger import SOURCE_LEDGER_SCHEMA, SOURCE_LEDGER_VERSION


def test_phase2_links_all_source_rows_and_withholds_network_claims(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    media = tmp_path / "media.parquet"
    occurrence = tmp_path / "occurrence.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _source_row(0, "s0", "EXCLUDED_PRE_1960"),
                _source_row(1, "s1", "RETAINED_V3"),
                _source_row(2, "s2", "RETAINED_V3"),
                _source_row(3, "s3", "EXCLUDED_EXPLICIT_MEDIA_RIGHTS"),
                _source_row(4, "s4", "RETAINED_V3"),
            ],
            schema=SOURCE_LEDGER_SCHEMA,
        ),
        source,
    )
    pq.write_table(
        pa.table(
            {
                "source_row_id": ["s1", "s2", "s4"],
                "overall_media_quality_status": ["PASS", "UNKNOWN", "CONFLICT"],
            }
        ),
        media,
    )
    pq.write_table(pa.table({"gbifID": ["1", "2"]}), occurrence)
    media_manifest = tmp_path / "media.json"
    occurrence_manifest = tmp_path / "occurrence.json"
    media_manifest.write_text(
        json.dumps({"counts": {"status_counts": {"overall": {"PASS": 1}}}}),
        encoding="utf-8",
    )
    occurrence_manifest.write_text(
        json.dumps({"counts": {"rows": 2}}), encoding="utf-8"
    )

    result = publish_phase2_summary(
        source_ledger_parquet=source,
        media_quality_parquet=media,
        occurrence_quality_parquet=occurrence,
        media_manifest=media_manifest,
        occurrence_manifest=occurrence_manifest,
        output_directory=tmp_path / "phase2",
        source_snapshot_id="sha256:test",
        expected_source_rows=5,
        expected_v3_rows=3,
        expected_occurrences=2,
        code_commit="abc",
        batch_rows=2,
    )

    assert all(result.manifest["validation"].values())
    statuses = pq.read_table(
        result.output_directory / "source_media_quality_status.parquet"
    ).to_pylist()
    assert [row["local_quality_status"] for row in statuses] == [
        "NOT_APPLICABLE",
        "PASS",
        "UNKNOWN",
        "NOT_APPLICABLE",
        "CONFLICT",
    ]
    coverage = pq.read_table(result.output_directory / "check_coverage.parquet")
    network = [
        row for row in coverage.to_pylist() if row["check_id"] == "MEDIA_URL_003"
    ]
    assert network == [
        {
            **network[0],
            "execution_status": "NOT_TESTED",
            "network_requests": 0,
            "evidence_path": None,
        }
    ]


def _source_row(position: int, row_id: str, status: str) -> dict[str, object]:
    retained = status == "RETAINED_V3"
    return {
        "ledger_version": SOURCE_LEDGER_VERSION,
        "source_snapshot_id": "sha256:test",
        "source_file": "multimedia.txt",
        "source_sort_position": position,
        "source_row_id": row_id,
        "gbifID": str(position),
        "media_join_status": "resolved_occurrence",
        "v3_funnel_status": status,
        "exclusion_reason": "NONE" if retained else "TEST",
        "local_quality_status": "NOT_TESTED" if retained else "NOT_APPLICABLE",
    }
