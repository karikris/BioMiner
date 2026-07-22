from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_quality.source_lineage import publish_source_assertion_lineage


def test_source_lineage_publishes_stable_locations_and_value_hashes(tmp_path) -> None:
    multimedia = tmp_path / "multimedia.parquet"
    status = tmp_path / "status.parquet"
    inventory = tmp_path / "source_inventory.json"
    source_manifest = tmp_path / "dwca_manifest.json"
    output = tmp_path / "lineage"
    pq.write_table(
        pa.table(
            {
                "gbifID": ["1", "2", "3"],
                "identifier": ["https://example/1.jpg", None, "https://example/3.jpg"],
                "license": ["CC BY 4.0", "CC0", None],
            }
        ),
        multimedia,
    )
    pq.write_table(
        pa.table(
            {
                "source_sort_position": pa.array([0, 1, 2], type=pa.int64()),
                "source_row_id": ["row-0", "row-1", "row-2"],
                "gbifID": ["1", "2", "3"],
                "media_join_status": ["resolved", "resolved", "resolved"],
                "v3_funnel_status": ["RETAINED", "EXCLUDED", "RETAINED"],
                "exclusion_reason": [None, "TEST", None],
            }
        ),
        status,
    )
    source_manifest.write_text(json.dumps({"generated_at": "2026-07-20T12:42:18Z"}))
    inventory.write_text(
        json.dumps(
            {
                "source_snapshot_id": "sha256:snapshot",
                "source_download_key": "download-key",
                "artifacts": [
                    {
                        "artifact_role": "multimedia_extension",
                        "sha256": "source-file-sha256",
                        "manifest_path": str(source_manifest),
                    }
                ],
            }
        )
    )

    manifest = publish_source_assertion_lineage(
        multimedia_parquet=multimedia,
        source_status_parquet=status,
        source_inventory_json=inventory,
        output_directory=output,
        expected_rows=3,
        code_commit="commit",
        partition_rows=2,
        memory_limit="256MB",
        threads=1,
    )

    rows = pq.read_table(output / "parts").sort_by("source_row_number").to_pylist()
    assert manifest["counts"] == {"rows": 3, "parts": 2}
    assert [row["source_partition"] for row in rows] == [0, 0, 1]
    assert [row["source_row_id"] for row in rows] == ["row-0", "row-1", "row-2"]
    assert all(row["source_value_hash"].startswith("sha256:") for row in rows)
    assert rows[0]["source_download_key"] == "download-key"
    assert rows[0]["source_doi"] is None
    assert rows[0]["ingestion_timestamp"] == "2026-07-20T12:42:18Z"
    assert (output / "manifest.json").is_file()
    with pytest.raises(FileExistsError):
        publish_source_assertion_lineage(
            multimedia_parquet=multimedia,
            source_status_parquet=status,
            source_inventory_json=inventory,
            output_directory=output,
            expected_rows=3,
            code_commit="commit",
            partition_rows=2,
            memory_limit="256MB",
            threads=1,
        )
