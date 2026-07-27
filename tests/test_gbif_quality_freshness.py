from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.freshness import publish_freshness_audit


def test_freshness_audit_classifies_provider_and_derived_evidence(tmp_path) -> None:
    source = tmp_path / "v3.parquet"
    inventory = tmp_path / "source_inventory.json"
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "freshness"
    pq.write_table(
        pa.table(
            {
                "gbifID": ["1", "2", "3"],
                "media_publisher": ["P1", "P1", "P2"],
                "publisher": [None, None, None],
                "datasetKey": ["D1", "D1", "D2"],
                "datasetName": ["one", "one", "two"],
                "modified": ["2026-07-01", "2026-07-02", None],
                "lastInterpreted": ["2026-07-03", "2026-07-04", None],
            }
        ),
        source,
    )
    inventory.write_text(
        json.dumps(
            {
                "source_snapshot_id": "sha256:snapshot",
                "source_publication_date": "2026-07-19",
            }
        )
    )
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "test/v1",
                "generated_at": "2026-07-20T00:00:00Z",
            }
        )
    )

    manifest = publish_freshness_audit(
        v3_parquet=source,
        source_inventory_json=inventory,
        data_root=data,
        output_directory=output,
        expected_rows=3,
        code_commit="commit",
        provider_stale_days=30,
        derived_stale_days=10_000,
        memory_limit="256MB",
        threads=1,
    )

    rows = pq.read_table(output / "provider_dataset_freshness.parquet").to_pylist()
    statuses = {row["provider"]: row["freshness_status"] for row in rows}
    assert statuses == {"P1": "PASS", "P2": "UNKNOWN"}
    assert manifest["counts"]["source_rows"] == 3
    assert manifest["counts"]["derived_manifest_rows"] == 1
    assert (output / "manifest.json").is_file()
