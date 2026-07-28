from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

import biominer.gbif_quality.recovery as recovery


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restart_inventory_covers_all_production_evidence_layers() -> None:
    required = {
        "source_lineage/identity_v2/manifest.json",
        "quality_results/phase2/manifest.json",
        "quality_results/phase3/manifest.json",
        "derived_assertions/geography_v3/manifest.json",
        "quality_results/phase3_v3/manifest.json",
        "quality_results/phase4_pilot_preflight/manifest.json",
        "representativeness_concentration/manifest.json",
        "quality_results/review_capsules/manifest.json",
        "incremental_state/manifest.json",
        "freshness/manifest.json",
        "provider_enrichment/manifest.json",
        "provider_enrichment_v2/manifest.json",
        "provider_enrichment_v4/manifest.json",
        "performance/manifest.json",
    }

    assert required <= set(recovery.STAGE_MANIFESTS)


def test_restart_validation_skips_committed_and_unchanged_work(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    incremental = data / "incremental_validation"
    incremental.mkdir(parents=True)
    baseline_artifact = data / "baseline.parquet"
    queue = incremental / "changed_row_queue.parquet"
    pq.write_table(pa.table({"value": [1]}), baseline_artifact)
    pq.write_table(pa.table({"value": pa.array([], type=pa.int64())}), queue)
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "source_snapshot_id": "sha256:snapshot",
                "artifact_inventory": [
                    {"path": baseline_artifact.name, "sha256": _sha256(baseline_artifact)}
                ],
                "manifest_policy": {"written_last": True},
            }
        )
    )
    (incremental / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {"queue_rows": 0},
                "validation": {
                    "unchanged_rows_not_queued": True,
                    "unchanged_rerun_semantically_identical": True,
                    "manifest_written_last": True,
                },
                "artifacts": [
                    {"path": queue.name, "sha256": _sha256(queue)}
                ],
            }
        )
    )
    monkeypatch.setattr(recovery, "STAGE_MANIFESTS", ("manifest.json",))

    output = tmp_path / "recovery"
    manifest = recovery.publish_restart_validation(
        data_root=data, output_directory=output, code_commit="commit"
    )

    rows = pq.read_table(output / "restart_validation.parquet").to_pylist()
    assert {row["restart_action"] for row in rows} == {
        "SKIP_COMMITTED",
        "SKIP_UNCHANGED",
        "CONTINUE",
    }
    assert manifest["validation"]["unchanged_rows_not_reprocessed"] is True
    assert manifest["counts"]["orphaned_staging_directories"] == 0
