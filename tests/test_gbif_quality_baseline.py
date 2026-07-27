from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_quality.baseline import publish_baseline
from biominer.gbif_quality.funnel import FUNNEL_SCHEMA_VERSION, SourceFunnel
from biominer.gbif_quality.inventory import INVENTORY_SCHEMA_VERSION, SourceInventory
from biominer.gbif_quality.policy import build_field_policy
from biominer.gbif_quality.profile import profile_completeness
from biominer.gbif_quality.schema_audit import audit_parquet_schema


def test_baseline_publication_is_create_only_and_manifest_last(tmp_path: Path) -> None:
    table = pa.table(
        {
            "gbifID": ["1", "1", "2", "3"],
            "taxonRank": ["SPECIES", "SPECIES", "GENUS", "SUBSPECIES"],
            "species": ["Papilio x", "Papilio x", None, None],
        }
    )
    source = tmp_path / "v3.parquet"
    pq.write_table(table, source, row_group_size=2)
    policies = build_field_policy(table.schema)
    profile = profile_completeness(source, policies, occurrence_batch_size=2)
    schema = audit_parquet_schema(source, policies, full_value_scan_status="PASS")
    inventory = _inventory()
    funnel = _funnel()
    data_root = tmp_path / "data" / "v4"
    report_root = tmp_path / "reports" / "v4"

    publication = publish_baseline(
        inventory=inventory,
        funnel=funnel,
        schema_audit=schema,
        policies=policies,
        completeness=profile,
        data_root=data_root,
        report_root=report_root,
        code_commit="abc123",
        generated_at="2026-07-22T00:00:00Z",
    )

    assert publication.data_manifest["acceptance_gate"]["passed"] is True
    assert publication.report_manifest["acceptance_gate"]["manifest_written_last"] is True
    assert (data_root / "completeness_by_applicability.parquet").is_file()
    assert (report_root / "source_funnel.md").is_file()
    assert (data_root / "manifest.json").stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns
        for path in data_root.iterdir()
        if path.name != "manifest.json"
    )
    assert len(publication.data_manifest["artifact_inventory"]) == 9
    assert len(publication.report_manifest["artifact_inventory"]) == 3
    comparisons = publication.data_manifest
    assert comparisons["publication_role"] == "phase_1_data"

    with pytest.raises(FileExistsError):
        publish_baseline(
            inventory=inventory,
            funnel=funnel,
            schema_audit=schema,
            policies=policies,
            completeness=profile,
            data_root=data_root,
            report_root=tmp_path / "reports-2",
            code_commit="abc123",
        )


def _inventory() -> SourceInventory:
    return SourceInventory(
        schema_version=INVENTORY_SCHEMA_VERSION,
        source_snapshot_id="sha256:test",
        source_download_key="test-download",
        source_package_id="test-package",
        source_title="test source",
        source_publication_date="2026-07-22",
        archive_members=(),
        artifacts=(
            {
                "source_snapshot_id": "sha256:test",
                "artifact_role": "dwca_archive",
                "path": "source.zip",
                "member": None,
                "physical_bytes": 1,
                "sha256": "0" * 64,
                "expected_sha256": "0" * 64,
                "checksum_status": "PASS",
                "row_count": None,
                "expected_row_count": None,
                "row_count_status": "NOT_APPLICABLE",
                "column_count": None,
                "expected_column_count": None,
                "column_count_status": "NOT_APPLICABLE",
                "row_group_count": None,
                "row_groups_complete": None,
                "schema_fingerprint": None,
                "manifest_path": "source-manifest.json",
            },
        ),
        validation={"all": True},
    )


def _funnel() -> SourceFunnel:
    return SourceFunnel(
        schema_version=FUNNEL_SCHEMA_VERSION,
        stages=(
            {
                "stage_order": 1,
                "stage_id": "TEST",
                "scope": "media_assertion",
                "input_row_count": 4,
                "output_row_count": 4,
                "excluded_row_count": 0,
                "input_occurrence_count": 3,
                "output_occurrence_count": 3,
                "fully_excluded_occurrence_count": 0,
                "exclusion_reason": "NO_EXCLUSION",
                "evidence_path": "fixture.parquet",
                "evidence_type": "fixture",
                "status": "PASS",
            },
        ),
        validation={"all": True},
        counts={
            "raw_occurrence_rows": 10,
            "raw_multimedia_rows": 4,
            "v3_media_rows": 4,
            "v3_occurrences": 3,
            "unresolved_multimedia_rows": 0,
            "unexplained_media_residual_rows": 0,
        },
    )
