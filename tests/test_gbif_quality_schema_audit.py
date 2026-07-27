from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_quality.policy import build_field_policy
from biominer.gbif_quality.schema_audit import audit_parquet_schema


def test_schema_audit_reconciles_row_groups_and_types(tmp_path: Path) -> None:
    table = pa.table(
        {
            "gbifID": ["1", "2", "3", "4", "5"],
            "year": ["2020", "2021", None, "2023", "2024"],
        }
    )
    path = tmp_path / "input.parquet"
    pq.write_table(table, path, row_group_size=2)

    audit = audit_parquet_schema(
        path, build_field_policy(table.schema), full_value_scan_status="PASS"
    )

    assert all(audit.validation.values())
    assert audit.counts == {
        "rows": 5,
        "columns": 2,
        "row_groups": 3,
        "row_group_rows": 5,
    }
    assert audit.column_table().num_rows == 2
    assert audit.row_group_table().num_rows == 3
    year = next(row for row in audit.columns if row["field_name"] == "year")
    assert year["recommended_valid_type"] == "nullable_integer"
    assert year["typed_derivative_recommended"] is True
    assert year["original_preservation_policy"] == "PRESERVE_RAW_AND_DERIVE_SEPARATELY"


def test_schema_audit_requires_full_scan_evidence(tmp_path: Path) -> None:
    table = pa.table({"gbifID": ["1"]})
    path = tmp_path / "input.parquet"
    pq.write_table(table, path)

    with pytest.raises(ValueError, match="full_value_scan_completed"):
        audit_parquet_schema(path, build_field_policy(table.schema))
