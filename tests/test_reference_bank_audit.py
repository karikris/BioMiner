from __future__ import annotations

import json

import polars as pl
import pytest

from biominer.evaluation.reference_bank_audit import (
    AUDIT_DIMENSIONS,
    ReferenceBankQualityPolicy,
    empty_reference_bank_quality_audit,
    empty_reference_bank_quality_summary,
    write_reference_bank_audit_contract,
)


def test_reference_bank_audit_contract_publishes_four_typed_artifacts(tmp_path) -> None:
    publication = write_reference_bank_audit_contract(
        tmp_path,
        audit=empty_reference_bank_quality_audit(),
        summary=empty_reference_bank_quality_summary(),
    )

    assert publication.audit_path.name == "reference_bank_quality_audit.parquet"
    assert publication.summary_path.name == "reference_bank_quality_summary.parquet"
    assert publication.policy_path.name == "reference_bank_quality_policy.json"
    assert publication.report_path.name == "reference_bank_quality_report.md"
    assert pl.read_parquet(publication.audit_path).schema == (
        empty_reference_bank_quality_audit().schema
    )
    assert set(AUDIT_DIMENSIONS) <= set(pl.read_parquet(publication.summary_path).columns)
    policy = json.loads(publication.policy_path.read_text(encoding="utf-8"))
    assert policy["require_sampling_weights_for_targeted_queues"] is True
    assert "Targeted queues require sampling weights" in publication.report_path.read_text(
        encoding="utf-8"
    )


def test_reference_bank_audit_schema_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        write_reference_bank_audit_contract(
            ".",
            audit=pl.DataFrame({"audit_record_id": []}, schema={"audit_record_id": pl.String}),
            summary=empty_reference_bank_quality_summary(),
        )


def test_reference_bank_quality_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="positive"):
        ReferenceBankQualityPolicy(minimum_group_sample_size=0)
    with pytest.raises(ValueError, match="confidence_level"):
        ReferenceBankQualityPolicy(confidence_level=1.0)
