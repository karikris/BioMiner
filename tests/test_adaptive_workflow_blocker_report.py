from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json

import polars as pl
import pytest

from biominer.reports.workflow_blockers import (
    ADAPTIVE_WORKFLOW_BLOCKER_SCHEMA,
    WORKFLOW_BLOCKER_KINDS,
    adaptive_workflow_blockers_frame,
    build_adaptive_workflow_blocker_report,
    validate_adaptive_workflow_blockers,
    write_adaptive_workflow_blocker_report,
)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _rows(*, unavailable: set[str] | None = None) -> list[dict[str, object]]:
    ids = {
        "failed_reference_downloads": ["media:1", "media:2"],
        "retryable_media": ["media:1"],
        "invalid_routes": ["media:3"],
        "stale_bank_artifacts": ["artifact:prototype:1"],
        "incomplete_audit_sample": ["species:Papilio-demoleus"],
        "pending_targeted_review": ["reference:4"],
        "pending_selective_rerun": ["score:5"],
    }
    missing = unavailable or set()
    return [
        {
            "blocker_kind": kind,
            "blocker_ids": [] if kind in missing else ids[kind],
            "measurement_status": (
                "unavailable" if kind in missing else "measured_complete"
            ),
            "blocks_initial_scoring": kind
            in {"stale_bank_artifacts", "invalid_routes"},
            "blocks_final_release": kind
            in {
                "stale_bank_artifacts",
                "incomplete_audit_sample",
                "pending_targeted_review",
                "pending_selective_rerun",
            },
            "source_artifact_id": f"artifact:{kind}",
            "source_artifact_fingerprint": _sha(f"artifact:{kind}"),
        }
        for kind in WORKFLOW_BLOCKER_KINDS
    ]


def test_blocker_report_covers_resume_and_human_input_states(tmp_path) -> None:
    blockers = adaptive_workflow_blockers_frame(_rows())
    result = build_adaptive_workflow_blocker_report(
        blockers,
        generated_at=datetime(2026, 7, 18, tzinfo=UTC),
    )

    assert blockers.schema == ADAPTIVE_WORKFLOW_BLOCKER_SCHEMA
    assert blockers["blocker_kind"].to_list() == list(WORKFLOW_BLOCKER_KINDS)
    assert result.report["summary"]["known_blocker_memberships"] == 8  # type: ignore[index]
    assert result.report["summary"][  # type: ignore[index]
        "membership_semantics"
    ] == "category_memberships_may_overlap"
    pending = blockers.filter(
        pl.col("blocker_kind") == "pending_targeted_review"
    ).row(0, named=True)
    assert pending["human_input_required"] is True
    assert pending["resume_action"] == "review_flagged_references"
    assert len(result.report["evidence_maturity"]["labels"]) == 6  # type: ignore[index]

    paths = write_adaptive_workflow_blocker_report(result, tmp_path)
    assert pl.read_parquet(paths["blockers"]).equals(blockers)
    assert json.loads(paths["json"].read_text()) == result.report
    assert paths["markdown"].read_text().startswith("# Adaptive workflow blockers")


def test_unavailable_blocker_evidence_is_not_zero() -> None:
    blockers = adaptive_workflow_blockers_frame(
        _rows(unavailable={"incomplete_audit_sample"})
    )
    row = blockers.filter(
        pl.col("blocker_kind") == "incomplete_audit_sample"
    ).row(0, named=True)

    assert row["blocker_count"] is None
    assert row["measurement_status"] == "unavailable"


def test_retryable_media_must_be_a_failed_download() -> None:
    rows = _rows()
    retryable = next(
        row for row in rows if row["blocker_kind"] == "retryable_media"
    )
    retryable["blocker_ids"] = ["media:unknown"]

    with pytest.raises(ValueError, match="must be failed reference downloads"):
        adaptive_workflow_blockers_frame(rows)


def test_blocker_validator_rejects_count_and_resume_semantic_tampering() -> None:
    blockers = adaptive_workflow_blockers_frame(_rows())
    bad_count = blockers.with_columns(
        pl.when(pl.col("blocker_kind") == "invalid_routes")
        .then(pl.lit(7, dtype=pl.UInt64))
        .otherwise(pl.col("blocker_count"))
        .alias("blocker_count")
    )
    with pytest.raises(ValueError, match="count mismatch"):
        validate_adaptive_workflow_blockers(bad_count)

    bad_action = blockers.with_columns(
        pl.when(pl.col("blocker_kind") == "pending_targeted_review")
        .then(pl.lit("automatically_clear"))
        .otherwise(pl.col("resume_action"))
        .alias("resume_action")
    )
    with pytest.raises(ValueError, match="resume semantics"):
        validate_adaptive_workflow_blockers(bad_action)
