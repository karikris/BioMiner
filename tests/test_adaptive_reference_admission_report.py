from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json

import polars as pl
import pytest

from biominer.reports.reference_admission import (
    ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_SCHEMA,
    REFERENCE_ADMISSION_STAGES,
    adaptive_reference_admission_funnel_frame,
    build_adaptive_reference_admission_report,
    validate_adaptive_reference_admission_funnel,
    write_adaptive_reference_admission_report,
)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _rows(*, unavailable: set[str] | None = None) -> list[dict[str, object]]:
    ids = {
        "candidates": ["media:1", "media:2", "media:3", "media:4", "media:5"],
        "downloaded": ["media:1", "media:2", "media:3", "media:4"],
        "decoded": ["media:1", "media:2", "media:3"],
        "deduplicated": ["media:1", "media:2"],
        "yoloe_routed": ["media:1", "media:2"],
        "provisionally_admitted": ["media:1", "media:2"],
        "human_verified": ["media:1"],
        "excluded": ["media:3", "media:4"],
        "flagged": ["media:2"],
        "reviewed_later": ["media:2"],
    }
    missing = unavailable or set()
    return [
        {
            "stage": stage,
            "reference_media_ids": [] if stage in missing else ids[stage],
            "measurement_status": (
                "unavailable" if stage in missing else "measured_complete"
            ),
            "source_artifact_id": f"artifact:{stage}",
            "source_artifact_fingerprint": _sha(f"artifact:{stage}"),
        }
        for stage in REFERENCE_ADMISSION_STAGES
    ]


def test_report_covers_every_admission_stage_with_derived_counts(tmp_path) -> None:
    funnel = adaptive_reference_admission_funnel_frame(_rows())
    result = build_adaptive_reference_admission_report(
        funnel,
        generated_at=datetime(2026, 7, 18, tzinfo=UTC),
    )

    assert funnel.schema == ADAPTIVE_REFERENCE_ADMISSION_FUNNEL_SCHEMA
    assert funnel["stage"].to_list() == list(REFERENCE_ADMISSION_STAGES)
    assert funnel["record_count"].to_list() == [5, 4, 3, 2, 2, 2, 1, 2, 1, 1]
    assert funnel.filter(pl.col("stage") == "provisionally_admitted")[
        "candidate_retention_rate"
    ].item() == pytest.approx(0.4)
    assert result.report["measurement_summary"] == {
        "measured_stage_count": 10,
        "unavailable_stages": [],
    }
    assert len(result.report["evidence_maturity"]["labels"]) == 6  # type: ignore[index]
    assert "Provider-asserted provisional" in " ".join(
        result.report["limitations"]
    )

    paths = write_adaptive_reference_admission_report(result, tmp_path)
    assert pl.read_parquet(paths["funnel"]).equals(funnel)
    assert json.loads(paths["json"].read_text()) == result.report
    assert paths["markdown"].read_text().startswith(
        "# Adaptive reference admission"
    )


def test_unavailable_stage_is_explicit_and_never_interpreted_as_zero() -> None:
    funnel = adaptive_reference_admission_funnel_frame(
        _rows(unavailable={"human_verified", "reviewed_later"})
    )
    report = build_adaptive_reference_admission_report(funnel).report

    assert funnel.filter(pl.col("stage") == "human_verified")[
        "record_count"
    ].item() is None
    assert report["measurement_summary"]["unavailable_stages"] == [  # type: ignore[index]
        "human_verified",
        "reviewed_later",
    ]


def test_funnel_rejects_impossible_stage_membership() -> None:
    rows = _rows()
    routed = next(row for row in rows if row["stage"] == "yoloe_routed")
    routed["reference_media_ids"] = ["media:1", "media:5"]

    with pytest.raises(ValueError, match="not a subset of deduplicated"):
        adaptive_reference_admission_funnel_frame(rows)


def test_funnel_rejects_tampered_count_and_fingerprint() -> None:
    funnel = adaptive_reference_admission_funnel_frame(_rows())
    bad_count = funnel.with_columns(
        pl.when(pl.col("stage") == "decoded")
        .then(pl.lit(99, dtype=pl.UInt64))
        .otherwise(pl.col("record_count"))
        .alias("record_count")
    )
    with pytest.raises(ValueError, match="count mismatch"):
        validate_adaptive_reference_admission_funnel(bad_count)

    bad_fingerprint = funnel.with_columns(
        pl.when(pl.col("stage") == "decoded")
        .then(pl.lit(_sha("tampered")))
        .otherwise(pl.col("stage_fingerprint"))
        .alias("stage_fingerprint")
    )
    with pytest.raises(ValueError, match="stage fingerprint mismatch"):
        validate_adaptive_reference_admission_funnel(bad_fingerprint)
