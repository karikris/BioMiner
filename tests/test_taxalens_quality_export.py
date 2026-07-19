"""Tests for TaxaLens review-design and quality sidecar exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_review import (
    ProbabilityAuditSamplingPolicy,
    RAW_SCORE_SEMANTICS,
)
from biominer.integration.taxalens_pool_export import (
    export_taxalens_score_pool_evidence,
    validate_taxalens_score_pool_export,
)
from biominer.integration.taxalens_pool_handoff import (
    build_taxalens_pool_handoff,
    validate_taxalens_pool_handoff,
)
from biominer.integration.taxalens_quality_export import (
    GEOGRAPHIC_CELLS_DOWNSTREAM_REASON,
    TAXALENS_REVIEW_SAMPLING_SCHEMA,
    build_taxalens_quality_sidecar,
    build_taxalens_review_sampling_frame,
    export_taxalens_review_quality_evidence,
    validate_taxalens_quality_sidecar,
    validate_taxalens_review_quality_export,
)
from helpers.dynamic_pool_handoff_fixture import (
    build_dynamic_pool_handoff_fixture,
    build_quality_report_fixture,
    build_review_selection_fixture,
)


def test_review_frame_preserves_probability_design_and_authority_boundary() -> None:
    selection, policy = build_review_selection_fixture()

    frame = build_taxalens_review_sampling_frame(selection, policy=policy)

    assert frame.schema == TAXALENS_REVIEW_SAMPLING_SCHEMA
    assert frame.height == selection.selected_count == 4
    assert frame["sampling_policy_fingerprint"].unique().to_list() == [
        policy.fingerprint
    ]
    assert frame["sampling_register_fingerprint"].unique().to_list() == [
        selection.register_fingerprint
    ]
    assert frame["selection_seed"].unique().to_list() == [17]
    assert frame["sampling_purpose"].unique().to_list() == ["representative_audit"]
    assert frame["representative"].all()
    assert frame["blind_review"].all()
    assert frame["raw_score_semantics"].unique().to_list() == [RAW_SCORE_SEMANTICS]
    assert not frame["raw_score_is_probability"].any()
    assert not frame["no_geo_is_biological_absence"].any()
    assert not frame["occurrence_release_authorized"].any()
    no_geo = frame.filter(pl.col("no_geo"))
    assert no_geo.height == 1
    assert no_geo["geographic_cluster_id"].null_count() == 1
    assert (
        frame["inclusion_probability"] * frame["sampling_weight"]
    ).to_list() == pytest.approx([1.0] * frame.height)


def test_quality_sidecar_preserves_review_metrics_without_release_authority() -> None:
    sidecar = build_taxalens_quality_sidecar(
        build_quality_report_fixture(sufficient=True)
    )

    assert sidecar["completed_review_count"] == 4
    assert sidecar["representative_evaluated_count"] == 4
    assert sidecar["targeted_review_excluded_count"] == 0
    assert sidecar["quality_estimate_available"] is True
    assert sidecar["quality_status"] == "available"
    assert sidecar["quality_unavailable_reasons"] == []
    assert [row["metric_name"] for row in sidecar["metrics"]] == sorted(
        row["metric_name"] for row in sidecar["metrics"]
    )
    assert sidecar["representative_and_targeted_are_separate"] is True
    assert sidecar["authorizes_occurrence_release"] is False
    assert sidecar["scientific_claim_allowed"] is False
    validate_taxalens_quality_sidecar(sidecar)


def test_insufficient_quality_is_an_available_report_not_a_zero_estimate() -> None:
    sidecar = build_taxalens_quality_sidecar(
        build_quality_report_fixture(sufficient=False)
    )

    assert sidecar["completed_review_count"] == 4
    assert sidecar["quality_estimate_available"] is False
    assert sidecar["quality_status"] == "insufficient_sample"
    assert sidecar["quality_unavailable_reasons"]
    assert all(row["estimate"] is None for row in sidecar["metrics"])
    assert all(
        row["metric_status"] == "insufficient_sample" for row in sidecar["metrics"]
    )


def test_no_quality_report_exports_explicit_unavailable_states(tmp_path: Path) -> None:
    root = tmp_path / "handoff"
    scores = export_taxalens_score_pool_evidence(
        **build_dynamic_pool_handoff_fixture(),
        output_root=root,
    )
    selection, policy = build_review_selection_fixture()

    sidecars = export_taxalens_review_quality_evidence(
        selection=selection,
        sampling_policy=policy,
        quality_report=None,
        output_root=root,
    )

    by_role = {row["role"]: row for row in sidecars.artifacts}
    assert by_role["review_sampling_frame"]["availability"] == "available"
    assert by_role["quality_sidecar"]["availability"] == "unavailable"
    assert by_role["geographic_cells"]["availability"] == "unavailable"
    assert by_role["geographic_cells"]["unavailable_reason"] == (
        GEOGRAPHIC_CELLS_DOWNSTREAM_REASON
    )
    assert sidecars.completed_review_count == 0
    assert sidecars.quality_estimate_available is False
    assert sidecars.quality_unavailable_reason == (
        "no validated reviewed quality report supplied"
    )
    validate_taxalens_score_pool_export(root, scores.artifacts)
    manifest = _manifest(
        artifacts=(*scores.artifacts, *sidecars.artifacts),
        completed_review_count=sidecars.completed_review_count,
        quality_estimate_available=sidecars.quality_estimate_available,
        quality_unavailable_reason=sidecars.quality_unavailable_reason,
    )
    assert manifest["evidence_maturity"]["quality_estimate"]["status"] == (
        "unavailable"
    )
    assert manifest["evidence_maturity"]["release"]["release_ready"] is False


@pytest.mark.parametrize("sufficient", [True, False])
def test_reviewed_quality_exports_and_builds_a_fail_closed_manifest(
    tmp_path: Path,
    sufficient: bool,
) -> None:
    root = tmp_path / "handoff"
    scores = export_taxalens_score_pool_evidence(
        **build_dynamic_pool_handoff_fixture(),
        output_root=root,
    )
    selection, policy = build_review_selection_fixture()

    sidecars = export_taxalens_review_quality_evidence(
        selection=selection,
        sampling_policy=policy,
        quality_report=build_quality_report_fixture(sufficient=sufficient),
        output_root=root,
    )

    quality = next(
        row for row in sidecars.artifacts if row["role"] == "quality_sidecar"
    )
    assert quality["availability"] == "available"
    payload = json.loads(
        (root / str(quality["relative_path"])).read_text(encoding="utf-8")
    )
    assert payload["quality_estimate_available"] is sufficient
    assert sidecars.completed_review_count == 4
    assert sidecars.quality_estimate_available is sufficient
    assert (sidecars.quality_unavailable_reason is None) is sufficient
    manifest = _manifest(
        artifacts=(*scores.artifacts, *sidecars.artifacts),
        completed_review_count=sidecars.completed_review_count,
        quality_estimate_available=sidecars.quality_estimate_available,
        quality_unavailable_reason=sidecars.quality_unavailable_reason,
    )
    validate_taxalens_pool_handoff(manifest)
    assert manifest["evidence_maturity"]["human_review"]["status"] == "available"
    assert manifest["evidence_maturity"]["release"]["release_ready"] is False


def test_export_is_create_only_and_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "handoff"
    selection, policy = build_review_selection_fixture()
    exported = export_taxalens_review_quality_evidence(
        selection=selection,
        sampling_policy=policy,
        quality_report=build_quality_report_fixture(),
        output_root=root,
    )

    with pytest.raises(FileExistsError, match="create-only"):
        export_taxalens_review_quality_evidence(
            selection=selection,
            sampling_policy=policy,
            quality_report=None,
            output_root=root,
        )

    tampered = deepcopy(exported.artifacts)
    tampered[0]["semantic_fingerprint"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="semantic identity differs"):
        validate_taxalens_review_quality_export(
            root,
            tampered,
            completed_review_count=exported.completed_review_count,
            quality_estimate_available=exported.quality_estimate_available,
        )

    review = next(
        row for row in exported.artifacts if row["role"] == "review_sampling_frame"
    )
    with (root / str(review["relative_path"])).open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ValueError, match="physical identity differs"):
        validate_taxalens_review_quality_export(
            root,
            exported.artifacts,
            completed_review_count=exported.completed_review_count,
            quality_estimate_available=exported.quality_estimate_available,
        )


def test_policy_mismatch_fails_before_writing_output(tmp_path: Path) -> None:
    selection, _ = build_review_selection_fixture()
    mismatched = ProbabilityAuditSamplingPolicy(review_budget=4, random_seed=18)
    root = tmp_path / "handoff"

    with pytest.raises(ValueError, match="sample and sampling policy differ"):
        export_taxalens_review_quality_evidence(
            selection=selection,
            sampling_policy=mismatched,
            quality_report=None,
            output_root=root,
        )

    assert not root.exists()


def _manifest(
    *,
    artifacts: Sequence[Mapping[str, object]],
    completed_review_count: int,
    quality_estimate_available: bool,
    quality_unavailable_reason: str | None,
) -> dict[str, object]:
    return build_taxalens_pool_handoff(
        producer_commit="1" * 40,
        created_at="2026-07-18T12:00:00+10:00",
        run_id="run-papilio-demoleus-001",
        registry_version="registry-2026-07-18",
        source_snapshot_fingerprints=["sha256:" + "2" * 64],
        model_fingerprint="sha256:" + "3" * 64,
        preprocessing_fingerprint="sha256:" + "4" * 64,
        artifacts=artifacts,
        completed_review_count=completed_review_count,
        quality_estimate_available=quality_estimate_available,
        quality_unavailable_reason=quality_unavailable_reason,
    )
