from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.target_aware_output import target_aware_object_scores_frame
from biominer.run.flickr_selective_rescore import (
    FLICKR_RESCORE_PLAN_FILE,
    calculate_flickr_rescore_plan,
    flickr_photo_ids_to_rescore,
    flickr_rescore_evidence_frame,
    flickr_rescore_evidence_from_target_scores,
    flickr_rescore_metrics,
    score_reference_dependency_frame,
    target_score_ids_to_rescore,
    validate_flickr_rescore_plan,
    write_flickr_rescore_plan,
)
from test_reference_revision_impact import _revision_and_graph
from test_target_aware_output import _row
from test_targeted_reference_review import SHA_A, SHA_B


def _evidence(
    score_id: str,
    *,
    bank_fingerprint: str,
    target: str = "gbif:unrelated-target",
    competitor: str = "gbif:unrelated-competitor",
    candidates: tuple[str, ...] | None = None,
    references: tuple[str, ...] = (),
    dependencies_complete: bool = True,
    margin: float | None = 0.5,
) -> dict[str, object]:
    return {
        "target_score_id": score_id,
        "source": "flickr",
        "flickr_photo_id": score_id.removeprefix("score:"),
        "scoring_unit_id": f"unit:{score_id}",
        "route": "adult_field",
        "prior_target_score_fingerprint": SHA_A,
        "prior_reference_bank_fingerprint": bank_fingerprint,
        "target_accepted_taxon_key": target,
        "best_competitor_accepted_taxon_key": competitor,
        "candidate_accepted_taxon_keys": list(
            candidates or (target, competitor)
        ),
        "reference_media_ids": list(references),
        "reference_dependencies_complete": dependencies_complete,
        "prior_target_competitor_margin": margin,
    }


def _revision_and_evidence():  # noqa: ANN202
    revision, _catalog, _edges, _media_ids = _revision_and_graph()
    changed_species = {
        str(row["accepted_taxon_key"])
        for row in revision.change_manifest.iter_rows(named=True)
        if not str(row["change_type"]).startswith("unchanged")
    }
    assert changed_species
    changed = sorted(changed_species)[0]
    removed = next(
        str(row["reference_media_id"])
        for row in revision.change_manifest.iter_rows(named=True)
        if row["old_support_eligible"] and not row["new_support_eligible"]
    )
    bank = revision.old_reference_bank_fingerprint
    evidence = flickr_rescore_evidence_frame(
        [
            _evidence("score:target", bank_fingerprint=bank, target=changed),
            _evidence(
                "score:competitor",
                bank_fingerprint=bank,
                competitor=changed,
            ),
            _evidence(
                "score:candidate",
                bank_fingerprint=bank,
                candidates=(
                    "gbif:unrelated-target",
                    "gbif:unrelated-competitor",
                    changed,
                ),
            ),
            _evidence(
                "score:removed",
                bank_fingerprint=bank,
                references=(removed,),
            ),
            _evidence("score:margin", bank_fingerprint=bank, margin=0.05),
            _evidence("score:unrelated", bank_fingerprint=bank, margin=0.5),
            _evidence(
                "score:missing-dependencies",
                bank_fingerprint=bank,
                dependencies_complete=False,
            ),
            _evidence("score:missing-margin", bank_fingerprint=bank, margin=None),
        ]
    )
    return revision, evidence


def test_selective_rescore_covers_each_impact_trigger_and_reuses_unrelated(
    tmp_path,
) -> None:
    revision, evidence = _revision_and_evidence()

    plan = calculate_flickr_rescore_plan(
        revision,
        evidence,
        margin_impact_band=0.1,
    )
    by_id = {
        str(row["target_score_id"]): row for row in plan.iter_rows(named=True)
    }

    assert by_id["score:target"]["target_bank_changed"] is True
    assert by_id["score:competitor"]["best_competitor_bank_changed"] is True
    assert by_id["score:candidate"]["candidate_union_changed"] is True
    assert by_id["score:candidate"]["target_bank_changed"] is False
    assert by_id["score:candidate"]["best_competitor_bank_changed"] is False
    assert by_id["score:removed"]["removed_reference_dependency"] is True
    assert by_id["score:margin"]["margin_in_impact_band"] is True
    assert by_id["score:unrelated"]["rescore_required"] is False
    assert by_id["score:unrelated"]["rescore_action"] == "reuse_prior_score"
    assert by_id["score:missing-dependencies"]["rescore_reasons"] == [
        "missing_reference_dependency_evidence"
    ]
    assert by_id["score:missing-margin"]["rescore_reasons"] == [
        "missing_prior_margin_evidence"
    ]
    assert set(target_score_ids_to_rescore(plan)) == set(by_id) - {
        "score:unrelated"
    }
    assert set(flickr_photo_ids_to_rescore(plan)) == {
        score_id.removeprefix("score:")
        for score_id in by_id
        if score_id != "score:unrelated"
    }
    metrics = flickr_rescore_metrics(plan)
    assert metrics["record_count"].sum() == plan.height
    output = write_flickr_rescore_plan(plan, tmp_path)
    assert output.name == FLICKR_RESCORE_PLAN_FILE
    assert pl.read_parquet(output).equals(plan)


def test_production_score_projection_retains_candidate_and_reference_evidence() -> None:
    revision, _catalog, _edges, media_ids = _revision_and_graph()
    scores = target_aware_object_scores_frame(
        [
            _row(
                reference_bank_fingerprint=(
                    revision.old_reference_bank_fingerprint
                )
            )
        ]
    )
    score_id = str(scores["target_score_id"].item())
    dependencies = score_reference_dependency_frame(
        [
            {
                "target_score_id": score_id,
                "reference_media_ids": [media_ids[0], media_ids[1]],
            }
        ]
    )

    evidence = flickr_rescore_evidence_from_target_scores(scores, dependencies)
    row = evidence.row(0, named=True)

    assert row["candidate_accepted_taxon_keys"] == ["gbif:1", "gbif:2"]
    assert row["reference_media_ids"] == sorted(media_ids[:2])
    assert row["reference_dependencies_complete"] is True
    assert row["prior_target_score_fingerprint"] == scores[
        "target_score_fingerprint"
    ].item()


def test_missing_projected_reference_dependencies_fail_closed_to_rescore() -> None:
    revision, _catalog, _edges, _media_ids = _revision_and_graph()
    scores = target_aware_object_scores_frame(
        [
            _row(
                reference_bank_fingerprint=(
                    revision.old_reference_bank_fingerprint
                )
            )
        ]
    )
    evidence = flickr_rescore_evidence_from_target_scores(
        scores,
        score_reference_dependency_frame(),
    )

    plan = calculate_flickr_rescore_plan(
        revision,
        evidence,
        margin_impact_band=0.1,
    )

    assert plan["rescore_required"].item() is True
    assert "missing_reference_dependency_evidence" in plan[
        "rescore_reasons"
    ].item()


def test_rescore_plan_rejects_stale_bank_and_tampering() -> None:
    revision, evidence = _revision_and_evidence()
    stale = flickr_rescore_evidence_frame(
        [
            {
                **evidence.row(0, named=True),
                "prior_reference_bank_fingerprint": SHA_B,
            }
        ]
    )
    with pytest.raises(ValueError, match="wrong reference bank"):
        calculate_flickr_rescore_plan(
            revision,
            stale,
            margin_impact_band=0.1,
        )

    plan = calculate_flickr_rescore_plan(
        revision,
        evidence,
        margin_impact_band=0.1,
    )
    tampered = plan.with_columns(pl.lit("reuse_prior_score").alias("rescore_action"))
    with pytest.raises(ValueError, match="action mismatch|plan_fingerprint"):
        validate_flickr_rescore_plan(tampered)


def test_margin_impact_band_must_be_finite_and_nonnegative() -> None:
    revision, evidence = _revision_and_evidence()
    for band in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="margin impact band"):
            calculate_flickr_rescore_plan(
                revision,
                evidence,
                margin_impact_band=band,
            )
