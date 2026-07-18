"""Contract tests for raw dynamic-pool scores and photo summaries."""

from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.dynamic_pool_scores import (
    DYNAMIC_POOL_CANDIDATE_SCORES_FILE,
    DYNAMIC_POOL_PHOTO_SUMMARY_FILE,
    build_dynamic_pool_candidate_scores,
    build_dynamic_pool_photo_summaries,
    dynamic_pool_candidate_score_schema,
    dynamic_pool_photo_summary_schema,
    validate_dynamic_pool_candidate_scores,
    validate_dynamic_pool_photo_summaries,
    validate_dynamic_pool_score_artifacts,
    write_dynamic_pool_score_artifacts,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint


TARGET = "gbif:5131359"
COMPETITOR = "gbif:5131360"


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _pool(character: str) -> str:
    return f"dynamic-reference-pool:{character * 64}"


def _row(
    *,
    candidate_key: str,
    candidate_name: str,
    target: bool,
    priority: int,
    fused: float,
    global_score: float,
    local_score: float | None,
    **changes: object,
) -> dict[str, object]:
    local_available = local_score is not None
    row: dict[str, object] = {
        "run_id": "run-20260718",
        "flickr_query_id": "query-papilio-demoleus",
        "flickr_photo_id": "flickr-photo-1",
        "organism_unit_id": "organism-unit-1",
        "visual_input_id": _sha("1"),
        "scoring_stage": "initial",
        "query_route": "adult_field",
        "plan_id": f"dynamic-pool-plan:{'2' * 64}",
        "plan_fingerprint": _sha("3"),
        "candidate_set_id": f"family-geo-candidate-set:{'4' * 64}",
        "candidate_set_fingerprint": _sha("5"),
        "target_accepted_taxon_key": TARGET,
        "target_scientific_name": "Papilio demoleus",
        "candidate_accepted_taxon_key": candidate_key,
        "candidate_scientific_name": candidate_name,
        "candidate_priority": priority,
        "target_candidate": target,
        "target_preserved": True,
        "global_pool_ids": [_pool("6")],
        "local_pool_ids": [_pool("7")] if local_available else [],
        "safety_pool_ids": [],
        "local_pool_status": "available" if local_available else "unavailable",
        "local_pool_unavailable_reason": (
            None if local_available else "no_geo_global_fallback"
        ),
        "global_score_status": "available",
        "global_score_unavailable_reason": None,
        "global_prototype_similarity": global_score - 0.02,
        "global_nearest_reference_similarity": global_score + 0.02,
        "global_top_k_mean_similarity": global_score,
        "global_raw_component_score": global_score,
        "global_configured_k": 3,
        "global_effective_k": 3,
        "global_reference_count": 3,
        "global_independent_observation_count": 3,
        "global_reference_shortfall_count": 0,
        "local_score_status": "available" if local_available else "unavailable",
        "local_score_unavailable_reason": (
            None if local_available else "local_pool_unavailable"
        ),
        "local_prototype_similarity": (
            local_score - 0.02 if local_score is not None else None
        ),
        "local_nearest_reference_similarity": (
            local_score + 0.02 if local_score is not None else None
        ),
        "local_top_k_mean_similarity": local_score,
        "local_raw_component_score": local_score,
        "local_configured_k": 3,
        "local_effective_k": 2 if local_available else 0,
        "local_reference_count": 2 if local_available else 0,
        "local_independent_observation_count": 2 if local_available else 0,
        "local_reference_shortfall_count": 1 if local_available else 3,
        "global_local_disagreement_status": (
            "available" if local_available else "unavailable"
        ),
        "global_local_disagreement_reason": (
            None if local_available else "local_score_unavailable"
        ),
        "global_local_raw_disagreement": (
            abs(global_score - local_score) if local_score is not None else None
        ),
        "family_evidence_status": "available",
        "family_evidence_reason": None,
        "family_evidence_rank": priority + 1,
        "family_evidence_raw_score": 0.9 - priority / 10,
        "family_priority_match": True,
        "family_changed_membership": False,
        "expansion_triggered": False,
        "expansion_rounds": 0,
        "expansion_triggers": [],
        "expansion_stop_reason": "initial_plan_sufficient",
        "score_policy_version": "dynamic-score-v1",
        "score_policy_fingerprint": _sha("8"),
        "model_fingerprint": _sha("9"),
        "fused_raw_score": fused,
        "probability_availability": "unavailable",
        "calibrated_probability": None,
        "probability_target": None,
        "calibrator_fingerprint": None,
        "probability_unavailable_reason": "calibrator_not_fitted",
        "human_review_required": True,
        "statistical_support_status": "not_evaluated",
        "statistical_support_report_fingerprint": None,
        "statistical_support_reason": "review_sample_not_available",
        "abstained": False,
        "abstention_reasons": [],
    }
    row.update(changes)
    return row


def _rows(*, local: bool = True) -> list[dict[str, object]]:
    return [
        _row(
            candidate_key=TARGET,
            candidate_name="Papilio demoleus",
            target=True,
            priority=0,
            fused=0.8,
            global_score=0.82,
            local_score=0.78 if local else None,
        ),
        _row(
            candidate_key=COMPETITOR,
            candidate_name="Papilio polytes",
            target=False,
            priority=1,
            fused=0.6,
            global_score=0.58,
            local_score=0.65 if local else None,
        ),
    ]


def test_score_schema_preserves_raw_components_and_pool_identities() -> None:
    fields = set(dynamic_pool_candidate_score_schema())

    assert {
        "plan_id",
        "candidate_set_id",
        "global_pool_ids",
        "local_pool_ids",
        "global_prototype_similarity",
        "global_nearest_reference_similarity",
        "global_top_k_mean_similarity",
        "global_raw_component_score",
        "local_prototype_similarity",
        "local_nearest_reference_similarity",
        "local_top_k_mean_similarity",
        "local_raw_component_score",
        "global_local_raw_disagreement",
        "global_effective_k",
        "local_effective_k",
        "probability_availability",
        "calibrated_probability",
        "score_fingerprint",
    } <= fields
    assert "confidence" not in fields


def test_builder_ranks_complete_set_and_derives_raw_margins() -> None:
    scores = build_dynamic_pool_candidate_scores(_rows())
    reversed_scores = build_dynamic_pool_candidate_scores(list(reversed(_rows())))

    assert scores.equals(reversed_scores)
    assert build_dynamic_pool_photo_summaries(scores).equals(
        build_dynamic_pool_photo_summaries(reversed_scores)
    )
    assert scores["candidate_accepted_taxon_key"].to_list() == [TARGET, COMPETITOR]
    assert scores["candidate_rank"].to_list() == [1, 2]
    assert scores["margin_to_next_raw"].to_list() == [pytest.approx(0.2), None]
    assert scores["target_preserved"].to_list() == [True, True]
    assert scores["score_id"].n_unique() == 2


def test_photo_summary_is_derived_and_never_authorizes_release() -> None:
    scores = build_dynamic_pool_candidate_scores(_rows())
    summaries = build_dynamic_pool_photo_summaries(scores)
    summary = summaries.row(0, named=True)

    assert summary["candidate_count"] == 2
    assert summary["top_candidate_accepted_taxon_key"] == TARGET
    assert summary["target_rank"] == 1
    assert summary["global_top_candidate_accepted_taxon_key"] == TARGET
    assert summary["local_top_candidate_accepted_taxon_key"] == TARGET
    assert summary["global_local_top1_agreement"] is True
    assert summary["probability_availability"] == "unavailable"
    assert summary["release_ready"] is False
    assert summary["scientific_claim_allowed"] is False


def test_global_only_scores_keep_local_components_null_not_zero() -> None:
    scores = build_dynamic_pool_candidate_scores(_rows(local=False))
    summaries = build_dynamic_pool_photo_summaries(scores)

    assert scores["local_pool_ids"].to_list() == [[], []]
    assert scores["local_raw_component_score"].to_list() == [None, None]
    summary = summaries.row(0, named=True)
    assert summary["local_top_candidate_accepted_taxon_key"] is None
    assert summary["global_local_top1_agreement"] is None
    assert summary["global_local_disagreement_reason"] == "local_score_unavailable"


def test_writer_creates_both_required_parquet_artifacts(tmp_path) -> None:
    scores = build_dynamic_pool_candidate_scores(_rows())
    summaries = build_dynamic_pool_photo_summaries(scores)

    paths = write_dynamic_pool_score_artifacts(scores, summaries, tmp_path / "scores")

    assert paths["candidate_scores"].name == DYNAMIC_POOL_CANDIDATE_SCORES_FILE
    assert paths["photo_summary"].name == DYNAMIC_POOL_PHOTO_SUMMARY_FILE
    loaded_scores = pl.read_parquet(paths["candidate_scores"])
    loaded_summaries = pl.read_parquet(paths["photo_summary"])
    validate_dynamic_pool_score_artifacts(loaded_scores, loaded_summaries)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"family_changed_membership": True}, "cannot change"),
        ({"global_effective_k": 4}, "effective k is inconsistent"),
        ({"local_reference_shortfall_count": 0}, "shortfall is inconsistent"),
        ({"local_raw_component_score": 0.0}, "disagreement does not match"),
        ({"family_evidence_status": "invented"}, "unsupported family"),
        ({"calibrated_probability": 0.9}, "requires null values"),
        ({"abstained": True}, "abstention state and reasons"),
        ({"fused_raw_score": 1.5}, "finite and in"),
    ],
)
def test_rejects_invalid_score_evidence(
    changes: dict[str, object], message: str
) -> None:
    rows = _rows()
    rows[1].update(changes)
    with pytest.raises(ValueError, match=message):
        build_dynamic_pool_candidate_scores(rows)


def test_available_probability_requires_calibrator_and_stays_separate() -> None:
    rows = _rows()
    rows[0].update(
        probability_availability="available",
        calibrated_probability=0.91,
        probability_target="decisive_positive_review",
        calibrator_fingerprint=_sha("a"),
        probability_unavailable_reason=None,
    )

    scores = build_dynamic_pool_candidate_scores(rows)
    top = scores.row(0, named=True)

    assert top["fused_raw_score"] == 0.8
    assert top["calibrated_probability"] == 0.91
    assert top["human_review_required"] is True


def test_expansion_requires_round_and_trigger_evidence() -> None:
    rows = _rows()
    rows[0].update(
        expansion_triggered=True,
        expansion_rounds=1,
        expansion_triggers=["small_raw_margin"],
        expansion_stop_reason="maximum_rounds_reached",
    )
    scores = build_dynamic_pool_candidate_scores(rows)
    assert scores.filter(pl.col("target_candidate"))["expansion_rounds"].item() == 1

    invalid = _rows()
    invalid[0]["expansion_triggered"] = True
    with pytest.raises(ValueError, match="flag and rounds"):
        build_dynamic_pool_candidate_scores(invalid)


def test_requires_one_preserved_target_and_unique_candidates() -> None:
    no_target = _rows()
    no_target[0]["target_candidate"] = False
    with pytest.raises(ValueError, match="exactly one matching target"):
        build_dynamic_pool_candidate_scores(no_target)

    duplicate = _rows()
    duplicate[1]["candidate_accepted_taxon_key"] = TARGET
    with pytest.raises(ValueError, match="duplicate candidate taxa"):
        build_dynamic_pool_candidate_scores(duplicate)


def test_validators_reject_score_summary_and_derived_set_tampering() -> None:
    scores = build_dynamic_pool_candidate_scores(_rows())
    summaries = build_dynamic_pool_photo_summaries(scores)
    first = scores["score_id"][0]
    tampered_scores = scores.with_columns(
        pl.when(pl.col("score_id") == first)
        .then(pl.lit(_sha("f")))
        .otherwise(pl.col("score_fingerprint"))
        .alias("score_fingerprint")
    )
    with pytest.raises(ValueError, match="score_fingerprint mismatch"):
        validate_dynamic_pool_candidate_scores(tampered_scores)

    tampered_summary = summaries.with_columns(
        pl.lit(_sha("f")).alias("summary_fingerprint")
    )
    with pytest.raises(ValueError, match="summary_fingerprint mismatch"):
        validate_dynamic_pool_photo_summaries(tampered_summary)

    rebound_summary_id = summaries.with_columns(
        pl.lit(f"dynamic-pool-photo-summary:{'e' * 64}").alias("photo_summary_id")
    ).with_columns(
        pl.struct(pl.all().exclude("summary_fingerprint"))
        .map_elements(canonical_semantic_fingerprint, return_dtype=pl.String)
        .alias("summary_fingerprint")
    )
    with pytest.raises(ValueError, match="summary identity mismatch"):
        validate_dynamic_pool_photo_summaries(rebound_summary_id)

    rebound = summaries.with_columns(
        pl.lit(COMPETITOR).alias("top_candidate_accepted_taxon_key")
    )
    with pytest.raises(ValueError):
        validate_dynamic_pool_score_artifacts(scores, rebound)


def test_photo_summary_schema_keeps_probability_and_release_distinct() -> None:
    fields = set(dynamic_pool_photo_summary_schema())

    assert "top_fused_raw_score" in fields
    assert "top_calibrated_probability" in fields
    assert "probability_availability" in fields
    assert "probability_unavailable_reason" in fields
    assert "human_review_required" in fields
    assert "statistical_support_report_fingerprint" in fields
    assert "statistical_support_reason" in fields
    assert "release_ready" in fields
    assert "scientific_claim_allowed" in fields
