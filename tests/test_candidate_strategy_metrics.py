"""Tests for evidence-bound candidate-strategy metrics."""

from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.family_geo_candidates import build_family_geo_candidate_sets
from biominer.candidates.strategy_ablation import (
    FAMILY_FIRST_SAFE_STRATEGY,
    GEOGRAPHY_FIRST_STRATEGY,
    PARALLEL_UNION_STRATEGY,
    build_candidate_strategy_plans,
)
from biominer.evaluation.candidate_strategies import (
    CANDIDATE_STRATEGY_METRICS_FILE,
    FAMILY_PRUNING_COUNTERFACTUAL_FILE,
    build_candidate_strategy_metrics,
    build_family_pruning_counterfactual,
    candidate_strategy_metric_schema,
    summarize_family_pruning_counterfactual,
    validate_candidate_strategy_metrics,
    validate_family_pruning_counterfactual,
    write_candidate_strategy_metrics,
    write_family_pruning_counterfactual,
)


TARGET = "gbif:target"
STRATEGIES = (
    GEOGRAPHY_FIRST_STRATEGY,
    FAMILY_FIRST_SAFE_STRATEGY,
    PARALLEL_UNION_STRATEGY,
)
KS = (1, 2, 5)


def test_candidate_strategy_metrics_cover_recall_cost_and_risk_slices() -> None:
    source, plans, labels, measurements = _evaluation_inputs()

    metrics = build_candidate_strategy_metrics(
        source,
        plans,
        evaluation_run_id="candidate-evaluation-1",
        labels=labels,
        measurements=measurements,
        ks=KS,
    )

    assert metrics.schema == candidate_strategy_metric_schema()
    assert metrics.height == len(STRATEGIES) * 2 * len(KS)
    assert set(metrics["target_candidate_recall_at_k"]) <= {0.0, 1.0}
    assert set(metrics["species_candidate_recall_at_k"]) <= {0.0, 1.0}
    assert set(metrics["family_candidate_recall_at_k"]) <= {0.0, 1.0}
    assert set(metrics["candidate_set_size"]) == {3}
    assert set(metrics["dot_product_count"]) == {2, 4, 6}
    assert set(metrics["reference_member_count"]) == {2, 4, 6}
    assert set(metrics["peak_memory_bytes"]) == {1024, 2048, 3072}
    assert set(metrics["cache_reuse_fraction"]) == {0.5}

    no_geo = metrics.filter(pl.col("no_geo"))
    wrong_family = metrics.filter(pl.col("wrong_family"))
    assert no_geo.height == wrong_family.height == len(STRATEGIES) * len(KS)
    assert set(no_geo["source_candidate_set_id"]) == {labels[1]["source_candidate_set_id"]}
    assert set(wrong_family["family_counterfactual_status"]) == {"wrong_family"}

    wrong_at_one = wrong_family.filter(pl.col("k") == 1)
    by_strategy = {
        row["strategy_name"]: row for row in wrong_at_one.to_dicts()
    }
    assert by_strategy[FAMILY_FIRST_SAFE_STRATEGY][
        "species_candidate_recall_at_k"
    ] == 0.0
    assert by_strategy[FAMILY_FIRST_SAFE_STRATEGY][
        "family_candidate_recall_at_k"
    ] == 0.0
    assert by_strategy[GEOGRAPHY_FIRST_STRATEGY][
        "species_candidate_recall_at_k"
    ] == 1.0
    assert by_strategy[PARALLEL_UNION_STRATEGY][
        "species_candidate_recall_at_k"
    ] == 1.0


def test_candidate_strategy_metrics_require_real_consistent_measurements() -> None:
    source, plans, labels, measurements = _evaluation_inputs()

    with pytest.raises(ValueError, match="missing measurement"):
        build_candidate_strategy_metrics(
            source,
            plans,
            evaluation_run_id="candidate-evaluation-1",
            labels=labels,
            measurements=measurements[:-1],
            ks=KS,
        )

    invalid = [dict(row) for row in measurements]
    invalid[0]["cache_new_reference_members"] = 99
    with pytest.raises(ValueError, match="partition reference members"):
        build_candidate_strategy_metrics(
            source,
            plans,
            evaluation_run_id="candidate-evaluation-1",
            labels=labels,
            measurements=invalid,
            ks=KS,
        )


def test_candidate_strategy_metrics_are_deterministic_and_round_trip(
    tmp_path,
) -> None:
    source, plans, labels, measurements = _evaluation_inputs()

    first = build_candidate_strategy_metrics(
        source,
        plans,
        evaluation_run_id="candidate-evaluation-1",
        labels=labels,
        measurements=measurements,
        ks=KS,
    )
    second = build_candidate_strategy_metrics(
        source,
        list(reversed(plans)),
        evaluation_run_id="candidate-evaluation-1",
        labels=list(reversed(labels)),
        measurements=list(reversed(measurements)),
        ks=KS,
    )

    assert first.equals(second)
    assert first["strategy_metric_id"].n_unique() == first.height
    path = write_candidate_strategy_metrics(first, tmp_path)
    assert path.name == CANDIDATE_STRATEGY_METRICS_FILE
    persisted = pl.read_parquet(path)
    validate_candidate_strategy_metrics(persisted)
    assert persisted.equals(first)

    tampered = first.with_columns(
        pl.when(pl.col("strategy_name") == FAMILY_FIRST_SAFE_STRATEGY)
        .then(pl.lit(1.0))
        .otherwise(pl.col("species_candidate_recall_at_k"))
        .alias("species_candidate_recall_at_k")
    )
    with pytest.raises(ValueError, match="inconsistent with its rank"):
        validate_candidate_strategy_metrics(tampered)


def test_family_pruning_counterfactual_quantifies_correct_species_loss(
    tmp_path,
) -> None:
    source, _plans, labels, _measurements = _evaluation_inputs()

    counterfactual = build_family_pruning_counterfactual(
        source,
        evaluation_run_id="candidate-evaluation-1",
        labels=labels,
    )
    summary = summarize_family_pruning_counterfactual(counterfactual)

    assert counterfactual.height == 2
    assert counterfactual["loss_eligible"].to_list() == [True, True]
    assert counterfactual["correct_species_lost"].sum() == 1
    lost = counterfactual.filter(pl.col("correct_species_lost")).row(0, named=True)
    assert lost["organism_unit_id"] == "organism-no-geo"
    assert lost["correct_species_family_priority_match"] is False
    assert lost["reviewed_species_in_complete_union"] is True
    assert lost["reviewed_species_in_hard_family_pool"] is False
    assert lost["no_geo"] is True
    assert summary["evaluated_label_count"] == 2
    assert summary["eligible_correct_species_count"] == 2
    assert summary["correct_species_lost_count"] == 1
    assert summary["correct_species_lost_rate"] == 0.5
    assert summary["wrong_family_evidence_count"] == 1
    assert summary["no_geo"] == {
        "eligible_correct_species_count": 1,
        "correct_species_lost_count": 1,
        "correct_species_lost_rate": 1.0,
    }
    assert summary["production_candidate_membership_changed"] is False

    path = write_family_pruning_counterfactual(counterfactual, tmp_path)
    assert path.name == FAMILY_PRUNING_COUNTERFACTUAL_FILE
    persisted = pl.read_parquet(path)
    validate_family_pruning_counterfactual(persisted)
    assert persisted.equals(counterfactual)


def test_family_pruning_loss_denominator_excludes_species_missing_from_union() -> None:
    source, _plans, labels, _measurements = _evaluation_inputs()
    missing_labels = [dict(row) for row in labels]
    missing_labels[0]["reviewed_accepted_taxon_key"] = "gbif:not-in-union"

    counterfactual = build_family_pruning_counterfactual(
        source,
        evaluation_run_id="candidate-evaluation-1",
        labels=missing_labels,
    )
    summary = summarize_family_pruning_counterfactual(counterfactual)

    missing = counterfactual.filter(
        pl.col("reviewed_accepted_taxon_key") == "gbif:not-in-union"
    ).row(0, named=True)
    assert missing["reviewed_species_in_complete_union"] is False
    assert missing["loss_eligible"] is False
    assert missing["correct_species_lost"] is False
    assert summary["evaluated_label_count"] == 2
    assert summary["eligible_correct_species_count"] == 1
    assert summary["correct_species_lost_count"] == 1
    assert summary["correct_species_lost_rate"] == 1.0
    assert summary["reviewed_species_missing_from_complete_union_count"] == 1


def _evaluation_inputs() -> tuple[
    pl.DataFrame,
    list[pl.DataFrame],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    source = build_family_geo_candidate_sets(
        [
            *_candidate_set_rows(
                photo="photo-geo",
                organism="organism-geo",
                geography_available=True,
                target_family_match=True,
            ),
            *_candidate_set_rows(
                photo="photo-no-geo",
                organism="organism-no-geo",
                geography_available=False,
                target_family_match=False,
            ),
        ]
    )
    plans = [
        build_candidate_strategy_plans(source, strategy=strategy)
        for strategy in STRATEGIES
    ]
    set_by_organism = {
        group["organism_unit_id"][0]: candidate_set_id
        for (candidate_set_id,), group in source.group_by("candidate_set_id")
    }
    labels = [
        {
            "source_candidate_set_id": set_by_organism[organism],
            "reviewed_accepted_taxon_key": TARGET,
            "reviewed_family_key": "family-papilionidae",
            "label_status": "human_reviewed",
            "label_source": "fixture-review-v1",
        }
        for organism in ("organism-geo", "organism-no-geo")
    ]
    measurements: list[dict[str, object]] = []
    for plan in plans:
        for (plan_id,), group in plan.group_by("strategy_plan_id"):
            for k in KS:
                evaluated = min(k, group.height)
                reference_members = evaluated * 2
                measurements.append(
                    {
                        "strategy_plan_id": plan_id,
                        "k": k,
                        "evaluated_candidate_count": evaluated,
                        "dot_product_count": reference_members,
                        "reference_member_count": reference_members,
                        "elapsed_time_ms": float(evaluated) / 10,
                        "peak_memory_bytes": evaluated * 1024,
                        "cache_reused_reference_members": evaluated,
                        "cache_new_reference_members": evaluated,
                        "measurement_source": "fixture-instrumentation-v1",
                    }
                )
    return source, plans, labels, measurements


def _candidate_set_rows(
    *,
    photo: str,
    organism: str,
    geography_available: bool,
    target_family_match: bool,
) -> list[dict[str, object]]:
    return [
        _candidate_row(
            photo=photo,
            organism=organism,
            key=TARGET,
            name="Papilio target",
            family_key="family-papilionidae",
            priority=0,
            target=True,
            family=True,
            family_match=target_family_match,
            geography=geography_available,
            query_has_geography=geography_available,
            safety=True,
        ),
        _candidate_row(
            photo=photo,
            organism=organism,
            key="gbif:priority-family",
            name="Pieris priority",
            family_key="family-pieridae",
            priority=1,
            family=True,
            family_match=not target_family_match,
            geography=False,
            query_has_geography=geography_available,
        ),
        _candidate_row(
            photo=photo,
            organism=organism,
            key="gbif:remainder",
            name="Danaus remainder",
            family_key="family-nymphalidae",
            priority=2,
            query_has_geography=geography_available,
        ),
    ]


def _candidate_row(
    *,
    photo: str,
    organism: str,
    key: str,
    name: str,
    family_key: str,
    priority: int,
    target: bool = False,
    family: bool = False,
    family_match: bool | None = None,
    geography: bool = False,
    query_has_geography: bool = False,
    safety: bool = False,
) -> dict[str, object]:
    genus = name.split()[0]
    return {
        "run_id": "run-candidate-evaluation",
        "flickr_query_id": f"query-{photo}",
        "flickr_photo_id": photo,
        "organism_unit_id": organism,
        "scoring_stage": "initial",
        "registry_version": "registry-v1",
        "target_accepted_taxon_key": TARGET,
        "target_scientific_name": "Papilio target",
        "query_geo_cluster_id": "geo-au-qld" if query_has_geography else None,
        "query_coordinate_quality": "local" if query_has_geography else "no_geo",
        "candidate_accepted_taxon_key": key,
        "candidate_scientific_name": name,
        "family_key": family_key,
        "family_name": family_key.removeprefix("family-").title(),
        "genus_key": f"genus-{genus.casefold()}",
        "genus_name": genus,
        "candidate_priority": priority,
        "candidate_reasons": ["target"] if target else ["complete_union"],
        "family_evidence_status": "available" if family else "unavailable",
        "family_evidence_reason": None if family else "outside_family_priority",
        "family_evidence_rank": priority + 1 if family else None,
        "family_evidence_raw_score": 0.9 - priority / 10 if family else None,
        "family_priority_match": family_match if family else None,
        "family_changed_membership": False,
        "geographic_evidence_status": "available" if geography else "unavailable",
        "geographic_evidence_reason": None if geography else "no_geo",
        "geographic_scopes": ["exact_local_cell"] if geography else [],
        "geographic_evidence_score": 0.8 if geography else None,
        "occurrence_support": 3 if geography else 0,
        "query_evidence_status": "available" if target else "not_applicable",
        "query_evidence_reason": None if target else "not_query_associated",
        "query_evidence_ids": ["query-evidence-1"] if target else [],
        "query_associated": target,
        "visual_neighbour_evidence_status": "not_applicable",
        "visual_neighbour_evidence_reason": "not_visual_neighbour",
        "visual_neighbour_graph_fingerprint": None,
        "visual_neighbour_rank": None,
        "visual_neighbour_raw_similarity": None,
        "visual_neighbour": False,
        "safety_union_membership": safety,
        "safety_union_reasons": ["target"] if safety else [],
        "target_candidate": target,
        "target_preserved": True,
        "included_in_complete_union": True,
        "source_versions": ["registry:v1", "candidate-evaluation:v1"],
    }
