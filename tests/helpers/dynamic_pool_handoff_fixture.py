"""Small, internally consistent dynamic-pooling handoff fixture."""

from __future__ import annotations

import polars as pl

from biominer.bioclip.dynamic_pool_contracts import (
    build_dynamic_reference_pool_members,
    build_dynamic_reference_pool_plans,
    build_dynamic_reference_pool_summaries,
    dynamic_reference_pool_plan_id,
)
from biominer.bioclip.dynamic_pool_scores import (
    build_dynamic_pool_candidate_scores,
    build_dynamic_pool_photo_summaries,
)
from biominer.bioclip.family_geo_candidates import build_family_geo_candidate_sets
from biominer.evaluation.dynamic_pool_quality import (
    DynamicPoolQualityObservation,
    DynamicPoolQualityPolicy,
    report_overall_pooling_quality,
)
from biominer.evaluation.dynamic_pool_review import (
    ProbabilityAuditSamplingPolicy,
    ProbabilityAuditSelection,
    build_dynamic_pool_audit_frame,
    build_probability_audit_sample,
)
from biominer.vision.bioclip_input_contract import (
    DYNAMIC_POOL_VISUAL_MODE,
    bioclip_visual_input_contract,
)
from biominer.vision.full_frame_attention import (
    FULL_FRAME_VISUAL_INPUT_VERSION,
    RAW_FULL_IMAGE_KIND,
)


TARGET = "gbif:5131359"
COMPETITOR = "gbif:5131360"


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _candidate_row(
    *,
    candidate_key: str,
    candidate_name: str,
    target: bool,
    priority: int,
) -> dict[str, object]:
    visual = not target
    return {
        "run_id": "run-tx-handoff-1",
        "flickr_query_id": "query-papilio-demoleus",
        "flickr_photo_id": "flickr-photo-1",
        "organism_unit_id": "organism-unit-1",
        "scoring_stage": "initial",
        "registry_version": "butterflies-v2-20260712",
        "target_accepted_taxon_key": TARGET,
        "target_scientific_name": "Papilio demoleus",
        "query_geo_cluster_id": "geo-au-qld",
        "query_coordinate_quality": "local",
        "candidate_accepted_taxon_key": candidate_key,
        "candidate_scientific_name": candidate_name,
        "family_key": "gbif:9417",
        "family_name": "Papilionidae",
        "genus_key": "gbif:1920494",
        "genus_name": "Papilio",
        "candidate_priority": priority,
        "candidate_reasons": ["target"] if target else ["visually_nearest"],
        "family_evidence_status": "available",
        "family_evidence_reason": None,
        "family_evidence_rank": priority + 1,
        "family_evidence_raw_score": 0.9 - priority / 10,
        "family_priority_match": True,
        "family_changed_membership": False,
        "geographic_evidence_status": "available",
        "geographic_evidence_reason": None,
        "geographic_scopes": ["exact_local_cell"],
        "geographic_evidence_score": 0.8 - priority / 10,
        "occurrence_support": 5 - priority,
        "query_evidence_status": "available" if target else "not_applicable",
        "query_evidence_reason": None if target else "not_query_associated",
        "query_evidence_ids": ["query-evidence-1"] if target else [],
        "query_associated": target,
        "visual_neighbour_evidence_status": (
            "available" if visual else "not_applicable"
        ),
        "visual_neighbour_evidence_reason": (
            None if visual else "not_a_visual_neighbour"
        ),
        "visual_neighbour_graph_fingerprint": _sha("a") if visual else None,
        "visual_neighbour_rank": 1 if visual else None,
        "visual_neighbour_raw_similarity": 0.72 if visual else None,
        "visual_neighbour": visual,
        "safety_union_membership": True,
        "safety_union_reasons": ["target"] if target else ["visual_neighbour"],
        "target_candidate": target,
        "target_preserved": True,
        "included_in_complete_union": True,
        "source_versions": ["registry:v2", "regional-candidate:v1"],
    }


def _plan_context(candidate_sets: pl.DataFrame) -> dict[str, object]:
    return {
        "run_id": "run-tx-handoff-1",
        "flickr_query_id": "query-papilio-demoleus",
        "flickr_photo_id": "flickr-photo-1",
        "organism_unit_id": "organism-unit-1",
        "visual_input_id": _sha("1"),
        "query_embedding_fingerprint": _sha("2"),
        "scoring_stage": "initial",
        "query_route": "adult_field",
        "registry_version": "butterflies-v2-20260712",
        "reference_bank_version": "reference-bank-v3",
        "reference_geography_index_fingerprint": _sha("3"),
        "candidate_set_id": candidate_sets["candidate_set_id"][0],
        "candidate_set_fingerprint": candidate_sets["candidate_set_fingerprint"][0],
        "query_geo_cluster_id": "geo-au-qld",
        "query_coordinate_quality": "local",
        "local_pool_status": "available",
        "local_pool_unavailable_reason": None,
        "selection_policy_version": "dynamic-pool-selection-v1",
        "selection_policy_fingerprint": _sha("4"),
        "model_id": "bioclip-2.5",
        "model_revision": "revision-1",
        "model_weights_sha256": _sha("5"),
        "model_fingerprint": _sha("6"),
        "preprocessing_fingerprint": _sha("7"),
        "configured_global_per_candidate": 1,
        "configured_local_per_candidate": 1,
        "configured_safety_per_candidate": 1,
        "maximum_expansion_rounds": 2,
    }


def _member_row(
    context: dict[str, object],
    *,
    candidate_key: str,
    candidate_name: str,
    suffix: str,
    local: bool,
) -> dict[str, object]:
    return {
        "plan_id": dynamic_reference_pool_plan_id(context),
        "run_id": context["run_id"],
        "flickr_query_id": context["flickr_query_id"],
        "flickr_photo_id": context["flickr_photo_id"],
        "organism_unit_id": context["organism_unit_id"],
        "visual_input_id": context["visual_input_id"],
        "query_embedding_fingerprint": context["query_embedding_fingerprint"],
        "scoring_stage": context["scoring_stage"],
        "query_route": context["query_route"],
        "candidate_set_id": context["candidate_set_id"],
        "candidate_set_fingerprint": context["candidate_set_fingerprint"],
        "candidate_accepted_taxon_key": candidate_key,
        "candidate_scientific_name": candidate_name,
        "reference_media_id": f"reference-media:{suffix * 64}",
        "reference_observation_id": f"reference-observation:{suffix * 64}",
        "reference_embedding_fingerprint": _sha(suffix),
        "reference_route": "adult_field",
        "reference_visual_input_kind": "raw_full_image",
        "pool_scope": "local" if local else "global",
        "pool_role": "nearest_local" if local else "global_core",
        "geographic_scope": "exact_local_cell" if local else "global",
        "geographic_distance_km": 12.5 if local else None,
        "geographic_distance_status": "available" if local else "not_applicable",
        "geographic_distance_reason": (
            None if local else "global_pool_has_no_query_distance"
        ),
        "fallback_level": 0,
        "selection_rank": 1,
        "independent_observation_group": f"observation-group-{suffix}",
        "observer_id_hash": _sha(suffix),
        "reference_country_code": "au",
        "inclusion_reason": "nearest_local" if local else "global_core",
        "selection_policy_fingerprint": context["selection_policy_fingerprint"],
        "source": "gbif",
        "source_dataset_key": f"dataset-{suffix}",
        "registry_version": context["registry_version"],
        "reference_bank_version": context["reference_bank_version"],
        "reference_geography_index_fingerprint": context[
            "reference_geography_index_fingerprint"
        ],
        "model_id": context["model_id"],
        "model_revision": context["model_revision"],
        "model_weights_sha256": context["model_weights_sha256"],
        "model_fingerprint": context["model_fingerprint"],
        "preprocessing_fingerprint": context["preprocessing_fingerprint"],
        "expansion_round": 0,
    }


def _score_row(
    plan: dict[str, object],
    *,
    candidate_key: str,
    candidate_name: str,
    target: bool,
    priority: int,
    fused: float,
    global_score: float,
    local_score: float,
) -> dict[str, object]:
    input_contract = bioclip_visual_input_contract(DYNAMIC_POOL_VISUAL_MODE)
    return {
        "run_id": plan["run_id"],
        "flickr_query_id": plan["flickr_query_id"],
        "flickr_photo_id": plan["flickr_photo_id"],
        "organism_unit_id": plan["organism_unit_id"],
        "visual_input_id": plan["visual_input_id"],
        "visual_input_kind": RAW_FULL_IMAGE_KIND,
        "visual_input_version": FULL_FRAME_VISUAL_INPUT_VERSION,
        "visual_input_contract_version": input_contract.contract_version,
        "visual_input_contract_fingerprint": input_contract.fingerprint,
        "spatial_crop_applied": False,
        "scoring_stage": plan["scoring_stage"],
        "query_route": plan["query_route"],
        "plan_id": plan["plan_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "candidate_set_id": plan["candidate_set_id"],
        "candidate_set_fingerprint": plan["candidate_set_fingerprint"],
        "target_accepted_taxon_key": TARGET,
        "target_scientific_name": "Papilio demoleus",
        "candidate_accepted_taxon_key": candidate_key,
        "candidate_scientific_name": candidate_name,
        "candidate_priority": priority,
        "target_candidate": target,
        "target_preserved": True,
        "global_pool_ids": plan["global_pool_ids"],
        "local_pool_ids": plan["local_pool_ids"],
        "safety_pool_ids": plan["safety_pool_ids"],
        "local_pool_status": plan["local_pool_status"],
        "local_pool_unavailable_reason": plan["local_pool_unavailable_reason"],
        "global_score_status": "available",
        "global_score_unavailable_reason": None,
        "global_prototype_similarity": global_score - 0.02,
        "global_nearest_reference_similarity": global_score + 0.02,
        "global_top_k_mean_similarity": global_score,
        "global_raw_component_score": global_score,
        "global_configured_k": 1,
        "global_effective_k": 1,
        "global_reference_count": 1,
        "global_independent_observation_count": 1,
        "global_reference_shortfall_count": 0,
        "local_score_status": "available",
        "local_score_unavailable_reason": None,
        "local_prototype_similarity": local_score - 0.02,
        "local_nearest_reference_similarity": local_score + 0.02,
        "local_top_k_mean_similarity": local_score,
        "local_raw_component_score": local_score,
        "local_configured_k": 1,
        "local_effective_k": 1,
        "local_reference_count": 1,
        "local_independent_observation_count": 1,
        "local_reference_shortfall_count": 0,
        "global_local_disagreement_status": "available",
        "global_local_disagreement_reason": None,
        "global_local_raw_disagreement": abs(global_score - local_score),
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
        "model_fingerprint": plan["model_fingerprint"],
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


def build_dynamic_pool_handoff_fixture() -> dict[str, pl.DataFrame]:
    """Build two candidates with global/local pools and raw score evidence."""

    candidate_sets = build_family_geo_candidate_sets(
        [
            _candidate_row(
                candidate_key=TARGET,
                candidate_name="Papilio demoleus",
                target=True,
                priority=0,
            ),
            _candidate_row(
                candidate_key=COMPETITOR,
                candidate_name="Papilio polytes",
                target=False,
                priority=1,
            ),
        ]
    )
    context = _plan_context(candidate_sets)
    members = build_dynamic_reference_pool_members(
        [
            _member_row(
                context,
                candidate_key=TARGET,
                candidate_name="Papilio demoleus",
                suffix="1",
                local=False,
            ),
            _member_row(
                context,
                candidate_key=TARGET,
                candidate_name="Papilio demoleus",
                suffix="2",
                local=True,
            ),
            _member_row(
                context,
                candidate_key=COMPETITOR,
                candidate_name="Papilio polytes",
                suffix="3",
                local=False,
            ),
            _member_row(
                context,
                candidate_key=COMPETITOR,
                candidate_name="Papilio polytes",
                suffix="4",
                local=True,
            ),
        ]
    )
    plans = build_dynamic_reference_pool_plans(
        [{"plan_id": dynamic_reference_pool_plan_id(context), **context}],
        members,
    )
    pool_summaries = build_dynamic_reference_pool_summaries(plans, members)
    plan = plans.row(0, named=True)
    candidate_scores = build_dynamic_pool_candidate_scores(
        [
            _score_row(
                plan,
                candidate_key=TARGET,
                candidate_name="Papilio demoleus",
                target=True,
                priority=0,
                fused=0.8,
                global_score=0.82,
                local_score=0.78,
            ),
            _score_row(
                plan,
                candidate_key=COMPETITOR,
                candidate_name="Papilio polytes",
                target=False,
                priority=1,
                fused=0.6,
                global_score=0.58,
                local_score=0.65,
            ),
        ]
    )
    return {
        "candidate_scores": candidate_scores,
        "photo_summaries": build_dynamic_pool_photo_summaries(candidate_scores),
        "pool_plans": plans,
        "pool_members": members,
        "pool_summaries": pool_summaries,
        "candidate_sets": candidate_sets,
    }


def build_review_selection_fixture() -> tuple[
    ProbabilityAuditSelection,
    ProbabilityAuditSamplingPolicy,
]:
    """Build a representative sample with geographic and no-geo units."""

    candidates: list[dict[str, object]] = []
    for index in range(4):
        no_geo = index == 3
        candidates.append(
            {
                "sampling_unit_id": f"review-unit-{index}",
                "source_record_hash": _sha(str(index + 1)),
                "source_artifact_fingerprint": _sha("9"),
                "flickr_photo_id": f"photo-{index}",
                "organism_unit_id": f"organism-{index}",
                "candidate_family_accepted_taxon_key": "col:Papilionidae",
                "candidate_family_scientific_name": "Papilionidae",
                "candidate_genus_accepted_taxon_key": "col:Papilio",
                "candidate_genus_scientific_name": "Papilio",
                "candidate_species_accepted_taxon_key": "col:Papilio-demoleus",
                "candidate_species_scientific_name": "Papilio demoleus",
                "geographic_cluster_id": None if no_geo else f"geo-au-{index % 2}",
                "no_geo": no_geo,
                "primary_query_tier": "T2",
                "raw_fusion_score": 0.72 - index / 20,
                "raw_competitor_margin": 0.04 + index / 100,
                "pool_disagreement": None if no_geo else 0.18 - index / 100,
                "route": "adult_field",
                "visual_domain": "field_photo",
                "subject_area_ratio": 0.08,
                "owner_group_id": f"owner-{index}",
                "duplicate_group_id": f"duplicate-{index}",
                "observation_group_id": f"observation-{index}",
                "final_release_candidate": True,
            }
        )
    policy = ProbabilityAuditSamplingPolicy(review_budget=4, random_seed=17)
    selection = build_probability_audit_sample(
        build_dynamic_pool_audit_frame(candidates),
        policy=policy,
    )
    return selection, policy


def build_quality_report_fixture(*, sufficient: bool = True) -> pl.DataFrame:
    """Build a small reviewed-evidence report with permissive fixture floors."""

    observations = [
        DynamicPoolQualityObservation(
            item_id=f"item-{index}",
            source_record_id=f"flickr:{index}",
            source_image_sha256=_sha(str(index + 1)),
            independence_component_id=f"component-{index}",
            family_key="family-papilionidae",
            family_name="Papilionidae",
            genus_key="genus-papilio",
            genus_name="Papilio",
            species_key="species-papilio-demoleus",
            scientific_name="Papilio demoleus",
            sampling_purpose="representative_audit",
            representative_estimation_eligible=True,
            sampling_weight=1.0,
            human_supported=index != 2,
            screening_selected=index != 3,
            model_abstained=False,
            family_routing_correct=True,
            global_local_disagreed=index % 2 == 0,
            local_support_available=True,
            reference_outlier_influenced_error=False,
            out_of_distribution=False,
            calibrated_supported_probability=0.8 if index != 2 else 0.3,
            country_code="AU",
            admin1="NSW",
            bioregion="Sydney Basin",
            geographic_cluster_id=f"geo-{index % 2}",
        )
        for index in range(4)
    ]
    policy = (
        DynamicPoolQualityPolicy(
            minimum_group_items=2,
            minimum_group_components=2,
            minimum_group_effective_sample_size=2.0,
            minimum_metric_denominator_items=1,
            minimum_metric_denominator_components=1,
            minimum_metric_effective_sample_size=1.0,
        )
        if sufficient
        else DynamicPoolQualityPolicy()
    )
    return report_overall_pooling_quality(observations, policy=policy)


__all__ = [
    "build_dynamic_pool_handoff_fixture",
    "build_quality_report_fixture",
    "build_review_selection_fixture",
]
