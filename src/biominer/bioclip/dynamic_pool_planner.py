"""Deterministic reference-observation planner for dynamic pools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from biominer.bioclip.dynamic_pool_contracts import (
    build_dynamic_reference_pool_members,
    build_dynamic_reference_pool_plans,
    build_dynamic_reference_pool_summaries,
    dynamic_reference_pool_plan_id,
    validate_dynamic_reference_pool_artifacts,
)
from biominer.bioclip.dynamic_pool_policy import DynamicReferencePoolPolicy
from biominer.bioclip.family_geo_candidates import (
    validate_family_geo_candidate_sets,
)
from biominer.bioclip.global_reference_anchors import (
    validate_global_reference_anchors,
)
from biominer.bioclip.reference_geography_index import (
    reference_geography_index_artifact_fingerprint,
    validate_reference_geography_index,
)
from biominer.vision.full_frame_attention import RAW_FULL_IMAGE_KIND


DYNAMIC_POOL_PLANNING_REQUEST_FIELDS = frozenset(
    {
        "run_id",
        "flickr_query_id",
        "flickr_photo_id",
        "organism_unit_id",
        "visual_input_id",
        "query_embedding_fingerprint",
        "scoring_stage",
        "query_route",
        "registry_version",
        "reference_bank_version",
        "reference_geography_index_fingerprint",
        "candidate_set_id",
        "candidate_set_fingerprint",
        "query_geo_cluster_id",
        "query_coordinate_quality",
        "model_id",
        "model_revision",
        "model_weights_sha256",
        "model_fingerprint",
        "preprocessing_fingerprint",
    }
)

_NO_LOCAL_QUERY_QUALITIES = frozenset(
    {"no_geo", "unassigned_geo", "withheld", "invalid"}
)
_REQUEST_SORT = (
    "run_id",
    "flickr_photo_id",
    "organism_unit_id",
    "scoring_stage",
    "candidate_set_id",
)


def plan_dynamic_reference_pools(
    requests: Sequence[Mapping[str, object]],
    candidate_sets: pl.DataFrame,
    reference_geography_index: pl.DataFrame,
    global_reference_anchors: pl.DataFrame,
    *,
    policy: DynamicReferencePoolPolicy,
    burst_group_by_observation: Mapping[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Select observations and emit the accepted plan/member/summary contracts."""

    if not isinstance(policy, DynamicReferencePoolPolicy):
        raise TypeError("policy must be a DynamicReferencePoolPolicy")
    validate_family_geo_candidate_sets(candidate_sets)
    validate_reference_geography_index(reference_geography_index)
    validate_global_reference_anchors(global_reference_anchors)
    normalized_requests = _normalize_requests(requests)
    index_fingerprint = reference_geography_index_artifact_fingerprint(
        reference_geography_index
    )
    _validate_shared_inputs(
        normalized_requests,
        candidate_sets=candidate_sets,
        reference_index=reference_geography_index,
        global_anchors=global_reference_anchors,
        index_fingerprint=index_fingerprint,
    )
    candidate_groups = {
        str(candidate_set_id): group.sort(
            "candidate_priority", "candidate_accepted_taxon_key"
        )
        for (candidate_set_id,), group in candidate_sets.group_by(
            "candidate_set_id", maintain_order=True
        )
    }
    anchor_ranks = {
        (
            str(row["reference_media_id"]),
            str(row["embedding_fingerprint"]),
        ): int(row["group_selection_rank"])
        for row in global_reference_anchors.iter_rows(named=True)
    }
    burst_groups = _normalize_burst_groups(
        burst_group_by_observation,
        reference_index=reference_geography_index,
    )
    plan_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for request in normalized_requests:
        candidate_group = candidate_groups[str(request["candidate_set_id"])]
        selections = _select_request_members(
            request,
            candidate_group=candidate_group,
            reference_index=reference_geography_index,
            anchor_ranks=anchor_ranks,
            burst_groups=burst_groups,
            policy=policy,
        )
        if not any(item["pool_scope"] == "global" for item in selections):
            raise ValueError("dynamic pool planning request has no global reference support")
        local_available = any(item["pool_scope"] == "local" for item in selections)
        local_reason = _local_unavailable_reason(
            request,
            local_available=local_available,
        )
        context = _plan_context(
            request,
            policy=policy,
            local_available=local_available,
            local_unavailable_reason=local_reason,
        )
        plan_id = dynamic_reference_pool_plan_id(context)
        plan_rows.append({"plan_id": plan_id, **context})
        member_rows.extend(
            _member_row(
                selection,
                request=request,
                context=context,
                plan_id=plan_id,
                policy=policy,
                index_fingerprint=index_fingerprint,
            )
            for selection in selections
        )
    members = build_dynamic_reference_pool_members(member_rows)
    plans = build_dynamic_reference_pool_plans(plan_rows, members)
    summaries = build_dynamic_reference_pool_summaries(plans, members)
    validate_dynamic_reference_pool_artifacts(plans, members, summaries)
    return plans, members, summaries


def _select_request_members(
    request: Mapping[str, object],
    *,
    candidate_group: pl.DataFrame,
    reference_index: pl.DataFrame,
    anchor_ranks: Mapping[tuple[str, str], int],
    burst_groups: Mapping[str, str],
    policy: DynamicReferencePoolPolicy,
) -> list[dict[str, object]]:
    queues: dict[str, list[dict[str, object]]] = {}
    candidate_order: list[str] = []
    for candidate in candidate_group.iter_rows(named=True):
        candidate_key = str(candidate["candidate_accepted_taxon_key"])
        candidate_order.append(candidate_key)
        selected_observations: set[str] = set()
        selected_duplicates: set[str] = set()
        candidate_index = reference_index.filter(
            (pl.col("accepted_taxon_key") == candidate["candidate_accepted_taxon_key"])
            & (pl.col("route") == request["query_route"])
        )
        _validate_candidate_reference_names(candidate, candidate_index)
        global_rows = _rank_reference_rows(
            candidate_index.filter(pl.col("global_anchor_eligible")),
            anchor_ranks=anchor_ranks,
            selected_observations=selected_observations,
            selected_duplicates=selected_duplicates,
            scope="global",
        )
        selected_global = global_rows[: policy.maximum_global_per_candidate]
        queue = [
            _selection_candidate(
                candidate,
                item,
                pool_scope="global",
                pool_role="global_core",
                geographic_scope="global",
                geographic_distance_status="not_applicable",
                geographic_distance_reason="global_pool_has_no_query_distance",
                fallback_level=0 if item["anchor_rank"] is not None else 1,
                inclusion_reason=(
                    "global_reference_anchor"
                    if item["anchor_rank"] is not None
                    else "global_index_fallback"
                ),
                burst_groups=burst_groups,
            )
            for item in selected_global
        ]
        for item in selected_global:
            _record_selected(
                item["reference"],
                observations=selected_observations,
                duplicates=selected_duplicates,
            )
        if not _query_supports_local(request):
            queues[candidate_key] = queue
            continue
        local_rows = candidate_index.filter(
            pl.col("local_anchor_eligible")
            & (pl.col("geo_cluster_id") == request["query_geo_cluster_id"])
        )
        ranked_local = _rank_reference_rows(
            local_rows,
            anchor_ranks={},
            selected_observations=selected_observations,
            selected_duplicates=selected_duplicates,
            scope="local",
        )
        selected_local = ranked_local[: policy.maximum_local_per_candidate]
        queue.extend(
            _selection_candidate(
                candidate,
                item,
                pool_scope="local",
                pool_role="nearest_local",
                geographic_scope="exact_local_cell",
                geographic_distance_status="unavailable",
                geographic_distance_reason="query_distance_not_materialized",
                fallback_level=0,
                inclusion_reason="exact_workload_cluster",
                burst_groups=burst_groups,
            )
            for item in selected_local
        )
        queues[candidate_key] = queue
    return _balance_candidate_queues(
        queues,
        candidate_order=candidate_order,
        stage=str(request["scoring_stage"]),
        policy=policy,
    )


def _rank_reference_rows(
    frame: pl.DataFrame,
    *,
    anchor_ranks: Mapping[tuple[str, str], int],
    selected_observations: set[str],
    selected_duplicates: set[str],
    scope: str,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in frame.iter_rows(named=True):
        observation = str(row["reference_observation_id"])
        duplicate = str(row["duplicate_group_id"])
        if observation in selected_observations or duplicate in selected_duplicates:
            continue
        anchor_rank = anchor_ranks.get(
            (str(row["reference_media_id"]), str(row["embedding_fingerprint"]))
        )
        candidates.append(
            {
                "reference": row,
                "anchor_rank": anchor_rank,
                "key": (
                    0 if anchor_rank is not None else 1,
                    anchor_rank if anchor_rank is not None else 2**32,
                    len(row["reference_quality_flags"]),
                    0 if row["visual_input_kind"] == RAW_FULL_IMAGE_KIND else 1,
                    _scope_geography_rank(row, scope=scope),
                    str(row["reference_observation_id"]),
                    str(row["reference_media_id"]),
                    str(row["embedding_fingerprint"]),
                ),
            }
        )
    candidates.sort(key=lambda item: item["key"])
    output: list[dict[str, object]] = []
    observations = set(selected_observations)
    duplicates = set(selected_duplicates)
    for item in candidates:
        row = item["reference"]
        observation = str(row["reference_observation_id"])
        duplicate = str(row["duplicate_group_id"])
        if observation in observations or duplicate in duplicates:
            continue
        output.append(item)
        observations.add(observation)
        duplicates.add(duplicate)
    return output


def _selection_candidate(
    candidate: Mapping[str, object],
    ranked: Mapping[str, object],
    *,
    pool_scope: str,
    pool_role: str,
    geographic_scope: str,
    geographic_distance_status: str,
    geographic_distance_reason: str,
    fallback_level: int,
    inclusion_reason: str,
    burst_groups: Mapping[str, str],
) -> dict[str, object]:
    reference = ranked["reference"]
    observation_id = str(reference["reference_observation_id"])
    return {
        "candidate": candidate,
        "reference": reference,
        "pool_scope": pool_scope,
        "pool_role": pool_role,
        "geographic_scope": geographic_scope,
        "geographic_distance_status": geographic_distance_status,
        "geographic_distance_reason": geographic_distance_reason,
        "geographic_distance_km": None,
        "fallback_level": fallback_level,
        "selection_rank": 0,
        "inclusion_reason": inclusion_reason,
        "independent_observation_group": burst_groups.get(
            observation_id, observation_id
        ),
    }


def _balance_candidate_queues(
    queues: Mapping[str, Sequence[dict[str, object]]],
    *,
    candidate_order: Sequence[str],
    stage: str,
    policy: DynamicReferencePoolPolicy,
) -> list[dict[str, object]]:
    budget = min(
        dict(policy.stage_member_limits)[stage],
        policy.maximum_total_reference_members,
    )
    positions = {candidate: 0 for candidate in candidate_order}
    class_counts = {candidate: 0 for candidate in candidate_order}
    observer_counts: dict[str, int] = {}
    locality_counts: dict[str, int] = {}
    seen_observations: set[str] = set()
    seen_duplicates: set[str] = set()
    seen_independent_groups: set[str] = set()
    selected: list[dict[str, object]] = []
    while len(selected) < budget:
        progress = False
        minimum_class_count = min(class_counts.values(), default=0)
        for candidate in candidate_order:
            if len(selected) >= budget:
                break
            if (
                class_counts[candidate] + 1
                > minimum_class_count + policy.maximum_class_count_difference
            ):
                continue
            queue = queues[candidate]
            while positions[candidate] < len(queue):
                item = queue[positions[candidate]]
                positions[candidate] += 1
                if not _passes_diversity_limits(
                    item,
                    policy=policy,
                    observer_counts=observer_counts,
                    locality_counts=locality_counts,
                    seen_observations=seen_observations,
                    seen_duplicates=seen_duplicates,
                    seen_independent_groups=seen_independent_groups,
                ):
                    continue
                selected.append(item)
                class_counts[candidate] += 1
                _record_diversity_selection(
                    item,
                    observer_counts=observer_counts,
                    locality_counts=locality_counts,
                    seen_observations=seen_observations,
                    seen_duplicates=seen_duplicates,
                    seen_independent_groups=seen_independent_groups,
                )
                progress = True
                break
        if not progress:
            break
    ranks: dict[tuple[str, str, str], int] = {}
    for item in selected:
        candidate_key = str(item["candidate"]["candidate_accepted_taxon_key"])
        rank_key = (candidate_key, str(item["pool_scope"]), str(item["pool_role"]))
        ranks[rank_key] = ranks.get(rank_key, 0) + 1
        item["selection_rank"] = ranks[rank_key]
    return selected


def _passes_diversity_limits(
    item: Mapping[str, object],
    *,
    policy: DynamicReferencePoolPolicy,
    observer_counts: Mapping[str, int],
    locality_counts: Mapping[str, int],
    seen_observations: set[str],
    seen_duplicates: set[str],
    seen_independent_groups: set[str],
) -> bool:
    reference = item["reference"]
    observation = str(reference["reference_observation_id"])
    duplicate = str(reference["duplicate_group_id"])
    independent = str(item["independent_observation_group"])
    if (
        observation in seen_observations
        or duplicate in seen_duplicates
        or independent in seen_independent_groups
    ):
        return False
    observer = _observer_diversity_key(reference)
    if observer_counts.get(observer, 0) >= policy.maximum_members_per_observer:
        return False
    locality = _locality_diversity_key(reference)
    return locality_counts.get(locality, 0) < policy.maximum_members_per_locality


def _record_diversity_selection(
    item: Mapping[str, object],
    *,
    observer_counts: dict[str, int],
    locality_counts: dict[str, int],
    seen_observations: set[str],
    seen_duplicates: set[str],
    seen_independent_groups: set[str],
) -> None:
    reference = item["reference"]
    observation = str(reference["reference_observation_id"])
    seen_observations.add(observation)
    seen_duplicates.add(str(reference["duplicate_group_id"]))
    seen_independent_groups.add(str(item["independent_observation_group"]))
    observer = _observer_diversity_key(reference)
    observer_counts[observer] = observer_counts.get(observer, 0) + 1
    locality = _locality_diversity_key(reference)
    locality_counts[locality] = locality_counts.get(locality, 0) + 1


def _observer_diversity_key(reference: Mapping[str, object]) -> str:
    observer = reference["observer_id_hash"]
    return (
        str(observer)
        if observer is not None
        else f"observer-unavailable:{reference['reference_observation_id']}"
    )


def _locality_diversity_key(reference: Mapping[str, object]) -> str:
    for field in (
        "local_cell_id",
        "regional_cell_id",
        "coarse_cell_id",
        "admin1",
        "country_code",
    ):
        value = reference[field]
        if value is not None:
            return f"{field}:{value}"
    return f"locality-unavailable:{reference['reference_observation_id']}"


def _member_row(
    selection: Mapping[str, object],
    *,
    request: Mapping[str, object],
    context: Mapping[str, object],
    plan_id: str,
    policy: DynamicReferencePoolPolicy,
    index_fingerprint: str,
) -> dict[str, object]:
    candidate = selection["candidate"]
    reference = selection["reference"]
    return {
        "plan_id": plan_id,
        "run_id": request["run_id"],
        "flickr_query_id": request["flickr_query_id"],
        "flickr_photo_id": request["flickr_photo_id"],
        "organism_unit_id": request["organism_unit_id"],
        "visual_input_id": request["visual_input_id"],
        "query_embedding_fingerprint": request["query_embedding_fingerprint"],
        "scoring_stage": request["scoring_stage"],
        "query_route": request["query_route"],
        "candidate_set_id": request["candidate_set_id"],
        "candidate_set_fingerprint": request["candidate_set_fingerprint"],
        "candidate_accepted_taxon_key": candidate["candidate_accepted_taxon_key"],
        "candidate_scientific_name": candidate["candidate_scientific_name"],
        "reference_media_id": reference["reference_media_id"],
        "reference_observation_id": reference["reference_observation_id"],
        "reference_embedding_fingerprint": reference["embedding_fingerprint"],
        "reference_route": reference["route"],
        "reference_visual_input_kind": reference["visual_input_kind"],
        "pool_scope": selection["pool_scope"],
        "pool_role": selection["pool_role"],
        "geographic_scope": selection["geographic_scope"],
        "geographic_distance_km": selection["geographic_distance_km"],
        "geographic_distance_status": selection["geographic_distance_status"],
        "geographic_distance_reason": selection["geographic_distance_reason"],
        "fallback_level": selection["fallback_level"],
        "selection_rank": selection["selection_rank"],
        "independent_observation_group": selection[
            "independent_observation_group"
        ],
        "observer_id_hash": reference["observer_id_hash"],
        "reference_country_code": reference["country_code"],
        "inclusion_reason": selection["inclusion_reason"],
        "selection_policy_fingerprint": policy.fingerprint,
        "source": reference["source"],
        "source_dataset_key": reference["source_dataset_key"],
        "registry_version": context["registry_version"],
        "reference_bank_version": context["reference_bank_version"],
        "reference_geography_index_fingerprint": index_fingerprint,
        "model_id": context["model_id"],
        "model_revision": context["model_revision"],
        "model_weights_sha256": context["model_weights_sha256"],
        "model_fingerprint": context["model_fingerprint"],
        "preprocessing_fingerprint": context["preprocessing_fingerprint"],
        "expansion_round": 0,
    }


def _plan_context(
    request: Mapping[str, object],
    *,
    policy: DynamicReferencePoolPolicy,
    local_available: bool,
    local_unavailable_reason: str | None,
) -> dict[str, object]:
    return {
        **dict(request),
        "local_pool_status": "available" if local_available else "unavailable",
        "local_pool_unavailable_reason": local_unavailable_reason,
        "selection_policy_version": policy.policy_version,
        "selection_policy_fingerprint": policy.fingerprint,
        "configured_global_per_candidate": policy.maximum_global_per_candidate,
        "configured_local_per_candidate": policy.maximum_local_per_candidate,
        "configured_safety_per_candidate": policy.maximum_safety_per_candidate,
        "maximum_expansion_rounds": policy.maximum_expansion_rounds,
    }


def _normalize_requests(
    requests: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(requests, str | bytes) or not isinstance(requests, Sequence):
        raise TypeError("dynamic pool planning requests must be a sequence")
    output: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for request in requests:
        if not isinstance(request, Mapping) or set(request) != DYNAMIC_POOL_PLANNING_REQUEST_FIELDS:
            raise ValueError("dynamic pool planning request fields do not match")
        normalized = dict(request)
        key = tuple(normalized[field] for field in _REQUEST_SORT)
        if key in seen:
            raise ValueError("dynamic pool planning requests contain a duplicate")
        seen.add(key)
        output.append(normalized)
    if not output:
        raise ValueError("at least one dynamic pool planning request is required")
    return sorted(output, key=lambda row: tuple(str(row[field]) for field in _REQUEST_SORT))


def _normalize_burst_groups(
    value: Mapping[str, str] | None,
    *,
    reference_index: pl.DataFrame,
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("burst_group_by_observation must be a mapping")
    known = set(str(item) for item in reference_index["reference_observation_id"])
    output: dict[str, str] = {}
    for raw_observation, raw_group in value.items():
        observation = str(raw_observation or "").strip()
        group = str(raw_group or "").strip()
        if not observation or observation not in known:
            raise ValueError("burst-group mapping references an unknown observation")
        if not group:
            raise ValueError("burst-group identity must be non-empty")
        output[observation] = group
    return output


def _validate_shared_inputs(
    requests: Sequence[Mapping[str, object]],
    *,
    candidate_sets: pl.DataFrame,
    reference_index: pl.DataFrame,
    global_anchors: pl.DataFrame,
    index_fingerprint: str,
) -> None:
    candidate_ids = set(candidate_sets["candidate_set_id"].to_list())
    index_pairs = set(
        reference_index.select(
            "reference_media_id", "embedding_fingerprint"
        ).iter_rows()
    )
    for request in requests:
        if request["candidate_set_id"] not in candidate_ids:
            raise ValueError("planning request references an unknown candidate set")
        group = candidate_sets.filter(
            pl.col("candidate_set_id") == request["candidate_set_id"]
        )
        if set(group["candidate_set_fingerprint"].to_list()) != {
            request["candidate_set_fingerprint"]
        }:
            raise ValueError("planning request candidate-set fingerprint mismatch")
        if set(group["registry_version"].to_list()) != {request["registry_version"]}:
            raise ValueError("planning request registry version mismatch")
        if request["reference_geography_index_fingerprint"] != index_fingerprint:
            raise ValueError("planning request reference-index fingerprint mismatch")
        if request["query_geo_cluster_id"] != group["query_geo_cluster_id"][0]:
            raise ValueError("planning request query geography conflicts with candidates")
        if request["query_coordinate_quality"] != group["query_coordinate_quality"][0]:
            raise ValueError("planning request coordinate quality conflicts with candidates")
    for anchor in global_anchors.iter_rows(named=True):
        key = (anchor["reference_media_id"], anchor["embedding_fingerprint"])
        if key not in index_pairs:
            raise ValueError("global anchor is absent from the reference index")
        if anchor["reference_geography_index_fingerprint"] != index_fingerprint:
            raise ValueError("global anchor reference-index fingerprint mismatch")


def _validate_candidate_reference_names(
    candidate: Mapping[str, object],
    frame: pl.DataFrame,
) -> None:
    if frame.is_empty():
        return
    if set(frame["scientific_name"].to_list()) != {
        candidate["candidate_scientific_name"]
    }:
        raise ValueError("candidate and reference scientific names conflict")


def _query_supports_local(request: Mapping[str, object]) -> bool:
    return (
        request["query_geo_cluster_id"] is not None
        and request["query_coordinate_quality"] not in _NO_LOCAL_QUERY_QUALITIES
    )


def _local_unavailable_reason(
    request: Mapping[str, object], *, local_available: bool
) -> str | None:
    if local_available:
        return None
    if not _query_supports_local(request):
        return "no_geo_global_fallback"
    return "no_exact_local_reference_support"


def _scope_geography_rank(row: Mapping[str, object], *, scope: str) -> int:
    if scope == "global":
        return 0
    quality_order = {
        "local": 0,
        "regional": 1,
        "coarse": 2,
        "unknown_precision": 3,
        "country_only": 4,
        "missing": 5,
        "withheld": 6,
        "invalid": 7,
    }
    return quality_order.get(str(row["coordinate_quality"]), 8)


def _record_selected(
    row: Mapping[str, object],
    *,
    observations: set[str],
    duplicates: set[str],
) -> None:
    observations.add(str(row["reference_observation_id"]))
    duplicates.add(str(row["duplicate_group_id"]))


__all__ = [
    "DYNAMIC_POOL_PLANNING_REQUEST_FIELDS",
    "plan_dynamic_reference_pools",
]
