"""Contract tests for dynamic reference-pool plans, members and summaries."""

from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.dynamic_pool_contracts import (
    DYNAMIC_POOL_MEMBERS_FILE,
    DYNAMIC_POOL_PLANS_FILE,
    DYNAMIC_POOL_SUMMARY_FILE,
    build_dynamic_reference_pool_members,
    build_dynamic_reference_pool_plans,
    build_dynamic_reference_pool_summaries,
    dynamic_reference_pool_member_schema,
    dynamic_reference_pool_plan_id,
    dynamic_reference_pool_plan_schema,
    dynamic_reference_pool_summary_schema,
    validate_dynamic_reference_pool_artifacts,
    validate_dynamic_reference_pool_members,
    validate_dynamic_reference_pool_plans,
    validate_dynamic_reference_pool_summaries,
    write_dynamic_reference_pool_artifacts,
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _plan_context(*, local: bool = True) -> dict[str, object]:
    return {
        "run_id": "run-20260718",
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
        "candidate_set_id": "candidate-set-1",
        "candidate_set_fingerprint": _sha("4"),
        "query_geo_cluster_id": "geo-au-qld" if local else None,
        "query_coordinate_quality": "local" if local else "no_geo",
        "local_pool_status": "available" if local else "unavailable",
        "local_pool_unavailable_reason": None if local else "no_geo_global_fallback",
        "selection_policy_version": "dynamic-pool-selection-v1",
        "selection_policy_fingerprint": _sha("5"),
        "model_id": "bioclip-2.5",
        "model_revision": "revision-1",
        "model_weights_sha256": _sha("6"),
        "model_fingerprint": _sha("7"),
        "preprocessing_fingerprint": _sha("8"),
        "configured_global_per_candidate": 2,
        "configured_local_per_candidate": 2,
        "configured_safety_per_candidate": 1,
        "maximum_expansion_rounds": 2,
    }


def _member(
    plan_context: dict[str, object],
    *,
    suffix: str,
    pool_scope: str = "global",
    pool_role: str = "global_core",
    selection_rank: int = 1,
    distance_km: float | None = None,
    distance_status: str = "not_applicable",
    distance_reason: str | None = "global_pool_has_no_query_distance",
    geographic_scope: str = "global",
    expansion_round: int = 0,
    **changes: object,
) -> dict[str, object]:
    plan_id = dynamic_reference_pool_plan_id(plan_context)
    row: dict[str, object] = {
        "plan_id": plan_id,
        "run_id": plan_context["run_id"],
        "flickr_query_id": plan_context["flickr_query_id"],
        "flickr_photo_id": plan_context["flickr_photo_id"],
        "organism_unit_id": plan_context["organism_unit_id"],
        "visual_input_id": plan_context["visual_input_id"],
        "query_embedding_fingerprint": plan_context["query_embedding_fingerprint"],
        "scoring_stage": plan_context["scoring_stage"],
        "query_route": plan_context["query_route"],
        "candidate_set_id": plan_context["candidate_set_id"],
        "candidate_set_fingerprint": plan_context["candidate_set_fingerprint"],
        "candidate_accepted_taxon_key": "gbif:5131359",
        "candidate_scientific_name": "Papilio demoleus",
        "reference_media_id": f"reference-media:{suffix * 64}",
        "reference_observation_id": f"reference-observation:{suffix * 64}",
        "reference_embedding_fingerprint": _sha(suffix),
        "reference_route": "adult_field",
        "reference_visual_input_kind": "raw_full_image",
        "pool_scope": pool_scope,
        "pool_role": pool_role,
        "geographic_scope": geographic_scope,
        "geographic_distance_km": distance_km,
        "geographic_distance_status": distance_status,
        "geographic_distance_reason": distance_reason,
        "fallback_level": 0,
        "selection_rank": selection_rank,
        "independent_observation_group": f"observation-group-{suffix}",
        "observer_id_hash": _sha(suffix),
        "reference_country_code": "au",
        "inclusion_reason": pool_role,
        "selection_policy_fingerprint": plan_context["selection_policy_fingerprint"],
        "source": "gbif",
        "source_dataset_key": f"dataset-{suffix}",
        "registry_version": plan_context["registry_version"],
        "reference_bank_version": plan_context["reference_bank_version"],
        "reference_geography_index_fingerprint": plan_context[
            "reference_geography_index_fingerprint"
        ],
        "model_id": plan_context["model_id"],
        "model_revision": plan_context["model_revision"],
        "model_weights_sha256": plan_context["model_weights_sha256"],
        "model_fingerprint": plan_context["model_fingerprint"],
        "preprocessing_fingerprint": plan_context["preprocessing_fingerprint"],
        "expansion_round": expansion_round,
    }
    row.update(changes)
    return row


def _artifacts(
    *, local: bool = True
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    context = _plan_context(local=local)
    member_rows = [
        _member(context, suffix="1", selection_rank=1),
        _member(context, suffix="2", selection_rank=2),
    ]
    if local:
        member_rows.append(
            _member(
                context,
                suffix="3",
                pool_scope="local",
                pool_role="nearest_local",
                geographic_scope="exact_local_cell",
                distance_km=12.5,
                distance_status="available",
                distance_reason=None,
            )
        )
    members = build_dynamic_reference_pool_members(member_rows)
    plans = build_dynamic_reference_pool_plans(
        [{"plan_id": dynamic_reference_pool_plan_id(context), **context}], members
    )
    summaries = build_dynamic_reference_pool_summaries(plans, members)
    return plans, members, summaries


def test_member_schema_contains_required_reconstructable_evidence() -> None:
    fields = set(dynamic_reference_pool_member_schema())

    assert {
        "pool_id",
        "flickr_query_id",
        "flickr_photo_id",
        "organism_unit_id",
        "scoring_stage",
        "query_route",
        "candidate_accepted_taxon_key",
        "reference_media_id",
        "reference_observation_id",
        "reference_route",
        "reference_visual_input_kind",
        "pool_role",
        "geographic_scope",
        "geographic_distance_km",
        "fallback_level",
        "selection_rank",
        "independent_observation_group",
        "inclusion_reason",
        "selection_policy_fingerprint",
        "source",
        "model_id",
        "model_revision",
        "model_weights_sha256",
        "member_fingerprint",
    } <= fields
    assert dynamic_reference_pool_plan_schema()["global_pool_ids"] == pl.List(pl.String)
    assert dynamic_reference_pool_summary_schema()["shortfall_count"] == pl.UInt32


def test_plan_member_and_summary_artifacts_cross_validate() -> None:
    plans, members, summaries = _artifacts()

    validate_dynamic_reference_pool_artifacts(plans, members, summaries)
    plan = plans.row(0, named=True)
    assert len(plan["global_pool_ids"]) == 1
    assert len(plan["local_pool_ids"]) == 1
    assert plan["safety_pool_ids"] == []
    assert summaries.height == 2
    local = summaries.filter(pl.col("pool_scope") == "local").row(0, named=True)
    assert local["configured_reference_count"] == 2
    assert local["effective_reference_count"] == 1
    assert local["shortfall_count"] == 1
    assert local["minimum_distance_km"] == 12.5


def test_member_and_pool_identities_are_input_order_independent() -> None:
    context = _plan_context()
    rows = [
        _member(context, suffix="1", selection_rank=1),
        _member(context, suffix="2", selection_rank=2),
    ]

    forward = build_dynamic_reference_pool_members(rows)
    reverse = build_dynamic_reference_pool_members(list(reversed(rows)))

    assert forward.equals(reverse)
    assert forward["pool_id"].n_unique() == 1
    assert (
        forward["member_fingerprint"].to_list()
        == reverse["member_fingerprint"].to_list()
    )


def test_no_geo_plan_is_explicitly_global_only() -> None:
    plans, members, summaries = _artifacts(local=False)
    plan = plans.row(0, named=True)

    assert plan["query_coordinate_quality"] == "no_geo"
    assert plan["local_pool_status"] == "unavailable"
    assert plan["local_pool_unavailable_reason"] == "no_geo_global_fallback"
    assert plan["local_pool_ids"] == []
    assert members["pool_scope"].unique().to_list() == ["global"]
    assert summaries["distance_available_count"].sum() == 0
    assert summaries["minimum_distance_km"].to_list() == [None]


def test_writer_creates_all_three_named_parquet_artifacts(tmp_path) -> None:
    plans, members, summaries = _artifacts()

    paths = write_dynamic_reference_pool_artifacts(
        plans, members, summaries, tmp_path / "pools"
    )

    assert paths["plans"].name == DYNAMIC_POOL_PLANS_FILE
    assert paths["members"].name == DYNAMIC_POOL_MEMBERS_FILE
    assert paths["summary"].name == DYNAMIC_POOL_SUMMARY_FILE
    loaded = {key: pl.read_parquet(path) for key, path in paths.items()}
    validate_dynamic_reference_pool_artifacts(
        loaded["plans"], loaded["members"], loaded["summary"]
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"geographic_distance_km": None, "geographic_distance_status": "available"},
            "requires value",
        ),
        (
            {"pool_scope": "local", "pool_role": "global_core"},
            "role conflicts",
        ),
        (
            {"pool_scope": "local", "geographic_scope": "global"},
            "role conflicts",
        ),
        ({"reference_route": "larval"}, "query and reference routes conflict"),
        ({"selection_rank": 0}, "must be positive"),
        ({"reference_country_code": "AUS"}, "ISO alpha-2"),
    ],
)
def test_member_builder_rejects_invalid_evidence(
    changes: dict[str, object], message: str
) -> None:
    context = _plan_context()
    with pytest.raises(ValueError, match=message):
        build_dynamic_reference_pool_members([_member(context, suffix="1", **changes)])


def test_selection_ranks_must_be_unique_and_contiguous() -> None:
    context = _plan_context()
    rows = [
        _member(context, suffix="1", selection_rank=1),
        _member(context, suffix="2", selection_rank=3),
    ]
    with pytest.raises(ValueError, match="contiguous from one"):
        build_dynamic_reference_pool_members(rows)


def test_one_observation_group_cannot_fill_multiple_pool_slots() -> None:
    context = _plan_context()
    rows = [
        _member(context, suffix="1", selection_rank=1),
        _member(
            context,
            suffix="2",
            selection_rank=2,
            independent_observation_group="observation-group-1",
        ),
    ]

    with pytest.raises(ValueError, match="multiple slots from one observation"):
        build_dynamic_reference_pool_members(rows)


def test_candidate_name_cannot_conflict_inside_one_pool() -> None:
    context = _plan_context()
    rows = [
        _member(context, suffix="1", selection_rank=1),
        _member(
            context,
            suffix="2",
            selection_rank=2,
            candidate_scientific_name="Papilio polytes",
        ),
    ]

    with pytest.raises(ValueError, match="candidate name conflicts"):
        build_dynamic_reference_pool_members(rows)


def test_plan_identity_rejects_context_drift() -> None:
    context = _plan_context()
    members = build_dynamic_reference_pool_members([_member(context, suffix="1")])
    changed = {**context, "model_revision": "revision-2"}

    with pytest.raises(ValueError, match="plan_id does not match"):
        build_dynamic_reference_pool_plans(
            [{"plan_id": dynamic_reference_pool_plan_id(context), **changed}], members
        )


def test_local_availability_requires_local_members() -> None:
    context = _plan_context(local=True)
    members = build_dynamic_reference_pool_members([_member(context, suffix="1")])

    with pytest.raises(ValueError, match="available local pool requires"):
        build_dynamic_reference_pool_plans(
            [{"plan_id": dynamic_reference_pool_plan_id(context), **context}], members
        )


def test_member_change_changes_pool_identity() -> None:
    context = _plan_context()
    first = build_dynamic_reference_pool_members([_member(context, suffix="1")])
    second = build_dynamic_reference_pool_members(
        [_member(context, suffix="2", source_dataset_key="dataset-2")]
    )

    assert first["pool_id"].item() != second["pool_id"].item()


def test_cross_validator_rejects_summary_tampering() -> None:
    plans, members, summaries = _artifacts()
    tampered = summaries.with_columns(
        pl.when(pl.col("pool_scope") == "local")
        .then(pl.lit(0, dtype=pl.UInt32))
        .otherwise(pl.col("shortfall_count"))
        .alias("shortfall_count")
    )

    with pytest.raises(ValueError, match="shortfall is inconsistent"):
        validate_dynamic_reference_pool_summaries(tampered)

    with pytest.raises(ValueError):
        validate_dynamic_reference_pool_artifacts(plans, members, tampered)


def test_cross_validator_rejects_expansion_beyond_plan_maximum() -> None:
    context = _plan_context(local=False)
    expanded = _member(
        context,
        suffix="1",
        pool_scope="safety_expansion",
        pool_role="visual_neighbour",
        geographic_scope="not_applicable",
        distance_reason="geography_not_applicable",
        expansion_round=3,
    )
    members = build_dynamic_reference_pool_members(
        [_member(context, suffix="2"), expanded]
    )
    plans = build_dynamic_reference_pool_plans(
        [{"plan_id": dynamic_reference_pool_plan_id(context), **context}], members
    )

    with pytest.raises(ValueError, match="expansion round exceeds"):
        build_dynamic_reference_pool_summaries(plans, members)


def test_validators_reject_fingerprint_tampering() -> None:
    plans, members, summaries = _artifacts()

    with pytest.raises(ValueError, match="plan_fingerprint mismatch"):
        validate_dynamic_reference_pool_plans(
            plans.with_columns(pl.lit(_sha("f")).alias("plan_fingerprint"))
        )
    with pytest.raises(ValueError, match="member_fingerprint mismatch"):
        validate_dynamic_reference_pool_members(
            members.with_columns(
                pl.when(
                    pl.col("reference_media_id") == members["reference_media_id"][0]
                )
                .then(pl.lit(_sha("f")))
                .otherwise(pl.col("member_fingerprint"))
                .alias("member_fingerprint")
            )
        )
    with pytest.raises(ValueError, match="summary_fingerprint mismatch"):
        validate_dynamic_reference_pool_summaries(
            summaries.with_columns(pl.lit(_sha("f")).alias("summary_fingerprint"))
        )
