from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path

import polars as pl
import pytest

from biominer.candidates.regional_union import (
    REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
    regional_candidate_species_schema,
)
from biominer.references.planner import (
    ReferencePlannerConfig,
    ReferencePlanResult,
    ReferenceStratumQuota,
    plan_geographically_balanced_support_bank,
    validate_reference_plan_result,
    write_reference_plan_result,
)
from biominer.references.schemas import (
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_observation_id,
    reference_acquisition_plan_schema,
    reference_acquisition_selection_schema,
    reference_media_candidates_frame,
    reference_observations_frame,
)


NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
TARGET_KEY = "gbif:1938069"
COMPETITOR_KEY = "gbif:888"


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_species(
    clusters: tuple[str, ...],
    *,
    include_competitor: bool = True,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    species = [(TARGET_KEY, "Papilio demoleus", True)]
    if include_competitor:
        species.append((COMPETITOR_KEY, "Papilio polytes", False))
    for cluster_id in clusters:
        candidate_set_id = f"regional:{cluster_id}"
        fingerprint = _sha(f"candidate-set:{cluster_id}")
        for priority, (taxon_key, name, target) in enumerate(species):
            rows.append(
                {
                    "schema_version": REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
                    "candidate_set_id": candidate_set_id,
                    "target_accepted_taxon_key": TARGET_KEY,
                    "geo_cluster_id": cluster_id,
                    "candidate_accepted_taxon_key": taxon_key,
                    "scientific_name": name,
                    "family": "Papilionidae",
                    "genus": "Papilio",
                    "candidate_reason": ["target" if target else "close_congener"],
                    "geographic_evidence_score": 1.0,
                    "occurrence_support": 10,
                    "same_genus": True,
                    "same_family": True,
                    "known_mimic": False,
                    "historical_false_positive": False,
                    "visually_nearest": False,
                    "target_candidate": target,
                    "candidate_priority": priority,
                    "source_versions": ["fixture-v1"],
                    "candidate_set_fingerprint": fingerprint,
                }
            )
    return pl.DataFrame(
        rows,
        schema=regional_candidate_species_schema(),
        strict=True,
    ).sort(["candidate_set_id", "candidate_priority", "candidate_accepted_taxon_key"])


def _reference_rows(
    specifications: list[dict[str, object]],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    observations: list[dict[str, object]] = []
    media: list[dict[str, object]] = []
    review: list[dict[str, object]] = []
    for index, specification in enumerate(specifications):
        taxon_key = str(specification["taxon_key"])
        cluster_id = str(specification["cluster_id"])
        observation_number = int(specification["observation_number"])
        source = str(specification.get("source", "iNaturalist"))
        source_observation_id = f"{taxon_key}:{cluster_id}:{observation_number}"
        observation_id = make_reference_observation_id(source, source_observation_id)
        observed_at = NOW - timedelta(days=observation_number % 7)
        fallback_level = int(specification.get("fallback_level", 0))
        observations.append(
            {
                "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
                "reference_observation_id": observation_id,
                "source": source,
                "source_observation_id": source_observation_id,
                "source_taxon_id": str(observation_number),
                "supplied_scientific_name": (
                    "Papilio demoleus" if taxon_key == TARGET_KEY else "Papilio polytes"
                ),
                "accepted_taxon_key": taxon_key,
                "reconciled_scientific_name": (
                    "Papilio demoleus" if taxon_key == TARGET_KEY else "Papilio polytes"
                ),
                "registry_version": "butterflies-v2-20260712",
                "taxon_reconciliation_status": "accepted_key_exact",
                "identification_quality": "research",
                "community_taxon_status": "species",
                "identification_disagreement": False,
                "captive_or_cultivated": False,
                "observer_id": specification.get(
                    "observer_id", f"observer-{index % 3}"
                ),
                "locality": specification.get("locality", f"locality-{index % 4}"),
                "life_stage": str(specification.get("life_stage", "adult")),
                "sex": None,
                "observed_at": observed_at,
                "latitude": -33.8,
                "longitude": 151.2,
                "coordinate_uncertainty": 20.0,
                "coordinates_obscured": False,
                "country": "Australia",
                "country_code": "AU",
                "geo_cluster_id": cluster_id,
                "distance_to_cluster_medoid_km": float(
                    specification.get("distance_km", observation_number)
                ),
                "source_dataset_key": None,
                "source_dataset_doi": None,
                "source_record_url": f"https://example.test/observations/{observation_number}",
                "source_record_hash": _sha(f"record:{source_observation_id}"),
                "retrieved_at": NOW,
                "source_snapshot_version": f"{source}-fixture-v1",
                "source_query_fingerprint": _sha(
                    f"query:{taxon_key}:{cluster_id}:{fallback_level}"
                ),
                "fallback_level": fallback_level,
                "geospatial_issue": False,
                "preserved_specimen": False,
                "fossil": False,
                "occurrence_absent": False,
                "uncertain_taxon_match": False,
                "basis_of_record_suitable": True,
            }
        )
        media_count = int(specification.get("media_count", 1))
        for media_index in range(media_count):
            provider_media_id = f"photo-{observation_number}-{media_index}"
            reference_media_id = make_reference_media_id(
                source,
                provider_media_id,
                observation_id,
            )
            media.append(
                {
                    "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
                    "reference_media_id": reference_media_id,
                    "reference_observation_id": observation_id,
                    "provider_media_id": provider_media_id,
                    "source": source,
                    "media_identifier": f"https://example.test/media/{provider_media_id}.jpg",
                    "media_type": "StillImage",
                    "width": 2048,
                    "height": 1365,
                    "creator": str(specification.get("observer_id", "observer")),
                    "rights_holder": str(specification.get("observer_id", "observer")),
                    "licence": str(specification.get("licence", "cc-by")),
                    "licence_uri": "https://creativecommons.org/licenses/by/4.0/",
                    "attribution": "Fixture / CC BY",
                    "occurrence_licence": "cc-by-nc",
                    "original_provider": source,
                    "media_position": media_index,
                    "source_checksum": None,
                    "source_checksum_algorithm": None,
                    "download_status": "pending",
                    "verification_status": "unreviewed",
                    "exclusion_reason": None,
                    "licence_policy_status": "allowed",
                    "retrieved_at": NOW,
                    "source_snapshot_version": f"{source}-fixture-v1",
                }
            )
            review.append(
                {
                    "reference_media_id": reference_media_id,
                    "visual_domain": str(specification.get("visual_domain", "field")),
                    "background_group_id": specification.get(
                        "background_group_id", f"background-{index % 3}"
                    ),
                    "reviewer": "fixture-reviewer",
                }
            )
    review_frame = (
        pl.DataFrame(review).sort("reference_media_id")
        if review
        else pl.DataFrame(schema={"reference_media_id": pl.String})
    )
    return (
        reference_observations_frame(observations),
        reference_media_candidates_frame(media),
        review_frame,
    )


def _config(*, quota: int) -> ReferencePlannerConfig:
    return ReferencePlannerConfig(
        strata=(
            ReferenceStratumQuota(
                life_stage="adult",
                visual_domain="field",
                requested_per_species=quota,
            ),
        ),
        minimum_per_sufficient_cluster=2,
        sufficiently_populated_candidate_count=2,
        distance_balance_band_km=25.0,
        selection_seed=77,
    )


def test_planner_allocates_minimum_plus_sqrt_and_balances_class_quota() -> None:
    specifications: list[dict[str, object]] = []
    for observation_number in range(1, 17):
        specifications.append(
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": observation_number,
            }
        )
    for observation_number in range(101, 105):
        specifications.append(
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-b",
                "observation_number": observation_number,
            }
        )
    for cluster_id, start in (("cluster-a", 201), ("cluster-b", 301)):
        for observation_number in range(start, start + 3):
            specifications.append(
                {
                    "taxon_key": COMPETITOR_KEY,
                    "cluster_id": cluster_id,
                    "observation_number": observation_number,
                }
            )
    observations, media, review = _reference_rows(specifications)
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species(("cluster-a", "cluster-b", "cluster-c")),
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=_config(quota=10),
        created_at=NOW,
    )

    target_plan = result.plan.filter(
        pl.col("candidate_accepted_taxon_key") == TARGET_KEY
    )
    target_requests = {
        str(row["geo_cluster_id"]): int(row["requested_count"])
        for row in target_plan.iter_rows(named=True)
    }
    assert target_requests == {"cluster-a": 6, "cluster-b": 4, "cluster-c": 0}
    requested_by_species = result.plan.group_by("candidate_accepted_taxon_key").agg(
        pl.col("requested_count").sum().alias("requested")
    )
    assert set(requested_by_species["requested"].to_list()) == {10}
    assert result.report["summary"]["balanced_requested_quota"] is True
    assert result.report["summary"]["target_selected"] == 10
    assert (
        result.plan.filter(pl.col("candidate_accepted_taxon_key") == COMPETITOR_KEY)[
            "shortfall_count"
        ].sum()
        == 4
    )


def test_planner_redistributes_hamilton_quota_when_a_cluster_saturates() -> None:
    specifications = [
        {
            "taxon_key": TARGET_KEY,
            "cluster_id": "cluster-a",
            "observation_number": value,
        }
        for value in range(1, 11)
    ]
    specifications.extend(
        {
            "taxon_key": TARGET_KEY,
            "cluster_id": "cluster-b",
            "observation_number": value,
        }
        for value in range(101, 110)
    )
    observations, media, review = _reference_rows(specifications)
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species(
            ("cluster-a", "cluster-b"),
            include_competitor=False,
        ),
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=ReferencePlannerConfig(
            strata=(
                ReferenceStratumQuota(
                    life_stage="adult",
                    visual_domain="field",
                    requested_per_species=19,
                ),
            ),
            minimum_per_sufficient_cluster=2,
            sufficiently_populated_candidate_count=10,
        ),
        created_at=NOW,
    )

    requests = {
        str(row["geo_cluster_id"]): int(row["requested_count"])
        for row in result.plan.iter_rows(named=True)
    }
    assert requests == {"cluster-a": 10, "cluster-b": 9}
    assert result.report["summary"]["selected"] == 19
    assert result.report["summary"]["shortfall"] == 0


def test_planner_prefers_independent_observations_and_diversifies_metadata() -> None:
    specifications = [
        {
            "taxon_key": TARGET_KEY,
            "cluster_id": "cluster-a",
            "observation_number": 1,
            "observer_id": "observer-a",
            "locality": "locality-a",
            "background_group_id": "background-a",
            "media_count": 2,
            "distance_km": 5.0,
        },
        {
            "taxon_key": TARGET_KEY,
            "cluster_id": "cluster-a",
            "observation_number": 2,
            "observer_id": "observer-b",
            "locality": "locality-b",
            "background_group_id": "background-b",
            "distance_km": 6.0,
        },
        {
            "taxon_key": TARGET_KEY,
            "cluster_id": "cluster-a",
            "observation_number": 3,
            "observer_id": "observer-a",
            "locality": "locality-c",
            "background_group_id": "background-c",
            "distance_km": 7.0,
        },
    ]
    observations, media, review = _reference_rows(specifications)
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species(
            ("cluster-a",),
            include_competitor=False,
        ),
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=_config(quota=3),
        created_at=NOW,
    )

    assert result.selections["reference_observation_id"].n_unique() == 3
    assert set(result.selections["selection_round"].to_list()) == {
        "independent_observation"
    }
    assert result.report["diversity"]["unique_observers"] == 2
    assert result.report["diversity"]["unique_localities"] == 3
    assert result.report["diversity"]["unique_background_groups"] == 3
    assert result.report["diversity_by_species"][0]["unique_observers"] == 2


def test_planner_penalizes_missing_diversity_metadata() -> None:
    observations, media, review = _reference_rows(
        [
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": 7,
                "observer_id": None,
                "locality": None,
                "background_group_id": None,
                "distance_km": 10.0,
            },
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": 8,
                "observer_id": "known-observer",
                "locality": "known-locality",
                "background_group_id": "known-background",
                "distance_km": 10.0,
            },
        ]
    )
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species(
            ("cluster-a",),
            include_competitor=False,
        ),
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=_config(quota=1),
        created_at=NOW,
    )

    assert result.selections["observer_id"].item() == "known-observer"


def test_planner_never_counts_extra_media_as_independent_quota_slots() -> None:
    observations, media, review = _reference_rows(
        [
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": 10,
                "media_count": 3,
            },
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": 11,
            },
        ]
    )
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species(
            ("cluster-a",),
            include_competitor=False,
        ),
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=_config(quota=3),
        created_at=NOW,
    )

    assert result.selections.height == 2
    assert result.selections["reference_observation_id"].n_unique() == 2
    assert set(result.selections["selection_round"].to_list()) == {
        "independent_observation"
    }
    assert result.report["summary"]["shortfall"] == 1
    assert result.report["diversity"]["same_observation_extra_images"] == 0


def test_planner_records_local_buffer_country_global_fallbacks_and_distances() -> None:
    specifications = [
        {
            "taxon_key": TARGET_KEY,
            "cluster_id": "cluster-a",
            "observation_number": 20 + fallback,
            "fallback_level": fallback,
            "distance_km": 10.0 * (fallback + 1),
            "source": "GBIF" if fallback % 2 else "iNaturalist",
            "licence": "cc-by" if fallback < 2 else "cc-by-nc",
        }
        for fallback in range(4)
    ]
    observations, media, review = _reference_rows(specifications)
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species(
            ("cluster-a",),
            include_competitor=False,
        ),
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=_config(quota=4),
        created_at=NOW,
    )

    assert result.report["fallback_distribution"] == {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
    }
    assert result.report["distance_distribution_km"] == {
        "count": 4,
        "p50": 20.0,
        "p95": 40.0,
        "max": 40.0,
    }
    assert result.report["source_distribution"] == {"GBIF": 2, "iNaturalist": 2}
    assert result.report["licence_distribution"] == {"cc-by": 2, "cc-by-nc": 2}
    assert result.plan["fallback_level"].item() == 3


def test_planner_reports_quota_shortfalls_without_crossing_species_boundaries() -> None:
    observations, media, review = _reference_rows(
        [
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": 40,
            },
            {
                "taxon_key": COMPETITOR_KEY,
                "cluster_id": "cluster-a",
                "observation_number": 41,
            },
        ]
    )
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species(("cluster-a",)),
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=_config(quota=5),
        created_at=NOW,
    )

    assert result.report["summary"]["requested"] == 10
    assert result.report["summary"]["selected"] == 2
    assert result.report["summary"]["shortfall"] == 8
    assert set(result.selections["candidate_accepted_taxon_key"].to_list()) == {
        TARGET_KEY,
        COMPETITOR_KEY,
    }


def test_planner_tops_up_only_the_existing_support_deficit() -> None:
    observations, media, review = _reference_rows(
        [
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": value,
            }
            for value in range(70, 73)
        ]
    )
    candidate_species = _candidate_species(
        ("cluster-a",),
        include_competitor=False,
    )
    first = plan_geographically_balanced_support_bank(
        candidate_species=candidate_species,
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=_config(quota=1),
        created_at=NOW,
    )
    topped_up = plan_geographically_balanced_support_bank(
        candidate_species=candidate_species,
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        existing_selections=first.selections,
        config=_config(quota=3),
        created_at=NOW,
    )

    assert topped_up.report["summary"]["configured_quota"] == 3
    assert topped_up.report["summary"]["existing_support"] == 1
    assert topped_up.report["summary"]["requested"] == 2
    assert topped_up.report["summary"]["selected"] == 2
    assert topped_up.report["summary"]["support_after_selection"] == 3
    assert topped_up.plan["existing_support_count"].item() == 1
    assert not (
        set(first.selections["reference_observation_id"].to_list())
        & set(topped_up.selections["reference_observation_id"].to_list())
    )


def test_planner_is_deterministic_for_equivalent_sorted_contract_inputs() -> None:
    specifications = [
        {
            "taxon_key": TARGET_KEY,
            "cluster_id": "cluster-a",
            "observation_number": value,
            "distance_km": 10.0,
        }
        for value in range(50, 56)
    ]
    observations, media, review = _reference_rows(specifications)
    arguments = {
        "candidate_species": _candidate_species(
            ("cluster-a",),
            include_competitor=False,
        ),
        "observations": observations,
        "media_candidates": media,
        "review_metadata": review,
        "config": _config(quota=3),
        "created_at": NOW,
    }
    first = plan_geographically_balanced_support_bank(**arguments)
    second = plan_geographically_balanced_support_bank(**arguments)

    assert first.plan.equals(second.plan)
    assert first.selections.equals(second.selections)
    assert first.report == second.report


def test_planner_identity_changes_with_output_relevant_media_metadata() -> None:
    observations, media, review = _reference_rows(
        [
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": 58,
            }
        ]
    )
    arguments = {
        "candidate_species": _candidate_species(
            ("cluster-a",),
            include_competitor=False,
        ),
        "observations": observations,
        "review_metadata": review,
        "config": _config(quota=1),
        "created_at": NOW,
    }
    first = plan_geographically_balanced_support_bank(
        **arguments,
        media_candidates=media,
    )
    changed_media = media.with_columns(pl.lit("cc0").alias("licence"))
    second = plan_geographically_balanced_support_bank(
        **arguments,
        media_candidates=changed_media,
    )

    assert (
        first.plan["acquisition_plan_id"].item()
        != second.plan["acquisition_plan_id"].item()
    )
    assert first.report["licence_distribution"] != second.report["licence_distribution"]


def test_planner_writes_selection_ledger_and_compact_reports(tmp_path: Path) -> None:
    observations, media, review = _reference_rows(
        [
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": 60,
            }
        ]
    )
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species(
            ("cluster-a",),
            include_competitor=False,
        ),
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=_config(quota=1),
        created_at=NOW,
    )
    paths = write_reference_plan_result(result, tmp_path)

    assert pl.read_parquet(paths["plan"]).schema == reference_acquisition_plan_schema()
    assert (
        pl.read_parquet(paths["selections"]).schema
        == reference_acquisition_selection_schema()
    )
    assert paths["metrics"].read_text(encoding="utf-8").startswith("{")
    assert "# Reference Acquisition Plan" in paths["summary"].read_text(
        encoding="utf-8"
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_planner_rejects_candidate_set_without_target() -> None:
    candidates = _candidate_species(("cluster-a",))
    candidates = candidates.with_columns(pl.lit(False).alias("target_candidate"))
    observations, media, review = _reference_rows([])

    with pytest.raises(ValueError, match="must contain one target"):
        plan_geographically_balanced_support_bank(
            candidate_species=candidates,
            observations=observations,
            media_candidates=media,
            review_metadata=review,
            config=_config(quota=1),
            created_at=NOW,
        )


def test_planner_rejects_competitor_misflagged_as_target() -> None:
    candidates = _candidate_species(("cluster-a",))
    candidates = candidates.with_columns(
        (pl.col("candidate_accepted_taxon_key") == COMPETITOR_KEY).alias(
            "target_candidate"
        )
    )
    observations, media, review = _reference_rows([])

    with pytest.raises(ValueError, match="target flag does not identify"):
        plan_geographically_balanced_support_bank(
            candidate_species=candidates,
            observations=observations,
            media_candidates=media,
            review_metadata=review,
            config=_config(quota=1),
            created_at=NOW,
        )


def test_planner_rejects_multiple_candidate_sets_for_one_cluster() -> None:
    candidates = _candidate_species(("cluster-a",))
    duplicate = candidates.with_columns(
        pl.lit("regional:duplicate-cluster-a").alias("candidate_set_id"),
        pl.lit(_sha("duplicate-cluster-a")).alias("candidate_set_fingerprint"),
    )
    candidates = pl.concat([candidates, duplicate]).sort(
        ["candidate_set_id", "candidate_priority", "candidate_accepted_taxon_key"]
    )
    observations, media, review = _reference_rows([])

    with pytest.raises(ValueError, match="multiple candidate sets"):
        plan_geographically_balanced_support_bank(
            candidate_species=candidates,
            observations=observations,
            media_candidates=media,
            review_metadata=review,
            config=_config(quota=1),
            created_at=NOW,
        )


def test_reference_plan_result_rejects_plan_selection_count_mismatch() -> None:
    observations, media, review = _reference_rows(
        [
            {
                "taxon_key": TARGET_KEY,
                "cluster_id": "cluster-a",
                "observation_number": 90,
            }
        ]
    )
    result = plan_geographically_balanced_support_bank(
        candidate_species=_candidate_species(
            ("cluster-a",),
            include_competitor=False,
        ),
        observations=observations,
        media_candidates=media,
        review_metadata=review,
        config=_config(quota=1),
        created_at=NOW,
    )
    inconsistent_plan = result.plan.with_columns(
        pl.lit(0, dtype=pl.UInt32).alias("selected_candidate_count"),
        pl.lit(1, dtype=pl.UInt32).alias("shortfall_count"),
    )

    with pytest.raises(ValueError, match="conflicts with selection ledger"):
        validate_reference_plan_result(
            ReferencePlanResult(
                plan=inconsistent_plan,
                selections=result.selections,
                report=result.report,
                markdown=result.markdown,
            )
        )
