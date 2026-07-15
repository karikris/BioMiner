from __future__ import annotations

from datetime import UTC, datetime
import hashlib

import polars as pl

from biominer.references.prototype_planner import (
    PLANNER_SCORE_SEMANTICS,
    PrototypeReferenceQuota,
    plan_trust_first_layered_references,
    prototype_reference_planner_evidence_schema,
)
from biominer.references.schemas import (
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_observation_id,
    reference_media_candidates_frame,
    reference_observations_frame,
)


NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
TARGET = "gbif:1938069"


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _fixtures(
    rows: list[dict[str, object]],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    observations: list[dict[str, object]] = []
    media: list[dict[str, object]] = []
    qualification: list[dict[str, object]] = []
    for index, values in enumerate(rows):
        source = str(values.get("source", "GBIF"))
        observation_id = make_reference_observation_id(source, f"obs-{index}")
        media_id = make_reference_media_id(source, f"media-{index}", observation_id)
        route = str(values.get("route", "adult_field"))
        life_stage = str(
            values.get("life_stage", "larva" if route == "larval" else "adult")
        )
        preserved = route == "pinned_specimen"
        cluster = str(values.get("cluster", "cluster-local"))
        observations.append(
            {
                "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
                "reference_observation_id": observation_id,
                "source": source,
                "source_observation_id": f"obs-{index}",
                "source_taxon_id": "1938069",
                "supplied_scientific_name": "Papilio demoleus",
                "accepted_taxon_key": TARGET,
                "reconciled_scientific_name": "Papilio demoleus",
                "registry_version": "fixture-v1",
                "taxon_reconciliation_status": "accepted_key_exact",
                "identification_quality": values.get(
                    "identification_quality", "research"
                ),
                "community_taxon_status": "species",
                "identification_disagreement": False,
                "captive_or_cultivated": False,
                "observer_id": f"observer-{index}",
                "locality": f"locality-{index}",
                "life_stage": life_stage,
                "sex": None,
                "observed_at": NOW,
                "latitude": None if cluster == "no_geo" else -33.8,
                "longitude": None if cluster == "no_geo" else 151.2,
                "coordinate_uncertainty": None,
                "coordinates_obscured": False,
                "country": "Australia",
                "country_code": "AU",
                "geo_cluster_id": cluster,
                "distance_to_cluster_medoid_km": None if cluster == "no_geo" else 5.0,
                "source_dataset_key": None,
                "source_dataset_doi": None,
                "source_record_url": f"https://example.test/obs/{index}",
                "source_record_hash": _sha(f"obs-{index}"),
                "retrieved_at": NOW,
                "source_snapshot_version": "fixture-v1",
                "source_query_fingerprint": _sha("query"),
                "fallback_level": int(values.get("fallback_level", 0)),
                "geospatial_issue": False,
                "preserved_specimen": preserved,
                "fossil": False,
                "occurrence_absent": False,
                "uncertain_taxon_match": False,
                "basis_of_record_suitable": True,
            }
        )
        checksum = values.get("checksum")
        media.append(
            {
                "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
                "reference_media_id": media_id,
                "reference_observation_id": observation_id,
                "provider_media_id": f"media-{index}",
                "source": source,
                "media_identifier": f"https://example.test/media/{index}.jpg",
                "media_type": "StillImage",
                "width": 2048,
                "height": 1365,
                "creator": "Fixture Creator",
                "rights_holder": "Fixture Rights Holder",
                "licence": "cc-by",
                "licence_uri": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": "Fixture Creator / CC BY 4.0",
                "occurrence_licence": "cc-by",
                "original_provider": source,
                "media_position": 0,
                "source_checksum": checksum,
                "source_checksum_algorithm": "sha256" if checksum else None,
                "download_status": "pending",
                "verification_status": "unreviewed",
                "exclusion_reason": None,
                "licence_policy_status": values.get("licence_status", "allowed"),
                "retrieved_at": NOW,
                "source_snapshot_version": "fixture-v1",
            }
        )
        qualification.append(
            {
                "reference_media_id": media_id,
                "verification_status": values.get(
                    "verification_status", "provider_supported"
                ),
                "trust_level": values.get("trust_level"),
                "verified_by": values.get("verified_by"),
                "geographic_layer": values.get("layer"),
                "route": route,
                "life_stage": life_stage,
                "visual_domain": values.get(
                    "visual_domain",
                    "pinned_specimen" if preserved else "live_field",
                ),
                "morphology_tags": values.get("morphology_tags", []),
                "image_quality_score": values.get("quality", 0.8),
                "provider_mirror_status": values.get(
                    "provider_mirror_status", "resolved"
                ),
            }
        )
    return (
        reference_observations_frame(observations),
        reference_media_candidates_frame(media),
        pl.DataFrame(qualification),
    )


def _quota(
    count: int,
    *,
    group: str = "target_adult",
    route: str = "adult_field",
    life_stage: str = "adult",
    visual_domain: str = "live_field",
) -> PrototypeReferenceQuota:
    return PrototypeReferenceQuota(
        reference_group=group,
        accepted_taxon_key=TARGET,
        scientific_name="Papilio demoleus",
        requested_count=count,
        route=route,
        life_stage=life_stage,
        visual_domain=visual_domain,
    )


def _plan(rows: list[dict[str, object]], count: int = 1):
    observations, media, qualification = _fixtures(rows)
    return plan_trust_first_layered_references(
        observations=observations,
        media_candidates=media,
        qualification_metadata=qualification,
        quotas=(_quota(count),),
    )


def test_r1_regional_beats_r4_exact_local() -> None:
    result = _plan(
        [
            {"trust_level": "R4", "layer": "A"},
            {
                "verification_status": "human_verified",
                "trust_level": "R1",
                "verified_by": "expert-1",
                "layer": "B",
            },
        ]
    )
    assert result.selected["trust_level"].item() == "R1"


def test_r2_global_beats_r5_exact_local() -> None:
    result = _plan(
        [
            {"verification_status": "provisional", "trust_level": "R5", "layer": "A"},
            {
                "verification_status": "provider_high_trust",
                "trust_level": "R2",
                "layer": "D",
            },
        ]
    )
    assert result.selected["trust_level"].item() == "R2"
    assert (
        result.evidence.filter(pl.col("trust_level") == "R5")["eligible"].item()
        is False
    )


def test_equivalent_trust_local_beats_global() -> None:
    result = _plan(
        [
            {"trust_level": "R3", "layer": "D"},
            {"trust_level": "R3", "layer": "A"},
        ]
    )
    assert result.selected["geographic_layer"].item() == "A"


def test_global_morphology_reference_fills_regional_visual_gap() -> None:
    result = _plan(
        [
            {"trust_level": "R3", "layer": "A", "morphology_tags": ["dorsal"]},
            {"trust_level": "R3", "layer": "D", "morphology_tags": ["ventral"]},
        ],
        count=2,
    )
    assert set(result.selected["geographic_layer"]) == {"A", "D"}
    assert set(result.selected["morphology_tags"].explode()) == {"dorsal", "ventral"}


def test_licence_ineligible_candidate_is_excluded() -> None:
    result = _plan([{"trust_level": "R2", "layer": "A", "licence_status": "denied"}])
    assert result.selected.is_empty()
    assert "licence_ineligible" in result.evidence["exclusion_reasons"].item()


def test_research_only_licence_is_eligible_for_prototype_planning() -> None:
    result = _plan(
        [{"trust_level": "R3", "layer": "A", "licence_status": "research_only"}]
    )

    assert result.selected.height == 1
    assert result.selected["licence_policy_status"].item() == "research_only"


def test_exact_duplicate_candidate_is_excluded() -> None:
    result = _plan(
        [
            {"trust_level": "R3", "layer": "A", "checksum": "same"},
            {"trust_level": "R4", "layer": "A", "checksum": "same"},
        ],
        count=2,
    )
    assert result.selected.height == 1
    assert result.evidence.filter(pl.col("exact_duplicate")).height == 1


def test_adult_larval_and_specimen_routes_remain_separate() -> None:
    observations, media, qualification = _fixtures(
        [
            {"route": "adult_field"},
            {"route": "larval", "life_stage": "larva"},
            {"route": "pinned_specimen"},
        ]
    )
    result = plan_trust_first_layered_references(
        observations=observations,
        media_candidates=media,
        qualification_metadata=qualification,
        quotas=(
            _quota(1),
            _quota(1, group="target_larva", route="larval", life_stage="larva"),
            _quota(
                1,
                group="target_specimen",
                route="pinned_specimen",
                visual_domain="pinned_specimen",
            ),
        ),
    )
    assert result.selected.group_by("reference_group").len().height == 3
    assert set(result.selected["route"]) == {"adult_field", "larval", "pinned_specimen"}


def test_trust_and_geography_components_are_persisted_separately() -> None:
    result = _plan([{"trust_level": "R3", "layer": "C"}])
    assert result.evidence.schema == prototype_reference_planner_evidence_schema()
    row = result.evidence.row(0, named=True)
    assert row["trust_component"] == 3.0
    assert row["geographic_component"] == 3.0


def test_planner_scores_are_explicitly_not_probabilities() -> None:
    result = _plan([{"trust_level": "R3", "layer": "A"}])
    assert result.evidence["score_semantics"].item() == PLANNER_SCORE_SEMANTICS
    assert result.report["score_is_probability"] is False
    assert not any("probability" in column for column in result.evidence.columns)


def test_actual_layer_proportions_and_quota_deviations_are_reported() -> None:
    result = _plan(
        [
            {"trust_level": "R3", "layer": "A"},
            {"trust_level": "R3", "layer": "C"},
            {"trust_level": "R3", "layer": "D"},
            {"trust_level": "R3", "layer": "E"},
        ],
        count=4,
    )
    assert result.report["actual_layer_bucket_proportions"] == {
        "AB": 0.25,
        "C": 0.25,
        "D": 0.25,
        "E": 0.25,
    }
    quota = result.report["quota_results"][0]
    assert quota["target_counts_by_layer_bucket"] == {"AB": 1, "C": 1, "D": 1, "E": 1}
    assert quota["quota_deviations"] == {"AB": 0, "C": 0, "D": 0, "E": 0}


def test_no_geo_uses_explicit_global_fallback() -> None:
    result = _plan([{"trust_level": "R3", "cluster": "no_geo", "layer": None}])
    assert result.selected["geographic_layer"].item() == "D"
    assert result.selected["no_geo_fallback"].item() is True
