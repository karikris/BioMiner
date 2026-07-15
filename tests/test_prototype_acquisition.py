from __future__ import annotations

from datetime import UTC, datetime
import hashlib

from biominer.references.prototype_acquisition import (
    PROTOTYPE_SHORTFALL_FILE,
    PROTOTYPE_SOURCE_SUMMARY_FILE,
    compile_prototype_acquisition,
    prototype_reference_shortfall_schema,
    prototype_reference_source_summary_schema,
    write_prototype_acquisition_result,
)
from biominer.references.schemas import (
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_observation_id,
    reference_acquisition_plan_schema,
    reference_media_candidates_frame,
    reference_observations_frame,
)


NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
TARGET = "gbif:1938069"


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _frames():
    observation_id = make_reference_observation_id("GBIF", "fixture-observation")
    media_id = make_reference_media_id("GBIF", "fixture-media", observation_id)
    observations = reference_observations_frame(
        [
            {
                "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
                "reference_observation_id": observation_id,
                "source": "GBIF",
                "source_observation_id": "fixture-observation",
                "source_taxon_id": "1938069",
                "supplied_scientific_name": "Papilio demoleus",
                "accepted_taxon_key": TARGET,
                "reconciled_scientific_name": "Papilio demoleus",
                "registry_version": "fixture-v1",
                "taxon_reconciliation_status": "accepted_key_exact",
                "identification_quality": "provider_supported",
                "community_taxon_status": None,
                "identification_disagreement": False,
                "captive_or_cultivated": False,
                "observer_id": "observer-1",
                "locality": "locality-1",
                "life_stage": "adult",
                "sex": None,
                "observed_at": NOW,
                "latitude": -33.8,
                "longitude": 151.2,
                "coordinate_uncertainty": 100.0,
                "coordinates_obscured": False,
                "country": "Australia",
                "country_code": "AU",
                "geo_cluster_id": "cluster-local",
                "distance_to_cluster_medoid_km": 5.0,
                "source_dataset_key": "fixture-dataset",
                "source_dataset_doi": None,
                "source_record_url": "https://example.test/observation",
                "source_record_hash": _sha("observation"),
                "retrieved_at": NOW,
                "source_snapshot_version": "fixture-v1",
                "source_query_fingerprint": _sha("query"),
                "fallback_level": 0,
                "geospatial_issue": False,
                "preserved_specimen": False,
                "fossil": False,
                "occurrence_absent": False,
                "uncertain_taxon_match": False,
                "basis_of_record_suitable": True,
            }
        ]
    )
    media = reference_media_candidates_frame(
        [
            {
                "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
                "reference_media_id": media_id,
                "reference_observation_id": observation_id,
                "provider_media_id": "fixture-media",
                "source": "GBIF",
                "media_identifier": "https://example.test/media.jpg",
                "media_type": "StillImage",
                "width": 2048,
                "height": 1365,
                "creator": "Fixture Creator",
                "rights_holder": "Fixture Rights Holder",
                "licence": "cc-by-nc",
                "licence_uri": "https://creativecommons.org/licenses/by-nc/4.0/",
                "attribution": "Fixture Creator / CC BY-NC 4.0",
                "occurrence_licence": "cc-by-nc",
                "original_provider": "Fixture Provider",
                "media_position": 0,
                "source_checksum": None,
                "source_checksum_algorithm": None,
                "download_status": "pending",
                "verification_status": "unreviewed",
                "exclusion_reason": None,
                "licence_policy_status": "unreviewed",
                "retrieved_at": NOW,
                "source_snapshot_version": "fixture-v1",
            }
        ]
    )
    return observations, media


def _result():
    observations, media = _frames()
    query_plan = {
        "target_accepted_taxon_key": TARGET,
        "queries": [
            {
                "accepted_taxon_key": TARGET,
                "scientific_name": "Papilio demoleus",
            }
        ],
        "acquisition_quotas": {
            "target_adult": {
                "species": [TARGET],
                "life_stage": "adult",
                "minimum_per_species": 1,
            }
        },
    }
    visual_manifest = {
        "target_accepted_taxon_key": TARGET,
        "manifest_version": "visual-fixture-v1",
        "candidates": [
            {
                "candidate_id": "visual-1",
                "visual_domain_category": "artwork",
                "source": "Wikimedia Commons",
                "source_record_id": "visual-1",
                "media_uri": "https://example.test/artwork.jpg",
                "attribution": "Fixture Artist / CC0",
                "licence_check_status": "allowed",
                "prototype_eligible": True,
                "contains_biological_butterfly": False,
            }
        ],
    }
    return compile_prototype_acquisition(
        observations=(observations,),
        media_candidates=(media,),
        query_plans=(query_plan,),
        visual_domain_manifest=visual_manifest,
        created_at=NOW,
    )


def test_prototype_acquisition_reports_taxonomic_and_visual_lanes() -> None:
    result = _result()

    assert result.plan.schema == reference_acquisition_plan_schema()
    assert result.source_summary.schema == prototype_reference_source_summary_schema()
    assert result.shortfalls.schema == prototype_reference_shortfall_schema()
    assert result.planner.selected["licence_policy_status"].item() == "research_only"
    assert result.report["summary"]["selected_for_download_count"] == 2
    assert result.report["selected_trust_distribution"]["R1"] == 0
    assert result.report["provider_supported_is_human_verified"] is False
    assert set(result.source_summary["candidate_scope_type"]) == {
        "accepted_taxon",
        "visual_domain",
    }


def test_prototype_acquisition_writes_required_parquet_outputs(tmp_path) -> None:
    paths = write_prototype_acquisition_result(_result(), tmp_path)

    assert paths["plan"].name == "reference_acquisition_plan.parquet"
    assert paths["source_summary"].name == PROTOTYPE_SOURCE_SUMMARY_FILE
    assert paths["shortfalls"].name == PROTOTYPE_SHORTFALL_FILE
    assert all(path.is_file() for path in paths.values())
