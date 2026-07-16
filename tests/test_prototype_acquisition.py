from __future__ import annotations

from datetime import UTC, datetime
import hashlib

import polars as pl

from biominer.references.prototype_acquisition import (
    PROTOTYPE_DOWNLOAD_CANDIDATES_FILE,
    PROTOTYPE_SELECTION_FILE,
    PROTOTYPE_SHORTFALL_FILE,
    PROTOTYPE_SOURCE_SUMMARY_FILE,
    compile_prototype_acquisition,
    prototype_reference_shortfall_schema,
    prototype_reference_selection_schema,
    prototype_reference_licence_policy,
    prototype_reference_source_summary_schema,
    write_prototype_acquisition_result,
)
from biominer.references.prototype_duplicates import (
    resolve_prototype_duplicates,
)
from biominer.references.schemas import (
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_observation_id,
    reference_acquisition_plan_schema,
    reference_media_candidate_schema,
    reference_media_candidates_frame,
    reference_media_objects_frame,
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


def _query_plan():
    return {
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


def _visual_manifest():
    return {
        "target_accepted_taxon_key": TARGET,
        "manifest_version": "visual-fixture-v1",
        "source_snapshot_version": "visual-fixture-v1",
        "candidates": [
            {
                "candidate_id": "visual-1",
                "visual_domain_category": "artwork",
                "source": "Wikimedia Commons",
                "source_record_id": "visual-1",
                "provider_media_id": "File:visual-1.jpg",
                "source_record_uri": "https://example.test/visual-1",
                "media_uri": "https://example.test/artwork.jpg",
                "width": 1200,
                "height": 800,
                "licence": "CC0",
                "licence_uri": "https://creativecommons.org/publicdomain/zero/1.0/",
                "attribution": "Fixture Artist / CC0",
                "verification_status": "provider_supported",
                "verification_actor": "wikimedia_commons",
                "agent_screening_status": "passed",
                "licence_check_status": "allowed",
                "prototype_eligible": True,
                "contains_biological_butterfly": False,
            }
        ],
    }


def _result():
    observations, media = _frames()
    return compile_prototype_acquisition(
        observations=(observations,),
        media_candidates=(media,),
        query_plans=(_query_plan(),),
        visual_domain_manifest=_visual_manifest(),
        created_at=NOW,
    )


def _downloaded_objects(result, *, exact: bool = False):
    rows = []
    for index, candidate in enumerate(result.download_candidates.iter_rows(named=True)):
        digest = _sha("exact" if exact else f"object-{index}")
        rows.append(
            {
                "schema_version": REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
                "reference_media_id": candidate["reference_media_id"],
                "source_object_uri": f"s3://fixture/{digest.removeprefix('sha256:')}.jpg",
                "content_type": "image/jpeg",
                "source_byte_count": 10_000 + index,
                "decoded_width": 1000,
                "decoded_height": 800,
                "sha256": digest,
                "perceptual_hash": (
                    "dhash128-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    if exact
                    else (
                        "dhash128-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        if index == 0
                        else "dhash128-v1:55555555555555555555555555555555"
                    )
                ),
                "duplicate_group_id": None,
                "duplicate_type": None,
                "canonical_reference_media_id": None,
                "provider_mirror_ids": [],
                "downloaded_at": NOW,
                "download_attempt_count": 1,
                "licence_policy_status": candidate["licence_policy_status"],
                "decode_status": "valid",
                "quarantine_reason": None,
                "object_fingerprint": _sha(
                    f"object:{candidate['reference_media_id']}:{digest}"
                ),
            }
        )
    return reference_media_objects_frame(rows)


def test_prototype_acquisition_reports_taxonomic_and_visual_lanes() -> None:
    result = _result()

    assert result.plan.schema == reference_acquisition_plan_schema()
    assert result.source_summary.schema == prototype_reference_source_summary_schema()
    assert result.shortfalls.schema == prototype_reference_shortfall_schema()
    assert result.selections.schema == prototype_reference_selection_schema()
    assert result.download_candidates.schema == reference_media_candidate_schema()
    assert result.selections["reference_observation_id"].n_unique() == 2
    assert set(result.selections["candidate_scope_type"]) == {
        "accepted_taxon",
        "visual_domain",
    }
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
    assert paths["selections"].name == PROTOTYPE_SELECTION_FILE
    assert paths["download_candidates"].name == PROTOTYPE_DOWNLOAD_CANDIDATES_FILE
    assert all(path.is_file() for path in paths.values())


def test_prototype_policy_keeps_public_domain_distinct_from_cc0() -> None:
    decision = prototype_reference_licence_policy().evaluate(
        media_licence="Public domain",
        licence_uri=None,
        attribution="Attributed public-domain work",
    )

    assert decision.status == "allowed"
    assert decision.canonical_licence == "public-domain"


def test_prototype_selections_are_direct_downloader_inputs() -> None:
    from biominer.references.downloader import _selected_media

    result = _result()
    selected = _selected_media(result.selections, result.download_candidates)

    assert len(selected) == result.selections.height
    assert {
        item.candidate["reference_media_id"] for item in selected
    } == set(result.selections["reference_media_id"])


def test_prototype_duplicate_resolution_builds_auditable_identity_groups() -> None:
    observations, _media = _frames()
    acquisition = _result()

    resolved = resolve_prototype_duplicates(
        selections=acquisition.selections,
        media_objects=_downloaded_objects(acquisition),
        media_candidates=acquisition.download_candidates,
        biological_observations=(observations,),
        visual_domain_manifest=_visual_manifest(),
        generated_at=NOW,
    )

    assert resolved.observations.height == 2
    assert resolved.identity_groups.height == 2
    assert resolved.report["counts"]["eligible"] == 2
    assert resolved.report["counts"]["relationships"] == 0
    assert resolved.report["counts"]["missing_owner_evidence"] == 1
    assert resolved.report["counts"]["missing_photographer_evidence"] == 1
    assert set(resolved.identity_groups["support_disposition"]) == {"eligible"}


def test_prototype_duplicate_resolution_flags_exact_copy_with_metadata_conflict() -> None:
    observations, _media = _frames()
    acquisition = _result()

    resolved = resolve_prototype_duplicates(
        selections=acquisition.selections,
        media_objects=_downloaded_objects(acquisition, exact=True),
        media_candidates=acquisition.download_candidates,
        biological_observations=(observations,),
        visual_domain_manifest=_visual_manifest(),
        generated_at=NOW,
    )

    relationship = resolved.deduplication.relationships.row(0, named=True)
    assert relationship["relationship_type"] == "exact"
    assert relationship["resolution_status"] == "conflict"
    assert "metadata_conflict" in relationship["evidence_types"]
    assert resolved.report["counts"]["eligible"] == 0
    assert resolved.report["counts"]["duplicate_conflicts"] == 2
    assert set(resolved.identity_groups["support_disposition"]) == {
        "duplicate_conflict"
    }


def test_prototype_duplicate_resolution_keeps_download_failures_retryable() -> None:
    observations, _media = _frames()
    acquisition = _result()
    rows = _downloaded_objects(acquisition).to_dicts()
    failed = rows[-1]
    for field in (
        "source_object_uri",
        "content_type",
        "source_byte_count",
        "decoded_width",
        "decoded_height",
        "sha256",
        "perceptual_hash",
        "downloaded_at",
    ):
        failed[field] = None
    failed["decode_status"] = "download_failed"
    failed["quarantine_reason"] = "retry_exhausted_http_429"
    failed["object_fingerprint"] = _sha("retryable-download-failure")

    resolved = resolve_prototype_duplicates(
        selections=acquisition.selections,
        media_objects=reference_media_objects_frame(rows),
        media_candidates=acquisition.download_candidates,
        biological_observations=(observations,),
        visual_domain_manifest=_visual_manifest(),
        generated_at=NOW,
    )

    assert resolved.report["counts"]["eligible"] == 1
    assert resolved.report["counts"]["operational_failures"] == 1
    failure = resolved.identity_groups.filter(
        pl.col("support_disposition") == "operational_failure"
    ).row(0, named=True)
    assert failure["duplicate_group_id"] is None
    assert failure["canonical_reference_media_id"] is None
    assert failure["exact_hash_group_id"] is None
