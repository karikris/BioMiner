from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from biominer.references.schemas import (
    REFERENCE_ACQUISITION_PLAN_FILE,
    REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
    REFERENCE_MEDIA_CANDIDATES_FILE,
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_FILE,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_acquisition_plan_id,
    make_reference_media_id,
    make_reference_observation_id,
    reference_acquisition_plan_frame,
    reference_acquisition_plan_schema,
    reference_media_candidate_schema,
    reference_media_candidates_frame,
    reference_observation_schema,
    reference_observations_frame,
    write_reference_acquisition_plan,
    write_reference_media_candidates,
    write_reference_observations,
)
from biominer.references.source_base import (
    ReferenceMetadataPage,
    ReferenceSourceQuery,
    validate_source_adapter,
)


NOW = datetime(2026, 7, 13, 0, 0, tzinfo=UTC)
OBSERVATION_HASH = "sha256:" + "a" * 64
PLAN_FINGERPRINT = "sha256:" + "b" * 64


def _observation(
    source_observation_id: str = "123",
    *,
    source: str = "GBIF",
) -> dict[str, object]:
    return {
        "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        "reference_observation_id": make_reference_observation_id(
            source,
            source_observation_id,
        ),
        "source": source,
        "source_observation_id": source_observation_id,
        "source_taxon_id": "1938069",
        "supplied_scientific_name": "Papilio demoleus",
        "accepted_taxon_key": "gbif:1938069",
        "reconciled_scientific_name": "Papilio demoleus",
        "registry_version": "butterflies-v2-20260712",
        "taxon_reconciliation_status": "accepted_key_exact",
        "identification_quality": "research_grade",
        "community_taxon_status": "species",
        "identification_disagreement": False,
        "captive_or_cultivated": False,
        "life_stage": "adult",
        "sex": None,
        "observed_at": datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
        "latitude": -33.87,
        "longitude": 151.21,
        "coordinate_uncertainty": 50.0,
        "coordinates_obscured": False,
        "country": "Australia",
        "country_code": "AU",
        "geo_cluster_id": "cluster-au",
        "distance_to_cluster_medoid_km": 4.2,
        "source_dataset_key": "dataset-1",
        "source_dataset_doi": "10.15468/example",
        "source_record_url": "https://example.test/occurrence/123",
        "source_record_hash": OBSERVATION_HASH,
        "retrieved_at": NOW,
        "source_snapshot_version": "gbif-occurrence-2026-07-13",
        "geospatial_issue": False,
        "preserved_specimen": False,
        "fossil": False,
        "occurrence_absent": False,
        "uncertain_taxon_match": False,
        "basis_of_record_suitable": True,
    }


def _media(
    provider_media_id: str = "media-1",
    *,
    observation: dict[str, object] | None = None,
) -> dict[str, object]:
    observation = observation or _observation()
    source = str(observation["source"])
    observation_id = str(observation["reference_observation_id"])
    return {
        "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
        "reference_media_id": make_reference_media_id(
            source,
            provider_media_id,
            observation_id,
        ),
        "reference_observation_id": observation_id,
        "provider_media_id": provider_media_id,
        "source": source,
        "media_identifier": "https://example.test/media/1.jpg",
        "media_type": "StillImage",
        "width": 2048,
        "height": 1365,
        "creator": "Example Observer",
        "rights_holder": "Example Observer",
        "licence": "CC-BY-4.0",
        "licence_uri": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Example Observer / CC BY 4.0",
        "occurrence_licence": "CC0-1.0",
        "original_provider": "Example Publisher",
        "media_position": 0,
        "source_checksum": None,
        "source_checksum_algorithm": None,
        "download_status": "pending",
        "verification_status": "unreviewed",
        "exclusion_reason": None,
        "licence_policy_status": "allowed",
        "retrieved_at": NOW,
        "source_snapshot_version": "gbif-occurrence-2026-07-13",
    }


def _plan() -> dict[str, object]:
    plan_id = make_acquisition_plan_id(
        target_accepted_taxon_key="gbif:1938069",
        candidate_set_id="regional:test",
        plan_configuration_fingerprint=PLAN_FINGERPRINT,
    )
    return {
        "schema_version": REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
        "acquisition_plan_id": plan_id,
        "target_accepted_taxon_key": "gbif:1938069",
        "candidate_set_id": "regional:test",
        "candidate_accepted_taxon_key": "gbif:1938069",
        "scientific_name": "Papilio demoleus",
        "geo_cluster_id": "cluster-au",
        "life_stage": "adult",
        "visual_domain": "field",
        "source": "GBIF",
        "requested_count": 20,
        "available_candidate_count": 15,
        "selected_candidate_count": 15,
        "shortfall_count": 5,
        "fallback_level": 0,
        "selection_strategy": "balanced-independent-observations-v1",
        "selection_seed": 42,
        "max_distance_km": 100.0,
        "licence_policy_version": "reference-licences-v1",
        "source_snapshot_version": "gbif-occurrence-2026-07-13",
        "plan_configuration_fingerprint": PLAN_FINGERPRINT,
        "created_at": NOW,
    }


def test_reference_frames_have_exact_deterministic_physical_schemas() -> None:
    observations = reference_observations_frame(
        [_observation("2"), _observation("1")]
    )
    media = reference_media_candidates_frame(
        [
            _media("2", observation=_observation("2")),
            _media("1", observation=_observation("1")),
        ]
    )
    plan = reference_acquisition_plan_frame([_plan()])

    assert observations.schema == reference_observation_schema()
    assert media.schema == reference_media_candidate_schema()
    assert plan.schema == reference_acquisition_plan_schema()
    assert observations["source_observation_id"].to_list() == ["1", "2"]
    assert media["provider_media_id"].to_list() == ["1", "2"]
    assert reference_observations_frame([]).schema == reference_observation_schema()
    assert reference_media_candidates_frame([]).schema == reference_media_candidate_schema()
    assert reference_acquisition_plan_frame([]).schema == reference_acquisition_plan_schema()


def test_reference_ids_are_stable_and_source_scoped() -> None:
    first = make_reference_observation_id("GBIF", "123")
    assert first == make_reference_observation_id("gbif", "123")
    assert first != make_reference_observation_id("iNaturalist", "123")
    media = make_reference_media_id("GBIF", "photo-1", first)
    assert media == make_reference_media_id("gbif", "photo-1", first)
    assert media != make_reference_media_id("GBIF", "photo-2", first)


def test_observation_contract_rejects_invalid_identity_and_geography() -> None:
    mismatched = _observation()
    mismatched["reference_observation_id"] = "reference-observation:wrong"
    with pytest.raises(ValueError, match="ID mismatch"):
        reference_observations_frame([mismatched])

    partial_coordinate = _observation()
    partial_coordinate["longitude"] = None
    with pytest.raises(ValueError, match="populated together"):
        reference_observations_frame([partial_coordinate])

    unresolved = _observation()
    unresolved["taxon_reconciliation_status"] = "unresolved"
    with pytest.raises(ValueError, match="marked uncertain"):
        reference_observations_frame([unresolved])


def test_media_contract_keeps_occurrence_and_media_licences_separate() -> None:
    row = _media()
    frame = reference_media_candidates_frame([row])

    assert frame["licence"].item() == "CC-BY-4.0"
    assert frame["occurrence_licence"].item() == "CC0-1.0"
    assert frame["verification_status"].item() == "unreviewed"

    checksum_without_algorithm = deepcopy(row)
    checksum_without_algorithm["source_checksum"] = "abc123"
    with pytest.raises(ValueError, match="populated together"):
        reference_media_candidates_frame([checksum_without_algorithm])


def test_acquisition_plan_contract_records_real_shortfalls() -> None:
    frame = reference_acquisition_plan_frame([_plan()])
    assert frame["shortfall_count"].item() == 5

    invalid = _plan()
    invalid["selected_candidate_count"] = 16
    with pytest.raises(ValueError, match="exceeds requested or available"):
        reference_acquisition_plan_frame([invalid])

    invalid = _plan()
    invalid["shortfall_count"] = 4
    with pytest.raises(ValueError, match="requested minus selected"):
        reference_acquisition_plan_frame([invalid])


def test_reference_writers_validate_and_round_trip_atomically(tmp_path: Path) -> None:
    observations = reference_observations_frame([_observation()])
    media = reference_media_candidates_frame([_media()])
    plan = reference_acquisition_plan_frame([_plan()])

    observation_path = write_reference_observations(observations, tmp_path)
    media_path = write_reference_media_candidates(media, tmp_path)
    plan_path = write_reference_acquisition_plan(plan, tmp_path)

    assert observation_path == tmp_path / REFERENCE_OBSERVATIONS_FILE
    assert media_path == tmp_path / REFERENCE_MEDIA_CANDIDATES_FILE
    assert plan_path == tmp_path / REFERENCE_ACQUISITION_PLAN_FILE
    assert pl.read_parquet(observation_path).equals(observations)
    assert pl.read_parquet(media_path).equals(media)
    assert pl.read_parquet(plan_path).equals(plan)
    assert not list(tmp_path.glob("*.tmp"))


def test_source_query_normalizes_scope_and_has_stable_fingerprint() -> None:
    first = ReferenceSourceQuery(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        geo_cluster_id="cluster-au",
        fallback_level=1,
        source_taxon_id="1938069",
        spatial_cell_ids=("cell-b", "cell-a", "cell-a"),
        country_codes=("au", "NZ"),
        page_size=300,
        source_snapshot_version="gbif-2026-07-13",
    )
    second = ReferenceSourceQuery(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        geo_cluster_id="cluster-au",
        fallback_level=1,
        source_taxon_id="1938069",
        spatial_cell_ids=("cell-a", "cell-b"),
        country_codes=("NZ", "AU"),
        page_size=300,
        source_snapshot_version="gbif-2026-07-13",
    )

    assert first.spatial_cell_ids == ("cell-a", "cell-b")
    assert first.country_codes == ("AU", "NZ")
    assert first.query_fingerprint == second.query_fingerprint


def test_source_adapter_protocol_and_normalized_page_linkage() -> None:
    observations = reference_observations_frame([_observation()])
    media = reference_media_candidates_frame([_media()])
    query = ReferenceSourceQuery(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        geo_cluster_id="cluster-au",
        fallback_level=0,
        source_snapshot_version="gbif-2026-07-13",
    )

    class FakeAdapter:
        source = "GBIF"
        source_version = "gbif-fixture-v1"
        user_agent = "BioMiner tests"

        def fetch_page(
            self,
            query: ReferenceSourceQuery,
            *,
            cursor: str | None = None,
        ) -> ReferenceMetadataPage:
            return ReferenceMetadataPage(
                source=self.source,
                source_version=self.source_version,
                query_fingerprint=query.query_fingerprint,
                page_cursor=cursor,
                next_cursor=None,
                observations=observations,
                media_candidates=media,
                request_count=1,
                retry_count=0,
                rate_limit_count=0,
                complete=True,
            )

    adapter = FakeAdapter()
    validate_source_adapter(adapter)
    page = adapter.fetch_page(query)
    assert page.complete is True
    assert page.observations.height == 1
    assert page.media_candidates.height == 1

    orphan = _media()
    orphan["reference_observation_id"] = make_reference_observation_id("GBIF", "missing")
    orphan["reference_media_id"] = make_reference_media_id(
        "GBIF",
        str(orphan["provider_media_id"]),
        str(orphan["reference_observation_id"]),
    )
    with pytest.raises(ValueError, match="absent from the page"):
        ReferenceMetadataPage(
            source="GBIF",
            source_version="gbif-fixture-v1",
            query_fingerprint=query.query_fingerprint,
            page_cursor=None,
            next_cursor=None,
            observations=observations,
            media_candidates=reference_media_candidates_frame([orphan]),
            request_count=1,
            retry_count=0,
            rate_limit_count=0,
            complete=True,
        )
