from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from biominer.references.licensing import (
    ReferenceLicencePolicy,
    canonicalise_creative_commons_licence,
)
from biominer.references.schemas import (
    REFERENCE_ACQUISITION_PLAN_FILE,
    REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
    REFERENCE_MEDIA_CANDIDATES_FILE,
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_MEDIA_OBJECTS_FILE,
    REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
    REFERENCE_MEDIA_RASTER_CONTENT_TYPES,
    REFERENCE_OBSERVATIONS_FILE,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_acquisition_plan_id,
    make_reference_media_id,
    make_reference_observation_id,
    reference_acquisition_plan_frame,
    reference_acquisition_plan_schema,
    reference_media_candidate_schema,
    reference_media_candidates_frame,
    reference_media_object_schema,
    reference_media_objects_frame,
    reference_observation_schema,
    reference_observations_frame,
    write_reference_acquisition_plan,
    write_reference_media_candidates,
    write_reference_media_objects,
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
OBJECT_SHA256 = "sha256:" + "d" * 64
OBJECT_FINGERPRINT = "sha256:" + "e" * 64


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("CC-BY-4.0", "cc-by"),
        ("CC0 1.0", "cc0"),
        ("https://creativecommons.org:443/licenses/by/4.0/", "cc-by"),
        ("http://creativecommons.org:80/licenses/by/4.0/", "cc-by"),
        ("CC-BY-5.0", None),
        ("https://creativecommons.org/licenses/by/5.0/", None),
        ("CC0 4.0", None),
        ("https://creativecommons.org/publicdomain/zero/4.0/", None),
        ("https://creativecommons.org:444/licenses/by/4.0/", None),
        ("http://creativecommons.org:443/licenses/by/4.0/", None),
    ],
)
def test_creative_commons_contract_rejects_unknown_versions_and_ports(
    value: str,
    canonical: str | None,
) -> None:
    assert canonicalise_creative_commons_licence(value) == canonical


def test_reference_licence_policy_rejects_conflicting_explicit_versions() -> None:
    policy = ReferenceLicencePolicy()

    conflict = policy.evaluate(
        media_licence="CC-BY-3.0",
        licence_uri="https://creativecommons.org/licenses/by/4.0/",
        attribution="Observer / CC BY",
    )
    matching = policy.evaluate(
        media_licence="CC-BY-4.0",
        licence_uri="https://creativecommons.org/licenses/by/4.0/",
        attribution="Observer / CC BY",
    )

    assert conflict.status == "quarantined"
    assert conflict.reason == "conflicting_media_licence"
    assert matching.status == "allowed"
    assert matching.reason is None


def test_reference_licence_policy_enforces_versioned_allowlist_and_aliases() -> None:
    policy = ReferenceLicencePolicy(
        broadly_reusable=("CC BY 4.0",),
        research_only=(),
        attribution_required=("CC BY 4.0",),
        licence_aliases=(
            (
                "provider-by-current",
                "https://creativecommons.org/licenses/by/4.0/",
            ),
        ),
    )

    assert policy.broadly_reusable == ("cc-by-4.0",)
    assert (
        policy.evaluate(
            media_licence="CC BY 4.0",
            licence_uri=None,
            attribution="Observer",
        ).status
        == "allowed"
    )
    assert (
        policy.evaluate(
            media_licence="CC BY 2.0",
            licence_uri=None,
            attribution="Observer",
        ).status
        == "denied"
    )
    assert (
        policy.evaluate(
            media_licence="cc-by",
            licence_uri=None,
            attribution="Observer",
        ).status
        == "denied"
    )
    conflict = policy.evaluate(
        media_licence="provider-by-current",
        licence_uri="https://creativecommons.org/licenses/by/2.0/",
        attribution="Observer",
    )
    assert conflict.status == "quarantined"
    assert conflict.reason == "conflicting_media_licence"


def test_reference_licence_policy_cannot_alias_or_allow_unknown_cc_versions() -> None:
    with pytest.raises(ValueError, match="invalid Creative Commons licence"):
        ReferenceLicencePolicy(
            broadly_reusable=("CC-BY-5.0",),
            research_only=(),
            attribution_required=(),
        )

    with pytest.raises(ValueError, match="cannot override Creative Commons"):
        ReferenceLicencePolicy(
            licence_aliases=(("CC-BY-5.0", "cc-by"),),
        )

    with pytest.raises(TypeError, match="tuple of strings"):
        ReferenceLicencePolicy(broadly_reusable=(123,))

    with pytest.raises(TypeError, match="must be strings"):
        ReferenceLicencePolicy(licence_aliases=((123, "cc-by"),))


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
        "observer_id": "observer-1",
        "locality": "Sydney",
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
        "source_query_fingerprint": "sha256:" + "c" * 64,
        "fallback_level": 0,
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
        "existing_support_count": 0,
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


def _media_object(
    reference_media_id: str = "reference-media:media-1",
    *,
    licence_policy_status: str = "allowed",
) -> dict[str, object]:
    digest = OBJECT_SHA256.removeprefix("sha256:")
    return {
        "schema_version": REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
        "reference_media_id": reference_media_id,
        "source_object_uri": f"s3://biominer/references/{digest}.jpg",
        "content_type": "image/jpeg",
        "source_byte_count": 4_096,
        "decoded_width": 32,
        "decoded_height": 24,
        "sha256": OBJECT_SHA256,
        "perceptual_hash": None,
        "duplicate_group_id": None,
        "duplicate_type": None,
        "canonical_reference_media_id": None,
        "provider_mirror_ids": [],
        "downloaded_at": NOW,
        "download_attempt_count": 1,
        "licence_policy_status": licence_policy_status,
        "decode_status": "valid",
        "quarantine_reason": None,
        "object_fingerprint": OBJECT_FINGERPRINT,
    }


def _quarantined_media_object(
    reference_media_id: str = "reference-media:media-1",
) -> dict[str, object]:
    row = _media_object(reference_media_id)
    row.update(
        {
            "source_object_uri": None,
            "content_type": None,
            "source_byte_count": None,
            "decoded_width": None,
            "decoded_height": None,
            "sha256": None,
            "downloaded_at": None,
            "download_attempt_count": 0,
            "licence_policy_status": "quarantined",
            "decode_status": "not_attempted",
            "quarantine_reason": "uncertain_media_licence",
        }
    )
    return row


def test_reference_frames_have_exact_deterministic_physical_schemas() -> None:
    observations = reference_observations_frame([_observation("2"), _observation("1")])
    media = reference_media_candidates_frame(
        [
            _media("2", observation=_observation("2")),
            _media("1", observation=_observation("1")),
        ]
    )
    plan = reference_acquisition_plan_frame([_plan()])
    objects = reference_media_objects_frame(
        [
            _media_object("reference-media:2"),
            _media_object("reference-media:1"),
        ]
    )

    assert observations.schema == reference_observation_schema()
    assert media.schema == reference_media_candidate_schema()
    assert plan.schema == reference_acquisition_plan_schema()
    assert objects.schema == reference_media_object_schema()
    assert observations["source_observation_id"].to_list() == ["1", "2"]
    assert media["provider_media_id"].to_list() == ["1", "2"]
    assert objects["reference_media_id"].to_list() == [
        "reference-media:1",
        "reference-media:2",
    ]
    assert reference_observations_frame([]).schema == reference_observation_schema()
    assert (
        reference_media_candidates_frame([]).schema
        == reference_media_candidate_schema()
    )
    assert (
        reference_acquisition_plan_frame([]).schema
        == reference_acquisition_plan_schema()
    )
    assert reference_media_objects_frame([]).schema == reference_media_object_schema()


def test_reference_media_object_schema_matches_the_locked_contract() -> None:
    assert reference_media_object_schema() == {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "source_object_uri": pl.String,
        "content_type": pl.String,
        "source_byte_count": pl.UInt64,
        "decoded_width": pl.UInt32,
        "decoded_height": pl.UInt32,
        "sha256": pl.String,
        "perceptual_hash": pl.String,
        "duplicate_group_id": pl.String,
        "duplicate_type": pl.String,
        "canonical_reference_media_id": pl.String,
        "provider_mirror_ids": pl.List(pl.String),
        "downloaded_at": pl.Datetime("us", "UTC"),
        "download_attempt_count": pl.UInt32,
        "licence_policy_status": pl.String,
        "decode_status": pl.String,
        "quarantine_reason": pl.String,
        "object_fingerprint": pl.String,
    }
    assert REFERENCE_MEDIA_RASTER_CONTENT_TYPES == frozenset(
        {"image/gif", "image/jpeg", "image/png", "image/tiff", "image/webp"}
    )


@pytest.mark.parametrize("licence_policy_status", ["allowed", "research_only"])
def test_reference_media_object_accepts_committed_licensed_media(
    licence_policy_status: str,
) -> None:
    frame = reference_media_objects_frame(
        [_media_object(licence_policy_status=licence_policy_status)]
    )

    assert frame["sha256"].item() == OBJECT_SHA256
    assert frame["licence_policy_status"].item() == licence_policy_status
    assert frame["perceptual_hash"].item() is None
    assert frame["provider_mirror_ids"].to_list() == [[]]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("licence_policy_status", "quarantined", "allowed or research-only"),
        (
            "source_object_uri",
            "s3://biominer/references/wrong.jpg",
            "contain the SHA-256",
        ),
        ("content_type", "text/html", "content_type must be one of"),
        ("download_attempt_count", 0, "positive object metrics"),
        ("quarantine_reason", "unexpected", "cannot have a quarantine reason"),
    ],
)
def test_reference_media_object_rejects_invalid_success_state(
    field: str,
    value: object,
    message: str,
) -> None:
    row = _media_object()
    row[field] = value

    with pytest.raises(ValueError, match=message):
        reference_media_objects_frame([row])


def test_reference_media_object_requires_closed_non_success_state() -> None:
    quarantined = _quarantined_media_object()
    frame = reference_media_objects_frame([quarantined])
    assert frame["decode_status"].item() == "not_attempted"
    assert frame["source_object_uri"].item() is None

    leaked_object = deepcopy(quarantined)
    leaked_object["source_object_uri"] = "s3://biominer/references/uncommitted.jpg"
    with pytest.raises(ValueError, match="cannot populate source_object_uri"):
        reference_media_objects_frame([leaked_object])

    missing_reason = deepcopy(quarantined)
    missing_reason["quarantine_reason"] = None
    with pytest.raises(ValueError, match="quarantine_reason must be nonblank"):
        reference_media_objects_frame([missing_reason])


@pytest.mark.parametrize(
    "provider_mirror_ids",
    [
        ["reference-media:2", "reference-media:1"],
        ["reference-media:1", "reference-media:1"],
    ],
)
def test_reference_media_object_requires_sorted_unique_provider_mirrors(
    provider_mirror_ids: list[str],
) -> None:
    row = _media_object()
    row["provider_mirror_ids"] = provider_mirror_ids

    with pytest.raises(ValueError, match="sorted and unique"):
        reference_media_objects_frame([row])


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
    objects = reference_media_objects_frame([_media_object()])

    observation_path = write_reference_observations(observations, tmp_path)
    media_path = write_reference_media_candidates(media, tmp_path)
    plan_path = write_reference_acquisition_plan(plan, tmp_path)
    object_path = write_reference_media_objects(objects, tmp_path)

    assert observation_path == tmp_path / REFERENCE_OBSERVATIONS_FILE
    assert media_path == tmp_path / REFERENCE_MEDIA_CANDIDATES_FILE
    assert plan_path == tmp_path / REFERENCE_ACQUISITION_PLAN_FILE
    assert object_path == tmp_path / REFERENCE_MEDIA_OBJECTS_FILE
    assert pl.read_parquet(observation_path).equals(observations)
    assert pl.read_parquet(media_path).equals(media)
    assert pl.read_parquet(plan_path).equals(plan)
    assert pl.read_parquet(object_path).equals(objects)
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


def test_reference_contracts_reject_undefined_fallback_levels() -> None:
    with pytest.raises(ValueError, match="between 0 and 3"):
        ReferenceSourceQuery(
            accepted_taxon_key="gbif:1938069",
            scientific_name="Papilio demoleus",
            geo_cluster_id="cluster-au",
            fallback_level=4,
            source_snapshot_version="gbif-2026-07-13",
        )

    observation = _observation()
    observation["fallback_level"] = 4
    with pytest.raises(ValueError, match="between 0 and 3"):
        reference_observations_frame([observation])


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
    orphan["reference_observation_id"] = make_reference_observation_id(
        "GBIF", "missing"
    )
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
