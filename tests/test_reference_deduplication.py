from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
import re

from PIL import Image
import polars as pl
import pytest

from biominer.references.deduplication import (
    _deduplication_markdown,
    _duplicate_group_id,
    _duplicate_relationship_id,
    _frame_fingerprint,
    _intrinsic_media_objects_fingerprint,
    REFERENCE_MEDIA_DEDUPLICATION_REPORT_FILE,
    REFERENCE_MEDIA_DEDUPLICATION_SUMMARY_FILE,
    ReferenceMediaDeduplicationConfig,
    ReferenceMediaDeduplicationResult,
    compute_reference_perceptual_hash,
    deduplicate_reference_media,
    perceptual_hash_distance,
    publish_reference_media_deduplication_result,
    validate_reference_media_deduplication_result,
    write_reference_media_deduplication_result,
)
from biominer.references.schemas import (
    DUPLICATE_RELATIONSHIP_TYPES,
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_FILE,
    REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_SCHEMA_VERSION,
    REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_observation_id,
    reference_media_candidates_frame,
    reference_media_duplicate_relationship_schema,
    reference_media_duplicate_relationships_frame,
    reference_media_objects_frame,
    reference_observations_frame,
    validate_reference_media_duplicate_relationships,
)
from biominer.storage.local import LocalStorageBackend


NOW = datetime(2026, 7, 14, 6, 0, tzinfo=UTC)
PHASH_BASE = "dhash128-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _phash(value: int) -> str:
    return f"dhash128-v1:{value:032x}"


def _observation(
    source: str,
    source_observation_id: str,
    *,
    source_record_url: str | None = None,
    identification_quality: str = "research_grade",
    taxon_reconciliation_status: str = "accepted_key_exact",
) -> dict[str, object]:
    accepted = taxon_reconciliation_status in {
        "accepted_key_exact",
        "accepted_name_synonym",
    }
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
        "accepted_taxon_key": "gbif:1938069" if accepted else None,
        "reconciled_scientific_name": "Papilio demoleus" if accepted else None,
        "registry_version": "butterflies-v2-20260714",
        "taxon_reconciliation_status": taxon_reconciliation_status,
        "identification_quality": identification_quality,
        "community_taxon_status": "species" if accepted else None,
        "identification_disagreement": False,
        "captive_or_cultivated": False,
        "observer_id": f"observer-{source_observation_id}",
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
        "source_dataset_key": f"dataset-{source}",
        "source_dataset_doi": None,
        "source_record_url": source_record_url
        or f"https://example.test/{source.casefold()}/{source_observation_id}",
        "source_record_hash": _sha(f"record:{source}:{source_observation_id}"),
        "retrieved_at": NOW,
        "source_snapshot_version": f"{source.casefold()}-2026-07-14",
        "source_query_fingerprint": _sha(f"query:{source}"),
        "fallback_level": 0,
        "geospatial_issue": False,
        "preserved_specimen": False,
        "fossil": False,
        "occurrence_absent": False,
        "uncertain_taxon_match": not accepted,
        "basis_of_record_suitable": True,
    }


def _candidate(
    observation: dict[str, object],
    provider_media_id: str,
    *,
    media_identifier: str | None = None,
    licence_policy_status: str = "allowed",
    verification_status: str = "unreviewed",
    download_status: str = "complete",
    exclusion_reason: str | None = None,
    original_provider: str | None = None,
    width: int = 96,
    height: int = 72,
) -> dict[str, object]:
    source = str(observation["source"])
    observation_id = str(observation["reference_observation_id"])
    licence = (
        "CC-BY-NC-4.0" if licence_policy_status == "research_only" else "CC-BY-4.0"
    )
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
        "media_identifier": media_identifier
        or f"https://media.example.test/{source}/{provider_media_id}.jpg",
        "media_type": "StillImage",
        "width": width,
        "height": height,
        "creator": "Example Observer",
        "rights_holder": "Example Observer",
        "licence": licence,
        "licence_uri": None,
        "attribution": f"Example Observer / {licence}",
        "occurrence_licence": "CC0-1.0",
        "original_provider": original_provider or source,
        "media_position": 0,
        "source_checksum": None,
        "source_checksum_algorithm": None,
        "download_status": download_status,
        "verification_status": verification_status,
        "exclusion_reason": exclusion_reason,
        "licence_policy_status": licence_policy_status,
        "retrieved_at": NOW,
        "source_snapshot_version": observation["source_snapshot_version"],
    }


def _media_object(
    candidate: dict[str, object],
    *,
    sha256: str,
    perceptual_hash: str = PHASH_BASE,
    width: int = 96,
    height: int = 72,
    byte_count: int = 10_000,
    licence_policy_status: str | None = None,
) -> dict[str, object]:
    media_id = str(candidate["reference_media_id"])
    digest = sha256.removeprefix("sha256:")
    return {
        "schema_version": REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "source_object_uri": f"s3://references/source_objects/{digest}.jpg",
        "content_type": "image/jpeg",
        "source_byte_count": byte_count,
        "decoded_width": width,
        "decoded_height": height,
        "sha256": sha256,
        "perceptual_hash": perceptual_hash,
        "duplicate_group_id": None,
        "duplicate_type": None,
        "canonical_reference_media_id": None,
        "provider_mirror_ids": [],
        "downloaded_at": NOW,
        "download_attempt_count": 1,
        "licence_policy_status": licence_policy_status
        or str(candidate["licence_policy_status"]),
        "decode_status": "valid",
        "quarantine_reason": None,
        "object_fingerprint": _sha(f"object:{media_id}:{sha256}"),
    }


def _quarantined_object(candidate: dict[str, object]) -> dict[str, object]:
    media_id = str(candidate["reference_media_id"])
    return {
        "schema_version": REFERENCE_MEDIA_OBJECTS_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "source_object_uri": None,
        "content_type": None,
        "source_byte_count": None,
        "decoded_width": None,
        "decoded_height": None,
        "sha256": None,
        "perceptual_hash": None,
        "duplicate_group_id": None,
        "duplicate_type": None,
        "canonical_reference_media_id": None,
        "provider_mirror_ids": [],
        "downloaded_at": None,
        "download_attempt_count": 0,
        "licence_policy_status": "quarantined",
        "decode_status": "not_attempted",
        "quarantine_reason": "uncertain_media_licence",
        "object_fingerprint": _sha(f"quarantined:{media_id}"),
    }


def _config(**overrides: object) -> ReferenceMediaDeduplicationConfig:
    values: dict[str, object] = {
        "same_observation_distance_threshold": 8,
        "cross_observation_distance_threshold": 4,
        "max_aspect_ratio_delta": 0.05,
        "minimum_informative_bits": 8,
        "policy_version": "reference-media-deduplication-policy-v1",
        "source_priority": ("iNaturalist", "GBIF"),
    }
    values.update(overrides)
    return ReferenceMediaDeduplicationConfig(**values)


def _deduplicate(
    observations: list[dict[str, object]],
    candidates: list[dict[str, object]],
    objects: list[dict[str, object]],
    *,
    config: ReferenceMediaDeduplicationConfig | None = None,
) -> ReferenceMediaDeduplicationResult:
    return deduplicate_reference_media(
        reference_media_objects_frame(objects),
        reference_media_candidates_frame(candidates),
        reference_observations_frame(observations),
        config=config or _config(),
        generated_at=NOW,
    )


def _rows_by_id(frame: pl.DataFrame) -> dict[str, dict[str, object]]:
    return {str(row["reference_media_id"]): row for row in frame.iter_rows(named=True)}


def _result_with_outputs(
    result: ReferenceMediaDeduplicationResult,
    *,
    media_objects: pl.DataFrame,
    relationships: pl.DataFrame,
) -> ReferenceMediaDeduplicationResult:
    report = deepcopy(result.report)
    valid = media_objects.filter(pl.col("decode_status") == "valid")
    report["inputs"]["media_objects_fingerprint"] = (
        _intrinsic_media_objects_fingerprint(media_objects)
    )
    report["counts"].update(
        {
            "valid_media": valid.height,
            "invalid_media": media_objects.height - valid.height,
            "duplicate_groups": valid["duplicate_group_id"].n_unique(),
            "canonical_media": valid["canonical_reference_media_id"].n_unique(),
            "relationships": relationships.height,
            "provider_mirror_relationships": relationships.filter(
                pl.col("provider_mirror")
            ).height,
            "review_required_relationships": relationships.filter(
                pl.col("resolution_status") == "review_required"
            ).height,
            "conflicting_relationships": relationships.filter(
                pl.col("resolution_status") == "conflict"
            ).height,
        }
    )
    report["duplicate_type_counts"] = dict(
        sorted(
            {
                str(value): valid["duplicate_type"].to_list().count(value)
                for value in valid["duplicate_type"].unique().to_list()
            }.items()
        )
    )
    report["relationship_type_counts"] = dict(
        sorted(
            {
                str(value): relationships["relationship_type"].to_list().count(value)
                for value in relationships["relationship_type"].unique().to_list()
            }.items()
        )
    )
    report["resolution_status_counts"] = dict(
        sorted(
            {
                str(value): relationships["resolution_status"].to_list().count(value)
                for value in relationships["resolution_status"].unique().to_list()
            }.items()
        )
    )
    report["outputs"]["media_objects_fingerprint"] = _frame_fingerprint(media_objects)
    report["outputs"]["relationships_fingerprint"] = _frame_fingerprint(relationships)
    return ReferenceMediaDeduplicationResult(
        media_objects=media_objects,
        relationships=relationships,
        media_candidates=result.media_candidates,
        observations=result.observations,
        report=report,
        markdown=_deduplication_markdown(report),
    )


def _ramp_image() -> Image.Image:
    image = Image.new("L", (9, 9))
    image.putdata(
        [
            (x * 29 + y * 47 + ((x * y) % 5) * 13) % 256
            for y in range(9)
            for x in range(9)
        ]
    )
    return image


def test_dhash128_has_a_locked_golden_and_raw_hamming_distance() -> None:
    uniform = Image.new("RGB", (37, 19), color=(127, 127, 127))
    ramp = _ramp_image()

    assert compute_reference_perceptual_hash(uniform) == (
        "dhash128-v1:00000000000000000000000000000000"
    )
    assert compute_reference_perceptual_hash(ramp) == (
        "dhash128-v1:fff5d2df7f7ff6d2fefde39fbd7ffce3"
    )
    assert compute_reference_perceptual_hash(ramp.copy()) == (
        compute_reference_perceptual_hash(ramp)
    )
    assert re.fullmatch(
        r"dhash128-v1:[0-9a-f]{32}",
        compute_reference_perceptual_hash(ramp),
    )
    assert perceptual_hash_distance(_phash(0), _phash(3)) == 2


def test_dhash128_normalises_exif_orientation_and_transparency() -> None:
    original = Image.new("RGB", (13, 9))
    original.putdata(
        [
            ((x * 31) % 256, (y * 47) % 256, ((x + y) * 19) % 256)
            for y in range(9)
            for x in range(13)
        ]
    )
    stored = original.transpose(Image.Transpose.ROTATE_90)
    stored.getexif()[274] = 6

    assert compute_reference_perceptual_hash(stored) == (
        compute_reference_perceptual_hash(original)
    )

    transparent_dark = Image.new("RGBA", (13, 9), (0, 0, 0, 0))
    transparent_bright = Image.new("RGBA", (13, 9), (255, 0, 255, 0))
    white = Image.new("RGB", (13, 9), "white")
    assert compute_reference_perceptual_hash(transparent_dark) == (
        compute_reference_perceptual_hash(white)
    )
    assert compute_reference_perceptual_hash(transparent_bright) == (
        compute_reference_perceptual_hash(white)
    )


def test_dhash128_uses_lanczos_for_equivalent_palette_and_bilevel_images() -> None:
    rgb = Image.new("RGB", (37, 19))
    rgb.putdata(
        [
            (255, 255, 255) if (x * 3 + y * 5) % 11 < 5 else (0, 0, 0)
            for y in range(19)
            for x in range(37)
        ]
    )
    palette = rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=2)
    bilevel = rgb.convert("1")

    assert compute_reference_perceptual_hash(palette) == (
        compute_reference_perceptual_hash(rgb)
    )
    assert compute_reference_perceptual_hash(bilevel) == (
        compute_reference_perceptual_hash(rgb)
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("dhash128-v1:0", _phash(0)),
        ("dhash128-v1:" + "A" * 32, _phash(0)),
        ("dhash64-v1:" + "0" * 16, _phash(0)),
    ],
)
def test_perceptual_distance_rejects_unversioned_or_incompatible_hashes(
    left: str,
    right: str,
) -> None:
    with pytest.raises(ValueError, match="perceptual hash"):
        perceptual_hash_distance(left, right)


def test_deduplication_config_requires_a_stricter_cross_observation_threshold() -> None:
    with pytest.raises(ValueError, match="cross-observation distance threshold"):
        ReferenceMediaDeduplicationConfig(
            same_observation_distance_threshold=3,
            cross_observation_distance_threshold=4,
        )


def test_dense_perceptual_hash_neighborhood_fails_closed_at_the_configured_bound() -> (
    None
):
    observation = _observation("iNaturalist", "dense-hashes")
    candidates = [_candidate(observation, f"dense-{index}") for index in range(4)]
    objects = [
        _media_object(
            candidate,
            sha256=_sha(f"dense-{index}"),
            perceptual_hash=_phash(1 << index),
        )
        for index, candidate in enumerate(candidates)
    ]

    with pytest.raises(ValueError, match="neighborhood exceeds"):
        _deduplicate(
            [observation],
            candidates,
            objects,
            config=_config(
                same_observation_distance_threshold=128,
                cross_observation_distance_threshold=0,
                max_perceptual_hash_neighbors=2,
            ),
        )


def test_candidate_source_must_match_observation_and_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observation = _observation("iNaturalist", "source-binding")
    candidate = _candidate(observation, "source-binding-media")
    candidate["source"] = "GBIF"
    candidate["reference_media_id"] = make_reference_media_id(
        "GBIF",
        str(candidate["provider_media_id"]),
        str(candidate["reference_observation_id"]),
    )
    caplog.set_level(logging.INFO, logger="biominer.references.deduplication")

    with pytest.raises(ValueError, match="source conflicts with observation"):
        _deduplicate(
            [observation],
            [candidate],
            [_media_object(candidate, sha256=_sha("source-binding"))],
        )

    events = [json.loads(record.message)["event"] for record in caplog.records]
    assert events[-2:] == [
        "reference_media_deduplication_started",
        "reference_media_deduplication_failed",
    ]


def test_invalid_generated_at_is_inside_the_failure_log_boundary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observation = _observation("GBIF", "invalid-generated-at")
    candidate = _candidate(observation, "invalid-generated-at")
    caplog.set_level(logging.INFO, logger="biominer.references.deduplication")

    with pytest.raises(ValueError, match="Invalid isoformat string"):
        deduplicate_reference_media(
            reference_media_objects_frame(
                [_media_object(candidate, sha256=_sha("invalid-generated-at"))]
            ),
            reference_media_candidates_frame([candidate]),
            reference_observations_frame([observation]),
            config=_config(),
            generated_at="not-a-timestamp",
        )

    events = [json.loads(record.message)["event"] for record in caplog.records]
    assert events[-2:] == [
        "reference_media_deduplication_started",
        "reference_media_deduplication_failed",
    ]


def test_large_exact_bucket_uses_a_sparse_evidence_tree() -> None:
    observation = _observation("iNaturalist", "exact-bucket")
    candidates = [_candidate(observation, f"exact-{index:03d}") for index in range(40)]
    exact_sha = _sha("one-content-object")
    result = _deduplicate(
        [observation],
        candidates,
        [_media_object(candidate, sha256=exact_sha) for candidate in candidates],
    )

    assert result.relationships.height == len(candidates) - 1
    assert set(result.relationships["relationship_type"]) == {"exact"}
    assert result.report["counts"]["duplicate_groups"] == 1


def test_exact_sha_group_retains_every_object_and_direct_relationship() -> None:
    observations = [_observation("GBIF", "101"), _observation("iNaturalist", "202")]
    candidates = [
        _candidate(observations[0], "gbif-photo"),
        _candidate(observations[1], "inat-photo", verification_status="accepted"),
    ]
    exact_sha = _sha("identical-source-bytes")
    objects = [
        _media_object(candidates[0], sha256=exact_sha),
        _media_object(candidates[1], sha256=exact_sha),
    ]
    original_candidates = reference_media_candidates_frame(candidates)
    original_observations = reference_observations_frame(observations)

    result = deduplicate_reference_media(
        reference_media_objects_frame(objects),
        original_candidates,
        original_observations,
        config=_config(),
        generated_at=NOW,
    )

    assert isinstance(result, ReferenceMediaDeduplicationResult)
    assert set(result.media_objects["reference_media_id"]) == {
        str(row["reference_media_id"]) for row in candidates
    }
    assert original_candidates.equals(reference_media_candidates_frame(candidates))
    assert original_observations.equals(reference_observations_frame(observations))
    rows = result.media_objects.to_dicts()
    assert len({row["duplicate_group_id"] for row in rows}) == 1
    assert {row["duplicate_type"] for row in rows} == {"exact"}
    assert len({row["canonical_reference_media_id"] for row in rows}) == 1

    assert (
        result.relationships.schema == reference_media_duplicate_relationship_schema()
    )
    validate_reference_media_duplicate_relationships(result.relationships)
    relationship = result.relationships.row(0, named=True)
    assert result.relationships.height == 1
    assert relationship["relationship_type"] == "exact"
    assert relationship["sha256_equal"] is True
    assert "exact_sha256" in relationship["evidence_types"]
    assert (
        relationship["left_reference_media_id"]
        < relationship["right_reference_media_id"]
    )
    assert relationship["resolution_status"] == "resolved"
    assert relationship["schema_version"] == (
        REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_SCHEMA_VERSION
    )
    assert REFERENCE_MEDIA_DUPLICATE_RELATIONSHIPS_FILE == (
        "reference_media_duplicate_relationships.parquet"
    )
    assert DUPLICATE_RELATIONSHIP_TYPES == frozenset(
        {
            "exact",
            "provider_mirror",
            "resized_copy",
            "near_identical_burst",
            "perceptual_candidate",
        }
    )
    assert (
        result.report["inputs"]["media_objects_fingerprint"]
        != (result.report["outputs"]["media_objects_fingerprint"])
    )


def test_equivalent_creative_commons_encodings_do_not_create_a_conflict() -> None:
    observations = [
        _observation("GBIF", "licence-1"),
        _observation("GBIF", "licence-2"),
    ]
    candidates = [
        _candidate(observations[0], "licence-code"),
        _candidate(observations[1], "licence-uri"),
    ]
    candidates[0]["licence"] = "CC-BY-4.0"
    candidates[1]["licence"] = "https://creativecommons.org/licenses/by/4.0/"
    digest = _sha("same-licensed-image")

    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )

    relationship = result.relationships.row(0, named=True)
    assert relationship["resolution_status"] == "resolved"
    assert "metadata_conflict" not in relationship["evidence_types"]
    assert set(result.media_objects["duplicate_type"]) == {"exact"}


def test_creative_commons_version_conflicts_are_not_collapsed() -> None:
    observations = [
        _observation("GBIF", "licence-version-1"),
        _observation("GBIF", "licence-version-2"),
    ]
    candidates = [
        _candidate(observations[0], "licence-version-left"),
        _candidate(observations[1], "licence-version-right"),
    ]
    candidates[0]["licence"] = "CC-BY-3.0"
    candidates[0]["licence_uri"] = "https://creativecommons.org/licenses/by/4.0/"
    candidates[1]["licence"] = "CC-BY-4.0"
    digest = _sha("same-bytes-conflicting-licence-versions")

    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )

    relationship = result.relationships.row(0, named=True)
    assert relationship["resolution_status"] == "conflict"
    assert "metadata_conflict" in relationship["evidence_types"]
    assert set(result.media_objects["duplicate_type"]) == {
        "unresolved_perceptual_candidate"
    }


def test_sparse_exact_component_detects_non_anchor_taxon_conflicts() -> None:
    observations = [
        _observation("GBIF", f"component-conflict-{index}") for index in range(3)
    ]
    candidates = [
        _candidate(observation, f"component-conflict-{index}")
        for index, observation in enumerate(observations)
    ]
    anchor_id = min(str(candidate["reference_media_id"]) for candidate in candidates)
    anchor_observation_id = next(
        str(candidate["reference_observation_id"])
        for candidate in candidates
        if candidate["reference_media_id"] == anchor_id
    )
    anchor_observation = next(
        observation
        for observation in observations
        if observation["reference_observation_id"] == anchor_observation_id
    )
    anchor_observation.update(
        {
            "accepted_taxon_key": None,
            "reconciled_scientific_name": None,
            "taxon_reconciliation_status": "unresolved",
            "community_taxon_status": None,
            "uncertain_taxon_match": True,
        }
    )
    conflicting_observation = next(
        observation
        for observation in observations
        if observation["reference_observation_id"] != anchor_observation_id
    )
    conflicting_observation.update(
        {
            "accepted_taxon_key": "gbif:other-species",
            "reconciled_scientific_name": "Papilio polytes",
        }
    )
    digest = _sha("sparse-component-conflict")

    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )

    assert result.relationships.height == 2
    assert set(result.relationships["resolution_status"]) == {"conflict"}
    assert all(
        "component_metadata_conflict" in evidence
        for evidence in result.relationships["evidence_types"]
    )
    assert set(result.media_objects["duplicate_type"]) == {
        "unresolved_perceptual_candidate"
    }


def test_resized_copy_and_near_burst_group_but_unrelated_same_observation_does_not() -> (
    None
):
    observation = _observation("iNaturalist", "301")
    candidates = [
        _candidate(observation, "original", width=128, height=96),
        _candidate(observation, "resized", width=64, height=48),
        _candidate(observation, "burst", width=128, height=96),
        _candidate(observation, "dorsal-view", width=128, height=96),
    ]
    base_value = int(PHASH_BASE.split(":", 1)[1], 16)
    objects = [
        _media_object(candidates[0], sha256=_sha("original"), width=128, height=96),
        _media_object(candidates[1], sha256=_sha("resized"), width=64, height=48),
        _media_object(
            candidates[2],
            sha256=_sha("burst"),
            perceptual_hash=_phash(base_value ^ 0b11),
            width=128,
            height=96,
        ),
        _media_object(
            candidates[3],
            sha256=_sha("different-view"),
            perceptual_hash=_phash(base_value ^ ((1 << 40) - 1)),
            width=128,
            height=96,
        ),
    ]

    result = _deduplicate([observation], candidates, objects)
    relationships = result.relationships.to_dicts()
    relationship_types = {row["relationship_type"] for row in relationships}

    assert "resized_copy" in relationship_types
    assert "near_identical_burst" in relationship_types
    dorsal_id = str(candidates[3]["reference_media_id"])
    assert all(
        dorsal_id
        not in {
            row["left_reference_media_id"],
            row["right_reference_media_id"],
        }
        for row in relationships
    )
    rows = _rows_by_id(result.media_objects)
    assert rows[dorsal_id]["duplicate_type"] == "unique"
    assert rows[dorsal_id]["canonical_reference_media_id"] == dorsal_id
    assert all(
        row["same_observation"] is True
        for row in relationships
        if row["relationship_type"] in {"resized_copy", "near_identical_burst"}
    )


def test_orientation_swapped_dimensions_do_not_block_a_resized_copy() -> None:
    observation = _observation("iNaturalist", "orientation-301")
    candidates = [
        _candidate(observation, "landscape", width=96, height=72),
        _candidate(observation, "exif-portrait", width=72, height=96),
    ]
    result = _deduplicate(
        [observation],
        candidates,
        [
            _media_object(
                candidates[0],
                sha256=_sha("landscape"),
                width=96,
                height=72,
            ),
            _media_object(
                candidates[1],
                sha256=_sha("exif-portrait"),
                width=72,
                height=96,
            ),
        ],
    )

    relationship = result.relationships.row(0, named=True)
    assert relationship["relationship_type"] == "resized_copy"
    assert relationship["resolution_status"] == "resolved"


def test_uniform_hash_collision_is_review_required_and_never_resolved() -> None:
    observations = [_observation("GBIF", "401"), _observation("GBIF", "402")]
    candidates = [
        _candidate(observations[0], "black"),
        _candidate(observations[1], "white"),
    ]
    uniform_hash = compute_reference_perceptual_hash(Image.new("L", (32, 24), 0))
    assert uniform_hash == compute_reference_perceptual_hash(
        Image.new("L", (32, 24), 255)
    )
    objects = [
        _media_object(
            candidates[0], sha256=_sha("black"), perceptual_hash=uniform_hash
        ),
        _media_object(
            candidates[1], sha256=_sha("white"), perceptual_hash=uniform_hash
        ),
    ]

    result = _deduplicate(observations, candidates, objects)
    relationship = result.relationships.row(0, named=True)

    assert result.relationships.height == 1
    assert relationship["relationship_type"] == "perceptual_candidate"
    assert relationship["resolution_status"] == "review_required"
    assert relationship["sha256_equal"] is False
    assert relationship["perceptual_hash_distance"] == 0
    assert set(result.media_objects["duplicate_type"]) == {
        "unresolved_perceptual_candidate"
    }
    assert "exact_sha256" not in relationship["evidence_types"]


def test_result_validator_cannot_promote_low_information_collision() -> None:
    observation = _observation("iNaturalist", "low-information-tamper")
    candidates = [
        _candidate(observation, "low-information-left"),
        _candidate(observation, "low-information-right"),
    ]
    uniform_hash = compute_reference_perceptual_hash(Image.new("L", (32, 24), 0))
    result = _deduplicate(
        [observation],
        candidates,
        [
            _media_object(
                candidates[0],
                sha256=_sha("low-information-left"),
                perceptual_hash=uniform_hash,
            ),
            _media_object(
                candidates[1],
                sha256=_sha("low-information-right"),
                perceptual_hash=uniform_hash,
            ),
        ],
    )
    relationship = result.relationships.row(0, named=True)
    relationship["relationship_type"] = "near_identical_burst"
    relationship["resolution_status"] = "resolved"
    tampered = ReferenceMediaDeduplicationResult(
        media_objects=result.media_objects,
        relationships=reference_media_duplicate_relationships_frame([relationship]),
        media_candidates=result.media_candidates,
        observations=result.observations,
        report=result.report,
        markdown=result.markdown,
    )

    with pytest.raises(ValueError, match="relationship type conflicts"):
        validate_reference_media_deduplication_result(tampered)


def test_same_observation_threshold_is_inclusive_without_merging_threshold_plus_one() -> (
    None
):
    observation = _observation("iNaturalist", "501")
    base = int(PHASH_BASE.split(":", 1)[1], 16)
    config = _config(same_observation_distance_threshold=4)

    at_candidates = [
        _candidate(observation, "at-left"),
        _candidate(observation, "at-right"),
    ]
    at_objects = [
        _media_object(at_candidates[0], sha256=_sha("at-left")),
        _media_object(
            at_candidates[1],
            sha256=_sha("at-right"),
            perceptual_hash=_phash(base ^ 0b1111),
        ),
    ]
    at_boundary = _deduplicate(
        [observation],
        at_candidates,
        at_objects,
        config=config,
    )
    assert at_boundary.relationships.height == 1
    assert at_boundary.relationships["perceptual_hash_distance"].item() == 4

    outside_candidates = [
        _candidate(observation, "outside-left"),
        _candidate(observation, "outside-right"),
    ]
    outside_objects = [
        _media_object(outside_candidates[0], sha256=_sha("outside-left")),
        _media_object(
            outside_candidates[1],
            sha256=_sha("outside-right"),
            perceptual_hash=_phash(base ^ 0b1_1111),
        ),
    ]
    outside = _deduplicate(
        [observation],
        outside_candidates,
        outside_objects,
        config=config,
    )
    assert outside.relationships.is_empty()
    assert outside.media_objects["duplicate_group_id"].n_unique() == 2
    assert outside.media_objects["duplicate_type"].to_list() == ["unique", "unique"]


def test_transitive_component_retains_only_direct_threshold_edges() -> None:
    observation = _observation("iNaturalist", "601")
    candidates = [_candidate(observation, name) for name in ("a", "b", "c")]
    base = int(PHASH_BASE.split(":", 1)[1], 16)
    objects = [
        _media_object(
            candidates[0], sha256=_sha("chain-a"), perceptual_hash=_phash(base)
        ),
        _media_object(
            candidates[1],
            sha256=_sha("chain-b"),
            perceptual_hash=_phash(base ^ 0b1),
        ),
        _media_object(
            candidates[2],
            sha256=_sha("chain-c"),
            perceptual_hash=_phash(base ^ 0b11),
        ),
    ]

    result = _deduplicate(
        [observation],
        candidates,
        objects,
        config=_config(
            same_observation_distance_threshold=1,
            cross_observation_distance_threshold=1,
        ),
    )
    pairs = {
        (row["left_reference_media_id"], row["right_reference_media_id"])
        for row in result.relationships.iter_rows(named=True)
    }
    ordered_ids = [str(candidate["reference_media_id"]) for candidate in candidates]
    expected_edges = {
        tuple(sorted((ordered_ids[0], ordered_ids[1]))),
        tuple(sorted((ordered_ids[1], ordered_ids[2]))),
    }

    assert pairs == expected_edges
    assert (
        len({row["duplicate_group_id"] for row in result.media_objects.to_dicts()}) == 1
    )
    assert (
        len(
            {
                row["canonical_reference_media_id"]
                for row in result.media_objects.to_dicts()
            }
        )
        == 1
    )
    assert sorted(result.relationships["perceptual_hash_distance"].to_list()) == [1, 1]


def test_output_is_shuffle_deterministic_and_rerun_idempotent() -> None:
    observations = [_observation("GBIF", "701"), _observation("GBIF", "702")]
    candidates = [
        _candidate(observations[0], "left"),
        _candidate(observations[1], "right"),
    ]
    digest = _sha("idempotent-exact")
    objects = [
        _media_object(candidates[0], sha256=digest),
        _media_object(candidates[1], sha256=digest),
    ]
    object_frame = reference_media_objects_frame(objects)
    candidate_frame = reference_media_candidates_frame(candidates)
    observation_frame = reference_observations_frame(observations)

    first = deduplicate_reference_media(
        object_frame,
        candidate_frame,
        observation_frame,
        config=_config(),
        generated_at=NOW,
    )
    shuffled = deduplicate_reference_media(
        object_frame.reverse(),
        candidate_frame.reverse(),
        observation_frame.reverse(),
        config=_config(),
        generated_at=NOW,
    )
    rerun = deduplicate_reference_media(
        first.media_objects,
        candidate_frame,
        observation_frame,
        config=_config(),
        generated_at=NOW,
    )

    assert shuffled.media_objects.equals(first.media_objects)
    assert shuffled.relationships.equals(first.relationships)
    assert shuffled.report == first.report
    assert shuffled.markdown == first.markdown
    assert rerun.media_objects.equals(first.media_objects)
    assert rerun.relationships.equals(first.relationships)
    assert rerun.report == first.report
    assert rerun.markdown == first.markdown


def _canonical_pair(
    *,
    left_source: str = "GBIF",
    right_source: str = "GBIF",
    left_licence: str = "allowed",
    right_licence: str = "allowed",
    left_verification: str = "accepted",
    right_verification: str = "accepted",
    left_quality: str = "research_grade",
    right_quality: str = "research_grade",
    left_reconciliation: str = "accepted_key_exact",
    right_reconciliation: str = "accepted_key_exact",
    left_size: tuple[int, int] = (96, 72),
    right_size: tuple[int, int] = (96, 72),
    left_bytes: int = 10_000,
    right_bytes: int = 10_000,
    resized: bool = False,
) -> tuple[str, list[dict[str, object]]]:
    observations = [
        _observation(
            left_source,
            "priority-left",
            identification_quality=left_quality,
            taxon_reconciliation_status=left_reconciliation,
        ),
        _observation(
            right_source,
            "priority-right",
            identification_quality=right_quality,
            taxon_reconciliation_status=right_reconciliation,
        ),
    ]
    if resized:
        observations[1] = observations[0]
    candidates = [
        _candidate(
            observations[0],
            "priority-left",
            licence_policy_status=left_licence,
            verification_status=left_verification,
            width=left_size[0],
            height=left_size[1],
        ),
        _candidate(
            observations[1],
            "priority-right",
            licence_policy_status=right_licence,
            verification_status=right_verification,
            width=right_size[0],
            height=right_size[1],
        ),
    ]
    left_sha = _sha("canonical-left")
    right_sha = _sha("canonical-right") if resized else left_sha
    objects = [
        _media_object(
            candidates[0],
            sha256=left_sha,
            width=left_size[0],
            height=left_size[1],
            byte_count=left_bytes,
            licence_policy_status=left_licence,
        ),
        _media_object(
            candidates[1],
            sha256=right_sha,
            width=right_size[0],
            height=right_size[1],
            byte_count=right_bytes,
            licence_policy_status=right_licence,
        ),
    ]
    result = _deduplicate(
        list(
            {str(row["reference_observation_id"]): row for row in observations}.values()
        ),
        candidates,
        objects,
    )
    canonical = result.media_objects["canonical_reference_media_id"].unique().item()
    return str(canonical), candidates


def test_canonical_selection_uses_locked_licence_and_quality_order() -> None:
    canonical, candidates = _canonical_pair(
        left_licence="allowed",
        right_licence="research_only",
        left_verification="unreviewed",
        right_verification="accepted",
        left_quality="casual",
        right_quality="research_grade",
        left_source="GBIF",
        right_source="iNaturalist",
    )
    assert canonical == candidates[0]["reference_media_id"]

    canonical, candidates = _canonical_pair(
        left_verification="accepted",
        right_verification="unreviewed",
        left_quality="casual",
        right_quality="research_grade",
        left_source="GBIF",
        right_source="iNaturalist",
    )
    assert canonical == candidates[0]["reference_media_id"]

    canonical, candidates = _canonical_pair(
        left_quality="research_grade",
        right_quality="casual",
        left_source="GBIF",
        right_source="iNaturalist",
    )
    assert canonical == candidates[0]["reference_media_id"]

    canonical, candidates = _canonical_pair(
        left_source="iNaturalist",
        right_source="GBIF",
    )
    assert canonical == candidates[0]["reference_media_id"]

    canonical, candidates = _canonical_pair(
        left_size=(128, 96),
        right_size=(64, 48),
        left_bytes=1_000,
        right_bytes=100_000,
        resized=True,
    )
    assert canonical == candidates[0]["reference_media_id"]

    canonical, candidates = _canonical_pair(
        left_bytes=20_000,
        right_bytes=10_000,
        resized=True,
    )
    assert canonical == candidates[0]["reference_media_id"]

    canonical, candidates = _canonical_pair(resized=True)
    assert canonical == min(
        str(candidate["reference_media_id"]) for candidate in candidates
    )


def test_quarantined_rows_are_retained_but_cannot_join_content_groups() -> None:
    observations = [_observation("GBIF", "801"), _observation("GBIF", "802")]
    candidates = [
        _candidate(observations[0], "valid"),
        _candidate(
            observations[1],
            "quarantined",
            licence_policy_status="quarantined",
            download_status="quarantined",
        ),
    ]
    objects = [
        _media_object(candidates[0], sha256=_sha("valid")),
        _quarantined_object(candidates[1]),
    ]

    result = _deduplicate(observations, candidates, objects)
    rows = _rows_by_id(result.media_objects)
    quarantined_id = str(candidates[1]["reference_media_id"])
    quarantined = rows[quarantined_id]

    assert set(rows) == {str(row["reference_media_id"]) for row in candidates}
    assert quarantined["decode_status"] == "not_attempted"
    assert quarantined["licence_policy_status"] == "quarantined"
    assert quarantined["quarantine_reason"] == "uncertain_media_licence"
    assert quarantined["perceptual_hash"] is None
    assert quarantined["duplicate_group_id"] is None
    assert quarantined["canonical_reference_media_id"] is None
    assert quarantined["provider_mirror_ids"] == []
    assert result.relationships.is_empty()
    assert isinstance(result.report, dict)
    assert isinstance(result.markdown, str)
    assert "deduplication" in result.markdown.casefold()


def test_explicit_gbif_inaturalist_metadata_mirror_retains_both_provenance_ids() -> (
    None
):
    observation_url = "https://www.inaturalist.org/observations/5001"
    photo_url = "https://static.inaturalist.org/photos/15001/original.jpg"
    gbif_observation = _observation(
        "GBIF",
        "9001",
        source_record_url=observation_url,
    )
    inaturalist_observation = _observation(
        "iNaturalist",
        "5001",
        source_record_url=observation_url,
    )
    gbif_candidate = _candidate(
        gbif_observation,
        "gbif-media-1",
        media_identifier=photo_url,
        download_status="excluded",
        exclusion_reason="duplicate_inaturalist_through_gbif",
        original_provider="iNaturalist",
    )
    inaturalist_candidate = _candidate(
        inaturalist_observation,
        "15001",
        media_identifier=photo_url,
        original_provider="iNaturalist",
    )
    observations = [gbif_observation, inaturalist_observation]
    candidates = [gbif_candidate, inaturalist_candidate]
    objects = [
        _media_object(inaturalist_candidate, sha256=_sha("direct-inaturalist-photo"))
    ]
    original_candidates = deepcopy(candidates)
    original_observations = deepcopy(observations)

    result = _deduplicate(observations, candidates, objects)
    direct_id = str(inaturalist_candidate["reference_media_id"])
    mirror_id = str(gbif_candidate["reference_media_id"])
    relationship = result.relationships.row(0, named=True)
    object_row = result.media_objects.row(0, named=True)

    assert result.media_objects.height == 1
    assert object_row["reference_media_id"] == direct_id
    assert object_row["canonical_reference_media_id"] == direct_id
    assert object_row["duplicate_type"] == "provider_mirror"
    assert object_row["provider_mirror_ids"] == [mirror_id]
    assert result.relationships.height == 1
    assert {
        relationship["left_reference_media_id"],
        relationship["right_reference_media_id"],
    } == {direct_id, mirror_id}
    assert relationship["relationship_type"] == "provider_mirror"
    assert relationship["provider_mirror"] is True
    assert relationship["sha256_equal"] is False
    assert relationship["perceptual_hash_distance"] is None
    assert relationship["resolution_status"] == "resolved"
    assert "provider_identifier" in relationship["evidence_types"]
    assert relationship["canonical_reference_media_id"] == direct_id
    assert {
        relationship["left_reference_observation_id"],
        relationship["right_reference_observation_id"],
    } == {
        str(gbif_observation["reference_observation_id"]),
        str(inaturalist_observation["reference_observation_id"]),
    }
    assert {
        relationship["left_source"],
        relationship["right_source"],
    } == {"GBIF", "iNaturalist"}
    assert {
        relationship["left_provider_media_id"],
        relationship["right_provider_media_id"],
    } == {"gbif-media-1", "15001"}
    assert candidates == original_candidates
    assert observations == original_observations


def test_exact_provider_mirror_remains_visible_in_provider_audit_counts() -> None:
    observation_url = "https://www.inaturalist.org/observations/6001"
    photo_url = "https://static.inaturalist.org/photos/16001/original.jpg"
    observations = [
        _observation("GBIF", "gbif-6001", source_record_url=observation_url),
        _observation("iNaturalist", "6001", source_record_url=observation_url),
    ]
    candidates = [
        _candidate(
            observations[0],
            "gbif-media-6001",
            media_identifier=photo_url,
            original_provider="iNaturalist",
        ),
        _candidate(
            observations[1],
            "16001",
            media_identifier=photo_url,
            original_provider="iNaturalist",
        ),
    ]
    digest = _sha("exact-provider-mirror")

    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )

    relationship = result.relationships.row(0, named=True)
    assert relationship["relationship_type"] == "exact"
    assert relationship["provider_mirror"] is True
    assert result.report["counts"]["provider_mirror_relationships"] == 1
    validate_reference_media_deduplication_result(result)


def test_deduplication_result_persists_inventory_relationships_and_audit_reports(
    tmp_path: Path,
) -> None:
    observations = [_observation("GBIF", "9001"), _observation("iNaturalist", "9002")]
    candidates = [
        _candidate(observations[0], "gbif-exact"),
        _candidate(observations[1], "inat-exact"),
    ]
    exact_sha = _sha("persisted-exact-source")
    result = _deduplicate(
        observations,
        candidates,
        [
            _media_object(candidates[0], sha256=exact_sha),
            _media_object(candidates[1], sha256=exact_sha),
        ],
    )

    validate_reference_media_deduplication_result(result)
    paths = write_reference_media_deduplication_result(result, tmp_path / "bank")

    assert set(paths) == {"media_objects", "relationships", "report", "summary"}
    assert pl.read_parquet(paths["media_objects"]).equals(result.media_objects)
    assert pl.read_parquet(paths["relationships"]).equals(result.relationships)
    assert json.loads(paths["report"].read_text(encoding="utf-8")) == result.report
    assert paths["summary"].read_text(encoding="utf-8") == result.markdown
    assert paths["report"].name == REFERENCE_MEDIA_DEDUPLICATION_REPORT_FILE
    assert paths["summary"].name == REFERENCE_MEDIA_DEDUPLICATION_SUMMARY_FILE
    assert not list((tmp_path / "bank").glob("*.tmp"))


def test_deduplication_result_publishes_run_scoped_artifacts_with_report_last(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    observation = _observation("GBIF", "cloud-publication")
    candidate = _candidate(observation, "cloud-publication")
    result = _deduplicate(
        [observation],
        [candidate],
        [_media_object(candidate, sha256=_sha("cloud-publication"))],
    )
    storage = LocalStorageBackend()
    caplog.set_level(logging.INFO, logger="biominer.references.deduplication")

    uris = publish_reference_media_deduplication_result(
        result,
        storage=storage,
        output_prefix=str(tmp_path / "bank"),
        run_id="Run / Unsafe",
        now=lambda: NOW,
    )

    assert all(storage.exists(uri) for uri in uris.values())
    assert "/run_id=run_unsafe-" in uris["report"]
    report = storage.read_json(uris["report"])
    assert report["status"] == "complete"
    assert report["run_id"] == "Run / Unsafe"
    assert report["elapsed_seconds"] == 0.0
    assert set(report["artifacts"]) == {
        "media_objects",
        "relationships",
        "summary",
    }
    assert all(
        artifact["byte_count"] > 0 and artifact["sha256"].startswith("sha256:")
        for artifact in report["artifacts"].values()
    )
    assert all(
        artifact["committed"] is True for artifact in report["artifacts"].values()
    )
    summary = Path(uris["summary"]).read_text(encoding="utf-8")
    assert "Run / Unsafe" in summary
    assert "references.deduplicate_media" in summary
    assert str(report["git_sha"] or "not_instrumented") in summary
    assert uris["media_objects"] in summary
    assert uris["relationships"] in summary
    events = [json.loads(record.message)["event"] for record in caplog.records]
    assert events[-2:] == [
        "reference_media_deduplication_publication_started",
        "reference_media_deduplication_publication_completed",
    ]


def test_failed_cloud_publication_uses_the_common_audit_shape(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailCompletionReportOnce(LocalStorageBackend):
        failed = False

        def write_json(self, uri: str | Path, payload: dict[str, object]) -> str:
            if (
                str(uri).endswith(REFERENCE_MEDIA_DEDUPLICATION_REPORT_FILE)
                and payload.get("status") == "complete"
                and not self.failed
            ):
                self.failed = True
                raise OSError("completion report write failed")
            return super().write_json(uri, payload)

    observation = _observation("GBIF", "failed-cloud-publication")
    candidate = _candidate(observation, "failed-cloud-publication")
    result = _deduplicate(
        [observation],
        [candidate],
        [_media_object(candidate, sha256=_sha("failed-cloud-publication"))],
    )
    storage = FailCompletionReportOnce()
    caplog.set_level(logging.INFO, logger="biominer.references.deduplication")

    with pytest.raises(OSError, match="completion report write failed"):
        publish_reference_media_deduplication_result(
            result,
            storage=storage,
            output_prefix=str(tmp_path / "bank"),
            run_id="failed-publication",
            now=lambda: NOW,
        )

    report_path = next(
        (tmp_path / "bank").rglob(REFERENCE_MEDIA_DEDUPLICATION_REPORT_FILE)
    )
    summary_path = next(
        (tmp_path / "bank").rglob(REFERENCE_MEDIA_DEDUPLICATION_SUMMARY_FILE)
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error_type"] == "OSError"
    assert {
        "pid",
        "git_sha",
        "inputs",
        "settings",
        "policy_fingerprint",
        "counts",
        "outputs",
        "artifacts",
    } <= set(report)
    summary = summary_path.read_text(encoding="utf-8")
    assert "failed-publication" in summary
    assert "`failed`" in summary
    assert "completion report write failed" in summary
    events = [json.loads(record.message)["event"] for record in caplog.records]
    assert "reference_media_deduplication_publication_failed" in events


def test_persistent_summary_failure_does_not_suppress_failed_json_report(
    tmp_path: Path,
) -> None:
    class FailEverySummaryWrite(LocalStorageBackend):
        def write_text(
            self,
            uri: str | Path,
            text: str,
            *,
            encoding: str = "utf-8",
        ) -> str:
            if str(uri).endswith(REFERENCE_MEDIA_DEDUPLICATION_SUMMARY_FILE):
                raise OSError("summary storage unavailable")
            return super().write_text(uri, text, encoding=encoding)

    observation = _observation("GBIF", "summary-failure")
    candidate = _candidate(observation, "summary-failure")
    result = _deduplicate(
        [observation],
        [candidate],
        [_media_object(candidate, sha256=_sha("summary-failure"))],
    )

    with pytest.raises(OSError, match="summary storage unavailable"):
        publish_reference_media_deduplication_result(
            result,
            storage=FailEverySummaryWrite(),
            output_prefix=str(tmp_path / "bank"),
            run_id="persistent-summary-failure",
            now=lambda: NOW,
        )

    report_path = next(
        (tmp_path / "bank").rglob(REFERENCE_MEDIA_DEDUPLICATION_REPORT_FILE)
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error_type"] == "OSError"
    assert report["error"] == "summary storage unavailable"
    assert not list(
        (tmp_path / "bank").rglob(REFERENCE_MEDIA_DEDUPLICATION_SUMMARY_FILE)
    )


def test_deduplication_result_writer_rejects_report_drift(tmp_path: Path) -> None:
    observation = _observation("GBIF", "9101")
    candidate = _candidate(observation, "single")
    result = _deduplicate(
        [observation],
        [candidate],
        [_media_object(candidate, sha256=_sha("single"))],
    )
    tampered_report = deepcopy(result.report)
    tampered_report["counts"]["valid_media"] = 2
    tampered = ReferenceMediaDeduplicationResult(
        media_objects=result.media_objects,
        relationships=result.relationships,
        media_candidates=result.media_candidates,
        observations=result.observations,
        report=tampered_report,
        markdown=result.markdown,
    )

    with pytest.raises(ValueError, match="counts are inconsistent"):
        write_reference_media_deduplication_result(tampered, tmp_path / "bank")
    assert not (tmp_path / "bank").exists()


def test_relationship_validator_rejects_forged_endpoint_provenance() -> None:
    observations = [_observation("GBIF", "forged-1"), _observation("GBIF", "forged-2")]
    candidates = [
        _candidate(observations[0], "forged-left"),
        _candidate(observations[1], "forged-right"),
    ]
    digest = _sha("forged-provenance-source")
    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )
    forged = result.relationships.with_columns(
        pl.lit("forged-source").alias("left_source")
    )

    with pytest.raises(ValueError, match="endpoint provenance"):
        validate_reference_media_duplicate_relationships(forged)


@pytest.mark.parametrize(
    "field",
    ["sha256_equal", "same_observation", "provider_mirror"],
)
def test_relationship_validator_rejects_null_boolean_evidence(field: str) -> None:
    observations = [_observation("GBIF", "null-1"), _observation("GBIF", "null-2")]
    candidates = [
        _candidate(observations[0], "null-left"),
        _candidate(observations[1], "null-right"),
    ]
    digest = _sha("null-evidence-source")
    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )
    nullable = result.relationships.with_columns(
        pl.lit(None, dtype=pl.Boolean).alias(field)
    )

    with pytest.raises(ValueError, match=f"{field} must be a non-null Boolean"):
        validate_reference_media_duplicate_relationships(nullable)


def test_result_validator_rejects_a_duplicate_group_without_evidence_edges() -> None:
    observations = [
        _observation("GBIF", "missing-1"),
        _observation("GBIF", "missing-2"),
    ]
    candidates = [
        _candidate(observations[0], "missing-left"),
        _candidate(observations[1], "missing-right"),
    ]
    digest = _sha("missing-ledger-source")
    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )
    empty_relationships = reference_media_duplicate_relationships_frame([])
    report = deepcopy(result.report)
    report["counts"].update(
        {
            "relationships": 0,
            "provider_mirror_relationships": 0,
            "review_required_relationships": 0,
            "conflicting_relationships": 0,
        }
    )
    report["relationship_type_counts"] = {}
    report["resolution_status_counts"] = {}
    report["outputs"]["relationships_fingerprint"] = _frame_fingerprint(
        empty_relationships
    )
    tampered = ReferenceMediaDeduplicationResult(
        media_objects=result.media_objects,
        relationships=empty_relationships,
        media_candidates=result.media_candidates,
        observations=result.observations,
        report=report,
        markdown=_deduplication_markdown(report),
    )

    with pytest.raises(
        ValueError,
        match="inconsistent with its evidence|connected evidence graph",
    ):
        validate_reference_media_deduplication_result(tampered)


def test_result_validator_rejects_equal_hash_objects_split_into_singletons() -> None:
    observations = [_observation("GBIF", "split-1"), _observation("GBIF", "split-2")]
    candidates = [
        _candidate(observations[0], "split-left"),
        _candidate(observations[1], "split-right"),
    ]
    digest = _sha("split-identical-content")
    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )
    singleton_rows = []
    for row in result.media_objects.iter_rows(named=True):
        media_id = str(row["reference_media_id"])
        row.update(
            {
                "duplicate_group_id": _duplicate_group_id([media_id]),
                "duplicate_type": "unique",
                "canonical_reference_media_id": media_id,
                "provider_mirror_ids": [],
            }
        )
        singleton_rows.append(row)
    media_objects = reference_media_objects_frame(singleton_rows)
    tampered = _result_with_outputs(
        result,
        media_objects=media_objects,
        relationships=reference_media_duplicate_relationships_frame([]),
    )

    with pytest.raises(ValueError, match="deterministic sparse result"):
        validate_reference_media_deduplication_result(tampered)


def test_result_validator_rejects_an_omitted_metadata_alias_edge() -> None:
    observation_url = "https://www.inaturalist.org/observations/94001"
    photo_url = "https://static.inaturalist.org/photos/95001/original.jpg"
    observations = [
        _observation("GBIF", "omitted-alias", source_record_url=observation_url),
        _observation("iNaturalist", "94001", source_record_url=observation_url),
    ]
    candidates = [
        _candidate(
            observations[0],
            "omitted-alias-gbif",
            media_identifier=photo_url,
            download_status="excluded",
            exclusion_reason="duplicate_inaturalist_through_gbif",
            original_provider="iNaturalist",
        ),
        _candidate(
            observations[1],
            "95001",
            media_identifier=photo_url,
            original_provider="iNaturalist",
        ),
    ]
    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidates[1], sha256=_sha("omitted-alias"))],
    )
    object_row = result.media_objects.row(0, named=True)
    media_id = str(object_row["reference_media_id"])
    object_row.update(
        {
            "duplicate_group_id": _duplicate_group_id([media_id]),
            "duplicate_type": "unique",
            "canonical_reference_media_id": media_id,
            "provider_mirror_ids": [],
        }
    )
    tampered = _result_with_outputs(
        result,
        media_objects=reference_media_objects_frame([object_row]),
        relationships=reference_media_duplicate_relationships_frame([]),
    )

    with pytest.raises(ValueError, match="deterministic sparse result"):
        validate_reference_media_deduplication_result(tampered)


def test_result_validator_rejects_a_noncanonical_extra_exact_edge() -> None:
    observations = [_observation("GBIF", f"extra-edge-{index}") for index in range(3)]
    candidates = [
        _candidate(observation, f"extra-edge-{index}")
        for index, observation in enumerate(observations)
    ]
    digest = _sha("extra-edge-identical-content")
    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )
    existing_pairs = {
        (
            str(row["left_reference_media_id"]),
            str(row["right_reference_media_id"]),
        )
        for row in result.relationships.iter_rows(named=True)
    }
    media_ids = sorted(str(candidate["reference_media_id"]) for candidate in candidates)
    extra_pair = next(
        (left, right)
        for index, left in enumerate(media_ids)
        for right in media_ids[index + 1 :]
        if (left, right) not in existing_pairs
    )
    candidate_by_id = {
        str(candidate["reference_media_id"]): candidate for candidate in candidates
    }
    left_candidate = candidate_by_id[extra_pair[0]]
    right_candidate = candidate_by_id[extra_pair[1]]
    evidence = ["exact_sha256", "perceptual_hash"]
    template = result.relationships.row(0, named=True)
    template.update(
        {
            "duplicate_relationship_id": _duplicate_relationship_id(
                *extra_pair,
                evidence_types=evidence,
            ),
            "left_reference_media_id": extra_pair[0],
            "right_reference_media_id": extra_pair[1],
            "left_reference_observation_id": left_candidate["reference_observation_id"],
            "right_reference_observation_id": right_candidate[
                "reference_observation_id"
            ],
            "left_source": left_candidate["source"],
            "right_source": right_candidate["source"],
            "left_provider_media_id": left_candidate["provider_media_id"],
            "right_provider_media_id": right_candidate["provider_media_id"],
            "evidence_types": evidence,
            "sha256_equal": True,
            "perceptual_hash_distance": 0,
            "same_observation": False,
            "provider_mirror": False,
            "relationship_type": "exact",
            "resolution_status": "resolved",
        }
    )
    relationships = reference_media_duplicate_relationships_frame(
        [*result.relationships.to_dicts(), template]
    )
    tampered = _result_with_outputs(
        result,
        media_objects=result.media_objects,
        relationships=relationships,
    )

    with pytest.raises(ValueError, match="deterministic sparse result"):
        validate_reference_media_deduplication_result(tampered)


def test_result_validator_binds_metadata_aliases_to_candidate_inventory() -> None:
    observation_url = "https://www.inaturalist.org/observations/71001"
    photo_url = "https://static.inaturalist.org/photos/81001/original.jpg"
    observations = [
        _observation("GBIF", "alias-gbif", source_record_url=observation_url),
        _observation("iNaturalist", "71001", source_record_url=observation_url),
    ]
    candidates = [
        _candidate(
            observations[0],
            "alias-gbif-media",
            media_identifier=photo_url,
            download_status="excluded",
            exclusion_reason="duplicate_inaturalist_through_gbif",
            original_provider="iNaturalist",
        ),
        _candidate(
            observations[1],
            "81001",
            media_identifier=photo_url,
            original_provider="iNaturalist",
        ),
    ]
    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidates[1], sha256=_sha("bound-alias"))],
    )
    retained_candidates = reference_media_candidates_frame([candidates[1]])
    report = deepcopy(result.report)
    report["inputs"]["media_candidate_rows"] = retained_candidates.height
    report["inputs"]["media_candidates_fingerprint"] = _frame_fingerprint(
        retained_candidates
    )
    tampered = ReferenceMediaDeduplicationResult(
        media_objects=result.media_objects,
        relationships=result.relationships,
        media_candidates=retained_candidates,
        observations=result.observations,
        report=report,
        markdown=_deduplication_markdown(report),
    )

    with pytest.raises(ValueError, match="unknown media provenance"):
        validate_reference_media_deduplication_result(tampered)


def test_result_validator_recomputes_exact_evidence_from_object_hashes() -> None:
    observations = [_observation("GBIF", "hash-1"), _observation("GBIF", "hash-2")]
    candidates = [
        _candidate(observations[0], "hash-left"),
        _candidate(observations[1], "hash-right"),
    ]
    digest = _sha("original-identical-content")
    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )
    rows = result.media_objects.to_dicts()
    right_id = str(candidates[1]["reference_media_id"])
    original_right = next(row for row in rows if row["reference_media_id"] == right_id)
    replacement = _media_object(candidates[1], sha256=_sha("different-content"))
    for field in (
        "duplicate_group_id",
        "duplicate_type",
        "canonical_reference_media_id",
        "provider_mirror_ids",
    ):
        replacement[field] = original_right[field]
    changed_objects = reference_media_objects_frame(
        [replacement if row["reference_media_id"] == right_id else row for row in rows]
    )
    report = deepcopy(result.report)
    report["inputs"]["media_objects_fingerprint"] = (
        _intrinsic_media_objects_fingerprint(changed_objects)
    )
    report["outputs"]["media_objects_fingerprint"] = _frame_fingerprint(changed_objects)
    tampered = ReferenceMediaDeduplicationResult(
        media_objects=changed_objects,
        relationships=result.relationships,
        media_candidates=result.media_candidates,
        observations=result.observations,
        report=report,
        markdown=_deduplication_markdown(report),
    )

    with pytest.raises(ValueError, match="exact evidence conflicts"):
        validate_reference_media_deduplication_result(tampered)


def test_result_validator_requires_exact_evidence_for_equal_object_hashes() -> None:
    observations = [_observation("GBIF", "equal-1"), _observation("GBIF", "equal-2")]
    candidates = [
        _candidate(observations[0], "equal-left"),
        _candidate(observations[1], "equal-right"),
    ]
    digest = _sha("equal-object-content")
    result = _deduplicate(
        observations,
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )
    relationship = result.relationships.row(0, named=True)
    evidence = sorted(set(relationship["evidence_types"]) - {"exact_sha256"})
    relationship.update(
        {
            "duplicate_relationship_id": _duplicate_relationship_id(
                str(relationship["left_reference_media_id"]),
                str(relationship["right_reference_media_id"]),
                evidence_types=evidence,
            ),
            "relationship_type": "perceptual_candidate",
            "evidence_types": evidence,
            "sha256_equal": False,
            "resolution_status": "review_required",
        }
    )
    tampered = ReferenceMediaDeduplicationResult(
        media_objects=result.media_objects,
        relationships=reference_media_duplicate_relationships_frame([relationship]),
        media_candidates=result.media_candidates,
        observations=result.observations,
        report=result.report,
        markdown=result.markdown,
    )

    with pytest.raises(ValueError, match="exact evidence conflicts"):
        validate_reference_media_deduplication_result(tampered)


def test_result_validator_requires_perceptual_evidence_for_valid_provider_mirrors() -> (
    None
):
    observation_url = "https://www.inaturalist.org/observations/92001"
    photo_url = "https://static.inaturalist.org/photos/93001/original.jpg"
    observations = [
        _observation("GBIF", "provider-gbif", source_record_url=observation_url),
        _observation("iNaturalist", "92001", source_record_url=observation_url),
    ]
    candidates = [
        _candidate(
            observations[0],
            "provider-gbif-media",
            media_identifier=photo_url,
            original_provider="iNaturalist",
        ),
        _candidate(
            observations[1],
            "93001",
            media_identifier=photo_url,
            original_provider="iNaturalist",
        ),
    ]
    base = int(PHASH_BASE.split(":", 1)[1], 16)
    result = _deduplicate(
        observations,
        candidates,
        [
            _media_object(candidates[0], sha256=_sha("provider-left")),
            _media_object(
                candidates[1],
                sha256=_sha("provider-right"),
                perceptual_hash=_phash(base ^ ((1 << 40) - 1)),
            ),
        ],
    )
    relationship = result.relationships.row(0, named=True)
    evidence = sorted(set(relationship["evidence_types"]) - {"perceptual_hash"})
    relationship.update(
        {
            "duplicate_relationship_id": _duplicate_relationship_id(
                str(relationship["left_reference_media_id"]),
                str(relationship["right_reference_media_id"]),
                evidence_types=evidence,
            ),
            "evidence_types": evidence,
            "perceptual_hash_distance": None,
            "resolution_status": "resolved",
        }
    )
    tampered = ReferenceMediaDeduplicationResult(
        media_objects=result.media_objects,
        relationships=reference_media_duplicate_relationships_frame([relationship]),
        media_candidates=result.media_candidates,
        observations=result.observations,
        report=result.report,
        markdown=result.markdown,
    )

    with pytest.raises(ValueError, match="perceptual evidence conflicts"):
        validate_reference_media_deduplication_result(tampered)


def test_result_validator_recomputes_same_observation_evidence() -> None:
    observation = _observation("iNaturalist", "same-observation-ledger")
    candidates = [
        _candidate(observation, "same-left"),
        _candidate(observation, "same-right"),
    ]
    digest = _sha("same-observation-content")
    result = _deduplicate(
        [observation],
        candidates,
        [_media_object(candidate, sha256=digest) for candidate in candidates],
    )
    relationship = result.relationships.row(0, named=True)
    evidence = sorted(set(relationship["evidence_types"]) - {"same_observation"})
    relationship["evidence_types"] = evidence
    relationship["same_observation"] = False
    relationship["duplicate_relationship_id"] = _duplicate_relationship_id(
        str(relationship["left_reference_media_id"]),
        str(relationship["right_reference_media_id"]),
        evidence_types=evidence,
    )
    changed_relationships = reference_media_duplicate_relationships_frame(
        [relationship]
    )
    report = deepcopy(result.report)
    report["outputs"]["relationships_fingerprint"] = _frame_fingerprint(
        changed_relationships
    )
    tampered = ReferenceMediaDeduplicationResult(
        media_objects=result.media_objects,
        relationships=changed_relationships,
        media_candidates=result.media_candidates,
        observations=result.observations,
        report=report,
        markdown=_deduplication_markdown(report),
    )

    with pytest.raises(ValueError, match="same-observation evidence conflicts"):
        validate_reference_media_deduplication_result(tampered)


def test_result_publication_leaves_no_partial_directory_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation("GBIF", "atomic-publication")
    candidate = _candidate(observation, "atomic-publication")
    result = _deduplicate(
        [observation],
        [candidate],
        [_media_object(candidate, sha256=_sha("atomic-publication"))],
    )

    def fail_relationship_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("relationship write failed")

    monkeypatch.setattr(
        "biominer.references.deduplication.write_reference_media_duplicate_relationships",
        fail_relationship_write,
    )
    output = tmp_path / "bank"
    with pytest.raises(OSError, match="relationship write failed"):
        write_reference_media_deduplication_result(result, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".bank.*.tmp"))
