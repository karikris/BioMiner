from __future__ import annotations

import hashlib
from typing import Any

import polars as pl
import pytest

from biominer.detection.detector_base import DecodedImage
from biominer.flickr_fetch.geography import build_flickr_geography_frame
from biominer.flickr_fetch.scoring_geography import build_flickr_scoring_geography
from biominer.flickr_fetch.scoring_partitions import (
    FLICKR_GEO_TAXON_PARTITIONS_FILE,
    FLICKR_PARTITION_SUMMARY_FILE,
    FlickrPartitionArtifacts,
    build_flickr_geo_taxon_partitions,
    validate_flickr_geo_taxon_partitions,
    write_flickr_partition_artifacts,
)
from biominer.flickr_fetch.scoring_units import build_flickr_scoring_unit_artifacts
from biominer.vision.target_full_frame import build_target_full_frame_plan


_ROUTING_POLICY_FINGERPRINT = "sha256:" + "a" * 64


def test_partitions_by_reusable_geo_taxon_and_model_contracts() -> None:
    scoring, geography = _partition_sources()

    result = build_flickr_geo_taxon_partitions(scoring, geography)

    assert result.partitions.height == 5
    assert result.summary.height == 4
    by_photo = {
        row["flickr_photo_id"]: row for row in result.partitions.to_dicts()
    }
    assert by_photo["photo-1"]["partition_id"] == by_photo["photo-2"][
        "partition_id"
    ]
    assert by_photo["photo-1"]["model_input_signature"] == by_photo["photo-2"][
        "model_input_signature"
    ]
    assert by_photo["photo-1"]["model_input_contract_signature"] == by_photo[
        "photo-2"
    ]["model_input_contract_signature"]
    assert by_photo["photo-1"]["association_set_signature"] != by_photo[
        "photo-2"
    ]["association_set_signature"]
    assert by_photo["photo-1"]["partition_id"] != by_photo["no-geo"][
        "partition_id"
    ]
    assert by_photo["photo-1"]["partition_id"] != by_photo["larval"][
        "partition_id"
    ]
    assert by_photo["photo-1"]["candidate_set_signature"] != by_photo[
        "expanded-candidates"
    ]["candidate_set_signature"]

    shared = result.summary.filter(
        pl.col("partition_id") == by_photo["photo-1"]["partition_id"]
    ).row(0, named=True)
    assert shared["organism_unit_count"] == 2
    assert shared["photo_embedding_unit_count"] == 2
    assert shared["visual_input_count"] == 1
    assert shared["model_input_count"] == 1
    assert shared["model_input_reuse_count"] == 1
    assert shared["candidate_species_count"] == 2
    assert shared["family_count"] == 1
    assert shared["association_set_count"] == 2
    assert shared["association_count"] == 2

    no_geo = by_photo["no-geo"]
    assert no_geo["geography_availability"] == "no_geo"
    assert no_geo["geographic_scope"] == "no_geo"
    columns = set(result.partitions.columns) | set(result.summary.columns)
    assert "embedding" not in columns
    assert "embedding_vector" not in columns
    assert "image_bytes" not in columns


def test_partition_artifacts_are_deterministic_and_round_trip(tmp_path) -> None:
    scoring, geography = _partition_sources()
    first = build_flickr_geo_taxon_partitions(scoring, geography)
    second = build_flickr_geo_taxon_partitions(scoring, geography)

    assert first.partitions.equals(second.partitions)
    assert first.summary.equals(second.summary)
    paths = write_flickr_partition_artifacts(
        first,
        scoring,
        geography,
        tmp_path,
    )
    assert {path.name for path in paths.values()} == {
        FLICKR_GEO_TAXON_PARTITIONS_FILE,
        FLICKR_PARTITION_SUMMARY_FILE,
    }
    persisted = FlickrPartitionArtifacts(
        partitions=pl.read_parquet(paths["partitions"]),
        summary=pl.read_parquet(paths["summary"]),
    )
    validate_flickr_geo_taxon_partitions(persisted, scoring, geography)
    assert persisted.partitions.equals(first.partitions)
    assert persisted.summary.equals(first.summary)

    alternate = build_flickr_geo_taxon_partitions(
        scoring,
        geography,
        partition_policy_version="flickr-geo-taxon-partition-policy-v2-test",
    )
    assert set(alternate.partitions["partition_id"]) != set(
        first.partitions["partition_id"]
    )


def test_empty_downstream_evidence_still_partitions_eligible_work() -> None:
    records = [
        _record(
            "pending-candidates",
            latitude=-27.4705,
            longitude=153.026,
            accuracy=16,
            country_code="AU",
        )
    ]
    plan = _plan(records)
    scoring = build_flickr_scoring_unit_artifacts(plan, run_id="run-empty")
    geography = build_flickr_scoring_geography(
        scoring.photo_embedding_units,
        build_flickr_geography_frame(records),
    )

    result = build_flickr_geo_taxon_partitions(scoring, geography)

    row = result.partitions.row(0, named=True)
    assert row["candidate_species_count"] == 0
    assert row["family_count"] == 0
    assert row["association_count"] == 0
    assert row["candidate_set_signature"].startswith("sha256:")
    assert row["family_pool_signature"].startswith("sha256:")
    assert row["association_set_signature"].startswith("sha256:")


def test_fails_closed_on_summary_or_source_tampering() -> None:
    scoring, geography = _partition_sources()
    result = build_flickr_geo_taxon_partitions(scoring, geography)
    tampered = FlickrPartitionArtifacts(
        partitions=result.partitions,
        summary=result.summary.with_columns(
            (pl.col("organism_unit_count") + 1).alias("organism_unit_count")
        ),
    )
    with pytest.raises(ValueError, match="summary does not match"):
        validate_flickr_geo_taxon_partitions(tampered, scoring, geography)

    broken_geography = geography.with_columns(
        pl.lit("sha256:" + "0" * 64).alias("row_fingerprint")
    )
    with pytest.raises(ValueError, match="row fingerprint mismatch"):
        build_flickr_geo_taxon_partitions(scoring, broken_geography)


def _partition_sources():
    records = [
        _record(
            "photo-1",
            latitude=-27.4705,
            longitude=153.026,
            accuracy=16,
            country_code="AU",
            admin1="Queensland",
        ),
        _record(
            "photo-2",
            latitude=-27.4705,
            longitude=153.026,
            accuracy=16,
            country_code="AU",
            admin1="Queensland",
        ),
        _record("no-geo", latitude=None, longitude=None, accuracy=None),
        _record(
            "larval",
            latitude=-27.4705,
            longitude=153.026,
            accuracy=16,
            country_code="AU",
            admin1="Queensland",
            route="larval",
            detection_route="caterpillar_field",
        ),
        _record(
            "expanded-candidates",
            latitude=-27.4705,
            longitude=153.026,
            accuracy=16,
            country_code="AU",
            admin1="Queensland",
        ),
    ]
    plan = _plan(records)
    unit_by_photo = {
        unit.flickr_photo_id: unit for unit in plan.scoring_units
    }
    associations = [
        _association(photo_id, query_index=index)
        for index, photo_id in enumerate(
            ("photo-1", "photo-2", "no-geo", "larval", "expanded-candidates"),
            start=1,
        )
    ]
    candidate_rows: list[dict[str, object]] = []
    for photo_id, unit in unit_by_photo.items():
        candidate_rows.extend(
            [
                _candidate(unit.scoring_unit_id, "7724053", priority=1),
                _candidate(unit.scoring_unit_id, "1938687", priority=2),
            ]
        )
        if photo_id == "expanded-candidates":
            candidate_rows.append(
                _candidate(unit.scoring_unit_id, "1938690", priority=3)
            )
    scoring = build_flickr_scoring_unit_artifacts(
        plan,
        run_id="run-partitions",
        associations=associations,
        candidate_species=candidate_rows,
    )
    geography = build_flickr_scoring_geography(
        scoring.photo_embedding_units,
        build_flickr_geography_frame(records),
    )
    return scoring, geography


def _plan(records: list[dict[str, object]]):
    shared_image = DecodedImage(
        width=2,
        height=2,
        mode="RGB",
        data=b"\x01" * 12,
    )
    images = {
        str(record["flickr_photo_id"]): (
            shared_image
            if record["flickr_photo_id"] in {"photo-1", "photo-2"}
            else DecodedImage(
                width=2,
                height=2,
                mode="RGB",
                data=hashlib.sha256(
                    str(record["flickr_photo_id"]).encode()
                ).digest()[:12],
            )
        )
        for record in records
    }
    return build_target_full_frame_plan(
        detection_rows=[
            _detection_row(
                str(record["flickr_photo_id"]),
                str(record["source_record_hash"]),
                route=str(record.get("route") or "adult_field"),
                detection_route=str(
                    record.get("detection_route") or "adult_butterfly_field"
                ),
            )
            for record in records
        ],
        image_loader=lambda row: images[str(row["flickr_photo_id"])],
    )


def _record(photo_id: str, **values: object) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": "sha256:" + hashlib.sha256(photo_id.encode()).hexdigest(),
        **values,
    }


def _association(photo_id: str, *, query_index: int) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "association_kind": "query",
        "association_source": "flickr_query_hits",
        "association_source_id": f"query-hash-{query_index}",
        "flickr_query_id": f"query-{query_index}",
        "query_hash": f"query-hash-{query_index}",
        "query_tier": "species_scientific:high:tags",
        "search_term": "Papilio demoleus",
        "accepted_taxon_key": "7724053",
        "scientific_name": "Papilio demoleus",
    }


def _candidate(
    organism_unit_id: str,
    taxon_key: str,
    *,
    priority: int,
) -> dict[str, object]:
    names = {
        "7724053": "Papilio demoleus",
        "1938687": "Papilio polytes",
        "1938690": "Papilio memnon",
    }
    return {
        "organism_unit_id": organism_unit_id,
        "candidate_accepted_taxon_key": taxon_key,
        "candidate_scientific_name": names[taxon_key],
        "family_key": "1933990",
        "family_name": "Papilionidae",
        "genus_key": "1938686",
        "genus_name": "Papilio",
        "candidate_priority": priority,
        "candidate_reasons": ["regional", "query_associated"],
        "candidate_source_ids": ["regional-union-1", "query-definition-1"],
    }


def _detection_row(
    photo_id: str,
    source_record_hash: str,
    *,
    route: str,
    detection_route: str,
) -> dict[str, Any]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": source_record_hash,
        "detection_id": f"detection:{photo_id}",
        "detection_status": "detected",
        "detector_score": 0.9,
        "detector_label": "butterfly_like",
        "bbox_xyxy": [0.0, 0.0, 2.0, 2.0],
        "bbox_xyxyn": [0.0, 0.0, 1.0, 1.0],
        "mask_polygon_xyn": None,
        "detection_route": detection_route,
        "routing_action": "score",
        "bioclip_route": route,
        "routing_policy_version": "detection-routing-policy-v1",
        "routing_policy_fingerprint": _ROUTING_POLICY_FINGERPRINT,
        "schema_version": "object-detection-v2",
    }
