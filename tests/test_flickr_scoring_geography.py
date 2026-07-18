from __future__ import annotations

import hashlib
from typing import Any

import polars as pl
import pytest

from biominer.detection.detector_base import DecodedImage
from biominer.flickr_fetch.geography import (
    FlickrGeographyConfig,
    build_flickr_geography_frame,
)
from biominer.flickr_fetch.scoring_geography import (
    FLICKR_SCORING_GEOGRAPHY_FILE,
    build_flickr_scoring_geography,
    flickr_scoring_geography_schema,
    validate_flickr_scoring_geography,
    write_flickr_scoring_geography,
)
from biominer.flickr_fetch.scoring_units import build_flickr_scoring_unit_artifacts
from biominer.geography import GeographicResolutions
from biominer.vision.target_full_frame import build_target_full_frame_plan


_ROUTING_POLICY_FINGERPRINT = "sha256:" + "a" * 64


def test_normalizes_scoring_geography_at_supported_precision() -> None:
    records = [
        _record(
            "street",
            latitude=-27.4705,
            longitude=153.026,
            accuracy=16,
            country_code="AU",
            admin1="Queensland",
            coordinate_uncertainty_m=125.0,
            coordinate_uncertainty_source="provider_metadata",
        ),
        _record(
            "country",
            latitude=-35.0,
            longitude=149.0,
            accuracy=3,
            country_code="AU",
        ),
        _record(
            "unknown",
            latitude=-33.0,
            longitude=151.0,
            accuracy=None,
        ),
        _record("missing", latitude=None, longitude=None, accuracy=None),
    ]
    photo_units = _photo_units(records)
    geography = build_flickr_geography_frame(records)

    frame = build_flickr_scoring_geography(
        photo_units,
        geography,
        bioregion_by_admin_region=(("AU:Queensland", "australasia-east"),),
        bioregion_mapping_version="au-bioregions-v1",
    )

    assert frame.schema == flickr_scoring_geography_schema()
    rows = {row["flickr_photo_id"]: row for row in frame.to_dicts()}
    street = rows["street"]
    assert street["geography_availability"] == "available"
    assert street["geographic_scope"] == "local_cell"
    assert street["geographic_scope_value"] == street["local_cell_id"]
    assert street["supported_cell_resolution"] == 7
    assert street["coordinate_uncertainty_m"] == pytest.approx(125.0)
    assert street["coordinate_uncertainty_source"] == "provider_metadata"
    assert street["geography_source_quality"] == "metric_uncertainty_available"
    assert street["bioregion"] == "australasia-east"
    assert street["bioregion_source"] == "bioregion_mapping:au-bioregions-v1"

    country = rows["country"]
    assert country["geography_availability"] == "available"
    assert country["geographic_scope"] == "country"
    assert country["geographic_scope_value"] == "AU"
    assert country["supported_cell_resolution"] is None
    assert country["coordinate_uncertainty_m"] is None

    unknown = rows["unknown"]
    assert unknown["geography_availability"] == "unassigned_geo"
    assert unknown["geography_unavailable_reason"] == (
        "no_supported_geographic_scope"
    )
    assert unknown["geographic_scope"] == "unassigned_geo"
    assert unknown["geographic_scope_value"] is None

    missing = rows["missing"]
    assert missing["geography_availability"] == "no_geo"
    assert missing["geography_unavailable_reason"] == "coordinates_missing"
    assert missing["latitude"] is None
    assert missing["longitude"] is None
    assert missing["geographic_scope"] == "no_geo"


def test_scoring_geography_is_deterministic_and_round_trips(tmp_path) -> None:
    records = [
        _record(
            "source-bioregion",
            latitude=-33.8688,
            longitude=151.2093,
            accuracy=11,
            country_code="AU",
            admin1="New South Wales",
            bioregion="Sydney Basin",
            bioregion_source="source_registry-v2",
        )
    ]
    photo_units = _photo_units(records)
    geography = build_flickr_geography_frame(records)

    first = build_flickr_scoring_geography(photo_units, geography)
    second = build_flickr_scoring_geography(photo_units, geography)

    assert first.equals(second)
    row = first.row(0, named=True)
    assert row["bioregion"] == "Sydney Basin"
    assert row["bioregion_source"] == "source_registry-v2"
    assert row["source_geography_row_fingerprint"] == geography[
        "row_fingerprint"
    ][0]
    assert row["geography_signature"].startswith("sha256:")

    path = write_flickr_scoring_geography(first, photo_units, tmp_path)
    assert path.name == FLICKR_SCORING_GEOGRAPHY_FILE
    persisted = pl.read_parquet(path)
    validate_flickr_scoring_geography(persisted, photo_units)
    assert persisted.equals(first)


def test_custom_cell_resolutions_and_fail_closed_lineage() -> None:
    records = [
        _record(
            "custom",
            latitude=-27.4705,
            longitude=153.026,
            accuracy=16,
            country_code="AU",
        )
    ]
    photo_units = _photo_units(records)
    geography = build_flickr_geography_frame(
        records,
        config=FlickrGeographyConfig(
            resolutions=GeographicResolutions(coarse=2, regional=4, local=6)
        ),
    )
    frame = build_flickr_scoring_geography(photo_units, geography)
    assert frame["supported_cell_resolution"].to_list() == [6]
    validate_flickr_scoring_geography(frame, photo_units)

    tampered = frame.with_columns(pl.lit("sha256:" + "0" * 64).alias("row_fingerprint"))
    with pytest.raises(ValueError, match="row fingerprint mismatch"):
        validate_flickr_scoring_geography(tampered, photo_units)
    with pytest.raises(ValueError, match="bioregion_mapping_version"):
        build_flickr_scoring_geography(
            photo_units,
            geography,
            bioregion_by_admin_region=(("AU:Queensland", "east"),),
        )


def _photo_units(records: list[dict[str, object]]) -> pl.DataFrame:
    image_by_id = {
        str(record["flickr_photo_id"]): DecodedImage(
            width=2,
            height=2,
            mode="RGB",
            data=hashlib.sha256(str(record["flickr_photo_id"]).encode()).digest()[:12],
        )
        for record in records
    }
    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row(
                str(record["flickr_photo_id"]),
                str(record["source_record_hash"]),
            )
            for record in records
        ],
        image_loader=lambda row: image_by_id[str(row["flickr_photo_id"])],
    )
    return build_flickr_scoring_unit_artifacts(
        plan,
        run_id="run-geography",
    ).photo_embedding_units


def _record(photo_id: str, **values: object) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": _source_hash(photo_id),
        **values,
    }


def _source_hash(photo_id: str) -> str:
    return "sha256:" + hashlib.sha256(photo_id.encode()).hexdigest()


def _detection_row(photo_id: str, source_record_hash: str) -> dict[str, Any]:
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
        "detection_route": "adult_butterfly_field",
        "routing_action": "score",
        "bioclip_route": "adult_field",
        "routing_policy_version": "detection-routing-policy-v1",
        "routing_policy_fingerprint": _ROUTING_POLICY_FINGERPRINT,
        "schema_version": "object-detection-v2",
    }
