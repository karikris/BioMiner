from __future__ import annotations

import polars as pl
import pytest

from biominer.flickr_fetch.geography import (
    FLICKR_GEOGRAPHY_SCHEMA_VERSION,
    FlickrGeographyConfig,
    build_flickr_geography_frame,
    flickr_geography_schema,
    geography_config_fingerprint,
    write_flickr_geography,
)
from biominer.geography import GeographicResolutions, cell_parent, is_valid_cell


def _record(photo_id: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": f"sha256:{photo_id.zfill(64)}",
        "latitude": -27.4705,
        "longitude": 153.026,
        "accuracy": 16,
    }
    record.update(overrides)
    return record


def test_normalizes_flickr_geography_and_gates_cells_by_source_precision() -> None:
    frame = build_flickr_geography_frame(
        [
            _record("4", accuracy=3, country="au"),
            _record("2", accuracy=11, country_code="au", admin1="Queensland"),
            _record("1", accuracy=16, location={"region": {"_content": "Queensland"}}),
            _record("3", accuracy=6),
        ]
    )

    assert frame.schema == flickr_geography_schema()
    assert frame["flickr_photo_id"].to_list() == ["1", "2", "3", "4"]
    rows = {row["flickr_photo_id"]: row for row in frame.to_dicts()}

    street = rows["1"]
    assert street["schema_version"] == FLICKR_GEOGRAPHY_SCHEMA_VERSION
    assert street["coordinate_source"] == "flickr_search_geo"
    assert street["coordinate_quality"] == "flickr_street"
    assert street["admin1"] == "Queensland"
    assert street["geotag_available"] is True
    assert street["geography_warnings"] == []
    assert all(
        is_valid_cell(street[field])
        for field in ("coarse_cell_id", "regional_cell_id", "local_cell_id")
    )
    assert street["coarse_cell_id"] == cell_parent(street["local_cell_id"], resolution=3)
    assert street["regional_cell_id"] == cell_parent(street["local_cell_id"], resolution=5)

    city = rows["2"]
    assert city["coordinate_quality"] == "flickr_city"
    assert city["country_code"] == "AU"
    assert city["coarse_cell_id"] is not None
    assert city["regional_cell_id"] is not None
    assert city["local_cell_id"] is None
    assert city["geography_warning"] == "coordinate_precision_limits_cells"

    region = rows["3"]
    assert region["coordinate_quality"] == "flickr_region"
    assert region["coarse_cell_id"] is not None
    assert region["regional_cell_id"] is None
    assert region["local_cell_id"] is None

    country = rows["4"]
    assert country["coordinate_quality"] == "flickr_country"
    assert country["country_code"] == "AU"
    assert country["coarse_cell_id"] is None
    assert country["regional_cell_id"] is None
    assert country["local_cell_id"] is None


def test_invalid_incomplete_and_missing_coordinates_remain_explicit() -> None:
    frame = build_flickr_geography_frame(
        [
            _record("missing", latitude=None, longitude=None, accuracy=None),
            _record("partial", longitude=None),
            _record("bad-latitude", latitude="south"),
            _record("bad-longitude", longitude=181),
        ]
    )
    rows = {row["flickr_photo_id"]: row for row in frame.to_dicts()}

    missing = rows["missing"]
    assert missing["geotag_available"] is False
    assert missing["coordinate_quality"] == "missing"
    assert missing["coordinate_source"] is None
    assert missing["geography_warning"] == "coordinates_missing"

    for photo_id, warning in (
        ("partial", "coordinate_pair_incomplete"),
        ("bad-latitude", "invalid_latitude"),
        ("bad-longitude", "invalid_longitude"),
    ):
        row = rows[photo_id]
        assert row["geotag_available"] is False
        assert row["coordinate_quality"] == "invalid"
        assert row["latitude"] is None
        assert row["longitude"] is None
        assert row["coarse_cell_id"] is None
        assert row["geography_warning"] == warning


def test_unknown_accuracy_preserves_value_without_claiming_spatial_precision() -> None:
    frame = build_flickr_geography_frame(
        [
            _record("missing-accuracy", accuracy=None),
            _record("nonintegral", accuracy=11.5),
            _record("out-of-range", accuracy=17, country_code="Australia"),
            _record("null-island", latitude=0, longitude=0, accuracy=16),
        ]
    )
    rows = {row["flickr_photo_id"]: row for row in frame.to_dicts()}

    assert rows["missing-accuracy"]["coordinate_accuracy"] is None
    assert rows["missing-accuracy"]["coordinate_quality"] == "unknown_precision"
    assert rows["missing-accuracy"]["local_cell_id"] is None
    assert rows["missing-accuracy"]["geography_warning"] == "coordinate_precision_unknown"

    assert rows["nonintegral"]["coordinate_accuracy"] == 11.5
    assert rows["nonintegral"]["coordinate_quality"] == "unknown_precision"
    assert rows["nonintegral"]["geography_warning"] == "coordinate_accuracy_nonintegral"

    out_of_range = rows["out-of-range"]
    assert out_of_range["coordinate_accuracy"] == 17.0
    assert out_of_range["country_code"] is None
    assert out_of_range["geotag_available"] is True
    assert out_of_range["coarse_cell_id"] is None
    assert out_of_range["geography_warning"] == "coordinate_accuracy_out_of_range"
    assert set(out_of_range["geography_warnings"]) == {
        "coordinate_accuracy_out_of_range",
        "coordinate_precision_unknown",
        "country_code_invalid",
    }

    assert rows["null-island"]["geotag_available"] is True
    assert rows["null-island"]["local_cell_id"] is not None
    assert rows["null-island"]["geography_warning"] == "coordinate_at_null_island"


def test_flickr_zero_geo_sentinel_is_missing_not_null_island() -> None:
    row = build_flickr_geography_frame(
        [_record("sentinel", latitude=0, longitude=0, accuracy=0)]
    ).to_dicts()[0]

    assert row["latitude"] is None
    assert row["longitude"] is None
    assert row["coordinate_accuracy"] == 0.0
    assert row["coordinate_source"] is None
    assert row["geotag_available"] is False
    assert row["coordinate_quality"] == "missing"
    assert row["coarse_cell_id"] is None
    assert row["regional_cell_id"] is None
    assert row["local_cell_id"] is None
    assert row["geography_warning"] == "flickr_zero_geo_sentinel"
    assert row["geography_warnings"] == [
        "coordinate_accuracy_out_of_range",
        "flickr_zero_geo_sentinel",
    ]


def test_nested_flickr_location_and_explicit_coordinate_source_are_preserved() -> None:
    frame = build_flickr_geography_frame(
        [
            {
                "source": "FLICKR",
                "id": "nested",
                "source_record_hash": "sha256:nested",
                "location": {
                    "latitude": "-33.8688",
                    "longitude": "151.2093",
                    "country_code": "au",
                    "region": {"_content": "New South Wales"},
                },
                "coordinate_accuracy": "16",
                "coordinate_source": "flickr.photos.geo.getLocation",
            }
        ]
    )
    row = frame.to_dicts()[0]

    assert row["source"] == "flickr"
    assert row["flickr_photo_id"] == "nested"
    assert row["latitude"] == pytest.approx(-33.8688)
    assert row["longitude"] == pytest.approx(151.2093)
    assert row["coordinate_accuracy"] == 16.0
    assert row["coordinate_source"] == "flickr.photos.geo.getLocation"
    assert row["country_code"] == "AU"
    assert row["admin1"] == "New South Wales"


def test_fingerprint_tracks_semantic_configuration_and_writer_is_atomic(tmp_path) -> None:
    default = FlickrGeographyConfig()
    changed = FlickrGeographyConfig(
        resolutions=GeographicResolutions(coarse=2, regional=4, local=6)
    )
    assert geography_config_fingerprint(default) == geography_config_fingerprint(default)
    assert geography_config_fingerprint(default) != geography_config_fingerprint(changed)

    output = tmp_path / "flickr_geography.parquet"
    assert write_flickr_geography([_record("1")], output) == output
    persisted = pl.read_parquet(output)
    assert persisted.schema == flickr_geography_schema()
    assert persisted["flickr_photo_id"].to_list() == ["1"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_rejects_missing_provenance_duplicate_identity_and_non_flickr_source() -> None:
    with pytest.raises(ValueError, match="source_record_hash"):
        build_flickr_geography_frame([_record("1", source_record_hash="")])
    with pytest.raises(ValueError, match="duplicate Flickr geography identity"):
        build_flickr_geography_frame([_record("1"), _record("1")])
    with pytest.raises(ValueError, match="source must be 'flickr'"):
        build_flickr_geography_frame([_record("1", source="gbif")])


def test_empty_projection_retains_typed_schema() -> None:
    frame = build_flickr_geography_frame(pl.DataFrame())
    assert frame.is_empty()
    assert frame.schema == flickr_geography_schema()
