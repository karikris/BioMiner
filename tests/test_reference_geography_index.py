"""Contract tests for the cached-reference geographic index."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from biominer.bioclip.reference_geography_index import (
    REFERENCE_GEOGRAPHY_INDEX_FILE,
    REFERENCE_GEOGRAPHY_INDEX_SCHEMA_VERSION,
    build_reference_geography_index,
    reference_geography_index_artifact_fingerprint,
    reference_geography_index_schema,
    validate_reference_geography_index,
    write_reference_geography_index,
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _row(suffix: str = "1", **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "registry_version": "butterflies-v2-20260712",
        "reference_bank_version": "reference-bank-v3",
        "reference_media_id": f"reference-media:{suffix * 64}",
        "reference_observation_id": f"reference-observation:{suffix * 64}",
        "source": "gbif",
        "source_dataset_key": "dataset-1",
        "accepted_taxon_key": "gbif:5131359",
        "scientific_name": "Papilio demoleus",
        "family_key": "gbif:9417",
        "family_name": "Papilionidae",
        "genus_key": "gbif:1920494",
        "genus_name": "Papilio",
        "route": "adult_field",
        "life_stage": "adult",
        "visual_domain": "live_field",
        "visual_input_kind": "raw_full_image",
        "country_code": "au",
        "admin1": "Queensland",
        "bioregion": "Wet Tropics",
        "geo_cluster_id": "geo-au-qld-wet-tropics",
        "coarse_cell_id": "h3-r3-a",
        "regional_cell_id": "h3-r5-a",
        "local_cell_id": "h3-r7-a",
        "latitude": -16.92,
        "longitude": 145.77,
        "coordinate_uncertainty_m": 25.0,
        "coordinate_quality": "local",
        "global_anchor_eligible": True,
        "local_anchor_eligible": True,
        "duplicate_group_id": f"reference-duplicate-group:{suffix * 32}",
        "observer_id_hash": _sha(suffix),
        "observation_date": "2026-01-03",
        "admission_mode": "adaptive_gbif_fast_start",
        "admission_policy_fingerprint": _sha("a"),
        "reference_quality_flags": ["observer_diversity_low", "provisional"],
        "embedding_fingerprint": _sha(suffix),
    }
    row.update(changes)
    return row


def test_schema_contains_every_required_reference_geography_field() -> None:
    schema = reference_geography_index_schema()

    assert list(schema) == [
        "schema_version",
        "registry_version",
        "reference_bank_version",
        "reference_media_id",
        "reference_observation_id",
        "source",
        "source_dataset_key",
        "accepted_taxon_key",
        "scientific_name",
        "family_key",
        "family_name",
        "genus_key",
        "genus_name",
        "route",
        "life_stage",
        "visual_domain",
        "visual_input_kind",
        "country_code",
        "admin1",
        "bioregion",
        "geo_cluster_id",
        "coarse_cell_id",
        "regional_cell_id",
        "local_cell_id",
        "latitude",
        "longitude",
        "coordinate_uncertainty_m",
        "coordinate_quality",
        "global_anchor_eligible",
        "local_anchor_eligible",
        "duplicate_group_id",
        "observer_id_hash",
        "observation_date",
        "admission_mode",
        "admission_policy_fingerprint",
        "reference_quality_flags",
        "embedding_fingerprint",
        "row_fingerprint",
    ]
    assert schema["reference_quality_flags"] == pl.List(pl.String)
    assert schema["observation_date"] == pl.Date


def test_build_is_canonical_and_order_independent() -> None:
    first = _row("1", reference_quality_flags=["provisional", "flag", "flag"])
    second = _row(
        "2",
        accepted_taxon_key="gbif:5131360",
        scientific_name="Papilio polytes",
        embedding_fingerprint=_sha("2"),
    )

    forward = build_reference_geography_index([first, second])
    reverse = build_reference_geography_index([second, first])

    assert forward.equals(reverse)
    assert forward["schema_version"].unique().to_list() == [
        REFERENCE_GEOGRAPHY_INDEX_SCHEMA_VERSION
    ]
    assert forward.row(0, named=True)["country_code"] == "AU"
    assert forward.row(0, named=True)["observation_date"] == date(2026, 1, 3)
    assert forward.row(0, named=True)["reference_quality_flags"] == [
        "flag",
        "provisional",
    ]
    assert reference_geography_index_artifact_fingerprint(forward) == (
        reference_geography_index_artifact_fingerprint(reverse)
    )


def test_write_uses_contract_filename_and_round_trips(tmp_path) -> None:
    frame = build_reference_geography_index([_row()])

    path = write_reference_geography_index(frame, tmp_path / "index")
    loaded = pl.read_parquet(path)

    assert path.name == REFERENCE_GEOGRAPHY_INDEX_FILE
    assert loaded.schema == reference_geography_index_schema()
    validate_reference_geography_index(loaded)
    assert reference_geography_index_artifact_fingerprint(loaded) == (
        reference_geography_index_artifact_fingerprint(frame)
    )


def test_global_only_reference_preserves_missing_geography() -> None:
    frame = build_reference_geography_index(
        [
            _row(
                country_code=None,
                admin1=None,
                bioregion=None,
                geo_cluster_id=None,
                coarse_cell_id=None,
                regional_cell_id=None,
                local_cell_id=None,
                latitude=None,
                longitude=None,
                coordinate_uncertainty_m=None,
                coordinate_quality="missing",
                global_anchor_eligible=True,
                local_anchor_eligible=False,
                observer_id_hash=None,
                observation_date=None,
            )
        ]
    )

    row = frame.row(0, named=True)
    assert row["global_anchor_eligible"] is True
    assert row["local_anchor_eligible"] is False
    assert row["latitude"] is None
    assert row["local_cell_id"] is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"longitude": None}, "both be set or null"),
        ({"latitude": 91.0}, "latitude is invalid"),
        ({"longitude": -181.0}, "longitude is invalid"),
        ({"coordinate_uncertainty_m": -1.0}, "finite and non-negative"),
        ({"local_cell_id": None}, "local coordinate quality"),
        ({"regional_cell_id": None}, "local_cell_id requires"),
        (
            {"coordinate_quality": "unknown_precision", "local_anchor_eligible": True},
            "cannot claim cell precision",
        ),
    ],
)
def test_rejects_invalid_geographic_semantics(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_reference_geography_index([_row(**changes)])


def test_country_only_reference_cannot_manufacture_local_support() -> None:
    with pytest.raises(ValueError, match="cannot carry coordinates or cells"):
        build_reference_geography_index(
            [
                _row(
                    coordinate_quality="country_only",
                    local_anchor_eligible=False,
                )
            ]
        )


def test_rejects_route_life_stage_or_domain_conflict() -> None:
    with pytest.raises(
        ValueError, match="route, life_stage and visual_domain conflict"
    ):
        build_reference_geography_index([_row(life_stage="larva")])


def test_rejects_unexpected_or_missing_input_fields() -> None:
    extra = _row()
    extra["path"] = "/tmp/reference.jpg"
    with pytest.raises(ValueError, match=r"unexpected=\['path'\]"):
        build_reference_geography_index([extra])

    missing = _row()
    missing.pop("embedding_fingerprint")
    with pytest.raises(ValueError, match=r"missing=\['embedding_fingerprint'\]"):
        build_reference_geography_index([missing])


def test_rejects_duplicate_index_grain() -> None:
    with pytest.raises(ValueError, match="grain is not unique"):
        build_reference_geography_index([_row(), _row()])


def test_validator_rejects_fingerprint_tampering() -> None:
    frame = build_reference_geography_index([_row()])
    tampered = frame.with_columns(pl.lit("Papilio polytes").alias("scientific_name"))

    with pytest.raises(ValueError, match="row fingerprint mismatch"):
        validate_reference_geography_index(tampered)


def test_validator_rejects_noncanonical_text_before_fingerprint() -> None:
    tampered = build_reference_geography_index([_row()]).with_columns(
        pl.lit(" gbif").alias("source")
    )

    with pytest.raises(ValueError, match="source is not canonically normalized"):
        validate_reference_geography_index(tampered)


def test_rejects_noncanonical_identity_hashes() -> None:
    with pytest.raises(ValueError, match="observer_id_hash"):
        build_reference_geography_index([_row(observer_id_hash="observer-1")])

    with pytest.raises(ValueError, match="embedding_fingerprint"):
        build_reference_geography_index([_row(embedding_fingerprint="abc")])
