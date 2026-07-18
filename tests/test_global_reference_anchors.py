"""Behavioral tests for deterministic global reference-anchor selection."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from biominer.bioclip.global_reference_anchors import (
    GLOBAL_REFERENCE_ANCHORS_FILE,
    GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION,
    GlobalReferenceAnchorPolicy,
    global_reference_anchors_artifact_fingerprint,
    global_reference_anchors_schema,
    select_global_reference_anchors,
    validate_global_reference_anchors,
    write_global_reference_anchors,
)
from biominer.bioclip.reference_geography_index import (
    build_reference_geography_index,
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _index_row(suffix: str, **changes: object) -> dict[str, object]:
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
        "country_code": "AU",
        "admin1": "Queensland",
        "bioregion": "Wet Tropics",
        "geo_cluster_id": f"geo-{suffix}",
        "coarse_cell_id": f"h3-r3-{suffix}",
        "regional_cell_id": f"h3-r5-{suffix}",
        "local_cell_id": f"h3-r7-{suffix}",
        "latitude": -16.92,
        "longitude": 145.77,
        "coordinate_uncertainty_m": 25.0,
        "coordinate_quality": "local",
        "global_anchor_eligible": True,
        "local_anchor_eligible": True,
        "duplicate_group_id": f"reference-duplicate-group:{suffix * 32}",
        "observer_id_hash": _sha(suffix),
        "observation_date": date(2026, 1, int(suffix, 16)),
        "admission_mode": "adaptive_gbif_fast_start",
        "admission_policy_fingerprint": _sha("a"),
        "reference_quality_flags": ["provisional"],
        "embedding_fingerprint": _sha(suffix),
    }
    row.update(changes)
    return row


def _index(*rows: dict[str, object]) -> pl.DataFrame:
    return build_reference_geography_index(list(rows))


def test_schema_carries_selection_diversity_shortfall_and_parent_provenance() -> None:
    assert list(global_reference_anchors_schema()) == [
        "schema_version",
        "registry_version",
        "reference_bank_version",
        "accepted_taxon_key",
        "scientific_name",
        "route",
        "visual_domain",
        "reference_media_id",
        "reference_observation_id",
        "duplicate_group_id",
        "observer_id_hash",
        "source",
        "country_code",
        "bioregion",
        "coarse_cell_id",
        "regional_cell_id",
        "local_cell_id",
        "coordinate_quality",
        "observation_date",
        "visual_input_kind",
        "view",
        "admission_mode",
        "reference_quality_flags",
        "embedding_fingerprint",
        "reference_geography_row_fingerprint",
        "anchor_group_id",
        "selection_rank",
        "group_selection_rank",
        "selection_round",
        "observer_reused",
        "geography_diversity_key",
        "date_diversity_key",
        "diversity_dimensions_added",
        "quality_flag_count",
        "eligible_observation_count",
        "requested_anchor_count",
        "selected_anchor_count",
        "anchor_shortfall",
        "selection_policy_fingerprint",
        "reference_geography_index_fingerprint",
        "row_fingerprint",
    ]


def test_selection_is_order_independent_and_excludes_ineligible_rows() -> None:
    rows = [
        _index_row("1"),
        _index_row("2", country_code="NZ", bioregion="Auckland"),
        _index_row("3", global_anchor_eligible=False),
    ]
    views = {
        str(row["reference_media_id"]): view
        for row, view in zip(rows, ["dorsal", "ventral", "lateral"], strict=True)
    }
    policy = GlobalReferenceAnchorPolicy(anchors_per_taxon_route=3)

    forward = select_global_reference_anchors(
        _index(*rows), views_by_media=views, policy=policy
    )
    reverse = select_global_reference_anchors(
        _index(*reversed(rows)), views_by_media=views, policy=policy
    )

    assert forward.equals(reverse)
    assert forward.height == 2
    assert set(forward["view"]) == {"dorsal", "ventral"}
    assert forward["schema_version"].unique().to_list() == [
        GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION
    ]
    assert forward["selected_anchor_count"].unique().to_list() == [2]
    assert forward["anchor_shortfall"].unique().to_list() == [1]
    assert global_reference_anchors_artifact_fingerprint(forward) == (
        global_reference_anchors_artifact_fingerprint(reverse)
    )


def test_one_observation_and_duplicate_group_can_fill_only_one_anchor() -> None:
    shared_observation = f"reference-observation:{'1' * 64}"
    shared_duplicate = f"reference-duplicate-group:{'2' * 32}"
    rows = [
        _index_row("1", duplicate_group_id=shared_duplicate),
        _index_row(
            "2",
            reference_observation_id=shared_observation,
            duplicate_group_id=shared_duplicate,
        ),
        _index_row("3", duplicate_group_id=shared_duplicate),
        _index_row("4"),
    ]

    anchors = select_global_reference_anchors(
        _index(*rows),
        policy=GlobalReferenceAnchorPolicy(anchors_per_taxon_route=4),
    )

    assert anchors.height == 2
    assert anchors["reference_observation_id"].n_unique() == 2
    assert anchors["duplicate_group_id"].n_unique() == 2
    assert anchors["eligible_observation_count"].unique().to_list() == [3]
    assert anchors["anchor_shortfall"].unique().to_list() == [2]


def test_uses_each_known_photographer_before_reuse() -> None:
    rows = [
        _index_row("1", observer_id_hash=_sha("a")),
        _index_row("2", observer_id_hash=_sha("a")),
        _index_row("3", observer_id_hash=_sha("b")),
    ]

    anchors = select_global_reference_anchors(
        _index(*rows),
        policy=GlobalReferenceAnchorPolicy(anchors_per_taxon_route=3),
    )

    assert anchors["selection_round"].to_list() == [
        "independent_photographer",
        "independent_photographer",
        "photographer_reuse",
    ]
    assert anchors["observer_reused"].to_list() == [False, False, True]
    assert anchors["observer_id_hash"][:2].n_unique() == 2


def test_known_photographer_precedes_missing_photographer() -> None:
    rows = [
        _index_row("1", observer_id_hash=None),
        _index_row("2", observer_id_hash=_sha("b")),
    ]

    anchors = select_global_reference_anchors(
        _index(*rows),
        policy=GlobalReferenceAnchorPolicy(anchors_per_taxon_route=2),
    )

    assert anchors["reference_media_id"].to_list()[0].endswith("2" * 64)
    assert anchors["selection_round"].to_list() == [
        "independent_photographer",
        "photographer_unavailable",
    ]


def test_greedy_selection_records_geography_date_view_and_input_gains() -> None:
    rows = [
        _index_row("1", observer_id_hash=_sha("a")),
        _index_row(
            "2",
            observer_id_hash=_sha("b"),
            country_code="NZ",
            bioregion="Auckland",
            coarse_cell_id="h3-r3-nz",
            regional_cell_id="h3-r5-nz",
            local_cell_id="h3-r7-nz",
            observation_date=date(2025, 6, 2),
            visual_input_kind="focused_full_frame",
        ),
    ]
    views = {
        str(rows[0]["reference_media_id"]): "dorsal",
        str(rows[1]["reference_media_id"]): "ventral",
    }

    anchors = select_global_reference_anchors(
        _index(*rows),
        views_by_media=views,
        policy=GlobalReferenceAnchorPolicy(anchors_per_taxon_route=2),
    )

    first, second = anchors.iter_rows(named=True)
    assert first["diversity_dimensions_added"] == [
        "route",
        "visual_domain",
        "photographer",
        "country",
        "bioregion",
        "coarse_cell",
        "regional_cell",
        "local_cell",
        "observation_month",
        "visual_view",
        "visual_input_kind",
    ]
    assert second["diversity_dimensions_added"] == [
        "photographer",
        "country",
        "bioregion",
        "coarse_cell",
        "regional_cell",
        "local_cell",
        "observation_month",
        "visual_view",
        "visual_input_kind",
    ]
    assert second["geography_diversity_key"] == "local_cell:h3-r7-nz"
    assert second["date_diversity_key"] == "month:2025-06"


def test_quality_flags_break_equal_diversity_ties_without_becoming_probability() -> (
    None
):
    rows = [
        _index_row("1", reference_quality_flags=["blur", "provisional"]),
        _index_row(
            "2",
            reference_quality_flags=[],
        ),
    ]

    anchors = select_global_reference_anchors(
        _index(*rows),
        policy=GlobalReferenceAnchorPolicy(anchors_per_taxon_route=1),
    )

    row = anchors.row(0, named=True)
    assert row["reference_media_id"].endswith("2" * 64)
    assert row["quality_flag_count"] == 0
    assert row["admission_mode"] == "adaptive_gbif_fast_start"
    assert "probability" not in anchors.columns


def test_route_domain_groups_have_independent_quotas() -> None:
    anchors = select_global_reference_anchors(
        _index(
            _index_row("1"),
            _index_row(
                "2",
                route="pinned_specimen",
                life_stage="unknown",
                visual_domain="pinned_specimen",
            ),
        ),
        policy=GlobalReferenceAnchorPolicy(anchors_per_taxon_route=1),
    )

    assert anchors.height == 2
    assert set(anchors["route"]) == {"adult_field", "pinned_specimen"}
    assert anchors["group_selection_rank"].to_list() == [1, 1]
    assert anchors["selected_anchor_count"].to_list() == [1, 1]


def test_missing_views_remain_explicit_and_do_not_claim_view_diversity() -> None:
    anchors = select_global_reference_anchors(
        _index(_index_row("1")),
        policy=GlobalReferenceAnchorPolicy(anchors_per_taxon_route=1),
    )

    row = anchors.row(0, named=True)
    assert row["view"] == "unknown"
    assert "visual_view" not in row["diversity_dimensions_added"]


def test_rejects_stale_or_invalid_view_mappings() -> None:
    index = _index(_index_row("1"))
    with pytest.raises(ValueError, match="unknown reference media"):
        select_global_reference_anchors(index, views_by_media={"media:stale": "dorsal"})
    with pytest.raises(ValueError, match="unsupported global-anchor view"):
        select_global_reference_anchors(
            index,
            views_by_media={str(index["reference_media_id"][0]): "top"},
        )


def test_rejects_observation_or_duplicate_identity_conflicts() -> None:
    shared_observation = f"reference-observation:{'1' * 64}"
    with pytest.raises(ValueError, match="observation spans conflicting"):
        select_global_reference_anchors(
            _index(
                _index_row("1"),
                _index_row(
                    "2",
                    reference_observation_id=shared_observation,
                    accepted_taxon_key="gbif:other",
                    scientific_name="Papilio other",
                ),
            )
        )

    shared_duplicate = f"reference-duplicate-group:{'3' * 32}"
    with pytest.raises(ValueError, match="duplicate group spans conflicting"):
        select_global_reference_anchors(
            _index(
                _index_row("3", duplicate_group_id=shared_duplicate),
                _index_row(
                    "4",
                    duplicate_group_id=shared_duplicate,
                    route="pinned_specimen",
                    life_stage="unknown",
                    visual_domain="pinned_specimen",
                ),
            )
        )


def test_write_round_trip_and_validator_detects_tampering(tmp_path) -> None:
    anchors = select_global_reference_anchors(_index(_index_row("1")))
    path = write_global_reference_anchors(anchors, tmp_path / "anchors")
    loaded = pl.read_parquet(path)

    assert path.name == GLOBAL_REFERENCE_ANCHORS_FILE
    assert loaded.schema == global_reference_anchors_schema()
    validate_global_reference_anchors(loaded)
    assert global_reference_anchors_artifact_fingerprint(loaded) == (
        global_reference_anchors_artifact_fingerprint(anchors)
    )

    tampered = loaded.with_columns(pl.lit(0).cast(pl.UInt32).alias("anchor_shortfall"))
    with pytest.raises(ValueError, match="row fingerprint mismatch"):
        validate_global_reference_anchors(tampered)


def test_empty_index_produces_closed_empty_artifact() -> None:
    anchors = select_global_reference_anchors(build_reference_geography_index([]))

    assert anchors.is_empty()
    assert anchors.schema == global_reference_anchors_schema()
    validate_global_reference_anchors(anchors)


@pytest.mark.parametrize("quota", [0, -1, True, 1.5])
def test_policy_rejects_invalid_quota(quota: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GlobalReferenceAnchorPolicy(anchors_per_taxon_route=quota)  # type: ignore[arg-type]
