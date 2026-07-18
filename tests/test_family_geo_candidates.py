"""Contract tests for target-preserving family/geography candidate sets."""

from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.family_geo_candidates import (
    FAMILY_GEO_CANDIDATE_FILE,
    build_family_geo_candidate_sets,
    family_geo_candidate_schema,
    validate_family_geo_candidate_sets,
    write_family_geo_candidate_sets,
)


TARGET = "gbif:5131359"
COMPETITOR = "gbif:5131360"


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _row(
    *, target: bool, priority: int, candidate_key: str, **changes: object
) -> dict[str, object]:
    visual = not target
    row: dict[str, object] = {
        "run_id": "run-20260718",
        "flickr_query_id": "query-papilio-demoleus",
        "flickr_photo_id": "flickr-photo-1",
        "organism_unit_id": "organism-unit-1",
        "scoring_stage": "initial",
        "registry_version": "butterflies-v2-20260712",
        "target_accepted_taxon_key": TARGET,
        "target_scientific_name": "Papilio demoleus",
        "query_geo_cluster_id": "geo-au-qld",
        "query_coordinate_quality": "local",
        "candidate_accepted_taxon_key": candidate_key,
        "candidate_scientific_name": (
            "Papilio demoleus" if target else "Papilio polytes"
        ),
        "family_key": "gbif:9417",
        "family_name": "Papilionidae",
        "genus_key": "gbif:1920494",
        "genus_name": "Papilio",
        "candidate_priority": priority,
        "candidate_reasons": ["target"] if target else ["visually_nearest"],
        "family_evidence_status": "available",
        "family_evidence_reason": None,
        "family_evidence_rank": priority + 1,
        "family_evidence_raw_score": 0.9 - priority / 10,
        "family_priority_match": True,
        "family_changed_membership": False,
        "geographic_evidence_status": "available",
        "geographic_evidence_reason": None,
        "geographic_scopes": ["exact_local_cell"],
        "geographic_evidence_score": 0.8 - priority / 10,
        "occurrence_support": 5 - priority,
        "query_evidence_status": "available" if target else "not_applicable",
        "query_evidence_reason": None if target else "not_query_associated",
        "query_evidence_ids": ["query-evidence-1"] if target else [],
        "query_associated": target,
        "visual_neighbour_evidence_status": (
            "available" if visual else "not_applicable"
        ),
        "visual_neighbour_evidence_reason": (
            None if visual else "not_a_visual_neighbour"
        ),
        "visual_neighbour_graph_fingerprint": _sha("a") if visual else None,
        "visual_neighbour_rank": 1 if visual else None,
        "visual_neighbour_raw_similarity": 0.72 if visual else None,
        "visual_neighbour": visual,
        "safety_union_membership": True,
        "safety_union_reasons": ["target"] if target else ["visual_neighbour"],
        "target_candidate": target,
        "target_preserved": True,
        "included_in_complete_union": True,
        "source_versions": ["registry:v2", "regional-candidate:v1"],
    }
    row.update(changes)
    return row


def _rows() -> list[dict[str, object]]:
    return [
        _row(target=True, priority=0, candidate_key=TARGET),
        _row(target=False, priority=1, candidate_key=COMPETITOR),
    ]


def test_schema_exposes_complete_union_and_all_evidence_axes() -> None:
    fields = set(family_geo_candidate_schema())

    assert {
        "candidate_reasons",
        "family_evidence_status",
        "family_evidence_raw_score",
        "family_changed_membership",
        "geographic_evidence_status",
        "geographic_scopes",
        "query_evidence_status",
        "query_evidence_ids",
        "visual_neighbour_evidence_status",
        "visual_neighbour_graph_fingerprint",
        "safety_union_membership",
        "target_preserved",
        "included_in_complete_union",
        "candidate_row_fingerprint",
        "candidate_set_fingerprint",
    } <= fields


def test_build_is_deterministic_and_preserves_target_and_safety_union() -> None:
    forward = build_family_geo_candidate_sets(_rows())
    reverse = build_family_geo_candidate_sets(list(reversed(_rows())))

    assert forward.equals(reverse)
    assert forward["candidate_set_id"].n_unique() == 1
    assert forward["candidate_set_fingerprint"].n_unique() == 1
    assert forward["target_preserved"].to_list() == [True, True]
    assert forward["included_in_complete_union"].to_list() == [True, True]
    target = forward.filter(pl.col("target_candidate")).row(0, named=True)
    assert target["candidate_accepted_taxon_key"] == TARGET
    assert target["safety_union_reasons"] == ["target"]


def test_family_evidence_is_diagnostic_and_cannot_change_membership() -> None:
    with pytest.raises(ValueError, match="must not change complete-union membership"):
        build_family_geo_candidate_sets(
            [
                _rows()[0],
                {
                    **_rows()[1],
                    "family_changed_membership": True,
                },
            ]
        )


def test_unavailable_family_and_geography_evidence_stay_null() -> None:
    rows = _rows()
    rows[1].update(
        family_evidence_status="unavailable",
        family_evidence_reason="family_model_not_run",
        family_evidence_rank=None,
        family_evidence_raw_score=None,
        family_priority_match=None,
        geographic_evidence_status="unavailable",
        geographic_evidence_reason="no_geo_global_fallback",
        geographic_scopes=[],
        geographic_evidence_score=None,
        occurrence_support=0,
    )

    result = build_family_geo_candidate_sets(rows)
    competitor = result.filter(
        pl.col("candidate_accepted_taxon_key") == COMPETITOR
    ).row(0, named=True)

    assert competitor["family_evidence_raw_score"] is None
    assert competitor["geographic_evidence_score"] is None
    assert competitor["occurrence_support"] == 0


def test_writer_uses_required_parquet_filename(tmp_path) -> None:
    frame = build_family_geo_candidate_sets(_rows())

    path = write_family_geo_candidate_sets(frame, tmp_path / "candidates")
    loaded = pl.read_parquet(path)

    assert path.name == FAMILY_GEO_CANDIDATE_FILE
    assert loaded.schema == family_geo_candidate_schema()
    validate_family_geo_candidate_sets(loaded)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"included_in_complete_union": False}, "complete union"),
        ({"safety_union_reasons": []}, "membership and reasons"),
        (
            {
                "query_evidence_status": "available",
                "query_evidence_reason": None,
                "query_associated": True,
                "query_evidence_ids": [],
            },
            "query evidence requires IDs",
        ),
        (
            {"visual_neighbour_graph_fingerprint": None},
            "visual-neighbour evidence is incomplete",
        ),
        ({"geographic_evidence_score": 1.5}, "finite and in"),
        ({"family_evidence_status": "verified"}, "unsupported"),
    ],
)
def test_rejects_invalid_candidate_evidence(
    changes: dict[str, object], message: str
) -> None:
    rows = _rows()
    rows[1].update(changes)
    with pytest.raises(ValueError, match=message):
        build_family_geo_candidate_sets(rows)


def test_requires_exactly_one_matching_target() -> None:
    without_target = [_rows()[1]]
    without_target[0]["candidate_priority"] = 0
    with pytest.raises(ValueError, match="exactly one target"):
        build_family_geo_candidate_sets(without_target)

    wrong_target = _rows()
    wrong_target[0]["candidate_accepted_taxon_key"] = "gbif:wrong"
    with pytest.raises(ValueError, match="does not match target"):
        build_family_geo_candidate_sets(wrong_target)


def test_priorities_are_contiguous_and_taxa_unique() -> None:
    noncontiguous = _rows()
    noncontiguous[1]["candidate_priority"] = 2
    with pytest.raises(ValueError, match="contiguous from zero"):
        build_family_geo_candidate_sets(noncontiguous)

    duplicate = _rows()
    duplicate[1]["candidate_accepted_taxon_key"] = TARGET
    with pytest.raises(ValueError, match="duplicate taxa"):
        build_family_geo_candidate_sets(duplicate)


def test_validator_rejects_row_and_set_fingerprint_tampering() -> None:
    frame = build_family_geo_candidate_sets(_rows())
    first_key = frame["candidate_accepted_taxon_key"][0]
    row_tampered = frame.with_columns(
        pl.when(pl.col("candidate_accepted_taxon_key") == first_key)
        .then(pl.lit(_sha("f")))
        .otherwise(pl.col("candidate_row_fingerprint"))
        .alias("candidate_row_fingerprint")
    )
    with pytest.raises(ValueError, match="row fingerprint mismatch"):
        validate_family_geo_candidate_sets(row_tampered)

    set_tampered = frame.with_columns(
        pl.lit(_sha("f")).alias("candidate_set_fingerprint")
    )
    with pytest.raises(ValueError, match="set fingerprint mismatch"):
        validate_family_geo_candidate_sets(set_tampered)
