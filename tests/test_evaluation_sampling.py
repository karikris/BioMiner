from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import polars as pl
import pytest

from biominer.cli import build_parser, run
from biominer.evaluation.sampling import (
    EVALUATION_SAMPLING_FRAME_SCHEMA,
    EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION,
    EvaluationSamplingConfig,
    build_evaluation_sampling_frame,
    empty_evaluation_sampling_frame,
    materialize_evaluation_sampling_frame,
    read_flickr_query_definitions,
)


def test_empty_sampling_frame_has_versioned_schema() -> None:
    frame = empty_evaluation_sampling_frame()

    assert frame.schema == EVALUATION_SAMPLING_FRAME_SCHEMA
    assert frame.is_empty()


def test_sampling_frame_stratifies_real_provenance_and_initial_scores() -> None:
    frame = build_evaluation_sampling_frame(
        _candidates(),
        _geo_assignments(),
        _query_definitions(),
        object_scores=_object_scores(),
        competitor_taxa=_competitor_taxa(),
        config=EvaluationSamplingConfig(target_text_terms=("Papilio demoleus",)),
    )

    assert frame.schema == EVALUATION_SAMPLING_FRAME_SCHEMA
    assert frame.height == 4
    by_id = {row["flickr_photo_id"]: row for row in frame.to_dicts()}

    adult = by_id["photo-a"]
    assert adult["schema_version"] == EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION
    assert adult["primary_query_tier"] == "T1"
    assert adult["primary_query_term"] == "Papilio demoleus"
    assert adult["query_tiers"] == ["T1", "T5"]
    assert adult["query_terms"] == ["Papilio demoleus", "Lime Butterfly"]
    assert [item["query_definition_id"] for item in adult["query_provenance"]] == [
        "query-1",
        "query-2",
    ]
    assert adult["metadata_target_text_evidence"] is True
    assert adult["initial_reference_score"] == pytest.approx(0.9)
    assert adult["initial_reference_score_tail"] == "high_tail"
    assert adult["initial_competitor_margin_tail"] == "high_tail"
    assert adult["subject_area_band"] == "large"
    assert adult["initial_visual_domain"] == "live_field"
    assert adult["text_image_reference_disagreement"] == "agreement_target"

    false_winner = by_id["photo-c"]
    assert false_winner["current_false_positive_genus"] == "Graphium"
    assert false_winner["false_positive_genus_stratum"] == "Graphium"
    assert false_winner["text_image_reference_disagreement"] == (
        "disagreement_text_target_reference_competitor"
    )

    unscored = by_id["photo-d"]
    assert unscored["primary_query_term"] == "Papilio demoleus"
    assert unscored["metadata_target_text_evidence"] is False
    assert unscored["initial_score_status"] == "not_scored"
    assert unscored["yoloe_route"] == "not_run"
    assert unscored["subject_area_band"] == "not_measured"
    assert unscored["initial_reference_score_tail"] == "not_scored"
    assert unscored["initial_competitor_margin_tail"] == "not_scored"
    assert unscored["initial_visual_domain"] == "artwork"
    assert unscored["visual_domain_source"] == "metadata_keyword_heuristic"

    assert set(frame["sampling_rank"].to_list()) == {1, 2, 3, 4}
    assert not {
        "all_query_labels",
        "is_target_positive",
        "target_present",
        "accepted_taxon_key",
    } & set(frame.columns)


def test_sampling_frame_is_deterministic_under_input_reordering() -> None:
    config = EvaluationSamplingConfig(
        target_text_terms=("Papilio demoleus",),
        random_seed=20260714,
    )
    first = build_evaluation_sampling_frame(
        _candidates(),
        _geo_assignments(),
        _query_definitions(),
        object_scores=_object_scores(),
        competitor_taxa=_competitor_taxa(),
        config=config,
    )
    second = build_evaluation_sampling_frame(
        _candidates().reverse(),
        _geo_assignments().reverse(),
        _query_definitions().reverse(),
        object_scores=_object_scores().reverse(),
        competitor_taxa=_competitor_taxa().reverse(),
        config=config,
    )

    assert first.equals(second)


def test_sampling_frame_requires_all_query_definitions() -> None:
    definitions = _query_definitions().filter(
        pl.col("query_definition_id") != "query-4"
    )

    with pytest.raises(ValueError, match="missing from Flickr state"):
        build_evaluation_sampling_frame(
            _candidates(),
            _geo_assignments(),
            definitions,
        )


def test_sampling_frame_decodes_flickr_upload_epoch_for_year_fallback() -> None:
    candidates = _candidates().with_columns(
        pl.when(pl.col("flickr_photo_id") == "photo-d")
        .then(pl.lit(""))
        .otherwise(pl.col("date_taken"))
        .alias("date_taken")
    )

    frame = build_evaluation_sampling_frame(
        candidates,
        _geo_assignments(),
        _query_definitions(),
    )
    row = frame.filter(pl.col("flickr_photo_id") == "photo-d").row(0, named=True)

    assert row["year"] == 2023
    assert row["year_source"] == "date_upload"
    assert row["year_stratum"] == "year:2023"


def test_sampling_frame_requires_exact_geo_assignment_set() -> None:
    geo = _geo_assignments().filter(pl.col("flickr_photo_id") != "photo-d")

    with pytest.raises(ValueError, match="exactly one row for every candidate"):
        build_evaluation_sampling_frame(
            _candidates(),
            geo,
            _query_definitions(),
        )


def test_sampling_frame_requires_taxonomy_for_false_winning_competitors() -> None:
    with pytest.raises(ValueError, match="competitor_taxa is required"):
        build_evaluation_sampling_frame(
            _candidates(),
            _geo_assignments(),
            _query_definitions(),
            object_scores=_object_scores(),
        )


def test_sampling_frame_rejects_out_of_range_subject_area() -> None:
    scores = _object_scores().with_columns(
        pl.when(pl.col("target_score_id") == "score-b")
        .then(pl.lit(1.1))
        .otherwise(pl.col("subject_area_ratio"))
        .alias("subject_area_ratio")
    )

    with pytest.raises(ValueError, match="subject_area_ratio"):
        build_evaluation_sampling_frame(
            _candidates(),
            _geo_assignments(),
            _query_definitions(),
            object_scores=scores,
            competitor_taxa=_competitor_taxa(),
        )


def test_read_flickr_query_definitions_deduplicates_pages(tmp_path: Path) -> None:
    state_db = tmp_path / "state.sqlite"
    _write_query_state(state_db)

    frame = read_flickr_query_definitions(state_db)

    assert frame.height == 4
    assert frame.filter(pl.col("query_definition_id") == "query-1").row(
        0,
        named=True,
    ) == {
        "query_definition_id": "query-1",
        "query_tier": "T1",
        "query_term": "Papilio demoleus",
        "query_field": "tags",
        "query_priority": 10,
    }


def test_materialize_sampling_frame_writes_parquet_and_compact_reports(
    tmp_path: Path,
) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    geo_path = tmp_path / "geo.parquet"
    state_db = tmp_path / "state.sqlite"
    output = tmp_path / "papilio_demoleus_evaluation_sampling_frame.parquet"
    _candidates().write_parquet(candidates_path)
    _geo_assignments().write_parquet(geo_path)
    _write_query_state(state_db)

    publication = materialize_evaluation_sampling_frame(
        candidates_path=candidates_path,
        geo_assignments_path=geo_path,
        query_state_db=state_db,
        output_path=output,
        config=EvaluationSamplingConfig(target_text_terms=("Papilio demoleus",)),
        run_id="sampling-test",
    )

    assert publication.frame_path == output
    assert publication.report_json_path == output.with_suffix(".report.json")
    assert publication.report_markdown_path == output.with_suffix(".report.md")
    assert pl.read_parquet(output).height == 4
    report = json.loads(publication.report_json_path.read_text(encoding="utf-8"))
    assert report["command"] == "evaluation.build_sampling_frame"
    assert report["run_id"] == "sampling-test"
    assert report["rows_out"] == 4
    assert report["scored_count"] == 0
    assert report["unscored_count"] == 4
    assert report["counts_by_query_tier"] == {
        "T1": 1,
        "T2": 1,
        "T3": 1,
        "T4": 0,
        "T5": 1,
    }
    assert report["artifact"]["sha256"].startswith("sha256:")


def test_sampling_frame_cli_parses_generic_target_inputs() -> None:
    args = build_parser().parse_args(
        [
            "evaluation",
            "build-sampling-frame",
            "--candidates",
            "candidates.parquet",
            "--geo-assignments",
            "geo.parquet",
            "--query-state-db",
            "poller.sqlite",
            "--target-text-term",
            "Papilio demoleus",
            "--random-seed",
            "17",
            "--output",
            "sampling.parquet",
        ]
    )

    assert args.command == "evaluation"
    assert args.evaluation_command == "build-sampling-frame"
    assert args.target_text_term == ["Papilio demoleus"]
    assert args.random_seed == 17


def test_sampling_frame_cli_materializes_unscored_register(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    geo_path = tmp_path / "geo.parquet"
    state_db = tmp_path / "state.sqlite"
    output = tmp_path / "sampling.parquet"
    _candidates().write_parquet(candidates_path)
    _geo_assignments().write_parquet(geo_path)
    _write_query_state(state_db)
    args = build_parser().parse_args(
        [
            "evaluation",
            "build-sampling-frame",
            "--candidates",
            str(candidates_path),
            "--geo-assignments",
            str(geo_path),
            "--query-state-db",
            str(state_db),
            "--target-text-term",
            "Papilio demoleus",
            "--run-id",
            "sampling-cli-test",
            "--output",
            str(output),
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["metrics"] == {
        "no_geo": 3,
        "rows": 4,
        "scored": 0,
        "unscored": 4,
    }
    assert output.exists()


def _candidates() -> pl.DataFrame:
    rows = [
        _candidate(
            "photo-a",
            query_ids=["query-2", "query-1"],
            title="Papilio demoleus on a leaf",
        ),
        _candidate(
            "photo-b",
            query_ids=["query-2"],
            title="Papilio polytes",
        ),
        _candidate(
            "photo-c",
            query_ids=["query-3"],
            description="A field record of Papilio demoleus",
            image_category="life_stage_non_adult",
            life_stage="caterpillar",
        ),
        _candidate(
            "photo-d",
            query_ids=["query-4"],
            title="Watercolour study",
            image_category="artwork",
        ),
    ]
    return pl.DataFrame(rows)


def _candidate(
    photo_id: str,
    *,
    query_ids: list[str],
    title: str = "",
    description: str = "",
    image_category: str = "unknown",
    life_stage: str = "unknown",
) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": f"hash-{photo_id}",
        "photo_page_url": f"https://example.test/photos/{photo_id}",
        "image_url": f"https://example.test/images/{photo_id}.jpg",
        "owner_id": f"owner-{photo_id[-1]}",
        "owner_name": f"Owner {photo_id[-1].upper()}",
        "date_taken": f"202{ord(photo_id[-1]) - ord('a')}-06-01 12:00:00",
        "date_upload": "1700000000",
        "raw_title": title,
        "raw_description": description,
        "raw_tags": "field observation",
        "machine_tags": "",
        "query_definition_ids": query_ids,
        "query_hit_count": len(query_ids),
        "image_category": image_category,
        "life_stage": life_stage,
    }


def _geo_assignments() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": photo_id,
                "source_record_hash": f"hash-{photo_id}",
                "geo_cluster_id": "cluster:one" if photo_id == "photo-a" else "no_geo",
            }
            for photo_id in ("photo-a", "photo-b", "photo-c", "photo-d")
        ]
    )


def _query_definitions() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "query_definition_id": "query-1",
                "query_tier": "T1",
                "query_term": "Papilio demoleus",
                "query_field": "tags",
                "query_priority": 10,
            },
            {
                "query_definition_id": "query-2",
                "query_tier": "T5",
                "query_term": "Lime Butterfly",
                "query_field": "text",
                "query_priority": 100,
            },
            {
                "query_definition_id": "query-3",
                "query_tier": "T2",
                "query_term": "Citrus Swallowtail",
                "query_field": "tags",
                "query_priority": 20,
            },
            {
                "query_definition_id": "query-4",
                "query_tier": "T3",
                "query_term": "Papilio demoleus",
                "query_field": "text",
                "query_priority": 30,
            },
        ]
    )


def _object_scores() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _score(
                "score-a-incompatible",
                "photo-a",
                reference_score=1.0,
                margin=0.3,
                scoring_unit_id="unit-a-incompatible",
                route_compatible=False,
            ),
            _score(
                "score-a-low",
                "photo-a",
                reference_score=0.2,
                margin=0.1,
                scoring_unit_id="unit-a-low",
            ),
            _score(
                "score-a",
                "photo-a",
                reference_score=0.9,
                margin=0.2,
                scoring_unit_id="unit-a",
                subject_area=0.4,
            ),
            _score(
                "score-b",
                "photo-b",
                reference_score=0.1,
                margin=-0.2,
                scoring_unit_id="unit-b",
                competitor_key="gbif:polytes",
                competitor_name="Papilio polytes",
                subject_area=0.005,
            ),
            _score(
                "score-c",
                "photo-c",
                reference_score=0.5,
                margin=-0.1,
                scoring_unit_id="unit-c",
                competitor_key="gbif:agamemnon",
                competitor_name="Graphium agamemnon",
                yoloe_route="larval",
                route="larval",
                subject_area=0.2,
                visual_disagreement=0.25,
            ),
        ]
    )


def _score(
    score_id: str,
    photo_id: str,
    *,
    reference_score: float,
    margin: float,
    scoring_unit_id: str,
    competitor_key: str = "gbif:polytes",
    competitor_name: str = "Papilio polytes",
    yoloe_route: str = "adult_field",
    route: str = "adult_field",
    subject_area: float = 0.1,
    visual_disagreement: float = 0.02,
    route_compatible: bool = True,
) -> dict[str, object]:
    return {
        "target_score_id": score_id,
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": f"hash-{photo_id}",
        "scoring_unit_id": scoring_unit_id,
        "route": route,
        "target_reference_centroid_similarity": reference_score,
        "target_competitor_margin": margin,
        "best_competitor_accepted_taxon_key": competitor_key,
        "best_competitor_scientific_name": competitor_name,
        "yoloe_route": yoloe_route,
        "subject_area_ratio": subject_area,
        "visual_input_disagreement": visual_disagreement,
        "route_compatible": route_compatible,
    }


def _competitor_taxa() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "candidate_accepted_taxon_key": "gbif:polytes",
                "genus": "Papilio",
            },
            {
                "candidate_accepted_taxon_key": "gbif:agamemnon",
                "genus": "Graphium",
            },
        ]
    )


def _write_query_state(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE flickr_work_items (
                work_item_id TEXT PRIMARY KEY,
                page INTEGER NOT NULL,
                query_definition_id TEXT,
                trust_tier TEXT,
                term TEXT,
                search_field TEXT,
                query_priority INTEGER
            )
            """
        )
        rows: list[tuple[object, ...]] = []
        for definition in _query_definitions().to_dicts():
            rows.append(
                (
                    f"{definition['query_definition_id']}:page:1",
                    1,
                    definition["query_definition_id"],
                    definition["query_tier"],
                    definition["query_term"],
                    definition["query_field"],
                    definition["query_priority"],
                )
            )
        rows.append(
            (
                "query-1:page:2",
                2,
                "query-1",
                "T1",
                "Papilio demoleus",
                "tags",
                10,
            )
        )
        connection.executemany(
            "INSERT INTO flickr_work_items VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()
