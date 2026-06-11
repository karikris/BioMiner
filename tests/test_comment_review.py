from __future__ import annotations

import json
import sqlite3

import polars as pl

from biominer.flickr_comments.comment_review import (
    CommentReviewState,
    comment_review_reasons,
    review_comments_for_record,
)


def _base_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "source": "flickr",
        "source_record_id": "1",
        "source_record_hash": "sha256:1",
        "flickr_photo_id": "1",
        "photo_page_url": "https://www.flickr.com/photos/example/1",
        "image_url": "https://live.staticflickr.com/1_l.jpg",
        "raw_title": "Papilio demoleus",
        "raw_tags": "Papilio demoleus",
        "bioclip_top1_label": "a photo of a butterfly",
        "bioclip_top1_score": 0.92,
        "is_target_positive": True,
        "occurrence_bin": "in_review/no_geo",
        "triage_bin": "in_review/no_geo",
        "image_category": "adult_butterfly",
        "life_stage": "adult_butterfly",
        "date_taken": "2024-01-15",
        "latitude": None,
        "longitude": None,
    }
    record.update(overrides)
    return record


def test_comment_review_tables_are_created(tmp_path) -> None:
    state = CommentReviewState(tmp_path / "comments.sqlite")

    with sqlite3.connect(state.path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert {"comment_review_queue", "comment_review_results", "comment_derived_terms", "missing_data_requests"}.issubset(tables)


def test_comment_review_queue_only_selected_records_and_skips_duplicates(tmp_path) -> None:
    state = CommentReviewState(tmp_path / "comments.sqlite")

    assert state.enqueue_record(_base_record()) == 1
    assert state.enqueue_record(_base_record()) == 0
    assert state.enqueue_record(_base_record(classification_status="skipped_existing", source_record_hash="sha256:duplicate")) == 0
    assert state.enqueue_record(_base_record(source_record_hash="sha256:gold", occurrence_bin="gold", triage_bin="gold", latitude=-27.0, longitude=153.0)) == 0
    assert state.pending_count() == 1


def test_comment_review_reasons_include_conflict_missing_data_unknown_and_low_score() -> None:
    record = _base_record(
        raw_tags="Papilio demoleus",
        bioclip_top1_label="a photo of a moth",
        bioclip_top1_score=0.31,
        image_category="unknown",
        life_stage="unknown",
        date_taken=None,
    )

    reasons = comment_review_reasons(record)

    assert "species_conflict" in reasons
    assert "missing_geo" in reasons
    assert "missing_event_date" in reasons
    assert "unknown_image_category" in reasons
    assert "unknown_life_stage" in reasons
    assert "low_bioclip_score" in reasons


def test_place_name_comment_creates_missing_geo_request_not_gold(tmp_path) -> None:
    state = CommentReviewState(tmp_path / "comments.sqlite")
    state.enqueue_record(_base_record())

    result = state.process_pending(
        fetch_comments=lambda photo_id: [{"author": "u1", "_content": "Confirmed Papilio demoleus, seen in Brisbane"}],
        max_api_calls=300,
    )

    assert result["missing_geo_requests_created"] == 1
    assert result["records_moved_to_gold"] == 0
    with sqlite3.connect(state.path) as conn:
        review = conn.execute("SELECT comment_review_decision FROM comment_review_results").fetchone()[0]
        request = conn.execute("SELECT request_type, evidence_text FROM missing_data_requests").fetchone()
    assert review == "request_missing_geo"
    assert request[0] == "missing_geo"
    assert "Brisbane" in request[1]


def test_structured_comment_geo_and_species_support_can_move_to_gold(tmp_path) -> None:
    state = CommentReviewState(tmp_path / "comments.sqlite")
    state.enqueue_record(_base_record(date_taken="2024-01-15"))

    result = state.process_pending(
        fetch_comments=lambda photo_id: [{"author": "u1", "_content": "Confirmed Papilio demoleus at -27.4698, 153.0251"}],
        max_api_calls=300,
    )

    assert result["records_moved_to_gold"] == 1
    with sqlite3.connect(state.path) as conn:
        review = conn.execute("SELECT comment_review_decision, geo_evidence_from_comments FROM comment_review_results").fetchone()
    assert review == ("move_to_gold", "-27.4698,153.0251")


def test_comment_review_decision_keeps_original_bioclip_result_as_extra_layer() -> None:
    result = review_comments_for_record(
        _base_record(raw_tags="Papilio demoleus", bioclip_top1_label="a photo of a moth", bioclip_tag_conflict=True, latitude=-27.0, longitude=153.0),
        [{"author": "u1", "_content": "confirmed Papilio demoleus"}],
    )

    assert result.bioclip_species_candidate == "non_target_insect"
    assert result.comment_species_candidate == "Papilio demoleus"
    assert result.comment_resolves_conflict is True
    assert result.comment_review_decision == "move_to_gold"


def test_apply_comment_review_decisions_updates_only_move_to_gold_records(tmp_path) -> None:
    state = CommentReviewState(tmp_path / "comments.sqlite")
    record = _base_record(source_record_hash="sha256:apply")
    state.enqueue_record(record)
    state.process_pending(
        fetch_comments=lambda photo_id: [{"author": "u1", "_content": "Confirmed Papilio demoleus at -27.4698, 153.0251"}],
        max_api_calls=300,
    )

    rows = state.apply_decisions_to_records([record])

    assert rows[0]["occurrence_bin"] == "gold"
    assert rows[0]["triage_bin"] == "gold"
    assert rows[0]["comment_review_decision"] == "move_to_gold"


def test_comment_review_cli_commands_run_once_without_network(tmp_path, capsys) -> None:
    from biominer.cli import build_parser, run

    input_path = tmp_path / "triage.parquet"
    output_path = tmp_path / "reviewed.parquet"
    state_db = tmp_path / "comments.sqlite"
    pl.DataFrame([_base_record()]).write_parquet(input_path)

    parser = build_parser()
    assert run(parser.parse_args(["build-comment-review-queue", "--input", str(input_path), "--state-db", str(state_db)])) == 0
    queue_payload = json.loads(capsys.readouterr().out)
    assert queue_payload["comment_review_queue_created"] == 1

    assert run(parser.parse_args(["apply-comment-review-decisions", "--input", str(input_path), "--output", str(output_path), "--state-db", str(state_db)])) == 0
    apply_payload = json.loads(capsys.readouterr().out)
    assert apply_payload["rows"] == 1
    assert output_path.exists()

    assert run(parser.parse_args(["review-comments-once", "--state-db", str(state_db), "--max-api-calls", "1"])) == 2
    error_payload = json.loads(capsys.readouterr().out)
    assert "Flickr API key is required" in error_payload["error"]
