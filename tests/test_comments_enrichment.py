from __future__ import annotations

import json
import sqlite3

from biominer.flickr_comments.comments_enrichment import CommentsEnrichmentState, mine_comment_terms


def test_comments_enrichment_queue_exists(tmp_path) -> None:
    state = CommentsEnrichmentState(tmp_path / "comments.sqlite")

    with sqlite3.connect(state.path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert {"comments_enrichment_queue", "comments_term_observations", "comment_promoted_terms"}.issubset(tables)


def test_obvious_hard_negative_bronze_not_queued_unless_selected_for_qa(tmp_path) -> None:
    state = CommentsEnrichmentState(tmp_path / "comments.sqlite")
    record = {
        "source": "flickr",
        "flickr_photo_id": "1",
        "triage_bin": "bronze",
        "triage_reason": "museum specimen",
    }

    assert state.queue_candidate(record) == 0
    assert state.pending_count() == 0

    assert state.queue_candidate(record, selected_for_qa=True) == 1
    assert state.pending_count() == 1


def test_comment_term_mining_finds_scientific_common_and_life_stage_terms() -> None:
    terms = mine_comment_terms("Papilio demoleus lime butterfly caterpillar")

    assert ("Papilio demoleus", "scientific_name") in {(term.term, term.term_kind) for term in terms}
    assert ("lime butterfly", "common_name") in {(term.term, term.term_kind) for term in terms}
    assert ("caterpillar", "life_stage") in {(term.term, term.term_kind) for term in terms}


def test_comments_derived_terms_are_thresholded_across_photos_and_users(tmp_path) -> None:
    state = CommentsEnrichmentState(tmp_path / "comments.sqlite")
    state.record_comments(
        flickr_photo_id="1",
        comments=[{"author": "same-user", "_content": "Papilio demoleus lime butterfly caterpillar"}],
    )
    state.record_comments(
        flickr_photo_id="2",
        comments=[{"author": "same-user", "_content": "Papilio demoleus lime butterfly caterpillar"}],
    )

    assert state.promote_supported_terms(min_photos=2, min_users=2) == []

    state.record_comments(
        flickr_photo_id="2",
        comments=[{"author": "second-user", "_content": "Papilio demoleus lime butterfly caterpillar"}],
    )
    promoted = state.promote_supported_terms(min_photos=2, min_users=2)

    promoted_keys = {(term.term, term.term_kind) for term in promoted}
    assert ("Papilio demoleus", "scientific_name") in promoted_keys
    assert ("lime butterfly", "common_name") in promoted_keys
    assert ("caterpillar", "life_stage") in promoted_keys


def test_promoted_comment_terms_create_new_work_items(tmp_path) -> None:
    state = CommentsEnrichmentState(tmp_path / "comments.sqlite")
    state.record_comments(flickr_photo_id="1", comments=[{"author": "u1", "_content": "lime butterfly"}])
    state.record_comments(flickr_photo_id="2", comments=[{"author": "u2", "_content": "lime butterfly"}])

    promoted = state.promote_supported_terms(min_photos=2, min_users=2)

    assert len(promoted) == 1
    with sqlite3.connect(state.path) as conn:
        query_payload = conn.execute("SELECT query_json FROM flickr_work_items").fetchone()[0]
    query = json.loads(query_payload)
    assert query["term"] == "lime butterfly"
    assert query["split_reason"] == "comment_promoted_term"
    assert query["lane"] == "count_probe"


def test_process_pending_fetches_only_selected_candidates(tmp_path) -> None:
    state = CommentsEnrichmentState(tmp_path / "comments.sqlite")
    state.queue_candidate({"source": "flickr", "flickr_photo_id": "selected", "triage_bin": "gold"})
    fetched: list[str] = []

    def fake_fetch(photo_id: str) -> list[dict[str, str]]:
        fetched.append(photo_id)
        return [{"author": "u1", "_content": "Papilio demoleus"}]

    result = state.process_pending(fetch_comments=fake_fetch, limit=5)

    assert fetched == ["selected"]
    assert result["comment_records_processed"] == 1
    assert result["term_observations_inserted"] == 1


def test_comments_do_not_override_bioclip_triage_automatically(tmp_path) -> None:
    state = CommentsEnrichmentState(tmp_path / "comments.sqlite")
    state.queue_candidate(
        {
            "source": "flickr",
            "flickr_photo_id": "1",
            "triage_bin": "silver",
            "triage_reason": "bioclip_score_below_threshold",
        }
    )

    state.record_comments(flickr_photo_id="1", comments=[{"author": "u1", "_content": "Papilio demoleus"}])

    with sqlite3.connect(state.path) as conn:
        row = conn.execute("SELECT triage_bin, triage_reason FROM comments_enrichment_queue WHERE flickr_photo_id = '1'").fetchone()
    assert row == ("silver", "bioclip_score_below_threshold")
